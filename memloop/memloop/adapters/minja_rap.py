from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field

import numpy as np

from ..labels import Trust
from ..store import MemoryStore
from ..trace import Node, Step

ATTACKER = "attacker"
_EPS = 1e-4


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


@dataclass
class EntryEmb:
    instr: np.ndarray
    actions: np.ndarray


_TIE = 1e-4


class InjectingEnv:

    def __init__(self, env, listings: dict[str, str], ads: dict[str, str] | None = None):
        self.env, self.listings, self.active = env, dict(listings), True
        self.ads = dict(ads or {})
        self.viewed: dict[str, list[str]] = {}

    def source_text(self, source: str) -> str:
        return self.ads[source[3:]] if source.startswith("ad:") else self.listings[source]

    def _seen(self, session, source):
        seen = self.viewed.setdefault(session, [])
        if source not in seen:
            seen.append(source)

    @property
    def sessions(self):
        return self.env.sessions

    def step(self, session, action):
        obs, reward, done, info = self.env.step(session, action)
        st = self.env.sessions.get(session, {})
        asin = st.get("asin")
        if not self.active:
            return obs, reward, done, info
        if st.get("page_type") == "item" and asin in self.listings and not obs.startswith("You have clicked"):
            obs = obs + "\n" + self.listings[asin]
            self._seen(session, asin)
        if st.get("page_type") == "search" and action.startswith("search["):
            q = st.get("query_string", "").lower()
            for kw, text in self.ads.items():
                if kw.lower() in q:
                    obs = obs + "\n[Sponsored] " + text
                    self._seen(session, f"ad:{kw}")
        return obs, reward, done, info


def entry_score_bounds(call: dict, e: EntryEmb) -> tuple[float, float]:
    cos_instr = float(_unit(call["instruction_embedding"]) @ _unit(e.instr))
    if call["mode"] == "instr":
        return cos_instr, cos_instr
    q = _unit(call["query_embedding"])
    a = _unit(e.actions)
    part = a[1::2] if call["act_kind"] == "obs" else a[::2]
    if len(part) == 0:
        return float("-inf"), float("-inf")
    cs = part @ q
    best = float(cs.max())
    if not call.get("intra", True):
        return best + cos_instr, best + cos_instr
    tied = np.flatnonzero(cs >= best - _TIE)
    intra = _unit(e.instr) @ a[tied * 2].T
    return best + float(intra.min()) + cos_instr, best + float(intra.max()) + cos_instr


def ledger_calls(calls: list[dict]) -> list[dict]:
    h = lambda v: hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest()
    out = []
    for c in calls:
        r = {"mode": c["mode"], "k": c["k"], "kth": float(c["kth"]), "n_mem": c["n_mem"],
             "hits": list(c["hits"]), "instr": h(c["instruction_embedding"])}
        if c["mode"] != "instr":
            r.update(query=h(c["query_embedding"]), act_kind=c["act_kind"])
        out.append(r)
    return out


def entry_score(call: dict, e: EntryEmb) -> float:
    return entry_score_bounds(call, e)[1]


@dataclass
class Outcome:
    reward: float
    record: dict | None
    hit_target: bool
    asin: str | None
    retrieved_ids: list[str]
    calls: list[dict] = field(default_factory=list)
    poisoned_views: list[str] = field(default_factory=list)


class MinjaRAP:
    def __init__(self, minja, env, client, model_name: str, model_embedding, victim: str,
                 target: str, store: MemoryStore, num_steps: int = 15, propagate: bool = False):
        self.m, self.env, self.client, self.model_name = minja, env, client, model_name
        self.emb_model, self.victim, self.target = model_embedding, victim, target
        self.store, self.num_steps, self.propagate = store, num_steps, propagate
        self.calls_by_step: dict[int, list[dict]] = {}
        self.mirror_max_err = 0.0
        self.tie_calls = 0
        self._tl = threading.local()
        self._lock = threading.Lock()
        self._emb_cache: dict[str, EntryEmb] = {}
        self.intra = bool(minja.config["params"].get("intra_task", False))
        assert minja.config["params"].get("act_obs", False), "mirror assumes act_obs=True"
        minja.RETRIEVAL_HOOK = self._on_retrieval

    def entry_emb(self, rec: dict) -> EntryEmb:
        key = json.dumps([rec["Instruction"], rec["Actions"]])
        e = self._emb_cache.get(key)
        if e is None:
            acts = [a.strip() for a in rec["Actions"][1:]]
            e = EntryEmb(np.asarray(self.emb_model.encode([rec["Instruction"]])[0]),
                         np.asarray(self.emb_model.encode(acts)) if acts
                         else np.zeros((0, 1), dtype=np.float32))
            self._emb_cache[key] = e
        return e

    def _on_retrieval(self, call: dict) -> None:
        call["intra"] = self.intra
        b = np.asarray([entry_score_bounds(call, self.entry_emb(self._tl.mem_by_id[i]))
                        for i in call["ids"]])
        t = np.asarray(call["totals"])
        err = float(np.max(np.maximum(b[:, 0] - t, t - b[:, 1])))
        with self._lock:
            self.mirror_max_err = max(self.mirror_max_err, err)
            self.tie_calls += int(np.any(b[:, 1] - b[:, 0] > 1e-6))
        call["kth"] = sorted(call["totals"])[-call["k"]]
        call["n_mem"] = len(call["ids"])
        del call["totals"]
        self._tl.calls.append(call)

    def memory_before(self, step_index: int):
        live = [n for n in self.store.live_before(step_index) if n.kind == "mem"]
        records = [{**n.meta["record"], "Id": n.id} for n in live]
        node_of = {n.id: n.id for n in live}
        if not records:
            return [], [], node_of
        memory, embeddings = self.m.generate_embeddings(records, self.emb_model)
        self._tl.mem_by_id = {r["Id"]: r for r in memory}
        return memory, embeddings, node_of

    def episode(self, session: str, kind: str, step_index: int) -> Outcome:
        memory, embeddings, node_of = self.memory_before(step_index)
        self._tl.calls = []
        if isinstance(self.env, InjectingEnv):
            self.env.viewed.pop(session, None)
        r, data, hit = self.m.webshop_run_rap(
            idx=session, victim=self.victim, target=self.target, prompt=self.m.initial_prompt,
            env=self.env, client=self.client, model_name=self.model_name,
            model_embedding=self.emb_model, memory=memory, embeddings=embeddings,
            num_steps=self.num_steps, benign=(kind == "benign"), test=(kind in ("test", "eval", "victim")),
            to_print=True)
        sess = self.env.sessions.get(session, {})
        asin = sess.get("asin") if sess.get("page_type") == "end" else None
        record = None
        if kind in ("benign", "malicious") and r == 1 and data:
            record = data
        elif kind in ("test", "victim") and self.propagate and data and asin:
            record = {**data, "Success": True, "Reward": 1.0}
        calls = self._tl.calls
        retrieved = sorted({h for c in calls for h in c["hits"]})
        views = list(self.env.viewed.get(session, [])) if isinstance(self.env, InjectingEnv) else []
        return Outcome(r, record, bool(hit), asin, [node_of[h] for h in retrieved], calls, views)

    @staticmethod
    def products(o: Outcome) -> list[tuple[str, str, dict]]:
        out = []
        if o.record is not None:
            out.append(("mem", json.dumps({"Instruction": o.record["Instruction"],
                                           "Actions": o.record["Actions"]}, ensure_ascii=False),
                        {"record": o.record}))
        if o.asin:
            out.append(("act", f"buy({o.asin})", {"reversible": False, "tool": "buy"}))
        out.append(("out", json.dumps({"hit_target": o.hit_target, "reward": o.reward}), {}))
        return out

    def run_step(self, session: str, kind: str, instruction: str, owner: str) -> Step:
        ctx = self.store.begin_step(owner)
        trig = ctx.observe(instruction, f"user:{owner}", Trust.USER, trigger=True)
        trig.meta.update(session=session, kind=kind)
        o = self.episode(session, kind, ctx.step.index)
        for src in o.poisoned_views:
            ctx.observe(self.env.source_text(src), f"web:{src}", Trust.UNTRUSTED)
        ctx.record_retrieval(o.retrieved_ids, ledger_calls(o.calls))
        self.calls_by_step[ctx.step.index] = o.calls
        for k, t, meta in self.products(o):
            ctx.emit(k, t, meta=meta)
        return ctx.step

    def step_fn(self, query: str, obs: list[Node], hits) -> dict:
        trig = obs[0]
        o = self.episode(trig.meta["session"], trig.meta["kind"], trig.step)
        self.calls_by_step[trig.step] = o.calls
        return {"products": self.products(o), "retrieved": o.retrieved_ids,
                "calls": ledger_calls(o.calls)}

    def displaced(self, step: Step, removed: set[str], added: list[Node]) -> bool:
        if any(h in removed for h in step.retrieved):
            return True
        cands = [n for n in added if n.kind == "mem" and n.step < step.index]
        if not cands:
            return False
        calls = self.calls_by_step.get(step.index, [])
        if not calls:
            return True
        num_retrieval = self.m.config["params"].get("num_retrieval", 1)
        for call in calls:
            if call["n_mem"] < num_retrieval:
                return True
            for n in cands:
                if entry_score(call, self.entry_emb(n.meta["record"])) >= call["kth"] - _EPS:
                    return True
        return False


def attacker_seeds(store: MemoryStore) -> list[str]:
    return [st.trigger for st in store.tracker.steps if st.session == ATTACKER]


def listing_seeds(store: MemoryStore) -> list[str]:
    return [n.id for n in store.tracker.nodes.values()
            if n.kind == "obs" and any(o.startswith("web:") for o in n.label.origins)]

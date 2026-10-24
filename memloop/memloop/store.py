from __future__ import annotations

import hashlib
import re
from typing import Callable

import numpy as np

from .labels import Label, Policy, Trust
from .trace import Node, Step, Tracker

Embedder = Callable[[list[str]], np.ndarray]


def vec_hash(v: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest()


def index_digest(pairs) -> str:
    lines = sorted(f"{i}:{h}" for i, h in pairs)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class HashEmbedder:

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class STEmbedder:
    def __init__(self, name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)


class MemoryStore:
    def __init__(self, embedder: Embedder, tracker: Tracker | None = None,
                 policy: Policy | None = None):
        self.embedder = embedder
        self.tracker = tracker if tracker is not None else Tracker()
        self.policy = policy
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._mat: np.ndarray | None = None
        self._qcache: dict[str, np.ndarray] = {}
        self._retired: dict[str, tuple[int, str]] = {}

    def __setstate__(self, state):
        state.setdefault("_retired", {})
        self.__dict__.update(state)

    def _insert(self, nid: str, vec: np.ndarray) -> None:
        self._ids.append(nid)
        self._vecs.append(np.asarray(vec, dtype=np.float32))
        self._mat = None

    def remove(self, nid: str) -> None:
        i = self._ids.index(nid)
        del self._ids[i]
        del self._vecs[i]
        self._retired.pop(nid, None)
        self._mat = None

    def has(self, nid: str) -> bool:
        return nid in self._ids

    def retire(self, nid: str, step: int, by: str) -> None:
        self._retired[nid] = (step, by)
        self.tracker.log({"t": "retire", "id": nid, "by": by, "step": step})

    def restore(self, nid: str, run: str | None = None) -> None:
        del self._retired[nid]
        self.tracker.log({"t": "restore", "id": nid, **({"run": run} if run else {})})

    def retired_by(self, nid: str) -> str | None:
        r = self._retired.get(nid)
        return r[1] if r else None

    def _visible(self, nid: str, before: int | None) -> bool:
        r = self._retired.get(nid)
        return r is None or (before is not None and r[0] >= before)

    def active_ids(self) -> list[str]:
        return [i for i in self._ids if i not in self._retired]

    def live_before(self, step_index: int) -> list[Node]:
        nodes = self.tracker.nodes
        return sorted((nodes[i] for i in self._ids
                       if nodes[i].step < step_index and self._visible(i, step_index)),
                      key=lambda n: n.order)

    def index_digest(self) -> str:
        return index_digest((i, vec_hash(v)) for i, v in zip(self._ids, self._vecs)
                            if i not in self._retired)

    def query_vec(self, query: str) -> np.ndarray:
        v = self._qcache.get(query)
        if v is None:
            v = self._qcache[query] = self.embedder([query])[0]
        return v

    def score(self, query: str, nid: str) -> float:
        return float(np.round(self._vecs[self._ids.index(nid)] @ self.query_vec(query), 9))

    def search(self, query: str, k: int, before: int | None = None,
               min_trust: Trust | None = None) -> list[Node]:
        if not self._ids or k <= 0:
            return []
        if self._mat is None:
            self._mat = np.stack(self._vecs)
        scores = np.round(self._mat @ self.query_vec(query), 9)
        nodes = self.tracker.nodes
        cands = []
        for nid, s in zip(self._ids, scores):
            n = nodes[nid]
            if before is not None and n.step >= before:
                continue
            if not self._visible(nid, before):
                continue
            if min_trust is not None and n.label.trust < min_trust:
                continue
            cands.append((-float(s), n.order, n))
        cands.sort(key=lambda c: (c[0], c[1]))
        return [c[2] for c in cands[:k]]

    def begin_step(self, session: str) -> "StepContext":
        return StepContext(self, self.tracker.new_step(session))


class StepContext:

    def __init__(self, store: MemoryStore, step: Step, replay: bool = False):
        self.store = store
        self.step = step
        self.replay = replay

    def observe(self, text: str, origin: str, trust: Trust, trigger: bool = False) -> Node:
        tr = self.store.tracker
        node = tr.add_node("obs", self.step, text, label=Label.source(origin, trust),
                           extra={"origin": origin, "trigger": trigger})
        self.step.observations.append(node.id)
        if trigger:
            self.step.trigger = node.id
        return node

    def retrieve(self, query: str, k: int, min_trust: Trust | None = None) -> list[Node]:
        st = self.step
        if st.query is not None or st.products:
            raise RuntimeError("one retrieval per step, before any product is emitted")
        hits = self.store.search(query, k, before=st.index, min_trust=min_trust)
        st.query, st.k, st.min_trust = query, k, min_trust
        st.retrieved = [h.id for h in hits]
        self.store.tracker.log({"t": "read", "step": st.index, "k": k,
                                "q": hashlib.sha256(query.encode()).hexdigest(),
                                "hits": st.retrieved})
        return hits

    def record_retrieval(self, ids: list[str], calls: list[dict] | None = None) -> None:
        st = self.step
        if st.query is not None or st.products:
            raise RuntimeError("one retrieval record per step, before any product is emitted")
        st.retrieved = list(ids)
        self.store.tracker.log({"t": "read", "step": st.index, "external": True,
                                "hits": st.retrieved, **({"calls": calls} if calls else {})})

    def emit(self, kind: str, text: str, parents=None, meta: dict | None = None) -> Node | None:
        store, tr, st = self.store, self.store.tracker, self.step
        parents = tuple(st.context if parents is None else parents)
        if not parents:
            raise ValueError("products need at least one parent")
        meta = dict(meta or {})
        if kind == "mem" and meta.get("supersedes") is not None:
            old = meta["supersedes"]
            if old not in parents or not store.has(old) or not store._visible(old, st.index):
                raise ValueError("a rewrite must derive from the live version it replaces")
        extra: dict = {}
        vec = None
        if kind == "act":
            tool = meta.setdefault("tool", text.split("(", 1)[0])
            meta.setdefault("reversible", True)
            label = Label.join(*(tr.nodes[p].label for p in parents))
            if store.policy is not None and label.trust < store.policy.required(tool):
                tr.log({"t": "blocked", "step": st.index, "tool": tool, "trust": int(label.trust)})
                return None
            extra = {"tool": tool, "rev": meta["reversible"]}
        elif kind == "mem":
            vec = store.embedder([text])[0]
            extra = {"vec": vec_hash(vec)}
        node = tr.add_node(kind, st, text, parents, meta=meta, extra=extra)
        if vec is not None:
            store._insert(node.id, vec)
            old = meta.get("supersedes")
            if old is not None:
                store.retire(old, st.index, node.id)
        st.products.append(node.id)
        return node

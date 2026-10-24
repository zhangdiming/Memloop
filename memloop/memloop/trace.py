from __future__ import annotations

import hashlib
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .labels import Label
from .ledger import Ledger

KINDS = ("obs", "mem", "act", "out")


def commit(salt: bytes, text: str) -> str:
    return hashlib.sha256(salt + text.encode()).hexdigest()


@dataclass
class Node:
    id: str
    kind: str
    step: int
    pos: int
    label: Label
    parents: tuple[str, ...]
    text: str | None
    salt: bytes | None
    meta: dict = field(default_factory=dict)
    status: str = "active"

    @property
    def order(self) -> tuple[int, int]:
        return (self.step, self.pos)


@dataclass
class Step:
    index: int
    session: str
    trigger: str | None = None
    query: str | None = None
    k: int = 0
    min_trust: int | None = None
    observations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    dropped: bool = False
    _pos: int = 0

    @property
    def context(self) -> list[str]:
        return self.observations + self.retrieved

    def next_pos(self) -> int:
        self._pos += 1
        return self._pos - 1


class Tracker:
    def __init__(self, ledger: Ledger | None = None):
        self.ledger = ledger if ledger is not None else Ledger()
        self.nodes: dict[str, Node] = {}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.steps: list[Step] = []
        self._seq = 0

    def log(self, record: dict) -> int:
        return self.ledger.append(record)

    def new_step(self, session: str) -> Step:
        step = Step(index=len(self.steps), session=session)
        self.steps.append(step)
        self.log({"t": "step", "step": step.index, "session": session})
        return step

    def add_node(self, kind: str, step: Step, text: str, parents: tuple[str, ...] = (),
                 label: Label | None = None, meta: dict | None = None,
                 extra: dict | None = None) -> Node:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind}")
        if parents:
            if label is not None:
                raise ValueError("derived nodes take their label from their parents")
            label = Label.join(*(self.nodes[p].label for p in parents))
        elif label is None:
            raise ValueError("source nodes need an explicit label")
        nid = f"{kind}:{self._seq}"
        self._seq += 1
        salt = os.urandom(16)
        node = Node(nid, kind, step.index, step.next_pos(), label, tuple(parents), text, salt,
                    dict(meta or {}))
        self.nodes[nid] = node
        for p in parents:
            self.children[p].append(nid)
        self.log({"t": "node", "id": nid, "kind": kind, "step": step.index, "pos": node.pos,
                  "parents": list(parents), "trust": int(label.trust),
                  "origins": sorted(label.origins), "c": commit(salt, text),
                  **(extra or {})})
        return node

    def descendants(self, seeds) -> set[str]:
        seen, queue = set(seeds), deque(seeds)
        while queue:
            for c in self.children.get(queue.popleft(), ()):
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
        return seen

    def ancestors(self, ids) -> set[str]:
        seen, queue = set(ids), deque(ids)
        while queue:
            for p in self.nodes[queue.popleft()].parents:
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        return seen

    def sources(self, ids) -> list[Node]:
        return sorted((self.nodes[a] for a in self.ancestors(ids) if not self.nodes[a].parents),
                      key=lambda n: n.order)

    def erase(self, nid: str) -> None:
        node = self.nodes[nid]
        node.text = None
        node.salt = None
        node.status = "deleted"

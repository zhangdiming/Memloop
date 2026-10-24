from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Trust(IntEnum):
    UNTRUSTED = 0
    TOOL = 1
    USER = 2
    SYSTEM = 3


@dataclass(frozen=True)
class Label:
    trust: Trust
    origins: frozenset[str]

    @staticmethod
    def source(origin: str, trust: Trust) -> "Label":
        return Label(Trust(trust), frozenset([origin]))

    @staticmethod
    def join(*labels: "Label") -> "Label":
        if not labels:
            raise ValueError("join of no labels is undefined; derived nodes need parents")
        return Label(min(l.trust for l in labels), frozenset().union(*(l.origins for l in labels)))


@dataclass
class Policy:
    act_min_trust: dict[str, Trust]
    default: Trust = Trust.UNTRUSTED

    def required(self, tool: str) -> Trust:
        return self.act_min_trust.get(tool, self.default)

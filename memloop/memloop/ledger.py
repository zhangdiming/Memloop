from __future__ import annotations

import hashlib
import hmac
import json


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def leaf_hash(data: bytes) -> bytes:
    return sha256(b"\x00" + data)


def node_hash(left: bytes, right: bytes) -> bytes:
    return sha256(b"\x01" + left + right)


def _split(n: int) -> int:
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


class Ledger:
    def __init__(self, key: bytes | None = None):
        self._records: list[dict] = []
        self._leaves: list[bytes] = []
        self._cache: dict[tuple[int, int], bytes] = {}
        self._key = key

    def __len__(self) -> int:
        return len(self._leaves)

    def append(self, record: dict) -> int:
        self._records.append(record)
        self._leaves.append(leaf_hash(canonical(record)))
        return len(self._leaves) - 1

    def record(self, index: int) -> dict:
        return self._records[index]

    def records(self, lo: int = 0, hi: int | None = None) -> list[dict]:
        return self._records[lo:hi]

    def _mth(self, lo: int, hi: int) -> bytes:
        n = hi - lo
        if n == 0:
            return sha256(b"")
        if n == 1:
            return self._leaves[lo]
        cached = self._cache.get((lo, hi))
        if cached is None:
            k = _split(n)
            cached = node_hash(self._mth(lo, lo + k), self._mth(lo + k, hi))
            self._cache[(lo, hi)] = cached
        return cached

    def root(self, size: int | None = None) -> bytes:
        size = len(self) if size is None else size
        return self._mth(0, size)

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        size = len(self) if size is None else size
        if not 0 <= index < size <= len(self):
            raise IndexError(f"leaf {index} not in tree of size {size}")

        def path(m: int, lo: int, hi: int) -> list[bytes]:
            n = hi - lo
            if n == 1:
                return []
            k = _split(n)
            if m < k:
                return path(m, lo, lo + k) + [self._mth(lo + k, hi)]
            return path(m - k, lo + k, hi) + [self._mth(lo, lo + k)]

        return path(index, 0, size)

    def consistency_proof(self, old_size: int, new_size: int | None = None) -> list[bytes]:
        new_size = len(self) if new_size is None else new_size
        if not 0 <= old_size <= new_size <= len(self):
            raise ValueError(f"bad sizes {old_size} -> {new_size}")
        if old_size in (0, new_size):
            return []

        def subproof(m: int, lo: int, hi: int, complete: bool) -> list[bytes]:
            n = hi - lo
            if m == n:
                return [] if complete else [self._mth(lo, hi)]
            k = _split(n)
            if m <= k:
                return subproof(m, lo, lo + k, complete) + [self._mth(lo + k, hi)]
            return subproof(m - k, lo + k, hi, False) + [self._mth(lo, lo + k)]

        return subproof(old_size, 0, new_size, True)

    def signed_head(self, size: int | None = None) -> dict:
        size = len(self) if size is None else size
        root = self.root(size).hex()
        mac = None
        if self._key is not None:
            mac = hmac.new(self._key, f"{size}:{root}".encode(), hashlib.sha256).hexdigest()
        return {"size": size, "root": root, "mac": mac}


def verify_head(head: dict, key: bytes) -> bool:
    expected = hmac.new(key, f"{head['size']}:{head['root']}".encode(), hashlib.sha256).hexdigest()
    return head["mac"] is not None and hmac.compare_digest(expected, head["mac"])


def verify_inclusion(leaf: bytes, index: int, size: int, proof: list[bytes], root: bytes) -> bool:
    if index >= size:
        return False
    fn, sn, r = index, size - 1, leaf
    for p in proof:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            r = node_hash(p, r)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def verify_consistency(old_size: int, new_size: int, old_root: bytes, new_root: bytes,
                       proof: list[bytes]) -> bool:
    if old_size > new_size:
        return False
    if old_size == new_size:
        return not proof and old_root == new_root
    if old_size == 0:
        return not proof
    if not proof:
        return False
    if old_size & (old_size - 1) == 0:
        proof = [old_root] + proof
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = proof[0]
    for c in proof[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            while not fn & 1 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1
    return fr == old_root and sr == new_root and sn == 0

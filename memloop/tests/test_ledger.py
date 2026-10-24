from memloop.ledger import (Ledger, canonical, leaf_hash, verify_consistency, verify_head,
                            verify_inclusion)

N = 40


def make(n, key=None):
    led = Ledger(key=key)
    for i in range(n):
        led.append({"t": "x", "i": i})
    return led


def test_inclusion_all_positions():
    led = make(N)
    for size in range(1, N + 1):
        root = led.root(size)
        for i in range(size):
            leaf = leaf_hash(canonical(led.record(i)))
            proof = led.inclusion_proof(i, size)
            assert verify_inclusion(leaf, i, size, proof, root)
            assert not verify_inclusion(leaf, (i + 1) % size, size, proof, root) or size == 1
            if proof:
                bad = [bytes(32)] + proof[1:]
                assert not verify_inclusion(leaf, i, size, bad, root)


def test_consistency_all_pairs():
    led = make(N)
    for m in range(0, N + 1):
        for n in range(m, N + 1):
            proof = led.consistency_proof(m, n)
            assert verify_consistency(m, n, led.root(m), led.root(n), proof), (m, n)
            if 0 < m < n:
                assert not verify_consistency(m, n, led.root(m), led.root(n - 1) if n - 1 > m
                                              else bytes(32), proof)


def test_rewritten_history_breaks_consistency():
    a, b = make(20), make(20)
    b._records[3] = {"t": "x", "i": "forged"}
    b._leaves[3] = leaf_hash(canonical(b._records[3]))
    b._cache.clear()
    for i in range(20, 30):
        a.append({"t": "x", "i": i})
        b.append({"t": "x", "i": i})
    assert not verify_consistency(20, 30, a.root(20), b.root(30), b.consistency_proof(20, 30))


def test_signed_head():
    led = make(5, key=b"k")
    head = led.signed_head()
    assert verify_head(head, b"k")
    assert not verify_head(head, b"other")
    assert not verify_head({**head, "size": 4}, b"k")

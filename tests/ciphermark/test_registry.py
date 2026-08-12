"""Tests du registre de tracabilite (nonce -> parite RS)."""

from distseal.ciphermark.registry import (
    NonceReuseError,
    TraceRegistry,
)


def test_put_get_roundtrip():
    with TraceRegistry() as reg:
        pi = b"\xab" * 32
        reg.put(nonce=7, parity=pi, n_bits=256, rs_nsym=32, session="s1")
        entry = reg.get(7)
        assert entry is not None
        assert entry.parity == pi
        assert entry.n_bits == 256
        assert entry.rs_nsym == 32
        assert entry.session == "s1"
        assert reg.parity_for(7) == pi


def test_nonce_inconnu():
    with TraceRegistry() as reg:
        assert reg.get(999) is None
        try:
            reg.parity_for(999)
        except KeyError:
            pass
        else:
            raise AssertionError("parity_for aurait du lever KeyError")


def test_rejet_du_nonce_reutilise():
    """Le coeur de la discipline OTP : jamais deux fois le meme nonce."""
    with TraceRegistry() as reg:
        reg.put(nonce=1, parity=b"\x00" * 32, n_bits=256, rs_nsym=32)
        try:
            reg.put(nonce=1, parity=b"\xff" * 32, n_bits=256, rs_nsym=32)
        except NonceReuseError:
            pass
        else:
            raise AssertionError("la reutilisation de nonce aurait du lever")
        # la premiere entree n'a pas ete ecrasee
        assert reg.parity_for(1) == b"\x00" * 32


def test_overwrite_explicite():
    with TraceRegistry() as reg:
        reg.put(nonce=1, parity=b"\x00" * 32, n_bits=256, rs_nsym=32)
        reg.put(nonce=1, parity=b"\xff" * 32, n_bits=256, rs_nsym=32,
                overwrite=True)
        assert reg.parity_for(1) == b"\xff" * 32


def test_next_free_nonce():
    with TraceRegistry() as reg:
        assert reg.next_free_nonce() == 0
        reg.put(nonce=0, parity=b"\x00" * 32, n_bits=256, rs_nsym=32)
        reg.put(nonce=5, parity=b"\x00" * 32, n_bits=256, rs_nsym=32)
        assert reg.next_free_nonce() == 6


def test_len_contains_iter():
    with TraceRegistry() as reg:
        for i in range(3):
            reg.put(nonce=i, parity=bytes([i]) * 32, n_bits=256, rs_nsym=32)
        assert len(reg) == 3
        assert 1 in reg
        assert 42 not in reg
        assert [e.nonce for e in reg] == [0, 1, 2]


if __name__ == "__main__":
    test_put_get_roundtrip()
    test_nonce_inconnu()
    test_rejet_du_nonce_reutilise()
    test_overwrite_explicite()
    test_next_free_nonce()
    test_len_contains_iter()
    print("[registry] OK")

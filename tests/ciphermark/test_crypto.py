"""Sanity-checks crypto. On evite pytest pour rester independant."""

import numpy as np

from distseal.crystal import crypto


def test_hmac_deterministe():
    k = b"\x00" * 32
    a = crypto.hmac_sha256(k, b"hello")
    b = crypto.hmac_sha256(k, b"hello")
    assert a == b
    assert len(a) == 32


def test_prg_reproductible():
    s = b"\x42" * 32
    n = crypto.make_nonce(7)
    a = crypto.prg(s, n, 64)
    b = crypto.prg(s, n, 64)
    assert a == b
    # change le nonce -> change la sortie
    c = crypto.prg(s, crypto.make_nonce(8), 64)
    assert a != c


def test_xor_otp_roundtrip():
    m = b"this is a tiny secret payload!!"
    k = crypto.random_seed(len(m))
    # OTP info-theoretique (ici k a la meme taille que m)
    c = crypto.xor_bytes(m, k)
    m2 = crypto.xor_bytes(c, k)
    assert m2 == m


def test_bits_gaussian_roundtrip():
    bits = np.random.randint(0, 2, 256, dtype=np.uint8)
    g = crypto.bits_to_gaussian(bits, block_size=16)
    assert g.shape == (256 // 16,)
    # ordres de grandeur attendus N(0, 1)
    assert -5.0 < float(g.mean()) < 5.0
    assert 0.1 < float(g.std()) < 3.0

    back = crypto.gaussian_to_bits(g, block_size=16)
    assert back.shape == bits.shape


def test_hkdf_expand_len():
    prk = crypto.hmac_sha256(b"\x00" * 32, b"seed")
    out = crypto.hkdf_expand(prk, b"info", 100)
    assert len(out) == 100


if __name__ == "__main__":
    test_hmac_deterministe()
    test_prg_reproductible()
    test_xor_otp_roundtrip()
    test_bits_gaussian_roundtrip()
    test_hkdf_expand_len()
    print("[crypto] tous les tests OK")

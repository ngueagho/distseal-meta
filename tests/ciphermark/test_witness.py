import numpy as np

from distseal.ciphermark import crypto
from distseal.ciphermark.witness import WitnessConfig, WitnessField


def _make_wf(n_bits=256):
    s = b"\x11" * 32
    k = b"\x22" * 32
    return WitnessField(s, k, WitnessConfig(n_bits=n_bits))


def test_omega_correct_verification():
    wf = _make_wf()
    h = b"\xab" * 32
    omega = wf.build_omega(h, image_id=1)
    res = wf.verify(omega, h, image_id=1)
    assert res.distance == 0
    assert res.ber == 0.0


def test_wrong_nonce_breaks():
    wf = _make_wf()
    h = b"\xab" * 32
    omega = wf.build_omega(h, image_id=1)
    res = wf.verify(omega, h, image_id=2)
    # decalage de keystream => moitie des bits errones en esperance
    assert res.distance > 50


def test_modified_hash_breaks():
    wf = _make_wf()
    h1 = b"\xab" * 32
    h2 = b"\xcd" * 32
    omega = wf.build_omega(h1, image_id=42)
    res = wf.verify(omega, h2, image_id=42)
    assert res.distance > 50  # HMAC change completement


def test_noisy_omega_partial():
    wf = _make_wf()
    h = b"\xab" * 32
    omega = wf.build_omega(h, image_id=7)
    # on flip 10 bits sur 256
    noisy = omega.copy()
    flip_idx = np.random.choice(omega.size, 10, replace=False)
    noisy[flip_idx] ^= 1
    res = wf.verify(noisy, h, image_id=7)
    assert res.distance == 10


if __name__ == "__main__":
    test_omega_correct_verification()
    test_wrong_nonce_breaks()
    test_modified_hash_breaks()
    test_noisy_omega_partial()
    print("[witness] OK")

"""
Tests du canal h : parite Reed-Solomon separee et correction.

Ces tests verifient la propriete qui fait vivre ou mourir CipherMark : une
derive du hash inferieure a la capacite RS doit etre corrigee exactement,
et au-dela elle ne doit PAS l'etre (sinon un contenu distinct serait ramene
sur le hash de reference, ce qui reouvrirait le rejeu).
"""

import numpy as np

from distseal.ciphermark.phash import _RSWrap
from distseal.ciphermark.witness import WitnessConfig, WitnessField


NSYM = 32          # parite de 32 octets -> corrige 16 octets
CAPACITY = NSYM // 2


def _h(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return bytes(rng.integers(0, 256, 32, dtype=np.uint8).tolist())


def _flip_bytes(data: bytes, n: int, seed: int = 1) -> bytes:
    """Corrompt exactement n octets distincts."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), n, replace=False)
    arr = bytearray(data)
    for i in idx:
        arr[i] ^= 0xFF
    return bytes(arr)


def test_capacite_annoncee():
    rs = _RSWrap(nsym=NSYM)
    assert rs.capacity == CAPACITY


def test_parite_longueur():
    rs = _RSWrap(nsym=NSYM)
    pi = rs.parity(_h())
    assert len(pi) == NSYM


def test_correction_sans_erreur():
    rs = _RSWrap(nsym=NSYM)
    h = _h()
    pi = rs.parity(h)
    out, ok = rs.correct(h, pi)
    assert ok and out == h


def test_correction_sous_capacite():
    """Jusqu'a 16 octets errones : correction exacte."""
    rs = _RSWrap(nsym=NSYM)
    h = _h(2)
    pi = rs.parity(h)
    for n_err in (1, 4, 8, 16):
        noisy = _flip_bytes(h, n_err, seed=n_err)
        out, ok = rs.correct(noisy, pi)
        assert ok, f"decodage echoue a {n_err} octets (capacite {CAPACITY})"
        assert out == h, f"correction fausse a {n_err} octets"


def test_pas_de_correction_au_dela_de_la_capacite():
    """
    Au-dela de 16 octets, le decodage ne doit pas ramener silencieusement
    vers h : soit il echoue, soit il rend autre chose. C'est ce qui empeche
    un contenu distinct (~31 octets de derive) d'etre corrige vers h.
    """
    rs = _RSWrap(nsym=NSYM)
    h = _h(3)
    pi = rs.parity(h)
    for n_err in (20, 25, 31):
        noisy = _flip_bytes(h, n_err, seed=100 + n_err)
        out, ok = rs.correct(noisy, pi)
        assert not (ok and out == h), (
            f"{n_err} octets errones corriges vers h : la fenetre de "
            f"discrimination est violee"
        )


def test_correction_puis_hmac_identique():
    """
    Bout en bout du canal dur : si RS corrige, le tag HMAC redevient
    exactement celui de reference. C'est la condition de la verification.
    """
    rs = _RSWrap(nsym=NSYM)
    wf = WitnessField(b"\x11" * 32, b"\x22" * 32, WitnessConfig(n_bits=256))

    h = _h(4)
    pi = rs.parity(h)
    tag_ref = wf.expected_tag_bits(h)

    noisy = _flip_bytes(h, 12, seed=7)          # sous la capacite
    h_corr, ok = rs.correct(noisy, pi)
    assert ok

    tag_corr = wf.expected_tag_bits(h_corr)
    assert int(np.sum(tag_ref != tag_corr)) == 0

    # sans correction, avalanche : ~50 % des bits du tag changent
    tag_noisy = wf.expected_tag_bits(noisy)
    d = int(np.sum(tag_ref != tag_noisy))
    assert d > 80, f"avalanche trop faible : {d}/256"


if __name__ == "__main__":
    test_capacite_annoncee()
    test_parite_longueur()
    test_correction_sans_erreur()
    test_correction_sous_capacite()
    test_pas_de_correction_au_dela_de_la_capacite()
    test_correction_puis_hmac_identique()
    print("[parity] OK")

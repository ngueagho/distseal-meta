import numpy as np

from distseal.ciphermark.witness import WitnessConfig, WitnessField
from distseal.ciphermark.equation import CipherMarkVerifier, Verdict, binomial_pvalue


def test_pvalue_extremes():
    # 0 bit errone sur 256 -> p tres petit
    p = binomial_pvalue(0, 256)
    assert p > 0 and p < 1e-50
    # bruit pur (~128/256) -> p proche de 0.5
    p_mid = binomial_pvalue(128, 256)
    assert 0.4 < p_mid < 0.6


def test_verdicts():
    s = b"\x55" * 32
    k = b"\x66" * 32
    wf = WitnessField(s, k, WitnessConfig(n_bits=256))
    v = CipherMarkVerifier(wf)
    h = b"\x77" * 32

    omega = wf.build_omega(h, image_id=3)
    # authentique
    r = v.verify(omega, h, image_id=3)
    assert r.verdict == Verdict.AUTHENTIC

    # bruit -> NOT_WATERMARKED
    noise = np.random.randint(0, 2, 256, dtype=np.uint8)
    r2 = v.verify(noise, h, image_id=3)
    # avec n=256 bits aleatoires on tombe ~50% -> NOT_WATERMARKED
    assert r2.verdict in (Verdict.NOT_WATERMARKED, Verdict.HEAVY_EDIT)


if __name__ == "__main__":
    test_pvalue_extremes()
    test_verdicts()
    print("[equation] OK")

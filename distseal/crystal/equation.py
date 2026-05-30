"""
Crystal Equation: verification cryptographique d'une image observee.

    Omega_obs  XOR  PRG(s_master, nonce)   ==  HMAC(K_secret, h_obs) ?

On regarde la distance de Hamming entre les deux cotes -> decision +
diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from . import crypto
from .witness import WitnessField, VerifyResult


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    AUTHENTIC = "authentic"
    LIGHT_EDIT = "light_edit"        # distorsion legitime
    HEAVY_EDIT = "heavy_edit"        # contenu modifie
    NOT_WATERMARKED = "no_wm"        # bruit pur
    UNKNOWN = "unknown"


@dataclass
class VerificationReport:
    verdict: Verdict
    distance: int
    total: int
    p_value: float
    confidence: float
    notes: str = ""

    @property
    def ber(self) -> float:
        return self.distance / max(1, self.total)

    def __str__(self) -> str:
        return (f"<{self.verdict.value} d={self.distance}/{self.total} "
                f"ber={self.ber:.3f} p={self.p_value:.2e}>")


# ---------------------------------------------------------------------------
# P-value binomiale
# ---------------------------------------------------------------------------

def binomial_pvalue(d: int, n: int) -> float:
    """
    Pr[Bin(n, 0.5) <= d]. Pour n grand on utilise une approximation normale,
    sinon le calcul exact log-binomial.
    """
    if n <= 0:
        return 1.0
    if d >= n:
        return 1.0
    if d < 0:
        return 0.0

    if n > 500:
        # approximation normale avec correction de continuite
        mean = n / 2.0
        std = math.sqrt(n / 4.0)
        z = (d + 0.5 - mean) / std
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # calcul exact en log-space
    log_p = -n * math.log(2.0)
    log_total = -math.inf
    log_coef = 0.0
    # coefficients binomiaux iteratifs
    for k in range(d + 1):
        if k > 0:
            log_coef += math.log(n - k + 1) - math.log(k)
        lp = log_p + log_coef
        log_total = _logaddexp(log_total, lp)
    return math.exp(log_total)


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


# ---------------------------------------------------------------------------
# Verifieur
# ---------------------------------------------------------------------------

@dataclass
class CrystalThresholds:
    """Seuils par defaut (fraction de bits errones)."""
    authentic: float = 0.10     # <= 10% d'erreur -> authentique
    light: float = 0.25         # 10-25% -> distorsion legere
    heavy: float = 0.40         # 25-40% -> edition lourde
    # > 0.40 -> non-watermarkee


class CrystalVerifier:
    """
    Verifie l'equation Crystal et produit un verdict + p-value.
    """

    def __init__(
        self,
        witness: WitnessField,
        thresholds: Optional[CrystalThresholds] = None,
    ):
        self.witness = witness
        self.th = thresholds or CrystalThresholds()

    def verify(
        self,
        omega_observed: np.ndarray,
        h_bytes: bytes,
        image_id: int,
    ) -> VerificationReport:
        res: VerifyResult = self.witness.verify(
            omega_observed, h_bytes, image_id,
        )
        ber = res.ber
        p = binomial_pvalue(res.distance, res.total)

        if ber <= self.th.authentic:
            verdict = Verdict.AUTHENTIC
            notes = "OK"
        elif ber <= self.th.light:
            verdict = Verdict.LIGHT_EDIT
            notes = "distorsion legitime probable"
        elif ber <= self.th.heavy:
            verdict = Verdict.HEAVY_EDIT
            notes = "contenu probablement modifie"
        else:
            verdict = Verdict.NOT_WATERMARKED
            notes = "pas de signal detecte"

        confidence = max(0.0, 1.0 - 2.0 * ber)  # 1 si ber=0, 0 si ber=0.5
        return VerificationReport(
            verdict=verdict,
            distance=res.distance,
            total=res.total,
            p_value=p,
            confidence=confidence,
            notes=notes,
        )

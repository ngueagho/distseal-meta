"""
CRYSTAL-OTP
===========

Tatouage numerique des modeles generatifs base sur la methode OTP.

Composants principaux:
    * crypto.py           -- HMAC, PRG, conversions bits/gaussiennes
    * phash.py            -- hash perceptuel (DINOv2 + LSH + RS)
    * witness.py          -- construction du champ temoin Omega
    * stable_subspace.py  -- sous-espace stable via Hessien (SSE)
    * equation.py         -- equation crystal + verdict + p-value
    * msg_processor.py    -- drop-in replacement de MsgProcessor
    * wam_crystal.py      -- orchestrateur avec point fixe

Voir docs/crystal.md pour le contexte theorique.
"""

from .crypto import (  # noqa: F401
    hmac_sha256,
    hkdf_expand,
    prg,
    make_nonce,
    random_seed,
    bits_to_gaussian,
    gaussian_to_bits,
    hamming,
)
from .phash import PerceptualHash, hamming_stability  # noqa: F401
from .witness import WitnessField, WitnessConfig, VerifyResult  # noqa: F401
from .stable_subspace import (  # noqa: F401
    StableProjector,
    LatentProjector,
    LanczosConfig,
    build_projector_from_loss,
    random_stable_projector,
    identity_projector,
)
from .equation import (  # noqa: F401
    CrystalVerifier,
    CrystalThresholds,
    VerificationReport,
    Verdict,
    binomial_pvalue,
)
from .msg_processor import CrystalMsgProcessor  # noqa: F401
from .wam_crystal import CrystalWam, CrystalKeys, CrystalConfig  # noqa: F401

__version__ = "0.1.0"

"""
OTP Witness Field (OWF).

   Omega = HMAC(K_secret, h)  XOR  PRG(s_master, nonce)

Le tout etendu si besoin avec HKDF pour avoir plus de bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import crypto


@dataclass
class WitnessConfig:
    n_bits: int = 256                # taille de Omega
    block_size: int = 16             # bits par sample gaussien (pour SSE)
    info_label: bytes = b"ciphermark/v1/omega"


class WitnessField:
    """
    Construit et verifie un champ temoin OTP a partir des cles + d'un hash.
    """

    def __init__(
        self,
        s_master: bytes,
        k_secret: bytes,
        cfg: Optional[WitnessConfig] = None,
    ):
        if len(s_master) != 32:
            raise ValueError("s_master doit etre 32 octets (256 bits)")
        if len(k_secret) != 32:
            raise ValueError("k_secret doit etre 32 octets (256 bits)")
        self.s_master = s_master
        self.k_secret = k_secret
        self.cfg = cfg or WitnessConfig()

    # ------------------------------------------------------------------ build

    def build_omega(self, h_bytes: bytes, image_id: int) -> np.ndarray:
        """
        h_bytes: hash perceptuel de l'image (bytes packes MSB-first)
        image_id: identifiant unique de l'image -> nonce
        Retour: bits Omega (taille cfg.n_bits), array uint8.
        """
        # 1) tag HMAC etendu
        tag = crypto.hmac_sha256(self.k_secret, h_bytes)
        n_bytes = (self.cfg.n_bits + 7) // 8
        tag_ext = crypto.hkdf_expand(tag, self.cfg.info_label, n_bytes)

        # 2) keystream PRG
        nonce = crypto.make_nonce(image_id)
        ks = crypto.prg(self.s_master, nonce, n_bytes)

        # 3) XOR
        omega_bytes = crypto.xor_bytes(tag_ext, ks)
        bits = crypto.bytes_to_bits(omega_bytes)[: self.cfg.n_bits]
        return bits

    def expected_tag_bits(self, h_bytes: bytes) -> np.ndarray:
        """HMAC(k, h) etendu en bits -- utile cote verifieur."""
        tag = crypto.hmac_sha256(self.k_secret, h_bytes)
        n_bytes = (self.cfg.n_bits + 7) // 8
        tag_ext = crypto.hkdf_expand(tag, self.cfg.info_label, n_bytes)
        return crypto.bytes_to_bits(tag_ext)[: self.cfg.n_bits]

    def keystream_bits(self, image_id: int) -> np.ndarray:
        nonce = crypto.make_nonce(image_id)
        n_bytes = (self.cfg.n_bits + 7) // 8
        ks = crypto.prg(self.s_master, nonce, n_bytes)
        return crypto.bytes_to_bits(ks)[: self.cfg.n_bits]

    # ------------------------------------------------------------------ gauss

    def omega_to_gaussian(self, omega_bits: np.ndarray) -> np.ndarray:
        """Bits -> samples N(0, 1) pour injection latente."""
        return crypto.bits_to_gaussian(
            omega_bits, block_size=self.cfg.block_size
        )

    def gaussian_to_omega(self, values: np.ndarray) -> np.ndarray:
        return crypto.gaussian_to_bits(
            values, block_size=self.cfg.block_size
        )

    # ------------------------------------------------------------------ check

    def verify(
        self,
        omega_observed: np.ndarray,
        h_bytes: bytes,
        image_id: int,
    ) -> "VerifyResult":
        ks = self.keystream_bits(image_id)
        expected = self.expected_tag_bits(h_bytes)
        recovered = crypto.xor_bits(omega_observed, ks)

        dist = crypto.hamming(recovered, expected)
        return VerifyResult(distance=dist, total=expected.size)


@dataclass
class VerifyResult:
    distance: int
    total: int

    @property
    def ber(self) -> float:
        return self.distance / max(1, self.total)

    @property
    def confidence(self) -> float:
        return max(0.0, 1.0 - self.ber)

    def __repr__(self) -> str:
        return (f"VerifyResult(d={self.distance}/{self.total}, "
                f"ber={self.ber:.3f})")

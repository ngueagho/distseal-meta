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
    # 64 bits, et pas 256 : mesure du 2026-08-21 sur WAM a 256x256, corpus
    # complet, conditions par ailleurs identiques (cf. docs/a-faire-memoire.md).
    #
    #   nbits   px/bit   bit_acc   PSNR   verdict
    #      64     1024    0.9998   22.5   utilisable  <-- 5/5 AUTHENTIC, d=0
    #     128      512    0.9616   22.4   insuffisant
    #     256      256    0.6076   22.0   inutilisable
    #
    # La qualite d'image est identique aux trois largeurs : ce n'est donc pas
    # un compromis mal regle, c'est la capacite de LECTURE du canal qui sature.
    # A 256 bits l'avalanche HMAC rend la verification impossible (0/5
    # AUTHENTIC, BER 39-55 %). 64 bits est aussi la valeur par defaut de
    # DistSeal (train.py:109 et les 5 configs d'origine), et 2^-64 reste le
    # minimum accepte en cryptographie moderne.
    n_bits: int = 64                 # taille de Omega
    block_size: int = 16             # bits par sample gaussien (pour SSE)
    info_label: bytes = b"ciphermark/v1/omega"
    # Etiquette HKDF pour deriver une cle par utilisateur (cf. derive_user_key).
    user_label: bytes = b"ciphermark/v1/user"


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
        self.user_id: Optional[str] = None   # rempli par for_user()

    # ------------------------------------------------------------------ user

    @staticmethod
    def derive_user_key(
        k_master: bytes,
        user_id: str,
        cfg: Optional[WitnessConfig] = None,
    ) -> bytes:
        """
        Derive une cle propre a un utilisateur a partir de la cle maitresse du
        fournisseur :

            k_user = HKDF-Expand(k_master, info = user_label || user_id, 32)

        Pourquoi c'est necessaire. Sans cela, Omega = HMAC(k_secret, h) ne
        contient AUCUNE identite : la chaine prouve qu'une image vient d'un
        detenteur de cle, mais pas DE QUI. C'est precisement l'argument oppose
        a WOUAF (Kim et al., CVPR 2024) et WMAdapter, dont le code utilisateur
        est arbitraire et donc forgeable par qui comprend le mecanisme.

        Avec la derivation, l'attribution devient une preuve : produire un
        Omega valide pour l'utilisateur U exige k_user, que seul le detenteur
        de k_master peut calculer. Deux utilisateurs distincts obtiennent des
        cles independantes -- HKDF garantit qu'on ne peut pas remonter de
        k_user a k_master, ni deviner la cle d'un autre.

        Le fournisseur n'a donc qu'UNE cle maitresse a proteger, et peut
        emettre autant d'identites que voulu sans stocker de secret par
        utilisateur.
        """
        if not isinstance(k_master, (bytes, bytearray)):
            raise TypeError("k_master doit etre bytes")
        if len(k_master) != 32:
            raise ValueError("k_master doit etre 32 octets (256 bits)")
        if not user_id:
            raise ValueError("user_id vide")
        cfg = cfg or WitnessConfig()
        info = cfg.user_label + b"|" + user_id.encode("utf-8")
        return crypto.hkdf_expand(k_master, info, 32)

    @classmethod
    def for_user(
        cls,
        s_master: bytes,
        k_master: bytes,
        user_id: str,
        cfg: Optional[WitnessConfig] = None,
    ) -> "WitnessField":
        """Construit un WitnessField dont la cle est derivee de `user_id`.

        Le registre doit alors stocker `nonce -> user_id` : c'est ce qui permet
        au verifieur de savoir quelle cle rederiver, sans jamais stocker de
        secret.
        """
        cfg = cfg or WitnessConfig()
        wf = cls(s_master=s_master,
                 k_secret=cls.derive_user_key(k_master, user_id, cfg),
                 cfg=cfg)
        wf.user_id = user_id
        return wf

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

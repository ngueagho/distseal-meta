"""
Orchestrateur CipherMark.

Ce module emballe un `distseal.models.wam.Wam` existant et lui ajoute:

  * la construction du champ temoin Omega via OTP + HMAC,
  * la boucle de point fixe (h_pred <-> h_real),
  * la verification via CipherMarkVerifier.

L'idee est de ne PAS toucher au Wam pour ne pas casser les checkpoints
existants. On lui passe juste un msg = Omega_bits pre-calcule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from .crypto import bytes_to_bits, random_seed
from .equation import CipherMarkVerifier, VerificationReport
from .phash import PerceptualHash
from .witness import WitnessConfig, WitnessField


@dataclass
class CipherMarkKeys:
    s_master: bytes
    k_secret: bytes

    @staticmethod
    def random() -> "CipherMarkKeys":
        return CipherMarkKeys(s_master=random_seed(32), k_secret=random_seed(32))

    @staticmethod
    def from_files(s_master_path: str, k_secret_path: str) -> "CipherMarkKeys":
        with open(s_master_path, "rb") as f:
            s = f.read()
        with open(k_secret_path, "rb") as f:
            k = f.read()
        return CipherMarkKeys(s_master=s, k_secret=k)


@dataclass
class CipherMarkConfig:
    n_bits: int = 256
    block_size: int = 16
    max_fixed_point_iters: int = 3
    fp_tol: int = 4               # bits de Hamming sous lequel on stoppe
    nonce_start: int = 0


class CipherMarkWam(nn.Module):
    """
    Wrapper autour de Wam.

    Usage minimal:

        wam = build_my_wam(...)
        phash = PerceptualHash(n_bits=256)
        ciphermark = CipherMarkWam(wam, phash, CipherMarkKeys.random())

        out = ciphermark.embed(imgs)              # genere les images watermarkees
        report = ciphermark.verify(out["imgs_w"], out["image_ids"])
    """

    def __init__(
        self,
        wam: nn.Module,
        phash: PerceptualHash,
        keys: CipherMarkKeys,
        cfg: Optional[CipherMarkConfig] = None,
    ):
        super().__init__()
        self.wam = wam
        self.phash = phash
        self.keys = keys
        self.cfg = cfg or CipherMarkConfig()
        self.witness = WitnessField(
            s_master=keys.s_master,
            k_secret=keys.k_secret,
            cfg=WitnessConfig(
                n_bits=self.cfg.n_bits,
                block_size=self.cfg.block_size,
            ),
        )
        self.verifier = CipherMarkVerifier(self.witness)
        self._next_image_id = self.cfg.nonce_start

    # ------------------------------------------------------------- helpers --

    def _allocate_ids(self, n: int) -> List[int]:
        ids = list(range(self._next_image_id, self._next_image_id + n))
        self._next_image_id += n
        return ids

    def _omega_bits_for(self, h_bytes_list: List[bytes],
                        image_ids: List[int]) -> torch.Tensor:
        rows = []
        for h, iid in zip(h_bytes_list, image_ids):
            rows.append(self.witness.build_omega(h, iid))
        return torch.from_numpy(np.stack(rows, axis=0)).to(torch.uint8)

    def _phash_bytes(self, imgs: torch.Tensor) -> List[bytes]:
        bits = self.phash(imgs).cpu().numpy()
        return [np.packbits(b, bitorder="big").tobytes() for b in bits]

    # ------------------------------------------------------------- embed ----

    @torch.no_grad()
    def embed(
        self,
        imgs: torch.Tensor,
        image_ids: Optional[List[int]] = None,
    ) -> dict:
        """
        Point fixe entre Omega et h_real:

           h_pred <- PHash(image_initiale)
           pour k in 1..K:
               Omega = HMAC(K, h_pred) XOR PRG(s, nonce)
               imgs_w = wam.embed(imgs, msg=Omega)
               h_real = PHash(imgs_w)
               si dist(h_real, h_pred) < tol: break
               h_pred <- h_real
        """
        B = imgs.shape[0]
        if image_ids is None:
            image_ids = self._allocate_ids(B)
        elif len(image_ids) != B:
            raise ValueError("image_ids doit avoir len == batch")

        # initialisation
        h_bytes = self._phash_bytes(imgs)

        last_imgs_w = None
        last_omega = None
        for it in range(self.cfg.max_fixed_point_iters):
            omega_bits = self._omega_bits_for(h_bytes, image_ids).to(imgs.device)

            out = self.wam.embed(imgs, msgs=omega_bits)
            imgs_w = out["imgs_w"]
            last_imgs_w = imgs_w
            last_omega = omega_bits

            h_real = self._phash_bytes(imgs_w)

            # convergence ?
            diffs = []
            for a, b in zip(h_bytes, h_real):
                ba = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
                bb = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
                diffs.append(int(np.sum(ba != bb)))
            max_d = max(diffs) if diffs else 0
            if max_d <= self.cfg.fp_tol:
                break
            h_bytes = h_real  # on continue avec l'observation

        return {
            "imgs_w": last_imgs_w,
            "omega": last_omega,
            "image_ids": image_ids,
            "n_iters": it + 1,
            "h_bytes": h_bytes,
        }

    # ------------------------------------------------------------- verify ---

    @torch.no_grad()
    def verify(
        self,
        imgs: torch.Tensor,
        image_ids: List[int],
    ) -> List[VerificationReport]:
        """
        Verifie chaque image du batch independamment.
        """
        if len(image_ids) != imgs.shape[0]:
            raise ValueError("image_ids et imgs incoherents")

        # extraction du temoin via le detector
        det_out = self.wam.detect(imgs)
        preds = det_out["preds"]  # (B, 1+nbits, H, W)

        # Heuristique: on aggrege le canal "msg" en moyennant sur H, W puis on
        # binarise. C'est la facon standard d'extraire un msg-bit dans
        # distseal.
        msg_logits = preds[:, 1:1 + self.cfg.n_bits].mean(dim=(-1, -2))
        omega_obs = (msg_logits > 0).to(torch.uint8).cpu().numpy()

        h_bytes = self._phash_bytes(imgs)
        reports = []
        for i, iid in enumerate(image_ids):
            r = self.verifier.verify(omega_obs[i], h_bytes[i], iid)
            reports.append(r)
        return reports

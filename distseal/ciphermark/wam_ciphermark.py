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
from .registry import TraceRegistry
from .witness import WitnessConfig, WitnessField

try:
    from distseal.models.video_wam import VideoWam
except Exception:  # pragma: no cover - distseal.models a des deps lourdes (torchvision...)
    VideoWam = None


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
    # Tolerance en bits APRES correction Reed-Solomon. 0 = egalite exacte,
    # ce qu'exige l'avalanche HMAC : le moindre bit d'ecart sur h fait diverger
    # le tag de ~50 %. Ne l'augmenter que pour du diagnostic.
    fp_tol: int = 0
    nonce_start: int = 0
    session: Optional[str] = None


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
        registry: Optional[TraceRegistry] = None,
    ):
        super().__init__()
        self.wam = wam
        self.phash = phash
        self.keys = keys
        self.cfg = cfg or CipherMarkConfig()
        # Base de tracabilite : nonce -> parite RS. Sans elle le verifieur ne
        # peut pas corriger le hash observe. En memoire par defaut (tests).
        self.registry = registry if registry is not None else TraceRegistry()
        self.witness = WitnessField(
            s_master=keys.s_master,
            k_secret=keys.k_secret,
            cfg=WitnessConfig(
                n_bits=self.cfg.n_bits,
                block_size=self.cfg.block_size,
            ),
        )
        self.verifier = CipherMarkVerifier(self.witness)
        # on repart apres le dernier nonce connu du registre : deux runs
        # successifs sur la meme base ne doivent pas reutiliser de keystream
        self._next_image_id = max(
            self.cfg.nonce_start, self.registry.next_free_nonce()
        )

    # ------------------------------------------------------------- helpers --

    def _batch_kwargs(self) -> dict:
        """
        `distseal.utils.cfg.setup_model` construit toujours un `VideoWam`, pas
        un `Wam` simple -- meme pour des checkpoints entraines sur des images.
        `VideoWam.embed`/`detect` traitent par defaut un batch comme les
        frames d'UNE SEULE video (`is_video=True`), ce qui casse l'hypothese
        de CipherMark : chaque element du batch est une image independante,
        avec son propre Omega. On force `is_video=False` uniquement si
        `self.wam` est bien un VideoWam (le simple `Wam` n'a pas ce parametre).
        """
        if VideoWam is not None and isinstance(self.wam, VideoWam):
            return {"is_video": False}
        return {}

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

    def _parity(self, h_bytes_list: List[bytes]) -> List[bytes]:
        """RS_parite(h) pour chaque image du batch."""
        return [self.phash.rs.parity(h) for h in h_bytes_list]

    def _corrige(
        self, h_obs_list: List[bytes], parity_list: List[bytes]
    ) -> List[bytes]:
        """RS_corrige(PHash(x), pi) pour chaque image du batch."""
        return [
            self.phash.correct(h, pi)[0]
            for h, pi in zip(h_obs_list, parity_list)
        ]

    @staticmethod
    def _bit_distance(a: bytes, b: bytes) -> int:
        ba = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
        bb = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
        return int(np.sum(ba != bb))

    # ------------------------------------------------------------- embed ----

    @torch.no_grad()
    def embed(
        self,
        imgs: torch.Tensor,
        image_ids: Optional[List[int]] = None,
    ) -> dict:
        """
        Algorithme 1 du memoire. Point fixe entre Omega et le hash corrige :

            h  <- PHash(image_initiale)
            pi <- RS_parite(h)
            pour k in 1..K:
                Omega <- HMAC(K_secret, h) XOR PRG(s_master, nonce)
                x_w   <- wam.embed(imgs, msg=Omega)
                h'    <- RS_corrige(PHash(x_w), pi)
                si h' == h : break        # point fixe atteint
                h  <- h' ; pi <- RS_parite(h)
            BD[nonce] <- (pi, metadonnees)

        Le critere de sortie porte sur le hash APRES correction, parce que
        c'est exactement ce que le verifieur recalculera. La parite finale est
        celle qui est enregistree : c'est elle qui permettra la correction.
        """
        B = imgs.shape[0]
        if image_ids is None:
            image_ids = self._allocate_ids(B)
        elif len(image_ids) != B:
            raise ValueError("image_ids doit avoir len == batch")

        # initialisation : h et sa parite sur l'image non marquee
        h_bytes = self._phash_bytes(imgs)
        parity = self._parity(h_bytes)

        last_imgs_w = None
        last_omega = None
        converged = False
        it = 0
        for it in range(self.cfg.max_fixed_point_iters):
            omega_bits = self._omega_bits_for(h_bytes, image_ids).to(imgs.device)

            out = self.wam.embed(imgs, msgs=omega_bits, **self._batch_kwargs())
            imgs_w = out["imgs_w"]
            last_imgs_w = imgs_w
            last_omega = omega_bits

            # hash de l'image marquee, corrige par la parite courante
            h_corr = self._corrige(self._phash_bytes(imgs_w), parity)

            max_d = max(
                (self._bit_distance(a, b) for a, b in zip(h_bytes, h_corr)),
                default=0,
            )
            if max_d <= self.cfg.fp_tol:
                converged = True
                break

            # sinon on repart du hash observe, avec SA parite
            h_bytes = h_corr
            parity = self._parity(h_bytes)

        # BD[nonce] <- (pi, metadonnees). Le registre refuse un nonce deja vu.
        for iid, pi in zip(image_ids, parity):
            self.registry.put(
                nonce=iid,
                parity=pi,
                n_bits=self.cfg.n_bits,
                rs_nsym=self.phash.rs.nsym,
                session=self.cfg.session,
            )

        return {
            "imgs_w": last_imgs_w,
            "omega": last_omega,
            "image_ids": image_ids,
            "n_iters": it + 1,
            "converged": converged,
            "h_bytes": h_bytes,
            "parity": parity,
        }

    # ------------------------------------------------------------- verify ---

    @torch.no_grad()
    def verify(
        self,
        imgs: torch.Tensor,
        image_ids: List[int],
    ) -> List[VerificationReport]:
        """
        Algorithme 2 du memoire. Verifie chaque image du batch independamment :

            Omega_obs <- extract(x_chapeau)
            pi        <- BD[nonce]
            h_chapeau <- RS_corrige(PHash(x_chapeau), pi)
            T         <- HMAC(K_secret, h_chapeau)
            BER       <- Ham(Omega_obs XOR PRG(s_master, nonce), T) / |Omega|

        Le hash est RECALCULE sur l'image observee : c'est ce qui lie le
        verdict au contenu et rend le rejeu inoperant. La base ne fournit que
        la parite, jamais le hash de reference.
        """
        if len(image_ids) != imgs.shape[0]:
            raise ValueError("image_ids et imgs incoherents")

        # extraction du temoin via le detector
        det_out = self.wam.detect(imgs, **self._batch_kwargs())
        preds = det_out["preds"]  # (B, 1+nbits) si pixelwise=False, (B, 1+nbits, H, W) sinon

        # Heuristique: on aggrege le canal "msg" en moyennant sur les eventuelles
        # dimensions spatiales restantes, puis on binarise. Certains extracteurs
        # DistSeal (ex. convnext_tiny, pixelwise=False) font deja ce pooling dans
        # le PixelDecoder et renvoient (B, 1+nbits) sans H,W -- on ne moyenne alors
        # plus rien, sous peine d'ecraser aussi la dimension batch.
        msg_logits = preds[:, 1:1 + self.cfg.n_bits]
        if msg_logits.dim() > 2:
            msg_logits = msg_logits.mean(dim=tuple(range(2, msg_logits.dim())))
        omega_obs = (msg_logits > 0).to(torch.uint8).cpu().numpy()

        h_obs = self._phash_bytes(imgs)
        parity = [self.registry.parity_for(iid) for iid in image_ids]
        h_hat = self._corrige(h_obs, parity)

        reports = []
        for i, iid in enumerate(image_ids):
            r = self.verifier.verify(omega_obs[i], h_hat[i], iid)
            reports.append(r)
        return reports

"""
Hash perceptuel "crypto-friendly" pour CipherMark.

Idee: extraire des features stables avec DINOv2 (ou un fallback rapide), les
projeter sur des hyperplans aleatoires fixes (LSH) -> bits stables, puis
passer dans un code Reed-Solomon pour tolerer quelques bit-flips dus aux
distorsions.

Si dinov2 n'est pas installable (cluster sans internet) on prend un
fallback en niveaux de gris + DCT, calque sur pHash classique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# Reed-Solomon: dependance optionnelle a l'import, mais obligatoire des qu'on
# touche a la parite (cf. _RSWrap.parity / _RSWrap.correct qui levent).
try:
    from reedsolo import RSCodec, ReedSolomonError  # type: ignore
    _HAS_RS = True
except Exception:
    _HAS_RS = False

    class ReedSolomonError(Exception):  # type: ignore
        """Stub quand reedsolo est absent."""


_RS_MISSING_MSG = (
    "reedsolo est requis pour la correction Reed-Solomon "
    "(pip install -r requirements.txt). Sans lui, le canal h ne peut pas "
    "absorber les derives legitimes et la verification echoue."
)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def _try_load_dinov2() -> Optional[nn.Module]:
    """Charge DINOv2-small si possible, sinon None."""
    try:
        # torch.hub est plus simple a manipuler que timm dans nos configs
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    except Exception as e:  # pragma: no cover
        print(f"[phash] DINOv2 indispo ({e}), fallback DCT")
        return None


class _DCTFallback(nn.Module):
    """Fallback ultra-simple si DINOv2 inaccessible: pHash a la 2010."""

    def __init__(self, feat_dim: int = 384):
        super().__init__()
        self.feat_dim = feat_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0, 1] ou [-1, 1] -> on s'en fout, on prend la luminance
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]
        gray = F.interpolate(
            gray.unsqueeze(1), size=(32, 32),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

        # DCT 2D maison (separable). Pas optimal mais on s'en sort.
        N = 32
        n = torch.arange(N, device=gray.device, dtype=gray.dtype)
        k = n.view(-1, 1)
        basis = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))
        # gray: (B, N, N)
        coeffs = basis @ gray @ basis.T  # (B, N, N)

        # on garde le coin haute-energie en zigzag
        flat = coeffs.flatten(1)  # (B, N*N)
        # tronque a feat_dim
        feats = flat[:, : self.feat_dim]
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)
        return feats


# ---------------------------------------------------------------------------
# LSH binarisation
# ---------------------------------------------------------------------------

@dataclass
class LSHParams:
    n_bits: int = 256
    seed: int = 0xC0FFEE


class RandomHyperplaneLSH(nn.Module):
    """
    LSH classique: on tire `n_bits` hyperplans aleatoires unitaires une fois
    pour toutes (seed fixe), puis on binarise par signe.
    """

    def __init__(self, in_dim: int, params: LSHParams):
        super().__init__()
        gen = torch.Generator().manual_seed(params.seed)
        H = torch.randn(params.n_bits, in_dim, generator=gen)
        H = H / (H.norm(dim=1, keepdim=True) + 1e-9)
        self.register_buffer("H", H)
        self.n_bits = params.n_bits

    @torch.no_grad()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, D), retourne (B, n_bits) en {0,1}
        proj = feats @ self.H.T  # (B, n_bits)
        return (proj > 0).to(torch.uint8)


# ---------------------------------------------------------------------------
# Code correcteur
# ---------------------------------------------------------------------------

class _RSWrap:
    """
    Wrapper autour de reedsolo.

    Deux niveaux d'API :
      * bits  (encode_bits / decode_bits) : codeword complet data+parite,
        conserve pour compatibilite ;
      * octets (parity / correct)         : la parite est renvoyee SEULE,
        pour etre stockee dans le registre de tracabilite et transmise au
        verifieur separement du hash. C'est le mode utilise par CipherMark.

    RSCodec(nsym) corrige jusqu'a nsym // 2 octets errones.
    """

    def __init__(self, nsym: int = 16):
        if not _HAS_RS:
            self.codec = None
        else:
            self.codec = RSCodec(nsym)
        self.nsym = nsym

    @property
    def capacity(self) -> int:
        """Nombre d'octets errones corrigeables."""
        return self.nsym // 2

    def _require(self) -> None:
        if self.codec is None:
            raise RuntimeError(_RS_MISSING_MSG)

    # ------------------------------------------------------------- octets ----

    def parity(self, data: bytes) -> bytes:
        """Parite seule (nsym octets) pour un bloc de donnees."""
        self._require()
        enc = bytes(self.codec.encode(bytes(data)))
        return enc[len(data):]

    def correct(self, data: bytes, parity: bytes) -> Tuple[bytes, bool]:
        """
        Corrige `data` a l'aide de `parity`.

        Retour (data_corrige, succes). En cas d'echec du decodage (plus de
        nsym//2 octets errones) on renvoie `data` inchange et succes=False :
        le verifieur constatera alors un BER ~50 % par avalanche HMAC, ce qui
        est le comportement voulu.
        """
        self._require()
        if len(parity) != self.nsym:
            raise ValueError(
                f"parite de {len(parity)} octets, attendu {self.nsym}"
            )
        try:
            dec = self.codec.decode(bytes(data) + bytes(parity))[0]
            return bytes(dec), True
        except ReedSolomonError:
            return bytes(data), False

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        if self.codec is None:
            # passthrough en mode degraded
            return bits.copy()
        data = np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()
        enc = bytes(self.codec.encode(data))
        return np.unpackbits(np.frombuffer(enc, dtype=np.uint8), bitorder="big")

    def decode_bits(self, bits: np.ndarray) -> np.ndarray:
        if self.codec is None:
            return bits.copy()
        try:
            data = np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()
            dec, _, _ = self.codec.decode(data)
            return np.unpackbits(np.frombuffer(bytes(dec), dtype=np.uint8),
                                 bitorder="big")
        except Exception:
            # decodage rate -> on rend les bits bruts moins la queue parity
            n_data_bits = bits.size - self.nsym * 8
            return bits[:n_data_bits].copy()


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------

class PerceptualHash(nn.Module):
    """
    PHash robuste:
        x (image, valeurs [0, 1]) -> bits stables (taille n_bits)

    Utilisation typique:

        phash = PerceptualHash(n_bits=256)
        h = phash(image_tensor)        # tensor uint8 (B, 256)
        h_np = h.cpu().numpy()
    """

    def __init__(
        self,
        n_bits: int = 256,
        backbone: Optional[nn.Module] = None,
        lsh_seed: int = 0xC0FFEE,
        # 32 octets de parite (corrige 16) : la derive legitime du hash DINOv2
        # atteint 10-15 octets sous jpeg/photometrie, et les contenus distincts
        # restent a ~31 octets -> 16 est le point d'equilibre mesure
        # (cf. results/phash_dino_rs16.csv vs rs32).
        rs_nsym: int = 32,
        input_size: int = 224,
    ):
        super().__init__()

        # backbone
        if backbone is None:
            backbone = _try_load_dinov2() or _DCTFallback()
        self.backbone = backbone

        # dimension de sortie du backbone: on infere une fois avec un dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            try:
                feat = backbone(dummy)
            except Exception:
                # certains backbones ont besoin de la methode 'forward_features'
                feat = getattr(backbone, "forward_features")(dummy)
            if isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
            feat = feat.reshape(feat.shape[0], -1)
        in_dim = int(feat.shape[1])

        self.lsh = RandomHyperplaneLSH(in_dim, LSHParams(n_bits=n_bits,
                                                        seed=lsh_seed))
        self.rs = _RSWrap(nsym=rs_nsym)
        self.input_size = input_size
        self.n_bits = n_bits

        # normalisation imagenet (DINO style)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        # x est suppose dans [0, 1]
        return (x - self.mean) / self.std

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._prep(x)
        try:
            feat = self.backbone(z)
        except Exception:
            feat = self.backbone.forward_features(z)
        if isinstance(feat, dict):
            feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
        feat = feat.reshape(feat.shape[0], -1)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-9)
        return feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        bits = self.lsh(feat)  # (B, n_bits)
        return bits

    # ------ helpers pour le pipeline ciphermark ------

    def hash_bytes(self, x: torch.Tensor) -> Tuple[bytes, ...]:
        bits = self.forward(x).cpu().numpy()
        return tuple(np.packbits(b, bitorder="big").tobytes() for b in bits)

    def encode_with_rs(self, x: torch.Tensor) -> np.ndarray:
        """Bits avec parite Reed-Solomon (utile cote generation)."""
        bits = self.forward(x).cpu().numpy()
        out = np.stack([self.rs.encode_bits(b) for b in bits], axis=0)
        return out

    def decode_with_rs(self, noisy_bits: np.ndarray) -> np.ndarray:
        return np.stack([self.rs.decode_bits(b) for b in noisy_bits], axis=0)

    # ------ parite separee : ce qui transite par le registre ------

    @property
    def rs_capacity(self) -> int:
        """Octets errones corrigeables (nsym // 2)."""
        return self.rs.capacity

    def parity_bytes(self, x: torch.Tensor) -> Tuple[bytes, ...]:
        """Parite Reed-Solomon de chaque hash du batch (nsym octets chacune)."""
        return tuple(self.rs.parity(h) for h in self.hash_bytes(x))

    def hash_and_parity(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[bytes, ...], Tuple[bytes, ...]]:
        """(hashs, parites) en un seul passage du backbone."""
        hs = self.hash_bytes(x)
        return hs, tuple(self.rs.parity(h) for h in hs)

    def correct(self, h_obs: bytes, parity: bytes) -> Tuple[bytes, bool]:
        """Corrige un hash observe vers le mot de code de generation."""
        return self.rs.correct(h_obs, parity)


def hamming_stability(h1: np.ndarray, h2: np.ndarray) -> float:
    """Mesure simple: 1 - frac_bits_differents."""
    h1 = np.asarray(h1).reshape(-1)
    h2 = np.asarray(h2).reshape(-1)
    if h1.size != h2.size:
        raise ValueError("tailles differentes")
    return 1.0 - float(np.mean(h1 != h2))

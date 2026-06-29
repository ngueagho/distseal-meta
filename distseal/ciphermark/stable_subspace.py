"""
Sous-espace stable (SSE).

L'idee est d'embeder le watermark dans les directions de plus grande
courbure du loss du modele pre-entraine. Empiriquement ces directions
"rigides" bougent peu sous LoRA fine-tuning, donc le watermark survit.

On expose deux modes:
  * 'random'    : projecteur identite ou aleatoire fixe (baseline rapide)
  * 'hessian'   : Lanczos sur le Hessien du loss + sous-espace top-k

Le mode hessian est lourd (1-2h sur DCAE-distill) -> on cache sur disque.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Projecteur (objet leger)
# ---------------------------------------------------------------------------

class StableProjector(nn.Module):
    """
    Projection orthogonale sur un sous-espace de dimension k de R^d.

    On stocke U: (d, k) avec colonnes orthonormales. Le projecteur est
    P = U U^T mais on ne le materialise jamais (couteux pour d ~ 8k).
    """

    def __init__(self, basis: torch.Tensor):
        super().__init__()
        # basis: (d, k)
        if basis.ndim != 2:
            raise ValueError("basis doit etre 2D")
        # QR pour s'assurer de l'orthonormalite (au cas ou)
        q, _ = torch.linalg.qr(basis)
        self.register_buffer("U", q.contiguous())

    @property
    def dim(self) -> int:
        return int(self.U.shape[0])

    @property
    def k(self) -> int:
        return int(self.U.shape[1])

    def project(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: (..., d). On applique P = U U^T sur la derniere dimension.
        """
        coords = v @ self.U          # (..., k)
        return coords @ self.U.T     # (..., d)

    def embed_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., k) -> vecteur dans R^d."""
        return coords @ self.U.T

    def get_coords(self, v: torch.Tensor) -> torch.Tensor:
        return v @ self.U

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.project(v)


# ---------------------------------------------------------------------------
# Construction depuis du bruit / identite (baseline)
# ---------------------------------------------------------------------------

def random_stable_projector(d: int, k: int, seed: int = 0) -> StableProjector:
    """Baseline 'aveugle' qui sert pour les tests rapides."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, k, generator=g)
    return StableProjector(A)


def identity_projector(d: int) -> StableProjector:
    """Projecteur identite (utile pour debug)."""
    return StableProjector(torch.eye(d))


# ---------------------------------------------------------------------------
# Hessien stochastique + Lanczos
# ---------------------------------------------------------------------------

@dataclass
class LanczosConfig:
    k: int = 100                # nombre de vecteurs propres voulus
    n_iter: int = 120           # un peu plus que k pour stabilite
    tol: float = 1e-6
    n_samples: int = 8          # batches pour estimation stochastique
    seed: int = 42
    verbose: bool = True


def _hvp(loss: torch.Tensor, params: List[torch.Tensor],
         v: List[torch.Tensor]) -> List[torch.Tensor]:
    """Produit Hessien-vecteur sans materialiser H."""
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * vi).sum() for g, vi in zip(grads, v))
    Hv = torch.autograd.grad(dot, params, retain_graph=False)
    return [h.detach() for h in Hv]


def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: torch.Tensor,
               like: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    offset = 0
    for t in like:
        n = t.numel()
        out.append(flat[offset:offset + n].view_as(t))
        offset += n
    return out


def estimate_top_eigenspace(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    cfg: LanczosConfig = LanczosConfig(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Renvoie (vals, basis) ou:
        vals  : (k,)  approximation des plus grandes valeurs propres
        basis : (d, k) vecteurs propres associes (colonnes)

    Lanczos avec re-orthogonalisation complete (un peu lent mais robuste).
    On accumule l'estimation stochastique sur cfg.n_samples batches.
    """
    if not params:
        raise ValueError("aucun parametre fourni")

    device = params[0].device
    d = sum(p.numel() for p in params)
    if cfg.verbose:
        print(f"[lanczos] d = {d}, k = {cfg.k}, n_iter = {cfg.n_iter}")

    torch.manual_seed(cfg.seed)
    # vecteur de depart aleatoire normalise
    v = torch.randn(d, device=device)
    v = v / v.norm()

    V = [v]
    alphas: List[float] = []
    betas: List[float] = []

    w_prev = torch.zeros(d, device=device)
    beta_prev = 0.0

    t0 = time.time()
    for j in range(cfg.n_iter):
        # estimation stochastique de H @ v
        Hv = torch.zeros(d, device=device)
        for s in range(cfg.n_samples):
            loss = loss_fn()
            v_unflat = _unflatten(V[-1], params)
            Hv_s = _hvp(loss, params, v_unflat)
            Hv = Hv + _flatten(Hv_s)
        Hv = Hv / cfg.n_samples

        # tridiagonalisation
        alpha = float((V[-1] * Hv).sum())
        alphas.append(alpha)
        w = Hv - alpha * V[-1] - beta_prev * w_prev

        # re-orthogonalisation complete (Gram-Schmidt sur tous V[i])
        for vi in V:
            w = w - (vi * w).sum() * vi

        beta = float(w.norm())
        betas.append(beta)
        if cfg.verbose and (j % 10 == 0):
            dt = time.time() - t0
            print(f"  iter {j:3d}: alpha={alpha:+.3e}, beta={beta:+.3e}, "
                  f"t={dt:.1f}s")

        if beta < cfg.tol:
            if cfg.verbose:
                print(f"[lanczos] convergence anticipee a iter {j}")
            break

        w_prev = V[-1]
        V.append(w / beta)
        beta_prev = beta

    # construire la matrice tridiagonale et la diagonaliser
    n = len(alphas)
    T = torch.zeros(n, n)
    for i in range(n):
        T[i, i] = alphas[i]
        if i + 1 < n:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    eigvals, eigvecs = torch.linalg.eigh(T)
    # plus grandes en magnitude (en valeur absolue ca pourrait avoir du sens,
    # mais ici on prend les plus positives = directions de courbure max).
    order = torch.argsort(eigvals, descending=True)
    top = order[: cfg.k]
    vals = eigvals[top]
    coefs = eigvecs[:, top]                    # (n, k)
    Vmat = torch.stack(V[:n], dim=1)           # (d, n)
    basis = Vmat @ coefs                       # (d, k)

    # re-orthonormalise (numeriquement)
    basis, _ = torch.linalg.qr(basis)
    if cfg.verbose:
        print(f"[lanczos] termine en {time.time() - t0:.1f}s, "
              f"vals top5 = {vals[:5].tolist()}")
    return vals.detach().cpu(), basis.detach().cpu()


# ---------------------------------------------------------------------------
# Construction "pratique" du projecteur a partir d'un modele
# ---------------------------------------------------------------------------

def build_projector_from_loss(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    k: int = 100,
    cache_path: Optional[str] = None,
    cfg: Optional[LanczosConfig] = None,
) -> StableProjector:
    """
    Pratique: si `cache_path` existe on le recharge. Sinon on calcule via
    Lanczos puis on sauvegarde.
    """
    cfg = cfg or LanczosConfig(k=k)
    if cache_path and os.path.exists(cache_path):
        print(f"[sse] chargement projecteur depuis {cache_path}")
        ckpt = torch.load(cache_path, map_location="cpu")
        return StableProjector(ckpt["basis"])

    _, basis = estimate_top_eigenspace(loss_fn, params, cfg)
    proj = StableProjector(basis)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({"basis": basis}, cache_path)
        print(f"[sse] projecteur sauve dans {cache_path}")
    return proj


# ---------------------------------------------------------------------------
# Wrapper "patch latent": prend un latent (B, C, H, W), applique P sur la
# perturbation aplatie.
# ---------------------------------------------------------------------------

class LatentProjector(nn.Module):
    """
    Adaptateur entre StableProjector (vit dans R^d) et un latent (B,C,H,W).

    Plutot que de faire un projecteur global de dimension C*H*W (lourd),
    on factorise: un petit projecteur sur la dimension canal C.

    En pratique c'est une bonne approximation: les directions LoRA-stables
    sont souvent dans le sous-espace des canaux.
    """

    def __init__(self, n_channels: int, k: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(n_channels, k, generator=g)
        q, _ = torch.linalg.qr(A)
        self.register_buffer("U", q[:, :k].contiguous())  # (C, k)
        self.k = k

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        # delta: (B, C, H, W)
        B, C, H, W = delta.shape
        flat = delta.permute(0, 2, 3, 1).reshape(-1, C)   # (BHW, C)
        coords = flat @ self.U                             # (BHW, k)
        back = coords @ self.U.T                           # (BHW, C)
        return back.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def embed_from_coords(
        self,
        coords_per_pixel: torch.Tensor,
        shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        # coords_per_pixel: (B, k, H, W)
        B, k, H, W = coords_per_pixel.shape
        flat = coords_per_pixel.permute(0, 2, 3, 1).reshape(-1, k)
        back = flat @ self.U.T  # (BHW, C)
        C = self.U.shape[0]
        return back.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.project(delta)

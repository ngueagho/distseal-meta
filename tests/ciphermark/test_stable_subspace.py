"""
Tests du sous-espace stable (SSE) : StableProjector, LatentProjector, et
l'estimation Lanczos du sous-espace de plus grande courbure du Hessien.

Le mode Lanczos est teste sur un jouet analytique (forme quadratique de
Hessien connu) plutot que sur un vrai reseau : ca suffit a verifier que
l'implementation retrouve les bonnes valeurs/vecteurs propres, sans le cout
d'un entrainement.
"""

import tempfile
import os

import numpy as np
import torch

from distseal.ciphermark.stable_subspace import (
    StableProjector,
    LatentProjector,
    LanczosConfig,
    build_projector_from_loss,
    estimate_top_eigenspace,
    random_stable_projector,
    identity_projector,
)


def test_stable_projector_orthonormal():
    torch.manual_seed(0)
    basis = torch.randn(8, 3)
    proj = StableProjector(basis)
    gram = proj.U.T @ proj.U
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)
    assert proj.dim == 8
    assert proj.k == 3


def test_project_is_idempotent():
    """P = U U^T doit etre un projecteur : P(P(v)) == P(v)."""
    proj = random_stable_projector(d=10, k=4, seed=1)
    v = torch.randn(5, 10)
    once = proj.project(v)
    twice = proj.project(once)
    assert torch.allclose(once, twice, atol=1e-5)


def test_identity_projector_preserves_vectors():
    proj = identity_projector(6)
    v = torch.randn(3, 6)
    assert torch.allclose(proj.project(v), v, atol=1e-5)


def test_embed_get_coords_roundtrip():
    """embed_coords(get_coords(v)) == U U^T v == project(v) (U orthonormale)."""
    proj = random_stable_projector(d=12, k=5, seed=2)
    v = torch.randn(4, 12)
    coords = proj.get_coords(v)
    back = proj.embed_coords(coords)
    assert torch.allclose(back, proj.project(v), atol=1e-5)


def test_latent_projector_shape_preserved():
    proj = LatentProjector(n_channels=8, k=3, seed=0)
    delta = torch.randn(2, 8, 5, 5)
    out = proj.project(delta)
    assert out.shape == delta.shape


def test_latent_projector_is_idempotent():
    proj = LatentProjector(n_channels=6, k=2, seed=3)
    delta = torch.randn(2, 6, 4, 4)
    once = proj.project(delta)
    twice = proj.project(once)
    assert torch.allclose(once, twice, atol=1e-5)


def test_latent_projector_embed_from_coords_matches_project():
    """
    Un delta construit depuis des coords vit deja dans le sous-espace :
    le projeter derechef ne doit rien changer.
    """
    n_channels, k, H, W = 8, 3, 4, 4
    proj = LatentProjector(n_channels=n_channels, k=k, seed=4)
    coords = torch.randn(2, k, H, W)
    delta = proj.embed_from_coords(coords, (2, n_channels, H, W))
    assert delta.shape == (2, n_channels, H, W)
    assert torch.allclose(proj.project(delta), delta, atol=1e-5)


def test_estimate_top_eigenspace_recovers_known_eigenvalues():
    """
    Loss quadratique 0.5 * p^T A p => Hessien exact = A (constant, independant
    de p). Lanczos doit retrouver les valeurs propres dominantes de A.
    """
    torch.manual_seed(0)
    d = 6
    M = torch.randn(d, d)
    A = (M + M.T) / 2  # symetrique
    true_vals, _ = torch.linalg.eigh(A)
    true_top2 = true_vals.flip(0)[:2]  # les 2 plus grandes

    p = torch.randn(d, requires_grad=True)

    def loss_fn():
        return 0.5 * (p @ A @ p)

    cfg = LanczosConfig(k=2, n_iter=d, n_samples=1, seed=0, verbose=False)
    vals, basis = estimate_top_eigenspace(loss_fn, [p], cfg)

    assert basis.shape == (d, 2)
    # Lanczos sur un Hessien exact converge en <= d iterations.
    assert torch.allclose(vals, true_top2, atol=1e-2), (
        f"vals={vals.tolist()} attendu={true_top2.tolist()}"
    )


def test_build_projector_from_loss_cache_roundtrip():
    """Le cache disque doit rendre exactement le meme projecteur."""
    torch.manual_seed(1)
    d = 5
    M = torch.randn(d, d)
    A = (M + M.T) / 2
    p = torch.randn(d, requires_grad=True)

    def loss_fn():
        return 0.5 * (p @ A @ p)

    cfg = LanczosConfig(k=2, n_iter=d, n_samples=1, seed=0, verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "proj.pt")
        assert not os.path.exists(cache_path)

        proj1 = build_projector_from_loss(loss_fn, [p], k=2, cache_path=cache_path, cfg=cfg)
        assert os.path.exists(cache_path)

        proj2 = build_projector_from_loss(loss_fn, [p], k=2, cache_path=cache_path, cfg=cfg)
        assert torch.allclose(proj1.U, proj2.U, atol=1e-6)


if __name__ == "__main__":
    test_stable_projector_orthonormal()
    test_project_is_idempotent()
    test_identity_projector_preserves_vectors()
    test_embed_get_coords_roundtrip()
    test_latent_projector_shape_preserved()
    test_latent_projector_is_idempotent()
    test_latent_projector_embed_from_coords_matches_project()
    test_estimate_top_eigenspace_recovers_known_eigenvalues()
    test_build_projector_from_loss_cache_roundtrip()
    print("[stable_subspace] OK")

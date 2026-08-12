"""
Test d'integration de la chaine complete embed -> registre -> verify.

On remplace le Wam entraine par un tatoueur factice qui ecrit Omega dans les
bits de poids faible des premiers pixels et sait le relire. Ce n'est pas un
watermark robuste, mais c'est un VRAI canal qui transite par le tenseur
image : la boucle de point fixe, le stockage de la parite et la correction
cote verifieur sont donc exerces pour de bon.

Ce que ces tests prouvent :
  * la parite est bien calculee, stockee sous le nonce, et relue au verify ;
  * un round-trip complet donne AUTHENTIC ;
  * un champ temoin transplante sur une autre image est rejete ;
  * un nonce ne peut pas etre reutilise.
"""

import numpy as np
import torch
from torch import nn

from distseal.ciphermark.phash import PerceptualHash, _DCTFallback
from distseal.ciphermark.registry import NonceReuseError, TraceRegistry
from distseal.ciphermark.wam_ciphermark import (
    CipherMarkConfig,
    CipherMarkKeys,
    CipherMarkWam,
)

N_BITS = 256
SIZE = 64


class FakeWam(nn.Module):
    """Tatoueur factice : Omega dans le canal 0, en clair, pixel par pixel."""

    def __init__(self, n_bits: int = N_BITS):
        super().__init__()
        self.n_bits = n_bits

    def _slots(self, W: int):
        """Positions (ligne, colonne) des n_bits premiers pixels."""
        idx = torch.arange(self.n_bits)
        return idx // W, idx % W

    def embed(self, imgs: torch.Tensor, msgs: torch.Tensor = None) -> dict:
        imgs_w = imgs.clone()
        rows, cols = self._slots(imgs.shape[-1])
        # 0.25 pour un bit a 0, 0.75 pour un bit a 1
        imgs_w[:, 0, rows, cols] = 0.25 + 0.5 * msgs.to(imgs_w.dtype)
        return {"imgs_w": imgs_w}

    def detect(self, imgs: torch.Tensor) -> dict:
        B, _, H, W = imgs.shape
        rows, cols = self._slots(W)
        bits = (imgs[:, 0, rows, cols] > 0.5).to(torch.float32)
        logits = (bits * 2.0 - 1.0)                      # {0,1} -> {-1,+1}
        preds = torch.zeros(B, 1 + self.n_bits, H, W)
        preds[:, 1:] = logits.view(B, self.n_bits, 1, 1).expand(
            B, self.n_bits, H, W
        )
        return {"preds": preds}


def _make(registry=None, n=4):
    torch.manual_seed(0)
    phash = PerceptualHash(n_bits=N_BITS, rs_nsym=32,
                           backbone=_DCTFallback())
    cm = CipherMarkWam(
        wam=FakeWam(),
        phash=phash,
        keys=CipherMarkKeys.random(),
        cfg=CipherMarkConfig(n_bits=N_BITS, max_fixed_point_iters=5),
        registry=registry if registry is not None else TraceRegistry(),
    )
    imgs = torch.rand(n, 3, SIZE, SIZE)
    return cm, imgs


def test_embed_remplit_le_registre():
    cm, imgs = _make(n=4)
    out = cm.embed(imgs)
    assert len(cm.registry) == 4
    for iid, pi in zip(out["image_ids"], out["parity"]):
        entry = cm.registry.get(iid)
        assert entry is not None
        assert entry.parity == pi
        assert len(entry.parity) == 32          # nsym octets
        assert entry.n_bits == N_BITS


def test_round_trip_authentic():
    """La chaine complete doit rendre un verdict authentique."""
    cm, imgs = _make(n=3)
    out = cm.embed(imgs)
    reports = cm.verify(out["imgs_w"], out["image_ids"])
    for r in reports:
        assert r.verdict.value == "authentic", (
            f"verdict={r.verdict.value} ber={r.ber:.3f} "
            f"(converge={out['converged']}, iters={out['n_iters']})"
        )
        assert r.distance == 0


def test_point_fixe_converge():
    cm, imgs = _make(n=2)
    out = cm.embed(imgs)
    assert out["converged"], "la boucle de point fixe n'a pas converge"
    assert 1 <= out["n_iters"] <= 5


def test_rejeu_transplante_rejete():
    """
    Omega de l'image A recopie sur une image C : le verifieur recalcule le
    hash sur C, ne retrouve pas le tag attendu, et doit refuser.
    """
    cm, imgs = _make(n=2)
    out = cm.embed(imgs)

    other = torch.rand(1, 3, SIZE, SIZE)
    # on transplante le champ temoin de l'image 0 sur une image tierce
    forged = cm.wam.embed(other, msgs=out["omega"][:1])["imgs_w"]

    reports = cm.verify(forged, [out["image_ids"][0]])
    assert reports[0].verdict.value != "authentic", (
        f"rejeu accepte : ber={reports[0].ber:.3f}"
    )


def test_verify_sans_entree_registre():
    cm, imgs = _make(n=1)
    out = cm.embed(imgs)
    try:
        cm.verify(out["imgs_w"], [999999])
    except KeyError:
        pass
    else:
        raise AssertionError("un nonce inconnu aurait du lever KeyError")


def test_nonce_non_reutilisable():
    cm, imgs = _make(n=2)
    cm.embed(imgs, image_ids=[10, 11])
    try:
        cm.embed(imgs, image_ids=[10, 12])
    except NonceReuseError:
        pass
    else:
        raise AssertionError("la reutilisation de nonce aurait du lever")


def test_nonces_distincts_donnent_omegas_distincts():
    cm, imgs = _make(n=2)
    out = cm.embed(imgs)
    o = out["omega"].cpu().numpy()
    assert int(np.sum(o[0] != o[1])) > 50, "Omega_i et Omega_j trop proches"


if __name__ == "__main__":
    test_embed_remplit_le_registre()
    test_round_trip_authentic()
    test_point_fixe_converge()
    test_rejeu_transplante_rejete()
    test_verify_sans_entree_registre()
    test_nonce_non_reutilisable()
    test_nonces_distincts_donnent_omegas_distincts()
    print("[pipeline] OK")

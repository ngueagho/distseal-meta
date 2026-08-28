"""Controle negatif de la chaine DistSeal + Omega variable.

POURQUOI CE SCRIPT EXISTE
-------------------------
La chaine rend 300 verdicts sur 300 generations. Pris seul, ce chiffre ne
prouve rien : un detecteur biaise, ou un message lisible dans le CONTENU plutot
que dans le conditionnement, donnerait le meme resultat. Il faut montrer que la
verification echoue quand elle doit echouer.

CE QU'IL TESTE
--------------
Le vrai Omega n'est pas relu depuis le disque -- il n'a pas ete enregistre --
mais les bits extraits de l'image AVEC temoin en sont a moins de 8 erreurs sur
64 : ils en tiennent lieu. On teste alors deux choses qui doivent etre vraies
si Omega est bien porte par le conditionnement :

  1. l'image SANS temoin, decodee du MEME latent, ne doit rien en dire ;
  2. verifier une image contre le temoin d'une AUTRE image ne doit rien rendre.

Les deux attendent une distance de 32/64 -- le hasard -- et aucun verdict. Si
le premier controle donnait une distance faible, le message viendrait du
contenu de l'image et non de notre conditionnement.

RESULTAT DU 2026-08-28 (300 paires, phase D-512 degelee)
--------------------------------------------------------
    controle                                mediane   min   verdicts a tort
    image SANS temoin, meme latent             31,0    22        0/300
    temoin d'une AUTRE image                   32,0    22        0/300

Lancement :
    PYTHONPATH=deps:. python3 scripts/ciphermark/controle_negatif_chaine.py \
        --images /workspace/runs/chaine_512_grande_echelle \
        --watermarker /workspace/ckpt/base_64bits.pth
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint
from distseal.ciphermark.equation import binomial_pvalue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="dossier contenant les paires *_avec_omega.png / *_sans_omega.png")
    ap.add_argument("--watermarker", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seuil-p", type=float, default=1e-6)
    args = ap.parse_args()

    cfg = get_config_from_checkpoint(args.watermarker)
    wam = setup_model_from_checkpoint(args.watermarker).cuda().eval()
    nbits, taille = int(cfg.args.nbits), int(cfg.args.img_size)
    kw = {}
    try:
        from distseal.models.videoseal import VideoWam
        if isinstance(wam, VideoWam):
            kw = {"is_video": False}
    except Exception:
        pass

    def bits(chemins):
        out = []
        for i in range(0, len(chemins), args.batch):
            lot = torch.stack([
                torch.from_numpy(np.asarray(Image.open(c).convert("RGB"))
                                 .astype(np.float32) / 255.).permute(2, 0, 1)
                for c in chemins[i:i + args.batch]]).cuda()
            if lot.shape[-1] != taille:
                lot = F.interpolate(lot, size=(taille, taille), mode="bilinear",
                                    align_corners=False, antialias=True)
            with torch.no_grad():
                p = wam.detect(lot, **kw)["preds"][:, 1:1 + nbits]
            out.append((p > 0).to(torch.uint8).cpu().numpy())
        return np.concatenate(out)

    av = sorted(glob.glob(os.path.join(args.images, "*_avec_omega.png")))
    sa = sorted(glob.glob(os.path.join(args.images, "*_sans_omega.png")))
    if not av:
        raise SystemExit(f"aucune image *_avec_omega.png dans {args.images}")
    if len(av) != len(sa):
        raise SystemExit(f"{len(av)} images avec temoin mais {len(sa)} sans : "
                         f"les paires ne correspondent pas")
    print(f"  {len(av)} paires avec/sans temoin", flush=True)

    Ba, Bs = bits(av), bits(sa)
    d_sans = (Ba != Bs).sum(1)
    perm = np.roll(np.arange(len(Ba)), 1)
    d_autre = (Ba != Ba[perm]).sum(1)
    pv_sans = np.array([binomial_pvalue(int(d), nbits) for d in d_sans])
    pv_autre = np.array([binomial_pvalue(int(d), nbits) for d in d_autre])

    print(f"\n  {'controle':<44}{'mediane':>9}{'min':>7}{'verdicts a tort':>18}")
    print(f"  {'-' * 78}")
    print(f"  {'1. image SANS temoin, meme latent':<44}"
          f"{np.median(d_sans):>9.1f}{d_sans.min():>7}"
          f"{int((pv_sans < args.seuil_p).sum()):>10}/{len(av)}")
    print(f"  {'2. temoin d une AUTRE image':<44}"
          f"{np.median(d_autre):>9.1f}{d_autre.min():>7}"
          f"{int((pv_autre < args.seuil_p).sum()):>10}/{len(av)}")
    print(f"\n  attendu sous l'hypothese nulle : mediane {nbits // 2}/{nbits}, "
          f"aucun verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

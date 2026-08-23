#!/usr/bin/env python3
"""
Reparer le hash perceptuel : diagnostic des marges et test de remedes.

Le probleme
-----------
`RandomHyperplaneLSH` binarise par le signe d'une projection :
`bit = (feats @ H.T) > 0`. Un bit dont la projection est proche de zero
bascule au moindre changement des features. Le marquage perturbe l'image,
donc les features, donc tous les bits qui etaient pres de leur frontiere.

Mesure du 2026-08-22 : marquer une image change 19.07 bits sur 64 (29.8 %),
alors qu'une image totalement differente n'en change que 28.47 (44.5 %). Les
plages se recouvrent (8-35 contre 18-35) : aucun seuil ne separe les deux cas,
et la liaison au contenu est donc inoperante -- 8/8 faux AUTHENTIC sous
transplantation.

Ce que ce script fait
---------------------
1. DIAGNOSTIC : distribution des marges |proj|. Si beaucoup de bits ont une
   marge quasi nulle, le mecanisme est confirme -- et on sait combien de bits
   sont structurellement fragiles.

2. REMEDES testes, chacun sur les deux criteres qui comptent :
     - derive sous marquage (a MINIMISER)
     - distance entre images differentes (a MAXIMISER)
   Le bon remede ecarte les deux distributions ; reduire la derive en ecrasant
   aussi la separation ne sert a rien.

   a) reference          -- l'existant, 224 px
   b) flou gaussien      -- sigma 1, 2, 3 avant le backbone. Hypothese : le
                            filigrane ecrit surtout en hautes frequences.
   c) sous-echantillon   -- 112 px, 64 px puis retour a 224. Meme idee, autre
                            moyen.
   d) marge (zone morte) -- ne compter comme "stables" que les bits dont
                            |proj| depasse un seuil ; mesure le gain potentiel
                            d'une selection de bits.

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/reparer_hash.py --n-images 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback  # noqa: E402
from distseal.ciphermark.registry import TraceRegistry  # noqa: E402
from distseal.ciphermark.wam_ciphermark import (  # noqa: E402
    CipherMarkConfig, CipherMarkKeys, CipherMarkWam)
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint  # noqa: E402
import distseal.utils.optim as uoptim  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[hash] {time.strftime('%H:%M:%S')} {m}", flush=True)


def replay_scaling(wam, cfg, ckpt):
    if getattr(cfg.args, "scaling_w_schedule", None) is None:
        return
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    ep = ck.get("epoch")
    if ep is None:
        return
    p = uoptim.parse_params(cfg.args.scaling_w_schedule)
    sc = uoptim.ScalingScheduler(obj=wam.blender, attribute="scaling_w",
                                 scaling_o=cfg.args.scaling_w, **p)
    log(f"scaling_w rejoue a l'epoque {ep} -> {sc.step(ep):.4f}")


def load_images(corpus, size, n, seed):
    exts = (".png", ".jpg", ".jpeg")
    files = [os.path.join(r, f) for r, _, ns in os.walk(corpus)
             for f in sorted(ns) if f.lower().endswith(exts)]
    random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f).convert("RGB").resize((size, size))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(out)


def gaussian_blur(x, sigma):
    """Flou gaussien separable, noyau dimensionne sur sigma."""
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    g = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    kh = g.view(1, 1, 1, k).expand(c, 1, 1, k)
    kv = g.view(1, 1, k, 1).expand(c, 1, k, 1)
    x = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"), kh, groups=c)
    x = F.conv2d(F.pad(x, (0, 0) + (k // 2,) * 2, mode="reflect"), kv, groups=c)
    return x


def sous_echantillonne(x, taille):
    y = F.interpolate(x, size=(taille, taille), mode="bilinear",
                      align_corners=False, antialias=True)
    return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


def bits_de(phash, x):
    return phash(x).cpu().numpy().astype(np.int8)


def dist_moy(A, B):
    return float((A != B).sum(axis=1).mean())


def paires(A):
    n = len(A)
    d = [(A[i] != A[j]).sum() for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d)), int(np.min(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-colab/val")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/reparer_hash.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() - 1)))
    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits, img_size = int(cfg.args.nbits), int(cfg.args.img_size)

    dino = _try_load_dinov2()
    log(f"PHash : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}, nbits={nbits}")
    phash = PerceptualHash(n_bits=nbits, backbone=dino or _DCTFallback()).to(DEVICE).eval()

    imgs = load_images(args.corpus, img_size, args.n_images, args.seed).to(DEVICE)
    cm = CipherMarkWam(wam=wam, phash=phash, keys=CipherMarkKeys.random(),
                       cfg=CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3),
                       registry=TraceRegistry())
    log("marquage des images")
    marquees = cm.embed(imgs)["imgs_w"]

    # ---------------------------------------------------- 1. diagnostic marges
    with torch.no_grad():
        f_o = phash.features(imgs)
        f_m = phash.features(marquees)
        proj_o = (f_o @ phash.lsh.H.T).cpu().numpy()
        proj_m = (f_m @ phash.lsh.H.T).cpu().numpy()

    marges = np.abs(proj_o)
    bascule = (np.sign(proj_o) != np.sign(proj_m))
    print()
    print("=" * 78)
    print("1. DIAGNOSTIC -- pourquoi les bits basculent")
    print("=" * 78)
    print(f"  marge |proj| : mediane {np.median(marges):.4f}   "
          f"p10 {np.percentile(marges,10):.4f}   max {marges.max():.4f}")
    print(f"  bits qui basculent au marquage : {bascule.mean():.1%}")
    for seuil in [0.01, 0.02, 0.05, 0.10]:
        petits = marges < seuil
        if petits.sum():
            print(f"    parmi les bits de marge < {seuil:.2f} "
                  f"({petits.mean():>5.1%} des bits) : {bascule[petits].mean():>5.1%} basculent")
    gros = marges >= np.percentile(marges, 50)
    print(f"    parmi la moitie la plus SURE      : {bascule[gros].mean():>5.1%} basculent")

    # ------------------------------------------------------------ 2. remedes
    variantes = [("reference", lambda x: x)]
    for s in (1.0, 2.0, 3.0):
        variantes.append((f"flou_sigma{s:g}", lambda x, s=s: gaussian_blur(x, s)))
    for t in (112, 64, 32):
        variantes.append((f"sousech_{t}", lambda x, t=t: sous_echantillonne(x, t)))
    variantes.append(("flou2_sousech112",
                      lambda x: sous_echantillonne(gaussian_blur(x, 2.0), 112)))

    print()
    print("=" * 78)
    print(f"2. REMEDES -- {args.n_images} images, hash de {nbits} bits")
    print("=" * 78)
    print(f"{'variante':<20}{'derive':>9}{'autre img':>11}{'ecart':>8}{'verdict':>12}")
    print("-" * 78)
    res = []
    for nom, fn in variantes:
        try:
            with torch.no_grad():
                bo = bits_de(phash, fn(imgs))
                bm = bits_de(phash, fn(marquees))
            derive = dist_moy(bo, bm)
            inter, inter_min = paires(bo)
            ecart = inter - derive
            ok = "utilisable" if ecart > 10 else ("limite" if ecart > 5 else "non")
            print(f"{nom:<20}{derive:>9.2f}{inter:>11.2f}{ecart:>8.2f}{ok:>12}")
            res.append({"variante": nom, "derive_marquage": derive,
                        "distance_autre_image": inter,
                        "distance_min_inter": inter_min, "ecart": ecart})
        except Exception as exc:
            print(f"{nom:<20} echec : {exc!r}")
    print("=" * 78)
    print("derive    = bits changes par le marquage de la MEME image (a minimiser)")
    print("autre img = bits differents entre deux images distinctes (a maximiser)")
    print("ecart     = marge de manoeuvre pour fixer un seuil")

    if res:
        best = max(res, key=lambda r: r["ecart"])
        base = next((r for r in res if r["variante"] == "reference"), None)
        print()
        if base:
            print(f"reference : derive {base['derive_marquage']:.2f}, "
                  f"ecart {base['ecart']:.2f}")
        print(f"meilleure : {best['variante']} -- derive {best['derive_marquage']:.2f}, "
              f"ecart {best['ecart']:.2f}")
        if best["ecart"] > 10:
            print("=> un seuil devient possible : piste a retenir.")
        else:
            print("=> aucune variante ne degage assez de marge. Il faut agir sur")
            print("   la BINARISATION (marge/zone morte, ou PCA au lieu")
            print("   d'hyperplans aleatoires), pas seulement sur l'image.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"nbits": nbits, "n_images": args.n_images,
                   "bascule_globale": float(bascule.mean()),
                   "marge_mediane": float(np.median(marges)),
                   "variantes": res}, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

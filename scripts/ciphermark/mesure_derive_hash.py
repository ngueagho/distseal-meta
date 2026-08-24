#!/usr/bin/env python3
"""
Dimensionner nsym : combien d'octets le code Reed-Solomon doit-il corriger ?

Le probleme a resoudre
----------------------
`RSCodec(nsym)` corrige jusqu'a nsym//2 octets. La configuration actuelle
(nsym=16) corrige donc 8 octets sur un hash de 64 bits qui en fait... 8. La
capacite de correction egale la taille de la donnee : la parite seule suffit a
reconstituer le hash de reference, SANS aucune information de l'image.

Consequence mesuree par eval_attaques_actives.py : la transplantation du
filigrane sur une autre image donne 8/8 faux AUTHENTIC. La liaison au contenu,
qui est l'argument central de CipherMark face a WOUAF et MetaSeal, est
entierement neutralisee.

Le bon nsym est donc un compromis :
  - assez GRAND pour absorber la derive legitime du hash (le marquage modifie
    l'image, donc son hash ; les attaques passives aussi)
  - assez PETIT pour qu'une image etrangere ne puisse pas etre "corrigee" vers
    le hash de reference

Ce script mesure la derive legitime pour trancher.

Ce qu'il mesure
---------------
  1. derive du marquage      : PHash(image) vs PHash(image marquee)
  2. derive sous attaque     : PHash(image marquee) vs PHash(image attaquee)
  3. derive d'une AUTRE image: la borne a NE PAS pouvoir corriger

Tout est exprime en OCTETS differents (l'unite de correction de Reed-Solomon),
pas en bits -- un seul bit faux dans un octet rend l'octet entier errone.

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/mesure_derive_hash.py --n-images 30
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
from distseal.augmentation import valuemetric, geometric  # noqa: E402
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint  # noqa: E402
import distseal.utils.optim as uoptim  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[derive] {time.strftime('%H:%M:%S')} {m}", flush=True)


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


def octets_differents(a: bytes, b: bytes) -> int:
    """Nombre d'octets differents -- l'unite que Reed-Solomon corrige."""
    return sum(1 for x, y in zip(a, b) if x != y)


def stats(name, vals, taille):
    if not vals:
        return None
    a = np.array(vals, dtype=float)
    return {"cas": name, "n": len(vals), "moyenne": float(a.mean()),
            "median": float(np.median(a)), "max": int(a.max()),
            "p95": float(np.percentile(a, 95)), "taille_hash_octets": taille}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-colab/val")
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--hash-bits", type=int, default=256,
                    help="largeur du hash perceptuel. Decouplee de celle d Omega : le hash ne traverse pas l image, il est recalcule par le verifieur, donc la capacite de l extracteur ne le contraint pas. En dessous de 128 bits les derives legitimes et le contenu etranger se recouvrent.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/mesure_derive_hash.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() - 1)))
    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits, img_size = int(cfg.args.nbits), int(cfg.args.img_size)
    taille = args.hash_bits // 8
    log(f"Omega de {nbits} bits, hash de {args.hash_bits} bits ({taille} octets)")

    dino = _try_load_dinov2()
    log(f"PHash : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}")
    phash = PerceptualHash(n_bits=args.hash_bits,
                           backbone=dino or _DCTFallback()).to(DEVICE).eval()

    imgs_all = load_images(args.corpus, img_size, args.n_images, args.seed)
    cm = CipherMarkWam(wam=wam, phash=phash, keys=CipherMarkKeys.random(),
                       cfg=CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3),
                       registry=TraceRegistry())

    # Les attaques passives sont definies avant la boucle : chaque lot les subit
    # toutes, sinon il faudrait re-marquer le corpus une fois par attaque.
    attaques = [
        ("jpeg_q50", lambda x: valuemetric.JPEG()(x, None, quality=50)[0]),
        ("jpeg_q30", lambda x: valuemetric.JPEG()(x, None, quality=30)[0]),
        ("flou_k5", lambda x: valuemetric.GaussianBlur()(x, None, kernel_size=5)[0]),
        ("resize_0.5", lambda x: geometric.Resize()(x, None, size=0.5)[0]),
        ("crop_0.9", lambda x: geometric.Crop()(x, None, size=0.9)[0]),
        ("crop_0.5", lambda x: geometric.Crop()(x, None, size=0.5)[0]),
    ]

    # Traitement par lots : a 5000 images, marquer tout d'un bloc epuise la
    # memoire du GPU. Seuls les hash (quelques octets par image) sont conserves.
    d_marquage = []
    d_attaque = {nom: [] for nom, _ in attaques}
    echecs = {}
    h_mar_tous = []
    nb = (len(imgs_all) + args.batch - 1) // args.batch
    log(f"marquage de {len(imgs_all)} images en {nb} lots de {args.batch}")
    for bi in range(nb):
        imgs = imgs_all[bi * args.batch:(bi + 1) * args.batch].to(DEVICE)
        marquees = cm.embed(imgs)["imgs_w"]
        h_ori = cm._phash_bytes(imgs)
        h_mar = cm._phash_bytes(marquees)
        h_mar_tous.extend(h_mar)
        d_marquage.extend(octets_differents(a, b) for a, b in zip(h_ori, h_mar))
        for nom, fn in attaques:
            if nom in echecs:
                continue
            try:
                x = fn(marquees.clone()).clamp(0, 1)
                if x.shape[-2:] != (img_size, img_size):
                    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                                      align_corners=False, antialias=True)
                h_att = cm._phash_bytes(x)
                d_attaque[nom].extend(octets_differents(a, b)
                                      for a, b in zip(h_mar, h_att))
            except Exception as exc:
                echecs[nom] = repr(exc)
                log(f"{nom} en echec : {exc!r}")
        if (bi + 1) % 20 == 0 or bi + 1 == nb:
            log(f"  lot {bi + 1}/{nb} -- {len(h_mar_tous)} images traitees")

    res = []

    # 1. derive due au marquage
    res.append(stats("marquage", d_marquage, taille))
    log(f"marquage : {np.mean(d_marquage):.2f} octets en moyenne, "
        f"max {max(d_marquage)}")

    # 2. derive sous attaques passives
    for nom, _ in attaques:
        d = d_attaque[nom]
        if d:
            res.append(stats(nom, d, taille))
            log(f"{nom} : {np.mean(d):.2f} octets en moyenne, max {max(d)}")

    # 3. la borne a NE PAS pouvoir corriger : une image etrangere
    d = [octets_differents(h_mar_tous[i], h_mar_tous[(i + 1) % len(h_mar_tous)])
         for i in range(len(h_mar_tous))]
    res.append(stats("AUTRE_image", d, taille))
    log(f"AUTRE image : {np.mean(d):.2f} octets en moyenne, min {min(d)}")

    # ------------------------------------------------------------- synthese
    print()
    print("=" * 80)
    print(f"Derive du hash perceptuel -- {args.n_images} images, hash de {taille} octets")
    print("=" * 80)
    print(f"{'cas':<16}{'moyenne':>9}{'median':>8}{'p95':>7}{'max':>6}")
    print("-" * 80)
    for r in res:
        if r:
            print(f"{r['cas']:<16}{r['moyenne']:>9.2f}{r['median']:>8.1f}"
                  f"{r['p95']:>7.1f}{r['max']:>6d}")
    print("=" * 80)

    legitimes = [r for r in res if r and r["cas"] != "AUTRE_image"]
    etranger = next((r for r in res if r and r["cas"] == "AUTRE_image"), None)
    if legitimes and etranger:
        besoin = max(r["max"] for r in legitimes)
        plancher = etranger["moyenne"]
        print()
        print(f"Derive legitime maximale observee : {besoin} octets")
        print(f"Distance moyenne a une AUTRE image : {plancher:.1f} octets")
        print()
        print("Il faut nsym//2 >= derive legitime, et nsym//2 < distance a une")
        print("autre image -- sinon la parite reconstruit n'importe quel hash.")
        lo, hi = 2 * besoin, int(2 * plancher)
        if lo < hi:
            print(f"=> nsym dans [{lo}, {hi}[  -- recommande : {lo} (marge minimale)")
        else:
            print(f"=> AUCUNE valeur ne satisfait les deux contraintes "
                  f"(besoin {besoin} >= distance {plancher:.1f}).")
            print("   Le hash perceptuel ne separe pas assez les images :")
            print("   il faut l'allonger (plus de bits) ou changer de backbone.")
        print(f"   Valeur actuelle : nsym=16 -> corrige 8 octets sur {taille} = TOUT.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"nbits": nbits, "hash_bits": args.hash_bits,
                   "taille_octets": taille, "resultats": res}, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

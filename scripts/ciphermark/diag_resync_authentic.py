#!/usr/bin/env python3
"""
Recadrage 0,9 avec resynchronisation : le VERDICT COMPLET, hash compris.

`diag_resync_recadrage.py` a montre que la recherche aveugle de position fait
passer la DETECTION de 0 % a 64 % au recadrage 0,9 (100 images, p*K < 1e-6).
Il ne mesurait ni le verdict AUTHENTIC (d_Omega <= 6 sur 64), ni la liaison
au contenu : un tatouage relu ne suffit pas, l'empreinte du morceau recadre
doit aussi rester a moins de tau de la reference.

Ce script passe par la vraie chaine CipherMark (embed avec boucle de point
fixe, registre, hash DINOv2) et mesure, par image :
  - d_Omega : distance minimale, sur la grille de positions, entre le temoin
    relu et le temoin attendu (le verifieur le connait : nonce -> registre)
  - d_hash  : distance entre l'empreinte du morceau OBSERVE et h_ref
  - AUTHENTIC = d_Omega <= 6 ET d_hash <= tau (69/256)

Controle : la meme recherche sur le morceau de l'image NON tatouee, presente
sous le nonce de l'image tatouee. Le contenu est le meme, le hash passe donc ;
seul le temoin peut rejeter. Tout AUTHENTIC y est un faux positif.

Multiplier les positions multiplie les chances du hasard : a d <= 6 sur 64,
P = 4,5e-12 par position, soit ~9e-10 pour K = 196, encore negligeable.

Usage (GPU conseille) :
    PYTHONPATH=deps:. python3 scripts/ciphermark/diag_resync_authentic.py \
        --corpus corpus-coco-5000 --n-images 500
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2  # noqa: E402
from distseal.ciphermark.registry import TraceRegistry  # noqa: E402
from distseal.ciphermark.wam_ciphermark import (  # noqa: E402
    CipherMarkConfig, CipherMarkKeys, CipherMarkWam)
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint  # noqa: E402
from reparer_hash import load_images, replay_scaling  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[resync] {time.strftime('%H:%M:%S')} {m}", flush=True)


def canevas(morceau, i, j, T):
    """Replace le morceau a sa taille d'origine ; fond = couleur moyenne."""
    c = morceau.mean(dim=(1, 2), keepdim=True).expand(3, T, T).clone()
    h, w = morceau.shape[-2:]
    c[:, i:i + h, j:j + w] = morceau
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-coco-5000")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--taille", type=float, default=0.9)
    ap.add_argument("--pas", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"runs/diag_resync_authentic_N{args.n_images}_c{args.taille}.json"

    torch.manual_seed(args.seed)
    rnd = random.Random(args.seed)
    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits, T = int(cfg.args.nbits), int(cfg.args.img_size)

    dino = _try_load_dinov2()
    if dino is None:
        raise SystemExit("DINOv2 introuvable : le repli DCT ne mesure pas le vrai hash")
    phash = PerceptualHash(n_bits=256, backbone=dino).to(DEVICE).eval()
    cm = CipherMarkWam(wam=wam, phash=phash, keys=CipherMarkKeys.random(),
                       cfg=CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3),
                       registry=TraceRegistry())
    tau = round(cm.cfg.hamming_threshold * phash.n_bits)
    d_omega_max = int(0.10 * nbits)          # seuil AUTHENTIC : BER <= 10 %
    kw = cm._batch_kwargs()

    c = int(args.taille * T)
    ax = sorted(set(list(range(0, T - c + 1, args.pas)) + [T - c]))
    grille = [(i, j) for i in ax for j in ax]
    K = len(grille)
    p_hasard = sum(math.comb(nbits, k) for k in range(d_omega_max + 1)) / 2 ** nbits
    log(f"recadrage {args.taille} ({c} px), grille {K} positions, tau hash {tau}/256, "
        f"AUTHENTIC si d_Omega <= {d_omega_max} ; hasard par image ~{K * p_hasard:.1e}")

    @torch.no_grad()
    def lire(x):
        pr = cm.wam.detect(x.to(DEVICE), **kw)["preds"][:, 1:1 + nbits]
        if pr.dim() > 2:
            pr = pr.mean(dim=tuple(range(2, pr.dim())))
        return (pr > 0).to(torch.uint8).cpu().numpy()

    imgs = load_images(args.corpus, T, args.n_images, args.seed)
    lignes = {"tatouee": [], "vierge": []}
    t0 = time.time()
    for b in range(0, len(imgs), args.batch):
        lot = imgs[b:b + args.batch].to(DEVICE)
        e = cm.embed(lot)
        imw = e["imgs_w"].clamp(0, 1).cpu()
        om = e["omega"]
        om = (om.detach().cpu().numpy() if torch.is_tensor(om) else np.asarray(om)).astype(np.uint8)
        for k, iid in enumerate(e["image_ids"]):
            h_ref = cm.registry.h_ref_for(iid)
            i, j = rnd.randint(0, T - c), rnd.randint(0, T - c)
            for nom, src in (("tatouee", imw[k]), ("vierge", lot[k].cpu())):
                morceau = src[:, i:i + c, j:j + c]
                bits = lire(torch.stack([canevas(morceau, a, bb, T) for a, bb in grille]))
                d_om = int((bits != om[k]).sum(1).min())
                d_h = cm._bit_distance(cm._phash_bytes(morceau[None].to(DEVICE))[0], h_ref)
                lignes[nom].append({"d_omega": d_om, "d_hash": int(d_h),
                                    "omega_ok": d_om <= d_omega_max, "hash_ok": d_h <= tau})
        log(f"{min(b + args.batch, len(imgs))}/{len(imgs)} images  ({time.time() - t0:.0f} s)")

    res = {"n_images": len(imgs), "taille": args.taille, "pas": args.pas, "K": K,
           "tau_hash": tau, "d_omega_max": d_omega_max,
           "hasard_par_image": K * p_hasard}
    print()
    print("=" * 84)
    print(f"Recadrage {args.taille} + resynchronisation -- {len(imgs)} images, K={K}")
    print("=" * 84)
    print(f"{'image':<10}{'d_Omega med':>12}{'Omega ok':>10}{'d_hash med':>12}"
          f"{'hash ok':>10}{'AUTHENTIC':>11}")
    for nom, L in lignes.items():
        om_ok = np.array([x["omega_ok"] for x in L])
        h_ok = np.array([x["hash_ok"] for x in L])
        r = {"d_omega_mediane": float(np.median([x["d_omega"] for x in L])),
             "taux_omega_ok": float(om_ok.mean()),
             "d_hash_mediane": float(np.median([x["d_hash"] for x in L])),
             "taux_hash_ok": float(h_ok.mean()),
             "taux_authentic": float((om_ok & h_ok).mean()),
             "n_authentic": int((om_ok & h_ok).sum())}
        res[nom] = r
        print(f"{nom:<10}{r['d_omega_mediane']:>12.0f}{r['taux_omega_ok']:>9.1%}"
              f"{r['d_hash_mediane']:>12.0f}{r['taux_hash_ok']:>9.1%}{r['taux_authentic']:>10.1%}")
    print("=" * 84)
    print("vierge = meme morceau, image NON tatouee, sous le nonce de la tatouee :")
    print("         tout AUTHENTIC y est un faux positif")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"ecrit : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

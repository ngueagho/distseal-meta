#!/usr/bin/env python3
"""
Marge souple sur le hash perceptuel : ne comparer que les bits fiables.

Le probleme
-----------
`RandomHyperplaneLSH` binarise par le signe d'une projection. Un bit dont la
projection est proche de zero bascule au moindre changement de l'image. La
verification tolere donc une distance tau = 0,27 x 256 = 69 bits entre le hash
observe et la reference : assez pour absorber la derive (95,8 % de verdicts
sous JPEG 30), mais la meme tolerance laisse passer 0,10 % des contenus
etrangers (transplantation, campagne a 5000 images).

Le remede teste (correction C3 du rapport d'examen)
---------------------------------------------------
A l'inscription, on garde un masque des bits FIABLES de la reference :
|proj| >= m. A la verification, la distance ne porte que sur ces bits et est
normalisee par leur nombre. Le masque tient en 256 bits, soit 32 octets : la
place du champ de parite Reed-Solomon, devenu inutile.

Si le remede marche, il ecarte les deux distributions : la derive legitime
baisse (on retire les bits qui basculent pour rien), la distance entre images
differentes ne baisse pas (un bit fiable reste discriminant).

Ce que le script mesure, pour chaque marge m
--------------------------------------------
- bits conserves (moyenne)
- derive legitime sous chaque attaque (fraction des bits compares)
- distance a un contenu etranger : la reference i confrontee a l'image
  marquee j, i != j -- exactement la transplantation
- au seuil actuel 0,27 : taux d'acceptation legitime et etranger
- au seuil qui garde 95 % des images legitimes sous JPEG 30 : taux etranger
  observe et estimation gaussienne de la queue (les paires etrangeres sont
  trop peu nombreuses a petite echelle pour voir 0,10 % directement)

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/diag_marge_hash.py --n-images 20
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

from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback  # noqa: E402
from distseal.ciphermark.registry import TraceRegistry  # noqa: E402
from distseal.ciphermark.wam_ciphermark import (  # noqa: E402
    CipherMarkConfig, CipherMarkKeys, CipherMarkWam)
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint  # noqa: E402
from distseal.augmentation import valuemetric  # noqa: E402
from reparer_hash import load_images, replay_scaling  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU_ACTUEL = 0.27


def log(m):
    print(f"[marge] {time.strftime('%H:%M:%S')} {m}", flush=True)


def attaques():
    j, g, b = valuemetric.JPEG(), valuemetric.GaussianBlur(), valuemetric.Brightness()
    gen = torch.Generator().manual_seed(1)
    return [
        ("aucune", lambda x: x),
        ("jpeg_q50", lambda x: j(x, None, quality=50)[0]),
        ("jpeg_q30", lambda x: j(x, None, quality=30)[0]),
        ("flou_k5", lambda x: g(x, None, kernel_size=5)[0]),
        ("lumin_1.2", lambda x: b(x, None, 1.2)[0]),
        ("bruit_0.1", lambda x: x + 0.1 * torch.randn(x.shape, generator=gen)),
    ]


def queue_gauss(mu, sd, seuil):
    """P(X <= seuil) pour X ~ N(mu, sd) : estimation de la queue basse."""
    if sd <= 0:
        return float(mu <= seuil)
    return 0.5 * (1 + math.erf((seuil - mu) / (sd * math.sqrt(2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-coco-5000")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--hash-bits", type=int, default=256)
    ap.add_argument("--marges", default="0,0.01,0.02,0.03,0.05")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    marges = [float(m) for m in args.marges.split(",")]
    out = args.out or f"runs/diag_marge_hash_N{args.n_images}.json"

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits, img_size = int(cfg.args.nbits), int(cfg.args.img_size)

    dino = _try_load_dinov2()
    if dino is None:
        raise SystemExit("DINOv2 introuvable : le repli DCT ne mesure pas le vrai hash")
    phash = PerceptualHash(n_bits=args.hash_bits, backbone=dino).to(DEVICE).eval()

    imgs = load_images(args.corpus, img_size, args.n_images, args.seed)
    # meme liste, meme ordre que load_images : pour nommer les paires
    fichiers = [os.path.join(r, f) for r, _, ns in os.walk(args.corpus)
                for f in sorted(ns) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.Random(args.seed).shuffle(fichiers)
    fichiers = fichiers[:args.n_images]
    cm = CipherMarkWam(wam=wam, phash=phash, keys=CipherMarkKeys.random(),
                       cfg=CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3),
                       registry=TraceRegistry())
    log(f"{len(imgs)} images, Omega {nbits} bits, hash {args.hash_bits} bits, "
        f"{args.threads} fils CPU")

    marquees = torch.cat([cm.embed(imgs[i:i + args.batch].to(DEVICE))["imgs_w"].cpu()
                          for i in range(0, len(imgs), args.batch)]).clamp(0, 1)
    log("images marquees")

    def projections(x):
        with torch.no_grad():
            return np.concatenate([
                (phash.features(x[i:i + args.batch].to(DEVICE)) @ phash.lsh.H.T).cpu().numpy()
                for i in range(0, len(x), args.batch)])

    # La reference est le hash de l'image MARQUEE : c'est ce que le registre
    # enregistre a l'inscription.
    p_ref = projections(marquees)
    p_att = {}
    # Attaques appliquees par lots : sur 1000 images d'un coup, le flou
    # epuisait la RAM de Colab et le processus etait tue sans trace.
    for nom, fn in attaques():
        p_att[nom] = np.concatenate([
            projections(fn(marquees[i:i + args.batch].clone()).clamp(0, 1))
            for i in range(0, len(marquees), args.batch)])
        log(f"  projections sous {nom}")

    n = len(imgs)
    signe_ref = p_ref > 0
    res = {"n_images": n, "hash_bits": args.hash_bits, "tau_actuel": TAU_ACTUEL,
           "marges": {}}
    bruts = {}
    D0 = None
    for m in marges:
        masque = np.abs(p_ref) >= m                     # (n, 256)
        garde = masque.sum(1)                           # bits compares par image
        if (garde == 0).any():
            log(f"marge {m} : une reference n'a plus aucun bit fiable, ignoree")
            continue

        def dist(i, p):
            diff = (signe_ref[i] != (p > 0)) & masque[i]
            return diff.sum() / garde[i]

        legit = {nom: np.array([dist(i, p[i]) for i in range(n)])
                 for nom, p in p_att.items()}
        # transplantation : reference i, image marquee j (vectorise sur j).
        # D[i, j] = distance de la reference i a l'image j ; diagonale exclue.
        signe_aut = p_att["aucune"] > 0
        D = np.stack([((signe_ref[i] != signe_aut) & masque[i]).sum(1) / garde[i]
                      for i in range(n)])
        hors_diag = ~np.eye(n, dtype=bool)
        etr = D[hors_diag]
        bruts[f"D_m{m}"] = D.astype(np.float16)
        for k, v in legit.items():
            bruts[f"legit_m{m}_{k}"] = v.astype(np.float16)

        # Calibrage CONJOINT : le seuil doit garder la cible d'images
        # legitimes sous TOUTES les attaques, pas seulement sous JPEG 30.
        conjoint = {}
        for cible in (0.95, 0.99):
            t = float(max(np.quantile(v, cible) for v in legit.values()))
            conjoint[str(cible)] = {
                "tau": t,
                "accepte_etranger": float((etr <= t).mean()),
                "n_etrangers_acceptes": int((etr <= t).sum()),
                "accepte_legit": {k: float((v <= t).mean()) for k, v in legit.items()},
            }

        tau95 = float(np.quantile(legit["jpeg_q30"], 0.95))
        mu, sd = float(etr.mean()), float(etr.std())
        r = {
            "bits_compares_moy": float(garde.mean()),
            "derive_moy": {k: float(v.mean()) for k, v in legit.items()},
            "accepte_legit_tau_actuel": {k: float((v <= TAU_ACTUEL).mean())
                                         for k, v in legit.items()},
            "etranger_moy": mu, "etranger_ecart_type": sd,
            "etranger_min": float(etr.min()),
            "accepte_etranger_tau_actuel": float((etr <= TAU_ACTUEL).mean()),
            "estim_etranger_tau_actuel": queue_gauss(mu, sd, TAU_ACTUEL),
            "tau_95_jpeg30": tau95,
            "accepte_etranger_tau95": float((etr <= tau95).mean()),
            "estim_etranger_tau95": queue_gauss(mu, sd, tau95),
            "n_paires_etrangeres": int(len(etr)),
            "conjoint": conjoint,
        }
        res["marges"][str(m)] = r
        if m == 0:
            D0 = D

    print()
    print("=" * 100)
    print(f"Marge souple sur le hash -- {n} images, {n * (n - 1)} paires etrangeres")
    print("=" * 100)
    print(f"{'marge':>6} {'bits':>6} {'der.jpeg30':>11} {'der.bruit':>10} "
          f"{'etranger':>13} {'tau95':>7} {'FA obs.':>8} {'FA estim.':>10} {'legit@0,27':>11}")
    print("-" * 100)
    for m, r in res["marges"].items():
        print(f"{float(m):>6.2f} {r['bits_compares_moy']:>6.1f} "
              f"{r['derive_moy']['jpeg_q30']:>11.3f} {r['derive_moy']['bruit_0.1']:>10.3f} "
              f"{r['etranger_moy']:>7.3f}+-{r['etranger_ecart_type']:.3f} "
              f"{r['tau_95_jpeg30']:>7.3f} {r['accepte_etranger_tau95']:>8.2%} "
              f"{r['estim_etranger_tau95']:>10.2e} "
              f"{r['accepte_legit_tau_actuel']['jpeg_q30']:>10.0%}")
    print("=" * 100)
    print("der.    = derive legitime moyenne (fraction des bits compares)")
    print("tau95   = seuil gardant 95 % des images legitimes sous JPEG 30")
    print("FA      = acceptation d'un contenu etranger a tau95 : observee, puis")
    print("          estimee par une loi normale (queue invisible a petite echelle)")

    print()
    print("Calibrage conjoint (le seuil garde la cible sous TOUTES les attaques)")
    print(f"{'marge':>6} {'cible':>6} {'tau':>7} {'FA obs.':>9} {'n FA':>7}  pire attaque legitime")
    for m, r in res["marges"].items():
        for cible, c in r["conjoint"].items():
            pire = min(c["accepte_legit"], key=c["accepte_legit"].get)
            print(f"{float(m):>6.2f} {float(cible):>6.0%} {c['tau']:>7.3f} "
                  f"{c['accepte_etranger']:>9.4%} {c['n_etrangers_acceptes']:>7}  "
                  f"{pire} {c['accepte_legit'][pire]:.1%}")

    # Les paires etrangeres les plus proches, a regarder a l'oeil : si ce
    # sont des scenes quasi identiques, le plancher de fausses acceptations
    # vient du contenu, pas de la binarisation.
    if D0 is not None:
        S = np.minimum(D0, D0.T)
        iu = np.triu_indices(n, 1)
        ordre = np.argsort(S[iu])[:12]
        paires = [(int(iu[0][k]), int(iu[1][k]), float(S[iu][k])) for k in ordre]
        res["paires_les_plus_proches"] = [
            {"a": os.path.basename(fichiers[a]), "b": os.path.basename(fichiers[b]),
             "distance": d} for a, b, d in paires]
        print("\nPaires etrangeres les plus proches (marge 0) :")
        for a, b, d in paires:
            print(f"  {d:.3f}  {os.path.basename(fichiers[a])}  {os.path.basename(fichiers[b])}")
        try:
            from torchvision.utils import save_image
            planche = torch.stack([marquees[k] for a, b, _ in paires for k in (a, b)])
            save_image(planche, out.replace(".json", "_paires.png"), nrow=6, padding=4)
            log(f"planche des paires : {out.replace('.json', '_paires.png')}")
        except Exception as exc:
            log(f"planche non ecrite : {exc!r}")

    np.savez_compressed(out.replace(".json", "_bruts.npz"), **bruts)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"ecrit : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

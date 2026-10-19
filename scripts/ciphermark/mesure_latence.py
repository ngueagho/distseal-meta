#!/usr/bin/env python3
"""
Latence de la chaine post-hoc CipherMark, image par image (question 5.1 du
rapport d'examen : « quel est le surcout exact de la boucle de point fixe ? »).

On chronometre, sur lot de 1, chaque maillon separement puis la chaine :
  - wam.embed seul      : un passage du tatoueur, sans cryptographie
  - phash               : DINOv2 + LSH sur une image
  - cm.embed            : la chaine complete d'inscription, boucle de point
                          fixe comprise (jusqu'a 3 iterations tatoueur + hash)
  - cm.verify           : extraction + hash + comparaison + verdict
Le surcout de la boucle se lit en comparant cm.embed a wam.embed.

Le surcout du DECODEUR CONDITIONNE se mesure ailleurs, dans
eval_grande_echelle_decodeur.py (decode_nu / decode_waf, ms par image).

Les chiffres dependent du materiel : le GPU utilise est inscrit dans le JSON.

Usage :
    PYTHONPATH=deps:. python3 scripts/ciphermark/mesure_latence.py --n 50
"""
import argparse
import json
import os
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


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def chrono(fn, n, echauffe=5):
    for _ in range(echauffe):
        fn()
    sync()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        t.append(1000 * (time.perf_counter() - t0))
    return {"ms_mediane": float(np.median(t)), "ms_moyenne": float(np.mean(t)),
            "ms_p95": float(np.percentile(t, 95))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-coco-5000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="runs/mesure_latence.json")
    args = ap.parse_args()

    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits, T = int(cfg.args.nbits), int(cfg.args.img_size)
    dino = _try_load_dinov2()
    if dino is None:
        raise SystemExit("DINOv2 introuvable")
    phash = PerceptualHash(n_bits=256, backbone=dino).to(DEVICE).eval()
    cm = CipherMarkWam(wam=wam, phash=phash, keys=CipherMarkKeys.random(),
                       cfg=CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3),
                       registry=TraceRegistry())
    kw = cm._batch_kwargs()

    imgs = load_images(args.corpus, T, args.n + 10, 0).to(DEVICE)
    x = imgs[:1]
    msg = torch.randint(0, 2, (1, nbits), device=DEVICE).float()
    k = {"i": 0}

    def suivante():
        k["i"] = (k["i"] + 1) % len(imgs)
        return imgs[k["i"]:k["i"] + 1]

    iters = []

    def embed_complet():
        e = cm.embed(suivante())
        iters.append(e["n_iters"])
        return e

    with torch.no_grad():
        res = {
            "materiel": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
            "taille_image": T, "n": args.n,
            "tatoueur_seul": chrono(lambda: wam.embed(suivante(), msgs=msg, **kw), args.n),
            "hash_perceptuel": chrono(lambda: phash(suivante()), args.n),
            "inscription_complete": chrono(embed_complet, args.n),
        }
        res["iterations_point_fixe_moy"] = float(np.mean(iters))
        e = cm.embed(x)
        xw, ids = e["imgs_w"], e["image_ids"]
        res["verification_complete"] = chrono(lambda: cm.verify(xw, ids), args.n)

    base = res["tatoueur_seul"]["ms_mediane"]
    print()
    print("=" * 70)
    print(f"Latence CipherMark post-hoc, lot de 1, {res['materiel']}, {T} px")
    print("=" * 70)
    for nom in ("tatoueur_seul", "hash_perceptuel", "inscription_complete",
                "verification_complete"):
        r = res[nom]
        print(f"{nom:<24}{r['ms_mediane']:>9.1f} ms (mediane)  p95 {r['ms_p95']:>7.1f} ms")
    print(f"iterations du point fixe : {res['iterations_point_fixe_moy']:.2f} en moyenne")
    print(f"surcout de l'inscription CipherMark sur le tatoueur seul : "
          f"x{res['inscription_complete']['ms_mediane'] / base:.1f}")
    print("=" * 70)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("ecrit :", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

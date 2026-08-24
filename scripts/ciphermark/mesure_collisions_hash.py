#!/usr/bin/env python3
"""
Le hash perceptuel separe-t-il les images ? Mesure de collision et de marge.

Pourquoi cette mesure
---------------------
Omega = HMAC(K_secret, h). Toute la liaison au contenu repose donc sur une
hypothese jamais verifiee : que h distingue les images. Deux exigences,
contradictoires par nature :

  SEPARATION  -- deux images differentes doivent donner des h eloignes,
                 sinon l'attribution est ambigue et un filigrane devient
                 transplantable.
  STABILITE   -- la MEME image, marquee puis attaquee, doit garder un h
                 proche, sinon la verification legitime echoue.

La mesure du 2026-08-22 (mesure_derive_hash.py) a montre que les deux
distributions se recouvrent : marquage 29.8 % de bits changes contre 44.5 %
pour une autre image, avec des plages 8-35 et 18-35. Ce script quantifie
precisement ce recouvrement sur un echantillon plus large, et repond a trois
questions distinctes :

  1. collisions exactes -- deux images donnent-elles le meme h ?
  2. distribution inter-images -- a quelle distance sont deux images
     quelconques ? (attendu pour un hash ideal : 50 % des bits, comme deux
     tirages aleatoires)
  3. les bits sont-ils equilibres et independants ? Un hash dont certains bits
     valent presque toujours 0 gaspille sa capacite : l'entropie effective est
     alors bien inferieure aux n_bits annonces.

Le point 3 est souvent la cause cachee d'un mauvais pouvoir de separation.

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/mesure_collisions_hash.py --n-images 200
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(m):
    print(f"[collisions] {time.strftime('%H:%M:%S')} {m}", flush=True)


def load_images(corpus, size, n, seed):
    exts = (".png", ".jpg", ".jpeg")
    files = [os.path.join(r, f) for r, _, ns in os.walk(corpus)
             for f in sorted(ns) if f.lower().endswith(exts)]
    if not files:
        raise SystemExit(f"aucune image dans {corpus}")
    random.Random(seed).shuffle(files)
    files = files[:n]
    out = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((size, size))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(out), files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus-colab/val")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--n-bits", type=int, default=256,
                    help="largeur du hash perceptuel teste (et non celle d Omega)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/mesure_collisions_hash.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() - 1)))
    dino = _try_load_dinov2()
    log(f"PHash : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}, "
        f"n_bits={args.n_bits}")
    phash = PerceptualHash(n_bits=args.n_bits,
                           backbone=dino or _DCTFallback()).to(DEVICE).eval()

    imgs, files = load_images(args.corpus, args.img_size, args.n_images, args.seed)
    log(f"{len(files)} images chargees depuis {args.corpus}")

    # --- calcul des hash, par lots -----------------------------------------
    # meme voie que CipherMarkWam._phash_bytes -- sinon les chiffres ne
    # seraient pas comparables a ceux de mesure_derive_hash.py
    hashes = []
    with torch.no_grad():
        for i in range(0, len(imgs), args.batch):
            b = imgs[i:i + args.batch].to(DEVICE)
            bits = phash(b).cpu().numpy()
            hashes.extend(np.packbits(x, bitorder="big").tobytes() for x in bits)
            log(f"  {min(i + args.batch, len(imgs))}/{len(imgs)} hash calcules")

    B = np.stack([np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:args.n_bits]
                  for h in hashes]).astype(np.int8)
    n, k = B.shape
    log(f"matrice de bits : {n} images x {k} bits")

    # --- 1. collisions exactes ---------------------------------------------
    uniques = len(set(hashes))
    collisions = n - uniques

    # --- 2. distances deux a deux ------------------------------------------
    # produit matriciel : d(i,j) = k - (accords) ; rapide et exact
    Bs = 2 * B.astype(np.int16) - 1          # 0/1 -> -1/+1
    accords = (Bs @ Bs.T + k) // 2
    dist = k - accords
    iu = np.triu_indices(n, k=1)
    d = dist[iu].astype(float)

    # Un nombre de collisions ne dit rien sans les images concernees : deux
    # photos reellement differentes au meme hash sont un defaut du hachage,
    # deux images quasi uniformes -- un mur blanc, un ciel -- sont un cas
    # degenere connu de tout hachage perceptuel. On nomme donc les paires, et
    # on donne l'ecart-type des pixels pour trancher entre les deux.
    plus_proches = []
    ordre = np.argsort(d)[:min(20, len(d))]
    for t in ordre:
        i, j = int(iu[0][t]), int(iu[1][t])
        ecart = lambda k: float(imgs[k].std())
        plus_proches.append({
            "distance": int(d[t]),
            "image_a": os.path.basename(files[i]), "ecart_type_a": ecart(i),
            "image_b": os.path.basename(files[j]), "ecart_type_b": ecart(j)})
    if plus_proches:
        print()
        print("  PAIRES LES PLUS PROCHES (ecart-type des pixels entre "
              "parentheses)")
        for pp in plus_proches[:10]:
            print(f"    {pp['distance']:>4} bits   "
                  f"{pp['image_a']} ({pp['ecart_type_a']:.3f})   "
                  f"{pp['image_b']} ({pp['ecart_type_b']:.3f})")

    # --- 3. equilibre et entropie effective des bits -----------------------
    p1 = B.mean(axis=0)                       # proportion de 1 par bit
    # entropie binaire par bit, sommee = entropie effective (bits independants)
    eps = 1e-12
    pc = np.clip(p1, eps, 1 - eps)
    H = float((-(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))).sum())
    bits_morts = int(((p1 < 0.05) | (p1 > 0.95)).sum())

    # correlation moyenne entre bits (hors diagonale)
    C = np.corrcoef(B.T.astype(float)) if k > 1 else np.array([[1.0]])
    C = np.nan_to_num(C)
    corr_moy = float(np.abs(C[np.triu_indices(k, 1)]).mean())

    # --- sortie -------------------------------------------------------------
    print()
    print("=" * 82)
    print(f"Hash perceptuel -- {n} images, {k} bits")
    print("=" * 82)
    print(f"  collisions exactes        : {collisions} / {n}  ({uniques} hash distincts)")
    print(f"  paires comparees          : {len(d)}")
    print()
    print("  DISTANCE ENTRE DEUX IMAGES QUELCONQUES (bits differents)")
    print(f"    moyenne  {d.mean():.2f} / {k}   soit {d.mean()/k:.1%}   (ideal : 50 %)")
    print(f"    mediane  {np.median(d):.1f}     min {int(d.min())}     max {int(d.max())}")
    print(f"    p1       {np.percentile(d,1):.1f}      p5 {np.percentile(d,5):.1f}")
    print()
    print("  QUALITE DES BITS")
    print(f"    entropie effective        : {H:.1f} bits sur {k} annonces "
          f"({H/k:.0%})")
    print(f"    bits quasi constants      : {bits_morts} / {k}")
    print(f"    correlation moyenne |r|   : {corr_moy:.3f}   (ideal : ~0)")
    print("=" * 82)
    print()
    print("LECTURE. La derive due au MARQUAGE vaut 19.07 bits en moyenne")
    print("(plage 8-35), mesuree par mesure_derive_hash.py. Pour que la liaison")
    print("au contenu tienne, il faudrait que deux images quelconques soient")
    print("nettement PLUS eloignees que cela -- sans recouvrement des plages.")
    marge = d.min() - 35
    print(f"  distance minimale entre deux images : {int(d.min())} bits")
    print(f"  derive maximale du marquage         : 35 bits")
    if marge > 0:
        print(f"  => marge de {int(marge)} bits : un seuil est possible.")
    else:
        print(f"  => RECOUVREMENT de {int(-marge)} bits : aucun seuil ne separe")
        print("     'meme image marquee' de 'image differente'.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_images": n, "n_bits": k, "collisions": collisions,
                   "distance_moyenne": float(d.mean()),
                   "distance_min": int(d.min()), "distance_max": int(d.max()),
                   "distance_p1": float(np.percentile(d, 1)),
                   "entropie_effective": H, "bits_quasi_constants": bits_morts,
                   "correlation_moyenne": corr_moy,
                   "paires_les_plus_proches": plus_proches}, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

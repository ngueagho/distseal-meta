"""
Protocole experimental n.1 : robustesse du canal hash perceptuel.

CipherMark met deux canaux en serie :
  * canal h (dur)     : le hash doit etre recupere exactement, sinon
                        l'avalanche HMAC detruit tout le signal ;
  * canal Omega (mou) : tolere ~10% d'erreurs (seuil de Hamming).

La robustesse de bout en bout vaut donc:
    P(detection) = P(h corrigeable) * P(BER_omega <= seuil)

Ce script mesure le canal dur, qui decide de la viabilite du systeme:
  phase 1 : bit-flips et octets errones du phash par distorsion
            (simples + combinees), face a la capacite Reed-Solomon
  phase 2 : demonstration de l'effet falaise (1 flip -> tag ~50%)
  phase 3 : taux de detection joint simule pour plusieurs BER de Omega

Usage:
    python -m scripts.ciphermark.eval_phash_robustness --n 50
    python -m scripts.ciphermark.eval_phash_robustness --data-dir ./imgs --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from distseal.ciphermark import (
    CipherMarkKeys,
    PerceptualHash,
    WitnessConfig,
    WitnessField,
)
from scripts.ciphermark.eval_ciphermark import (
    DISTORTIONS,
    _crop,
    _jpeg,
    _noise,
    _rotate,
)


# les attaques combinees sont le point faible mesure de DistSeal
# (Tab. 1: 84.28 vs 97.29) -- c'est ici qu'on attend la falaise
COMBINED = {
    "crop-80+jpeg-60":  lambda x: _jpeg(_crop(x, 0.80), 60),
    "crop-70+jpeg-50":  lambda x: _jpeg(_crop(x, 0.70), 50),
    "rot-5+jpeg-70":    lambda x: _jpeg(_rotate(x, 5.0), 70),
    "noise+jpeg-60":    lambda x: _jpeg(_noise(x, 0.02), 60),
}

ALL_DISTORTIONS = {**DISTORTIONS, **COMBINED}


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

def load_images(data_dir: str, n: int, size: int = 256) -> torch.Tensor:
    from PIL import Image

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    paths = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(exts)
    )[:n]
    if not paths:
        raise FileNotFoundError(f"aucune image dans {data_dir}")

    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1) / 255.0)
    return torch.stack(imgs, dim=0).float()


def synthetic_images(n: int, size: int = 256) -> torch.Tensor:
    # meme recette que eval_ciphermark: degrade doux + moyenne locale
    import torch.nn.functional as F
    torch.manual_seed(0)
    base = torch.rand(n, 3, size, size)
    return 0.5 * base + 0.5 * F.avg_pool2d(base, 7, stride=1, padding=3)


# ---------------------------------------------------------------------------
# Metriques du canal h
# ---------------------------------------------------------------------------

def byte_errors(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    """Nombre d'octets differents -- c'est l'unite que RS corrige."""
    ba = np.packbits(bits_a.astype(np.uint8), bitorder="big")
    bb = np.packbits(bits_b.astype(np.uint8), bitorder="big")
    return int(np.sum(ba != bb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--data-dir", default=None,
                    help="dossier d'images reelles (sinon: synthetiques)")
    ap.add_argument("--n-bits", type=int, default=256)
    ap.add_argument("--rs-nsym", type=int, default=32,
                    help="octets de parite RS (corrige nsym/2 octets)")
    ap.add_argument("--backbone", choices=("dino", "dct"), default="dino",
                    help="dct = fallback local, sans telechargement")
    ap.add_argument("--csv", default=None, help="export csv des resultats")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rs_capacity = args.rs_nsym // 2

    if args.data_dir:
        base = load_images(args.data_dir, args.n).to(device)
        src = args.data_dir
    else:
        base = synthetic_images(args.n).to(device)
        src = "synthetique"
    n = base.shape[0]

    backbone = None
    if args.backbone == "dct":
        from distseal.ciphermark.phash import _DCTFallback
        backbone = _DCTFallback()
    phash = PerceptualHash(n_bits=args.n_bits, rs_nsym=args.rs_nsym,
                           backbone=backbone).to(device)

    print(f"\n[phash-eval] n={n} ({src}), n_bits={args.n_bits}, "
          f"capacite RS = {rs_capacity} octets\n")

    h_ref = phash(base).cpu().numpy()

    # ------------------------------------------------------------- phase 1
    print("Phase 1 -- canal h : flips et recuperabilite par distorsion")
    print("-" * 76)
    print(f"  {'distorsion':16s} {'flips moy':>9s} {'flips max':>9s} "
          f"{'octets moy':>10s} {'recuperable':>11s}")

    rows = []
    hash_ok = {}   # name -> bool array (n,)
    for name, fn in ALL_DISTORTIONS.items():
        with torch.no_grad():
            h_d = phash(fn(base.clone())).cpu().numpy()
        flips = (h_ref != h_d).sum(axis=1)
        octets = np.array([byte_errors(h_ref[i], h_d[i]) for i in range(n)])
        ok = octets <= rs_capacity
        hash_ok[name] = ok
        print(f"  {name:16s} {flips.mean():9.1f} {flips.max():9d} "
              f"{octets.mean():10.1f} {ok.mean():10.0%}")
        rows.append({
            "distortion": name,
            "flips_mean": float(flips.mean()),
            "flips_max": int(flips.max()),
            "bytes_mean": float(octets.mean()),
            "recoverable": float(ok.mean()),
        })

    # ------------------------------------------------------------- phase 2
    print("\nPhase 2 -- effet falaise : 1 bit de hash non corrige")
    print("-" * 76)
    keys = CipherMarkKeys.random()
    wf = WitnessField(keys.s_master, keys.k_secret,
                      WitnessConfig(n_bits=args.n_bits))

    h0 = np.packbits(h_ref[0], bitorder="big").tobytes()
    tag0 = wf.expected_tag_bits(h0)
    for k_flips in (1, 2, 8):
        h_mod = h_ref[0].copy()
        h_mod[:k_flips] ^= 1
        hb = np.packbits(h_mod, bitorder="big").tobytes()
        tag = wf.expected_tag_bits(hb)
        d = int(np.sum(tag0 != tag))
        print(f"  {k_flips} flip(s) sur h -> distance tag = {d}/{args.n_bits} "
              f"({d / args.n_bits:.0%})")
    print("  (attendu ~50% des le premier flip : c'est l'avalanche HMAC)")

    # ------------------------------------------------------------- phase 3
    print("\nPhase 3 -- detection jointe simulee")
    print("-" * 76)
    # seuil 'authentic' du verifieur: 10% de bits errones sur Omega
    max_err = int(0.10 * args.n_bits)
    rng = np.random.default_rng(1)
    omega_bers = (0.00, 0.05, 0.10)

    header = "  " + f"{'distorsion':16s}" + "".join(
        f"  BER={b:.2f}" for b in omega_bers)
    print(header)
    for row in rows:
        name = row["distortion"]
        ok = hash_ok[name]
        dets = []
        for ber in omega_bers:
            # si h est perdu, le tag attendu est aleatoire -> BER ~0.5, echec
            errs = rng.binomial(args.n_bits, max(ber, 1e-9), size=n)
            det = np.mean(ok & (errs <= max_err))
            dets.append(det)
            row[f"detect_ber{ber:.2f}"] = float(det)
        print("  " + f"{name:16s}" + "".join(f"  {d:8.0%}" for d in dets))

    print("\n  lecture : la colonne BER=0.00 isole le canal h ; si elle chute,")
    print("  aucun progres sur l'extracteur Omega ne sauvera la detection.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[phash-eval] resultats -> {args.csv}")

    print("\n[phash-eval] termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

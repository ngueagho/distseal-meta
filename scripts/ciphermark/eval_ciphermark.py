"""
Script d'evaluation rapide de CipherMark.

Ce script ne charge PAS le Wam complet (qui demande les checkpoints DCAE
+ extracteur entraines). Il fait une evaluation end-to-end de la chaine
crypto (OWF + Equation CipherMark) + du PHash sur les 11 distorsions usuelles.

Usage:
    python -m scripts.ciphermark.eval_ciphermark --n 50 --keys-dir ./keys
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from distseal.ciphermark import (
    CipherMarkKeys,
    PerceptualHash,
    WitnessConfig,
    WitnessField,
    CipherMarkVerifier,
    hamming_stability,
)


# ---------------------------------------------------------------------------
# Distorsions standard
# ---------------------------------------------------------------------------

def _jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG via torchvision (requiert recent torchvision).

    encode_jpeg ne prend que des tenseurs CPU : on fait l'aller-retour
    explicitement pour ne pas retomber silencieusement dans le fallback
    quand les images vivent sur GPU.
    """
    try:
        import torchvision.io as tvio
        out = []
        for i in range(img.shape[0]):
            x = (img[i].clamp(0, 1) * 255).to(torch.uint8).cpu()
            enc = tvio.encode_jpeg(x, quality=quality)
            dec = tvio.decode_jpeg(enc).float() / 255.0
            out.append(dec.to(img.device))
        return torch.stack(out, dim=0)
    except Exception as e:
        # fallback: pas de JPEG -> petit bruit gaussien (et on le DIT)
        print(f"[distorsions] JPEG indisponible ({e}), fallback bruit leger")
        return img + 0.005 * torch.randn_like(img)


def _crop(img: torch.Tensor, frac: float) -> torch.Tensor:
    _, _, H, W = img.shape
    h = int(H * frac)
    w = int(W * frac)
    top = (H - h) // 2
    left = (W - w) // 2
    cr = img[:, :, top:top + h, left:left + w]
    return F.interpolate(cr, size=(H, W), mode="bilinear", align_corners=False)


def _rotate(img: torch.Tensor, deg: float) -> torch.Tensor:
    try:
        import torchvision.transforms.functional as TF
        return TF.rotate(img, deg)
    except Exception:
        return img


def _brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
    return (img * factor).clamp(0, 1)


def _contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    mean = img.mean(dim=(2, 3), keepdim=True)
    return ((img - mean) * factor + mean).clamp(0, 1)


def _noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return (img + sigma * torch.randn_like(img)).clamp(0, 1)


DISTORTIONS = {
    "identity":    lambda x: x,
    "jpeg-80":     lambda x: _jpeg(x, 80),
    "jpeg-50":     lambda x: _jpeg(x, 50),
    "crop-90":     lambda x: _crop(x, 0.90),
    "crop-70":     lambda x: _crop(x, 0.70),
    "rot-5":       lambda x: _rotate(x, 5.0),
    "rot-15":      lambda x: _rotate(x, 15.0),
    "bright-0.7":  lambda x: _brightness(x, 0.7),
    "bright-1.5":  lambda x: _brightness(x, 1.5),
    "contrast-1.5": lambda x: _contrast(x, 1.5),
    "noise-0.02":  lambda x: _noise(x, 0.02),
}


# ---------------------------------------------------------------------------
# Eval principal
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="nb images synthetiques")
    ap.add_argument("--keys-dir", default=None,
                    help="dossier contenant s_master.bin et k_secret.bin")
    ap.add_argument("--n-bits", type=int, default=64,
                    help="largeur d Omega. 64 et non 256 : mesure du 2026-08-21, "
                         "cf. docs/a-faire-memoire.md -- a 256 bits le canal "
                         "plafonne (bit_acc 0.61, 0/5 AUTHENTIC), a 64 il rend "
                         "5/5 a zero bit d erreur.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.keys_dir:
        keys = CipherMarkKeys.from_files(
            os.path.join(args.keys_dir, "s_master.bin"),
            os.path.join(args.keys_dir, "k_secret.bin"),
        )
    else:
        keys = CipherMarkKeys.random()
        print("(cles ephemeres -- utiliser --keys-dir pour des cles persistantes)")

    # phash + witness
    phash = PerceptualHash(n_bits=args.n_bits).to(device)
    witness = WitnessField(keys.s_master, keys.k_secret,
                           WitnessConfig(n_bits=args.n_bits))
    verifier = CipherMarkVerifier(witness)

    # echantillon synthetique: images aleatoires douces (degrade + gaussien)
    torch.manual_seed(0)
    base = torch.rand(args.n, 3, 256, 256, device=device)
    base = 0.5 * base + 0.5 * F.avg_pool2d(base, 7, stride=1, padding=3)

    print(f"\n[eval] n={args.n}, n_bits={args.n_bits}, device={device}\n")
    print("Phase 1 -- stabilite du PHash sous distorsions")
    print("-" * 60)
    hashes_id = phash(base).cpu().numpy()
    for name, fn in DISTORTIONS.items():
        with torch.no_grad():
            distorted = fn(base.clone())
        h_d = phash(distorted).cpu().numpy()
        sims = [hamming_stability(hashes_id[i], h_d[i]) for i in range(args.n)]
        print(f"  {name:14s}  similarity = {np.mean(sims):.4f} +/- {np.std(sims):.4f}")

    print("\nPhase 2 -- Equation CipherMark (round-trip + replay)")
    print("-" * 60)
    # On simule un round-trip parfait: Omega(h) -> verify(h) doit etre OK.
    correct = 0
    forgeable = 0
    image_ids = list(range(args.n))
    for i in range(args.n):
        h = np.packbits(hashes_id[i], bitorder="big").tobytes()
        omega = witness.build_omega(h, image_ids[i])
        r = verifier.verify(omega, h, image_ids[i])
        if r.verdict.value == "authentic":
            correct += 1
        # replay attack: meme Omega mais image differente -> doit echouer
        other = (i + 1) % args.n
        h_other = np.packbits(hashes_id[other], bitorder="big").tobytes()
        r2 = verifier.verify(omega, h_other, image_ids[i])
        if r2.verdict.value == "authentic":
            forgeable += 1

    print(f"  round-trip OK     : {correct}/{args.n}")
    print(f"  replay reussi (BAD): {forgeable}/{args.n}")

    print("\nPhase 3 -- p-value baseline")
    print("-" * 60)
    # un Omega tire au hasard sur une image quelconque
    np.random.seed(1)
    n_fake = 50
    fake_pass = 0
    for i in range(n_fake):
        h = np.packbits(hashes_id[i % args.n], bitorder="big").tobytes()
        omega = np.random.randint(0, 2, args.n_bits, dtype=np.uint8)
        r = verifier.verify(omega, h, image_ids[i % args.n])
        if r.verdict.value == "authentic":
            fake_pass += 1
    print(f"  faux positifs (bruit aleatoire): {fake_pass}/{n_fake}")

    print("\n[eval] termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Profil du couple embedder/extracteur : de quelle machine a-t-on VRAIMENT besoin ?

Mesure trois choses que `nvidia-smi` ne dit pas :

  1. La taille reelle du modele (parametres, memoire des poids et des gradients).
  2. Le debit en images/seconde selon la taille de lot. C'est LE test decisif :
     si le debit croit avec la taille de lot, le GPU n'est PAS sature et on
     paie de la puissance inutilisee. S'il plafonne, on est bien limite par le
     calcul et la machine est adaptee.
  3. Le pic memoire par taille de lot, qui determine la VRAM minimale requise.

Attention : `nvidia-smi` affichant 98-100 % ne signifie pas que le GPU est
sature -- ce compteur indique seulement qu'au moins un noyau tournait pendant
l'echantillonnage. Un petit modele avec de petits lots peut afficher 100 %
en n'exploitant qu'une fraction des unites de calcul.

Usage :
    python3 scripts/ciphermark/profile_model.py --nbits 64 --img_size 256
    python3 scripts/ciphermark/profile_model.py --batches 8,16,32,64,128
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from distseal.models import Wam, build_embedder, build_extractor  # noqa: E402
from distseal.augmentation.augmenter import Augmenter  # noqa: E402


def human(n):
    for u in ["", "K", "M", "G"]:
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}T"


def build(nbits, img_size, device):
    ecfg = OmegaConf.load("configs/embedder.yaml")
    embedder = build_embedder("unet_small2_yuv_quant",
                              ecfg["unet_small2_yuv_quant"], nbits, 1)
    xcfg = OmegaConf.load("configs/extractor.yaml")
    extractor = build_extractor("convnext_tiny", xcfg["convnext_tiny"],
                                img_size, nbits)
    acfg = OmegaConf.load("configs/augmentation/identity_only.yaml")
    acfg.num_augs = 1
    wam = Wam(embedder, extractor, Augmenter(**acfg), None,
              scaling_w=0.2, scaling_i=1.0, img_size=img_size)
    return wam.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batches", default="8,16,32,64")
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wam = build(args.nbits, args.img_size, device)

    n_emb = sum(p.numel() for p in wam.embedder.parameters())
    n_ext = sum(p.numel() for p in wam.detector.parameters())
    tot = n_emb + n_ext
    print(f"device = {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU    = {p.name}, {p.total_memory/1e9:.1f} Go, {p.multi_processor_count} SM")
    print()
    print("TAILLE DU MODELE")
    print(f"  embedder  (U-Net)      : {human(n_emb)} parametres")
    print(f"  extracteur (ConvNeXt)  : {human(n_ext)} parametres")
    print(f"  total                  : {human(tot)}")
    print(f"  poids en fp32          : {tot*4/1e6:.0f} Mo")
    print(f"  + gradients + Adam     : {tot*16/1e6:.0f} Mo  (4 copies : poids, grad, m, v)")
    print()

    if device != "cuda":
        print("Pas de GPU ici -- le banc de debit doit tourner sur le pod.")
        return 0

    print("DEBIT ET MEMOIRE PAR TAILLE DE LOT")
    print(f"{'lot':>5} {'img/s':>9} {'ms/iter':>9} {'pic VRAM':>10} {'gain':>7}")
    ref = None
    opt = torch.optim.AdamW(wam.parameters(), lr=1e-4)
    for bs in [int(x) for x in args.batches.split(",")]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            imgs = torch.rand(bs, 3, args.img_size, args.img_size, device=device)
            masks = torch.ones(bs, 1, args.img_size, args.img_size, device=device)
            msgs = torch.randint(0, 2, (bs, args.nbits), device=device).float()
            # rodage
            for _ in range(3):
                out = wam(imgs, masks, msgs)
                loss = out["preds"][:, 1:].float().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                out = wam(imgs, masks, msgs)
                loss = out["preds"][:, 1:].float().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.iters
            ips = bs / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            if ref is None:
                ref = ips; gain = "ref"
            else:
                gain = f"x{ips/ref:.2f}"
            print(f"{bs:>5} {ips:>9.1f} {dt*1000:>9.1f} {peak:>9.1f}G {gain:>7}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>5} {'OOM':>9}")
            break
        except Exception as exc:
            print(f"{bs:>5}  erreur : {exc!r}")
            break

    print()
    print("LECTURE : si le debit (img/s) croit encore avec la taille de lot, le")
    print("GPU n'est PAS sature -- une machine moins chere ferait le meme travail,")
    print("ou une taille de lot plus grande accelererait l'entrainement. S'il")
    print("plafonne, on est bien limite par le calcul.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

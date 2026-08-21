#!/usr/bin/env python3
"""
Evaluation de recuperation CipherMark, a l'echelle et sous attaques.

Ce que ce script mesure, et pourquoi la decomposition compte
-----------------------------------------------------------
Un verdict AUTHENTIC exige DEUX conditions independantes, et les confondre
empeche de comprendre les echecs :

  1. le TEMOIN survit      -- le detector relit Omega correctement depuis
                              l'image attaquee (bit_acc_omega)
  2. le HASH reste stable  -- PHash(image attaquee), apres correction
                              Reed-Solomon, redonne le hash de reference
                              (hash_stable)

La deuxieme est propre a CipherMark et n'existe pas dans un tatouage
classique : comme Omega = HMAC(K, h), le verifieur RECALCULE h sur l'image
recue. Si le hash bouge d'un seul bit, l'avalanche HMAC fait diverger le tag
d'environ 50 % -- meme avec un temoin parfaitement lu. Un echec peut donc
venir du canal OU du hash, et seule la decomposition le dit.

On mesure aussi la BER finale (apres comparaison au tag recalcule), le verdict,
la p-value, et la qualite d'image (PSNR/SSIM).

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/eval_recovery_robustness.py \
        --checkpoint runs/ciphermark_64bits_checkpoint.pth --n-images 50
    PYTHONPATH=. python3 scripts/ciphermark/eval_recovery_robustness.py \
        --n-images 997 --corpus corpus-50k --no-attacks
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
from distseal.utils.cfg import setup_model_from_checkpoint  # noqa: E402
import distseal.utils.optim as uoptim  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(f"[eval] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def replay_scaling(wam, cfg, ckpt_path):
    """setup_model() reconstruit scaling_w depuis la valeur STATIQUE de la
    config, pas sa valeur effective a l'epoque du checkpoint. On rejoue le
    calendrier -- sans cela on evalue un filigrane a la mauvaise amplitude."""
    if getattr(cfg.args, "scaling_w_schedule", None) is None:
        return
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        epoch = ck.get("epoch")
        if epoch is None:
            return
        params = uoptim.parse_params(cfg.args.scaling_w_schedule)
        sc = uoptim.ScalingScheduler(obj=wam.blender, attribute="scaling_w",
                                     scaling_o=cfg.args.scaling_w, **params)
        effective = sc.step(epoch)
        log(f"scaling_w rejoue a l'epoque {epoch} : {cfg.args.scaling_w} (statique) "
            f"-> {effective:.4f} (effectif)")
    except Exception as exc:
        log(f"replay scaling_w IMPOSSIBLE ({exc!r}) -- resultats non fiables")


def load_images(corpus, size, n, seed):
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for root, _, names in os.walk(corpus):
        for nm in sorted(names):
            if nm.lower().endswith(exts):
                files.append(os.path.join(root, nm))
    if not files:
        raise SystemExit(f"aucune image dans {corpus}")
    random.Random(seed).shuffle(files)
    files = files[:n]
    out = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((size, size))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(out), files


def psnr_ssim(a, b):
    mse = F.mse_loss(a, b, reduction="none").mean(dim=(1, 2, 3)).clamp_min(1e-12)
    p = (10 * torch.log10(1.0 / mse)).cpu().numpy()
    # SSIM global simplifie (moyenne sur canaux, fenetre globale)
    mu_a, mu_b = a.mean(dim=(1, 2, 3)), b.mean(dim=(1, 2, 3))
    va, vb = a.var(dim=(1, 2, 3)), b.var(dim=(1, 2, 3))
    cov = ((a - mu_a[:, None, None, None]) * (b - mu_b[:, None, None, None])).mean(dim=(1, 2, 3))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return p, s.cpu().numpy()


def build_attacks():
    """Attaques issues des modules du projet, parametres FIXES (pas aleatoires)
    pour que les chiffres soient reproductibles."""
    A = []
    A.append(("aucune", lambda x: x))
    for q in (90, 70, 50, 30):
        j = valuemetric.JPEG()
        A.append((f"jpeg_q{q}", lambda x, j=j, q=q: j(x, None, quality=q)[0]))
    for k in (3, 5, 7):
        g = valuemetric.GaussianBlur()
        A.append((f"flou_k{k}", lambda x, g=g, k=k: g(x, None, kernel_size=k)[0]))
    for s in (0.75, 0.5):
        r = geometric.Resize()
        A.append((f"resize_{s}", lambda x, r=r, s=s: r(x, None, size=s)[0]))
    for s in (0.9, 0.7, 0.5):
        c = geometric.Crop()
        A.append((f"crop_{s}", lambda x, c=c, s=s: c(x, None, size=s)[0]))
    for f in (0.8, 1.2):
        b = valuemetric.Brightness()
        A.append((f"lumin_{f}", lambda x, b=b, f=f: b(x, None, f)[0]))
    for f in (0.8, 1.2):
        c = valuemetric.Contrast()
        A.append((f"contraste_{f}", lambda x, c=c, f=f: c(x, None, f)[0]))
    return A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-colab/val")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-attacks", action="store_true")
    ap.add_argument("--out", default="runs/eval_recovery_robustness.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    log(f"device={DEVICE}  checkpoint={args.checkpoint}")

    wam, cfg = setup_model_from_checkpoint(args.checkpoint), None
    from distseal.utils.cfg import get_config_from_checkpoint
    cfg = get_config_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits = int(cfg.args.nbits)
    img_size = int(cfg.args.img_size)
    log(f"Wam charge : nbits={nbits}, img_size={img_size}, "
        f"scaling_w effectif={wam.blender.scaling_w:.4f}")

    dino = _try_load_dinov2()
    log(f"PHash backbone : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}")
    phash = PerceptualHash(n_bits=nbits, backbone=dino or _DCTFallback()).to(DEVICE).eval()

    imgs_all, files = load_images(args.corpus, img_size, args.n_images, args.seed)
    log(f"{len(files)} images chargees depuis {args.corpus}")

    attacks = [("aucune", lambda x: x)] if args.no_attacks else build_attacks()
    log(f"{len(attacks)} conditions a evaluer")

    cm_cfg = CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3)
    keys = CipherMarkKeys.random()

    acc = {name: {"bit_acc": [], "ber": [], "verdicts": [], "hash_stable": [], "p": []}
           for name, _ in attacks}
    psnr_all, ssim_all, iters_all, conv_all = [], [], [], []

    nb = (len(imgs_all) + args.batch - 1) // args.batch
    for bi in range(nb):
        imgs = imgs_all[bi * args.batch:(bi + 1) * args.batch].to(DEVICE)
        registry = TraceRegistry()
        cm = CipherMarkWam(wam=wam, phash=phash, keys=keys, cfg=cm_cfg, registry=registry)
        out = cm.embed(imgs)
        imgs_w, omega_true, ids = out["imgs_w"], out["omega"], out["image_ids"]
        h_ref = out["h_bytes"]
        iters_all.append(out["n_iters"]); conv_all.append(bool(out["converged"]))
        p, s = psnr_ssim(imgs, imgs_w)
        psnr_all += list(p); ssim_all += list(s)

        for name, fn in attacks:
            try:
                x = fn(imgs_w.clone()).clamp(0, 1)
                if x.shape[-2:] != (img_size, img_size):
                    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                                      align_corners=False, antialias=True)
                # 1) le temoin survit-il ?
                det = cm.wam.detect(x, **cm._batch_kwargs())
                ml = det["preds"][:, 1:1 + nbits]
                if ml.dim() > 2:
                    ml = ml.mean(dim=tuple(range(2, ml.dim())))
                om = (ml > 0).to(torch.uint8).cpu().numpy()
                ot = omega_true
                if isinstance(ot, torch.Tensor):
                    ot = ot.detach().cpu().numpy()
                ot = np.asarray(ot).astype(np.uint8)
                acc[name]["bit_acc"] += list((om == ot).astype(np.float32).mean(axis=1))
                # 2) le hash reste-t-il stable ?
                h_obs = cm._phash_bytes(x)
                h_hat = cm._corrige(h_obs, [registry.parity_for(i) for i in ids])
                acc[name]["hash_stable"] += [int(h_hat[k] == h_ref[k]) for k in range(len(ids))]
                # 3) verdict complet
                for r in cm.verify(x, ids):
                    acc[name]["ber"].append(r.ber)
                    acc[name]["verdicts"].append(r.verdict.value)
                    acc[name]["p"].append(r.p_value)
            except Exception as exc:
                log(f"  attaque {name} en echec : {exc!r}")
        log(f"lot {bi + 1}/{nb} traite")

    # ---------------------------------------------------------------- sortie
    print()
    print("=" * 96)
    print(f"CipherMark -- recuperation sur {len(files)} images, nbits={nbits}")
    print(f"qualite du filigrane : PSNR {np.mean(psnr_all):.2f} dB  "
          f"SSIM {np.mean(ssim_all):.3f}  |  point fixe : {np.mean(iters_all):.2f} iters, "
          f"{100 * np.mean(conv_all):.0f}% converges")
    print("=" * 96)
    print(f"{'attaque':<16} {'bit_acc':>8} {'hash ok':>8} {'BER':>8} {'AUTH':>7} {'p med':>10}")
    print("-" * 96)
    rows = []
    for name, _ in attacks:
        d = acc[name]
        if not d["ber"]:
            continue
        ba = float(np.mean(d["bit_acc"])); hs = float(np.mean(d["hash_stable"]))
        be = float(np.mean(d["ber"]))
        au = d["verdicts"].count("authentic") / len(d["verdicts"])
        pm = float(np.median(d["p"]))
        print(f"{name:<16} {ba:>8.4f} {hs:>7.0%} {be:>8.4f} {au:>6.0%} {pm:>10.2e}")
        rows.append({"attaque": name, "bit_acc_omega": ba, "hash_stable": hs,
                     "ber": be, "taux_authentic": au, "p_median": pm,
                     "verdicts": {v: d["verdicts"].count(v) for v in set(d["verdicts"])}})
    print("=" * 96)
    print("bit_acc = fraction des bits d'Omega relus correctement")
    print("hash ok = fraction des images dont le hash corrige redonne la reference")
    print("BER     = ecart final au tag recalcule (c'est LUI qui decide du verdict)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_images": len(files), "nbits": nbits,
                   "psnr": float(np.mean(psnr_all)), "ssim": float(np.mean(ssim_all)),
                   "fixed_point_iters": float(np.mean(iters_all)),
                   "resultats": rows}, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

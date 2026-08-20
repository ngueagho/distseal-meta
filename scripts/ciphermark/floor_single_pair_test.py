#!/usr/bin/env python3
"""
TEST PLANCHER ABSOLU -- une seule image, un seul message FIXE.

But
---
Repondre sans ambiguite a : "le pipeline embedder/extracteur est-il capable
d'apprendre quoi que ce soit ?"

C'est le test le plus favorable qui puisse exister :
  * UNE image, toujours la meme
  * UN message, toujours le meme (contrairement a train.py qui tire un message
    aleatoire a chaque iteration -- cf. wam.py:94-96)
  * aucune augmentation, aucune attaque
  * aucune pression de fidelite, aucun discriminateur

Dans ces conditions le modele n'a rien a generaliser : il lui suffit de
memoriser un unique motif. Si bit_acc ne monte pas a ~1.0, le probleme est
STRUCTUREL (gradient, appariement message/lecture, saturation) et aucun budget
GPU ne le resoudra.

Le test balaie aussi plusieurs scaling_w, parce qu'on a mesure que le clamp
`imgs_w = torch.clamp(imgs_w, 0, 1)` (wam.py:186) tue le gradient des que la
perturbation sature : a scaling_w=0.5, 17 a 33 % des pixels etaient satures.

Usage:
    python3 scripts/ciphermark/floor_single_pair_test.py
    python3 scripts/ciphermark/floor_single_pair_test.py --nbits 16 --steps 1500
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from distseal.models import Wam, build_embedder, build_extractor  # noqa: E402
from distseal.augmentation.augmenter import Augmenter  # noqa: E402


def build_wam(nbits: int, img_size: int, scaling_w: float, device: str) -> Wam:
    embedder_cfg = OmegaConf.load("configs/embedder.yaml")
    embedder = build_embedder("unet_small2_yuv_quant",
                              embedder_cfg["unet_small2_yuv_quant"], nbits, 1)

    extractor_cfg = OmegaConf.load("configs/extractor.yaml")
    extractor = build_extractor("convnext_tiny", extractor_cfg["convnext_tiny"],
                                img_size, nbits)

    aug_cfg = OmegaConf.load("configs/augmentation/identity_only.yaml")
    aug_cfg.num_augs = 1
    augmenter = Augmenter(**aug_cfg)

    wam = Wam(embedder, extractor, augmenter, None,
              scaling_w=scaling_w, scaling_i=1.0, img_size=img_size)
    return wam.to(device)


def load_one_image(path: str, img_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1 c h w
    return t.to(device)


def run_one(scaling_w: float, nbits: int, img_size: int, steps: int,
            lr: float, img_path: str, device: str, seed: int = 0,
            n_messages: int = 1) -> dict:
    """n_messages : taille du vivier de messages distincts.
    1 = un seul message fixe (memorisation pure).
    N = on tire a chaque pas parmi N messages fixes.
    0 = message entierement aleatoire a chaque pas (regime de train.py)."""
    torch.manual_seed(seed)
    wam = build_wam(nbits, img_size, scaling_w, device)
    imgs = load_one_image(img_path, img_size, device)

    g = torch.Generator().manual_seed(1234)
    if n_messages > 0:
        pool = torch.randint(0, 2, (n_messages, nbits),
                             generator=g).float().to(device)
    else:
        pool = None  # tirage frais a chaque pas

    masks = torch.ones_like(imgs[:, 0:1])  # NoMaskEmbedder -> tout l'image

    params = list(wam.embedder.parameters()) + list(wam.detector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    history = []
    sat_first = None
    for step in range(steps):
        if pool is None:
            msgs = torch.randint(0, 2, (1, nbits), device=device).float()
        else:
            msgs = pool[torch.randint(0, pool.shape[0], (1,))]
        out = wam(imgs, masks, msgs)
        preds = out["preds"]                      # b (1+k) [h w]
        msg_preds = preds[:, 1:]                  # b k [h w]
        # l'extracteur predit soit un message global (b k), soit un message par
        # pixel (b k h w) -- cf. detperceptual.py:216
        if msg_preds.dim() == 2:
            targs = msgs
        else:
            targs = msgs.unsqueeze(-1).unsqueeze(-1).expand_as(msg_preds)
        loss = F.binary_cross_entropy_with_logits(msg_preds, targs)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9).item()
        opt.step()

        if step % max(1, steps // 8) == 0 or step == steps - 1:
            with torch.no_grad():
                imgs_w = out["imgs_w"]
                sat = (((imgs_w <= 1e-3) | (imgs_w >= 1 - 1e-3)).float()
                       .mean().item())
                if sat_first is None:
                    sat_first = sat
                psnr = 10 * torch.log10(1.0 /
                                        F.mse_loss(imgs_w, imgs).clamp_min(1e-12))
                # bit_acc evaluee sur TOUT le vivier (ou 32 messages frais si
                # le vivier est infini) -- sinon on ne mesure que le message
                # du pas courant, ce qui n'a pas de sens des que n_messages > 1
                if pool is None:
                    eval_msgs = torch.randint(0, 2, (16, nbits),
                                              device=device).float()
                else:
                    eval_msgs = pool[:16]
                accs = []
                for m in eval_msgs:
                    o = wam(imgs, masks, m.unsqueeze(0))
                    mp = o["preds"][:, 1:]
                    if mp.dim() == 2:
                        b = (mp > 0).float()
                    else:  # vote majoritaire spatial
                        b = ((mp > 0).float().mean(dim=(-2, -1)) > 0.5).float()
                    accs.append((b == m.unsqueeze(0)).float().mean().item())
                bit_acc = float(np.mean(accs))
            history.append((step, loss.item(), bit_acc, psnr.item(), sat, gnorm))
            print(f"    step {step:5d} | loss {loss.item():.4f} | "
                  f"bit_acc {bit_acc:.3f} | psnr {psnr.item():5.1f} | "
                  f"satures {sat*100:4.1f}% | |grad| {gnorm:.2e}")

    final = history[-1]
    return {"scaling_w": scaling_w, "n_messages": n_messages,
            "final_loss": final[1], "final_bit_acc": final[2],
            "final_psnr": final[3], "sat_first": sat_first,
            "sat_final": final[4]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbits", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image", default="corpus-smoketest/train/bsds100075_0.png")
    ap.add_argument("--scalings", default="0.05,0.1,0.3,0.5")
    ap.add_argument("--messages", default=None,
                    help="balayage du nombre de messages distincts, ex "
                         "'1,2,4,16,64,0' (0 = aleatoire pur, regime train.py). "
                         "Si fourni, --scalings est fige a sa premiere valeur.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  nbits={args.nbits}  img_size={args.img_size}  "
          f"steps={args.steps}  lr={args.lr}")
    print(f"image={args.image}")

    results = []
    if args.messages:
        sw = float(args.scalings.split(",")[0])
        print(f"BALAYAGE du vivier de messages, scaling_w fige a {sw}.")
        print("UNE image, aucune augmentation, aucune pression de fidelite.\n")
        for n in [int(x) for x in args.messages.split(",")]:
            label = "aleatoire pur" if n == 0 else f"{n} message(s)"
            print(f"--- vivier = {label} ---")
            results.append(run_one(sw, args.nbits, args.img_size, args.steps,
                                   args.lr, args.image, device,
                                   n_messages=n))
            print()
        key, head = "n_messages", "n_msgs"
    else:
        print("UNE image, UN message fixe, aucune augmentation, aucune "
              "pression de fidelite.\n")
        for sw in [float(x) for x in args.scalings.split(",")]:
            print(f"--- scaling_w = {sw} ---")
            results.append(run_one(sw, args.nbits, args.img_size, args.steps,
                                   args.lr, args.image, device))
            print()
        key, head = "scaling_w", "scaling_w"

    print("=" * 78)
    print(f"{head:>10} {'bit_acc':>9} {'loss':>9} {'psnr':>7} "
          f"{'sat.debut':>10} {'sat.fin':>9}")
    for r in results:
        v = r[key]
        vs = "aleat." if (key == "n_messages" and v == 0) else f"{v:g}"
        print(f"{vs:>10} {r['final_bit_acc']:>9.3f} "
              f"{r['final_loss']:>9.4f} {r['final_psnr']:>7.1f} "
              f"{r['sat_first']*100:>9.1f}% {r['sat_final']*100:>8.1f}%")
    print("=" * 78)

    best = max(r["final_bit_acc"] for r in results)
    print()
    if args.messages:
        # En mode balayage, le verdict "structurel" n'a pas de sens : il ne
        # s'applique qu'au cas UN message fixe, ou l'echec serait forcement un
        # bug. Ici on mesure une degradation, pas une panne.
        base = next((r["final_bit_acc"] for r in results
                     if r["n_messages"] == 1), None)
        print("Lecture : la degradation avec la taille du vivier mesure la")
        print("difficulte de GENERALISATION a des messages arbitraires.")
        if base is not None and base >= 0.95:
            print(f"Le cas a 1 message atteint {base:.3f} -- le pipeline")
            print("fonctionne, donc ce qui suit n'est pas un bug mais un mur")
            print("de capacite.")
        return 0
    if best >= 0.95:
        print(f"VERDICT : le pipeline APPREND (bit_acc max {best:.3f}).")
        print("Le blocage a l'echelle reelle est un probleme de capacite, de")
        print("volume de donnees ou d'optimisation -- pas un bug structurel.")
    elif best >= 0.75:
        print(f"VERDICT : apprentissage PARTIEL (bit_acc max {best:.3f}).")
        print("Le pipeline n'est pas mort mais quelque chose le bride.")
    else:
        print(f"VERDICT : le pipeline N'APPREND PAS (bit_acc max {best:.3f}).")
        print("Sur UNE image et UN message fixe, c'est un probleme STRUCTUREL.")
        print("Aucun budget GPU ne le resoudra -- il faut debuguer le code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Teste les tatoueurs pre-entraines publies par Meta (dossier cards/).

Question posee
--------------
Les quatre fiches `cards/detector_*.yaml` pointent vers des checkpoints Meta.
Malgre leur nom, `load_watermarker()` en charge le modele COMPLET -- embedder
ET extracteur. Trois inconnues a lever :

  1. Combien de bits transportent-ils ?
  2. Travaillent-ils en espace PIXEL ou en espace LATENT ? (les scaling_w des
     fiches et les configs `latent_layer: input_after_quantize` suggerent du
     latent, ce qui les rendrait inutilisables tels quels pour CipherMark qui
     opere en pixel post-hoc)
  3. Lisent-ils un message ARBITRAIRE, ou seulement celui fige avec lequel ils
     ont ete distilles ? C'est LA question qui decide de tout : un extracteur
     mono-message ne sert a rien pour Omega, qui change a chaque image.

Le test compare systematiquement deux regimes :
  * le message de reference livre avec la fiche (.npy / .txt)
  * des messages tires au hasard

Si bit_acc est eleve sur le message de reference mais au hasard sur les
messages aleatoires, le detecteur est mono-message -> inutilisable pour
CipherMark. S'il est eleve dans les deux cas, c'est un canal multi-bits
reutilisable, et cela dispense d'entrainer le sien.

Usage (depuis /workspace/code-memoire sur le pod) :
    python3 scripts/ciphermark/test_meta_detectors.py
    python3 scripts/ciphermark/test_meta_detectors.py --cards detector_dc-ae
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from distseal.loader import load_watermarker  # noqa: E402
import distseal.augmentation.neuralcompression as neuralcompression  # noqa: E402

CARDS = ["detector_dc-ae.yaml", "detector_rar.yaml",
         "detector_maskgit_prequant.yaml", "detector_maskgit_postquant.yaml"]

# Ces tatoueurs operent en espace LATENT : leur embedder attend le latent du
# modele generatif associe (128 canaux pour le DC-AE), pas des pixels RGB.
# Il faut leur brancher l'autoencodeur via les wrappers de neuralcompression,
# qui exposent encode_pre_quant/quantize/decode -- l'interface que Wam attend
# (cf. train.py: `getattr(neuralcompression, params.autoencoder)()`). Le modele
# HuggingFace brut ne l'expose pas.
DECODER_FOR_CARD = {
    "detector_dc-ae.yaml": "DCAEf64c128",
    "detector_maskgit_prequant.yaml": "MaskgitVqgan",
    "detector_maskgit_postquant.yaml": "MaskgitVqgan",
    "detector_rar.yaml": "MaskgitVqgan",
}


def load_images(paths, size, device):
    ims = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        ims.append(torch.from_numpy(
            np.asarray(img).astype(np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(ims).to(device)


def bits_from_preds(preds, nbits):
    """preds: b x (1+k) [x h x w] -> b x k binaire, meme regle que
    distseal/utils/metrics.bit_accuracy (vote majoritaire spatial)."""
    mp = preds[:, 1:1 + nbits]
    if mp.dim() == 2:
        return (mp > 0).float()
    return ((mp > 0).float().mean(dim=(-2, -1)) > 0.5).float()


def try_roundtrip(wm, imgs, msgs, scaling_w):
    """Retourne (bit_acc, psnr) ou leve."""
    if scaling_w is not None:
        try:
            wm.blender.scaling_w = float(scaling_w)
        except Exception:
            pass
    with torch.no_grad():
        out = wm.embed(imgs, msgs=msgs, is_video=False)
        imgs_w = out["imgs_w"]
        det = wm.detect(imgs_w, is_video=False)
        preds = det["preds"] if isinstance(det, dict) else det
        nbits = msgs.shape[1]
        got = bits_from_preds(preds, nbits)
        acc = (got == msgs.float()).float().mean().item()
        mse = F.mse_loss(imgs_w.clamp(0, 1), imgs).clamp_min(1e-12)
        psnr = (10 * torch.log10(1.0 / mse)).item()
    return acc, psnr


def test_card(card, imgs_paths, size, device, n_random=5):
    print("=" * 72)
    print(f"FICHE : {card}")
    print("=" * 72)
    try:
        wm, ref_msg, scaling_w = load_watermarker(card, device=device)
    except Exception as exc:
        print(f"  ECHEC DE CHARGEMENT : {exc!r}")
        traceback.print_exc()
        return {"card": card, "status": "load_failed"}

    # brancher l'autoencodeur : l'embedder attend le latent, pas les pixels
    ae_name = DECODER_FOR_CARD.get(card)
    if getattr(wm, "autoencoder", None) is None and ae_name:
        try:
            wm.autoencoder = getattr(neuralcompression, ae_name)().to(device)
            print(f"  autoencodeur branche : neuralcompression.{ae_name}")
        except Exception as exc:
            print(f"  autoencodeur NON branche ({ae_name}) : {exc!r}")

    has_ae = getattr(wm, "autoencoder", None) is not None
    latent_layer = getattr(wm, "latent_layer", None)
    nbits = None
    if ref_msg is not None:
        nbits = ref_msg.shape[-1]
    print(f"  scaling_w de la fiche : {scaling_w}")
    print(f"  autoencodeur present  : {has_ae}   (True => espace LATENT)")
    print(f"  latent_layer          : {latent_layer}")
    print(f"  message de reference  : {'oui, %d bits' % nbits if nbits else 'aucun'}")

    if nbits is None:
        try:
            nbits = wm.get_random_msg(1).shape[1]
            print(f"  nbits deduit du modele : {nbits}")
        except Exception:
            print("  impossible de deduire nbits -> abandon")
            return {"card": card, "status": "nbits_unknown"}

    imgs = load_images(imgs_paths, size, device)
    res = {"card": card, "status": "ok", "nbits": nbits,
           "latent": has_ae, "scaling_w": scaling_w}

    # 1) message de reference livre avec la fiche
    if ref_msg is not None:
        try:
            m = ref_msg.float().to(device).repeat(imgs.shape[0], 1)
            acc, psnr = try_roundtrip(wm, imgs, m, scaling_w)
            res["acc_ref"] = acc
            res["psnr_ref"] = psnr
            print(f"  [message de reference] bit_acc {acc:.4f}   psnr {psnr:.1f} dB")
        except Exception as exc:
            print(f"  [message de reference] ECHEC : {exc!r}")
            res["acc_ref"] = None

    # 2) messages ARBITRAIRES -- la question qui decide
    accs = []
    for t in range(n_random):
        try:
            m = torch.randint(0, 2, (imgs.shape[0], nbits), device=device).float()
            acc, psnr = try_roundtrip(wm, imgs, m, scaling_w)
            accs.append(acc)
            print(f"  [aleatoire {t + 1}/{n_random}]     bit_acc {acc:.4f}   psnr {psnr:.1f} dB")
        except Exception as exc:
            print(f"  [aleatoire {t + 1}/{n_random}]     ECHEC : {exc!r}")
            break
    if accs:
        res["acc_random"] = float(np.mean(accs))
        print(f"  --> moyenne sur messages aleatoires : {res['acc_random']:.4f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", nargs="*", default=CARDS)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--corpus", default="corpus-colab/val")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(os.listdir(args.corpus))[:args.n_images]
    paths = [os.path.join(args.corpus, f) for f in files]
    print(f"device={device}  images={len(paths)}  taille={args.img_size}\n")

    results = [test_card(c, paths, args.img_size, device) for c in args.cards]

    print("\n" + "=" * 72)
    print("SYNTHESE")
    print("=" * 72)
    print(f"{'fiche':<30} {'nbits':>6} {'latent':>7} {'acc_ref':>8} {'acc_alea':>9}")
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['card']:<30} {r.get('status')}")
            continue
        ar = r.get("acc_ref")
        aa = r.get("acc_random")
        print(f"{r['card']:<30} {r['nbits']:>6} {str(r['latent']):>7} "
              f"{('%.4f' % ar) if ar is not None else '-':>8} "
              f"{('%.4f' % aa) if aa is not None else '-':>9}")
    print()
    print("LECTURE : si acc_alea est proche de 0.5 alors que acc_ref est elevee,")
    print("le detecteur est MONO-MESSAGE et ne peut pas transporter Omega.")
    print("Si acc_alea est elevee, c'est un canal multi-bits reutilisable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

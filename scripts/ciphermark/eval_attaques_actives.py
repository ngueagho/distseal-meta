#!/usr/bin/env python3
"""
Attaques ACTIVES contre CipherMark : un adversaire qui cherche a forger.

Difference avec eval_recovery_robustness.py
-------------------------------------------
Ce script-la mesure des degradations PASSIVES (JPEG, recadrage, flou) :
personne n'attaque, l'image est simplement abimee. C'est la robustesse.

Ici l'adversaire VEUT tromper le verifieur. C'est l'infalsifiabilite -- la
contribution reelle de CipherMark, et la seule chose qui la distingue de
WOUAF/WMAdapter, dont le code utilisateur est arbitraire donc forgeable.

Modele de menace (repris de MetaSeal, TMLR 02/2026, section 3.1)
----------------------------------------------------------------
L'attaquant connait l'algorithme, dispose d'images marquees, mais n'a PAS les
cles. On teste quatre strategies, de la plus naive a la plus informee :

  A. TRANSPLANTATION  -- copier le filigrane d'une image sur une autre.
     C'est l'attaque que la dependance au contenu doit rendre inoperante :
     Omega = HMAC(K, h) est lie au hash de SON image ; sur une autre image le
     verifieur recalcule un h different, donc un tag different.

  B. REJEU            -- reutiliser une image marquee valide en pretendant
     qu'elle correspond a un autre enregistrement du registre (autre nonce).

  C. MIXUP            -- estimer le signal du filigrane en moyennant les
     residus (marquee - originale) de N images, puis l'appliquer a une image
     vierge. C'est l'attaque classique contre les tatouages additifs
     content-agnostiques (cf. MetaSeal eq. 1-2).

  D. COLLAGE          -- moyenner deux images marquees par des cles
     differentes, pour voir si l'une des deux identites ressort.

Ce qu'on mesure a chaque fois : le verdict rendu, le BER, et surtout le
nombre de FAUX AUTHENTIC -- une seule occurrence suffirait a invalider la
propriete d'infalsifiabilite.

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/eval_attaques_actives.py \
        --checkpoint runs/ciphermark_64bits_checkpoint.pth --n-images 20
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
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint  # noqa: E402
import distseal.utils.optim as uoptim  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(f"[attaques] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def replay_scaling(wam, cfg, ckpt_path):
    """scaling_w est un float Python, pas un buffer : setup_model() le
    reconstruit a sa valeur STATIQUE. Sans rejouer le calendrier on evalue un
    filigrane a la mauvaise amplitude (cf. full_chain_real_weights_test.py)."""
    if getattr(cfg.args, "scaling_w_schedule", None) is None:
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    epoch = ck.get("epoch")
    if epoch is None:
        return
    params = uoptim.parse_params(cfg.args.scaling_w_schedule)
    sc = uoptim.ScalingScheduler(obj=wam.blender, attribute="scaling_w",
                                 scaling_o=cfg.args.scaling_w, **params)
    log(f"scaling_w rejoue a l'epoque {epoch} : {cfg.args.scaling_w} -> {sc.step(epoch):.4f}")


def load_images(corpus, size, n, seed):
    exts = (".png", ".jpg", ".jpeg")
    files = [os.path.join(r, f) for r, _, ns in os.walk(corpus)
             for f in sorted(ns) if f.lower().endswith(exts)]
    if len(files) < n:
        raise SystemExit(f"{corpus} ne contient que {len(files)} images")
    random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f).convert("RGB").resize((size, size))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(out)


def summarize(name, reports, note=""):
    """Compte les faux AUTHENTIC. C'est LA metrique : une seule occurrence
    invaliderait l'infalsifiabilite."""
    verdicts = [r.verdict.value for r in reports]
    bers = [r.ber for r in reports]
    n_auth = verdicts.count("authentic")
    return {"attaque": name, "n": len(reports), "faux_authentic": n_auth,
            "ber_moyen": float(np.mean(bers)) if bers else None,
            "verdicts": {v: verdicts.count(v) for v in set(verdicts)},
            "note": note}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-colab/val")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval_attaques_actives.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() - 1)))
    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    nbits, img_size = int(cfg.args.nbits), int(cfg.args.img_size)
    log(f"Wam charge : nbits={nbits}, img_size={img_size}, device={DEVICE}")

    dino = _try_load_dinov2()
    log(f"PHash : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}")
    phash = PerceptualHash(n_bits=nbits, backbone=dino or _DCTFallback()).to(DEVICE).eval()

    n = args.n_images
    imgs = load_images(args.corpus, img_size, 2 * n, args.seed).to(DEVICE)
    src, other = imgs[:n], imgs[n:]      # src = marquees, other = images vierges
    cm_cfg = CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3)

    # --- la victime : un utilisateur legitime qui marque ses images ---------
    keys_v = CipherMarkKeys.random()
    reg_v = TraceRegistry()
    victime = CipherMarkWam(wam=wam, phash=phash, keys=keys_v, cfg=cm_cfg, registry=reg_v)
    out_v = victime.embed(src)
    marquees, ids_v = out_v["imgs_w"], out_v["image_ids"]
    log(f"{n} images marquees par la victime")

    base = victime.verify(marquees, ids_v)
    n_ok = sum(1 for r in base if r.verdict.value == "authentic")
    log(f"reference (aller-retour honnete) : {n_ok}/{n} AUTHENTIC")

    resultats = [summarize("reference_honnete", base,
                            "doit etre 100 % AUTHENTIC -- sinon le reste n'a pas de sens")]

    # --- A. TRANSPLANTATION -------------------------------------------------
    # L'attaquant colle le filigrane de l'image i sur l'image vierge i.
    # Approximation realiste du residu : (marquee - originale).
    log("A. transplantation du filigrane sur une AUTRE image")
    residu = marquees - src
    forgees = (other + residu).clamp(0, 1)
    rep = victime.verify(forgees, ids_v)          # meme nonce que l'original
    resultats.append(summarize("A_transplantation", rep,
                                "la dependance au contenu doit la rendre inoperante"))

    # --- B. REJEU -----------------------------------------------------------
    # Image marquee valide, mais presentee sous le nonce d'une AUTRE image.
    log("B. rejeu -- image valide presentee sous un autre nonce")
    ids_permutes = ids_v[1:] + ids_v[:1]
    rep = victime.verify(marquees, ids_permutes)
    resultats.append(summarize("B_rejeu_autre_nonce", rep,
                                "le nonce fixe le keystream : un mauvais nonce doit echouer"))

    # --- C. MIXUP -----------------------------------------------------------
    # L'attaquant estime le signal en moyennant les residus de toutes les
    # images marquees dont il dispose, puis l'applique a une image vierge.
    log("C. mixup -- signal estime par moyenne des residus")
    signal_estime = residu.mean(dim=0, keepdim=True)
    forgees = (other + signal_estime).clamp(0, 1)
    rep = victime.verify(forgees, ids_v)
    resultats.append(summarize("C_mixup", rep,
                                "efficace contre les tatouages content-agnostiques"))

    # --- D. COLLAGE de deux identites --------------------------------------
    # Deux utilisateurs marquent la MEME image ; on moyenne les deux resultats
    # et on regarde si l'une des identites ressort.
    log("D. collage -- moyenne de deux images marquees par des cles differentes")
    keys_b = CipherMarkKeys.random()
    reg_b = TraceRegistry()
    autre = CipherMarkWam(wam=wam, phash=phash, keys=keys_b, cfg=cm_cfg, registry=reg_b)
    out_b = autre.embed(src)
    melange = ((marquees + out_b["imgs_w"]) / 2).clamp(0, 1)
    rep_v = victime.verify(melange, ids_v)
    rep_b = autre.verify(melange, out_b["image_ids"])
    resultats.append(summarize("D_collage_vu_par_A", rep_v, "moyenne de deux identites"))
    resultats.append(summarize("D_collage_vu_par_B", rep_b, "moyenne de deux identites"))

    # ------------------------------------------------------------------ bilan
    print()
    print("=" * 92)
    print(f"CipherMark -- attaques ACTIVES, {n} images, nbits={nbits}")
    print("=" * 92)
    print(f"{'attaque':<24}{'n':>4}{'faux AUTH':>11}{'BER moyen':>11}   verdicts")
    print("-" * 92)
    total_faux = 0
    for r in resultats:
        if r["attaque"] != "reference_honnete":
            total_faux += r["faux_authentic"]
        v = ", ".join(f"{k}={n_}" for k, n_ in sorted(r["verdicts"].items()))
        print(f"{r['attaque']:<24}{r['n']:>4}{r['faux_authentic']:>11}"
              f"{r['ber_moyen']:>11.4f}   {v}")
    print("=" * 92)
    if total_faux == 0:
        print("VERDICT : aucune attaque n'a produit de faux AUTHENTIC.")
        print("L'infalsifiabilite tient sur les quatre strategies testees.")
    else:
        print(f"VERDICT : {total_faux} faux AUTHENTIC -- l'infalsifiabilite est PRISE EN DEFAUT.")
        print("A rapporter tel quel : c'est un resultat, pas un bug.")
    print()
    print("Rappel : l'attaquant connait l'algorithme et dispose d'images")
    print("marquees, mais pas des cles. Aucun seuil du verifieur n'est modifie.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_images": n, "nbits": nbits, "resultats": resultats}, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

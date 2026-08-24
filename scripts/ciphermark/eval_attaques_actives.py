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
# largeur du hash perceptuel, independante de celle d'Omega
HASH_BITS = 256


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
    ap.add_argument("--batch", type=int, default=10)
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
    # hash decouple d'Omega : il ne traverse pas l'image, sa largeur n'est pas
    # contrainte par la capacite de l'extracteur (cf. CipherMarkConfig)
    phash = PerceptualHash(n_bits=HASH_BITS, backbone=dino or _DCTFallback()).to(DEVICE).eval()

    n = args.n_images
    imgs_all = load_images(args.corpus, img_size, 2 * n, args.seed)
    src_all, other_all = imgs_all[:n], imgs_all[n:]   # src = a marquer, other = vierges
    cm_cfg = CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3)

    # --- la victime, et l'utilisateur B du scenario de collage --------------
    keys_v = CipherMarkKeys.random()
    reg_v = TraceRegistry()
    victime = CipherMarkWam(wam=wam, phash=phash, keys=keys_v, cfg=cm_cfg, registry=reg_v)
    keys_b = CipherMarkKeys.random()
    reg_b = TraceRegistry()
    autre = CipherMarkWam(wam=wam, phash=phash, keys=keys_b, cfg=cm_cfg, registry=reg_b)

    # Traitement par lots. A 5000 images, garder tout le corpus marque en VRAM
    # est impossible ; les scenarios sont donc joues lot par lot et seuls les
    # rapports de verification (quelques scalaires) sont accumules.
    #
    # Le mixup fait exception : il exige la moyenne des residus de TOUTES les
    # images dont dispose l'attaquant. La moyenne est donc accumulee ici puis
    # appliquee dans une seconde passe, pour que l'attaquant reste exactement
    # aussi fort qu'avec la version non decoupee.
    acc = {k: [] for k in ("reference_honnete", "A_transplantation",
                           "B_rejeu_autre_nonce", "D_collage_vu_par_A",
                           "D_collage_vu_par_B")}
    somme_residus, n_vus, ids_tous = None, 0, []
    nb = (n + args.batch - 1) // args.batch
    log(f"{n} images, {nb} lots de {args.batch}")
    for bi in range(nb):
        sl = slice(bi * args.batch, (bi + 1) * args.batch)
        src, other = src_all[sl].to(DEVICE), other_all[sl].to(DEVICE)
        out_v = victime.embed(src)
        marquees, ids_v = out_v["imgs_w"], out_v["image_ids"]
        ids_tous.extend(ids_v)

        # reference : aller-retour honnete
        acc["reference_honnete"].extend(victime.verify(marquees, ids_v))

        residu = marquees - src
        s = residu.sum(dim=0, keepdim=True)
        somme_residus = s if somme_residus is None else somme_residus + s
        n_vus += residu.shape[0]

        # A. transplantation : le residu de l'image i colle sur la vierge i,
        #    presentee sous le nonce d'origine.
        acc["A_transplantation"].extend(
            victime.verify((other + residu).clamp(0, 1), ids_v))

        # B. rejeu : image valide presentee sous le nonce de la suivante.
        acc["B_rejeu_autre_nonce"].extend(
            victime.verify(marquees, ids_v[1:] + ids_v[:1]))

        # D. collage : la meme image marquee par deux identites, moyennee.
        out_b = autre.embed(src)
        melange = ((marquees + out_b["imgs_w"]) / 2).clamp(0, 1)
        acc["D_collage_vu_par_A"].extend(victime.verify(melange, ids_v))
        acc["D_collage_vu_par_B"].extend(autre.verify(melange, out_b["image_ids"]))
        if (bi + 1) % 10 == 0 or bi + 1 == nb:
            log(f"  lot {bi + 1}/{nb} -- {n_vus} images")

    # --- C. MIXUP : seconde passe, signal moyen sur tout le corpus ----------
    log("C. mixup -- signal estime par moyenne des residus de tout le corpus")
    signal_estime = somme_residus / n_vus
    acc["C_mixup"] = []
    for bi in range(nb):
        sl = slice(bi * args.batch, (bi + 1) * args.batch)
        forgees = (other_all[sl].to(DEVICE) + signal_estime).clamp(0, 1)
        acc["C_mixup"].extend(victime.verify(forgees, ids_tous[sl]))

    resultats = [
        summarize("reference_honnete", acc["reference_honnete"],
                  "doit etre 100 % AUTHENTIC -- sinon le reste n'a pas de sens"),
        summarize("A_transplantation", acc["A_transplantation"],
                  "la dependance au contenu doit la rendre inoperante"),
        summarize("B_rejeu_autre_nonce", acc["B_rejeu_autre_nonce"],
                  "le nonce fixe le keystream : un mauvais nonce doit echouer"),
        summarize("C_mixup", acc["C_mixup"],
                  "efficace contre les tatouages content-agnostiques"),
        summarize("D_collage_vu_par_A", acc["D_collage_vu_par_A"],
                  "moyenne de deux identites"),
        summarize("D_collage_vu_par_B", acc["D_collage_vu_par_B"],
                  "moyenne de deux identites"),
    ]
    log(f"reference (aller-retour honnete) : "
        f"{resultats[0]['faux_authentic']}/{n} AUTHENTIC")

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

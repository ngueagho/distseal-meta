#!/usr/bin/env python3
"""
Le hash a 256 bits repare-t-il la liaison au contenu, avec Omega a 64 bits ?

Pourquoi cette piste
--------------------
Omega a ete ramene a 64 bits parce que l'EXTRACTEUR ne peut pas porter
davantage (bit_acc 0.61 a 256 bits). Mais le hash, lui, ne traverse jamais
l'image : il est recalcule par le verifieur et sa parite vit dans le registre.
Rien n'oblige donc h a faire la meme taille qu'Omega. Le code le permet deja :
`hmac_sha256(k, h_bytes)` accepte un h de longueur quelconque et sort 32
octets, que `hkdf_expand` etire ensuite vers la largeur d'Omega.

Ce que le script mesure
-----------------------
1. Les trois derives, en octets (l'unite que corrige Reed-Solomon) :
     - marquage      : PHash(image) vs PHash(image marquee)   -> doit etre CORRIGE
     - attaques      : PHash(marquee) vs PHash(attaquee)      -> doit etre CORRIGE
     - AUTRE image   : la borne a ne JAMAIS corriger
2. S'il existe un nsym qui separe les deux : nsym//2 >= derive legitime max
   ET nsym//2 < derive d'une autre image.
3. Le juge de paix : l'attaque par transplantation est-elle encore possible ?

Usage :
    PYTHONPATH=. python3 scripts/ciphermark/test_hash256_omega64.py
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback
from distseal.ciphermark.registry import TraceRegistry
from distseal.ciphermark.wam_ciphermark import (CipherMarkConfig, CipherMarkKeys, CipherMarkWam)
from distseal.augmentation import valuemetric, geometric
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint
import distseal.utils.optim as uoptim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def log(m): print(f"[h256] {time.strftime('%H:%M:%S')} {m}", flush=True)

def replay_scaling(wam, cfg, ckpt):
    if getattr(cfg.args, "scaling_w_schedule", None) is None: return
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    ep = ck.get("epoch")
    if ep is None: return
    p = uoptim.parse_params(cfg.args.scaling_w_schedule)
    sc = uoptim.ScalingScheduler(obj=wam.blender, attribute="scaling_w",
                                 scaling_o=cfg.args.scaling_w, **p)
    log(f"scaling_w rejoue a l'epoque {ep} -> {sc.step(ep):.4f}")

def load_images(corpus, size, n, seed):
    exts = (".png", ".jpg", ".jpeg")
    files = [os.path.join(r, f) for r, _, ns in os.walk(corpus)
             for f in sorted(ns) if f.lower().endswith(exts)]
    random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f).convert("RGB").resize((size, size))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.).permute(2, 0, 1))
    return torch.stack(out)

def oct_diff(a, b): return sum(1 for x, y in zip(a, b) if x != y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ciphermark_64bits_checkpoint.pth")
    ap.add_argument("--corpus", default="corpus-colab/val")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--hash-bits", type=int, default=256)
    ap.add_argument("--omega-bits", type=int, default=64)
    ap.add_argument("--rs-nsym", type=int, default=32)
    ap.add_argument("--out", default="runs/test_hash256_omega64.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) - 1)))
    cfg = get_config_from_checkpoint(args.checkpoint)
    wam = setup_model_from_checkpoint(args.checkpoint)
    replay_scaling(wam, cfg, args.checkpoint)
    wam = wam.to(DEVICE).eval()
    img_size = int(cfg.args.img_size)
    nbits_wam = int(cfg.args.nbits)

    taille = args.hash_bits // 8
    log(f"hash {args.hash_bits} bits ({taille} oct) | Omega {args.omega_bits} bits "
        f"| nsym={args.rs_nsym} -> corrige {args.rs_nsym//2} oct")
    assert args.omega_bits == nbits_wam, \
        f"Omega ({args.omega_bits}) doit egaler nbits du WAM ({nbits_wam})"

    dino = _try_load_dinov2()
    log(f"PHash : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}")
    phash = PerceptualHash(n_bits=args.hash_bits, rs_nsym=args.rs_nsym,
                           backbone=dino or _DCTFallback()).to(DEVICE).eval()

    imgs_all = load_images(args.corpus, img_size, args.n_images, 0)
    cm = CipherMarkWam(wam=wam, phash=phash, keys=CipherMarkKeys.random(),
                       cfg=CipherMarkConfig(n_bits=args.omega_bits, max_fixed_point_iters=3),
                       registry=TraceRegistry())

    attaques = [("jpeg_q50", lambda x: valuemetric.JPEG()(x, None, quality=50)[0]),
                ("jpeg_q30", lambda x: valuemetric.JPEG()(x, None, quality=30)[0]),
                ("flou_k5",  lambda x: valuemetric.GaussianBlur()(x, None, kernel_size=5)[0]),
                ("resize_0.5", lambda x: geometric.Resize()(x, None, size=0.5)[0]),
                ("crop_0.9", lambda x: geometric.Crop()(x, None, size=0.9)[0])]

    # Traitement par lots : a 5000 images le corpus marque ne tient pas en VRAM.
    # La transplantation est jouee dans la meme boucle -- elle a besoin des
    # originales ET des marquees, qui ne survivent pas au lot suivant.
    d_marquage, d_attaque, echecs, h_mar_tous = [], {n: [] for n, _ in attaques}, {}, []
    n_faux, n_total = 0, 0
    nb = (len(imgs_all) + args.batch - 1) // args.batch
    log(f"marquage de {len(imgs_all)} images en {nb} lots de {args.batch}")
    for bi in range(nb):
        imgs = imgs_all[bi * args.batch:(bi + 1) * args.batch].to(DEVICE)
        if imgs.shape[0] < 2:
            break                      # un lot d'une image ne permet aucun echange
        out = cm.embed(imgs)
        marquees = out["imgs_w"]
        h_ori, h_mar = cm._phash_bytes(imgs), cm._phash_bytes(marquees)
        h_mar_tous.extend(h_mar)
        d_marquage.extend(oct_diff(a, b) for a, b in zip(h_ori, h_mar))

        for nom, fn in attaques:
            if nom in echecs:
                continue
            try:
                x = fn(marquees.clone()).clamp(0, 1)
                if x.shape[-2:] != (img_size, img_size):
                    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                                      align_corners=False, antialias=True)
                d_attaque[nom].extend(oct_diff(a, b)
                                      for a, b in zip(h_mar, cm._phash_bytes(x)))
            except Exception as e:
                echecs[nom] = repr(e)
                log(f"{nom} echec : {e!r}")

        # JUGE DE PAIX -- la vraie transplantation : le residu de l'image i est
        # greffe sur une AUTRE image, puis presente sous le nonce d'origine.
        # (Presenter l'image marquee j sous le nonce i serait un rejeu, pas une
        # transplantation : c'est une autre attaque, mesuree ailleurs.)
        autres = torch.roll(imgs, shifts=-1, dims=0)
        forgees = (autres + (marquees - imgs)).clamp(0, 1)
        try:
            for r in cm.verify(forgees, image_ids=out["image_ids"]):
                n_faux += int(r.verdict.value == "authentic")
                n_total += 1
        except Exception as e:
            log(f"transplantation en echec sur le lot {bi} : {e!r}")
        if (bi + 1) % 20 == 0 or bi + 1 == nb:
            log(f"  lot {bi + 1}/{nb} -- {len(h_mar_tous)} images traitees")

    res = [("marquage", float(np.mean(d_marquage)), int(max(d_marquage)))]
    log(f"marquage : {np.mean(d_marquage):.2f} oct (max {max(d_marquage)})")
    for nom, _ in attaques:
        d = d_attaque[nom]
        if d:
            res.append((nom, float(np.mean(d)), int(max(d))))
            log(f"{nom} : {np.mean(d):.2f} oct (max {max(d)})")

    d = [oct_diff(h_mar_tous[i], h_mar_tous[(i + 1) % len(h_mar_tous)])
         for i in range(len(h_mar_tous))]
    autre_moy, autre_min = float(np.mean(d)), int(min(d))
    res.append(("AUTRE_image", autre_moy, autre_min))
    log(f"AUTRE image : {autre_moy:.2f} oct (min {autre_min})")

    print("\n" + "=" * 78)
    print(f"DERIVES -- hash {args.hash_bits} bits ({taille} octets), {args.n_images} images")
    print("=" * 78)
    print(f"{'cas':<16}{'moyenne':>10}{'extreme':>10}")
    for nom, moy, ext in res: print(f"{nom:<16}{moy:>10.2f}{ext:>10d}")

    legit = [r for r in res if r[0] != "AUTRE_image"]
    besoin = max(r[2] for r in legit)
    print("\n" + "-" * 78)
    print(f"Derive legitime maximale : {besoin} octets  (a corriger)")
    print(f"Distance a une AUTRE image : {autre_moy:.1f} moyenne, {autre_min} minimum  (a NE PAS corriger)")
    lo, hi = 2 * besoin, 2 * autre_min
    if lo < hi:
        print(f"=> FENETRE OUVERTE : nsym dans [{lo}, {hi}[ , recommande {lo}")
        verdict = f"fenetre [{lo},{hi}["
    else:
        print(f"=> AUCUNE FENETRE : besoin {besoin} >= min autre image {autre_min}")
        verdict = "aucune fenetre"

    print("\n" + "=" * 78)
    print("JUGE DE PAIX -- la transplantation est-elle encore possible ?")
    print("=" * 78)
    print(f"  faux AUTHENTIC par transplantation : {n_faux}/{n_total}")
    print("  => liaison au contenu " + ("OPERANTE" if n_faux == 0 else "TOUJOURS INOPERANTE"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"hash_bits": args.hash_bits, "omega_bits": args.omega_bits,
               "rs_nsym": args.rs_nsym, "derives": res, "verdict_fenetre": verdict,
               "faux_authentic_transplantation": n_faux,
               "n_transplantations": n_total, "n_images": len(h_mar_tous)},
              open(args.out, "w"), indent=2)
    log(f"resultats -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

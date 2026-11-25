#!/usr/bin/env python3
"""
Trois tests que DistSeal rapporte et que la campagne CipherMark n'avait pas.

  --mode transfo   Transformations supplementaires du protocole DistSeal :
                   retournement horizontal, teinte, saturation, niveaux de
                   gris, et deux attaques combinees. Les scripts de campagne
                   existants sont reutilises tels quels ; seule leur liste
                   d'attaques est remplacee.
                     --canal pixel          eval_recovery_robustness.py
                     --canal diffusion      eval_grande_echelle_decodeur.py
                     --canal autoregressif  idem

  --mode fid       FID sur des images GENEREES par le U-ViT (diffusion). Le
                   meme latent est decode trois fois : decodeur d'origine
                   pre-entraine, decodeur affine sans Omega, decodeur affine
                   avec Omega. On compare les trois distributions deux a deux.
                   DistSeal compare au jeu de validation d'ImageNet, que l'on
                   n'a pas ici : ce FID est RELATIF (meme latents, memes
                   classes), il mesure ce que le tatouage deplace dans la
                   distribution des images, pas la qualite absolue du modele.

  --mode exemples  Quelques images generees, avec et sans Omega, pour la
                   figure d'exemples ; s'arrete si Omega n'est pas lu.

  --mode lora      Affinage LoRA du decodeur conditionne sur des images non
                   tatouees, comme le test d'oubli de DistSeal (annexe E).
                   ATTENTION A LA LECTURE : chez DistSeal la marque est gravee
                   dans les poids, et l'affinage est la seule facon de
                   l'effacer. Chez CipherMark, Omega est FOURNI au decodeur :
                   qui detient les poids et le code n'a qu'a ne pas le fournir.
                   Ce test mesure donc la STABILITE du canal quand le decodeur
                   est reaffine (mise a jour par l'operateur, ou affinage des
                   poids par un tiers qui conserve la chaine de generation),
                   pas la resistance a un attaquant qui controle le code.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(RACINE, "deps"), RACINE]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATEUR = "mit-han-lab/dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k"


def log(m):
    print(f"[complements] {time.strftime('%H:%M:%S')} {m}", flush=True)


# ===========================================================================
# 1. Transformations supplementaires
# ===========================================================================
def attaques_supplementaires():
    """Conditions du protocole DistSeal absentes de la campagne de cloture.
    Parametres FIXES, comme toutes les autres conditions, pour que les
    chiffres soient reproductibles."""
    from distseal.augmentation import valuemetric, geometric
    A = [("aucune", lambda x: x)]
    hf = geometric.HorizontalFlip()
    A.append(("retournement_h", lambda x, hf=hf: hf(x, None)[0]))
    for f in (-0.1, 0.1):
        h = valuemetric.Hue()
        A.append((f"teinte_{f:+.1f}", lambda x, h=h, f=f: h(x, None, f)[0]))
    for f in (0.5, 1.5):
        s = valuemetric.Saturation()
        A.append((f"saturation_{f}", lambda x, s=s, f=f: s(x, None, f)[0]))
    g = valuemetric.Grayscale()
    A.append(("niveaux_de_gris", lambda x, g=g: g(x, None)[0]))
    # Attaque combinee de DistSeal (JPEG, recadrage 0,71, luminosite 0,5),
    # et la meme sans recadrage, pour separer l'effet du recadrage.
    j, c, b = valuemetric.JPEG(), geometric.Crop(), valuemetric.Brightness()
    A.append(("combinee_distseal",
              lambda x, j=j, c=c, b=b: b(c(j(x, None, quality=60)[0], None, size=0.71)[0], None, 0.5)[0]))
    A.append(("combinee_sans_recadrage",
              lambda x, j=j, b=b: b(j(x, None, quality=60)[0], None, 0.8)[0]))
    return A


def mode_transfo(args, reste):
    if args.canal == "pixel":
        import scripts.ciphermark.eval_recovery_robustness as mod
    else:
        import scripts.ciphermark.eval_grande_echelle_decodeur as mod
        if "--famille" not in reste:
            reste = ["--famille", args.canal] + reste
    mod.build_attacks = attaques_supplementaires          # seule substitution
    sys.argv = [mod.__file__] + reste
    log(f"canal {args.canal} : {len(attaques_supplementaires())} conditions")
    return mod.main() or 0


# ===========================================================================
# Outils communs aux modes fid et lora (famille diffusion, DC-AE 512 px)
# ===========================================================================
def charge_extracteur(chemin):
    from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint
    cfg_w = get_config_from_checkpoint(chemin)
    wam = setup_model_from_checkpoint(chemin).to(DEVICE).eval()
    kw = {}
    try:
        from distseal.models.videoseal import VideoWam
        if isinstance(wam, VideoWam):
            kw = {"is_video": False}
    except Exception:
        pass
    return wam, kw, int(cfg_w.args.nbits), int(cfg_w.args.img_size)


@torch.no_grad()
def lit_bits(wam, kw, img01, nbits, img_size):
    xr = F.interpolate(img01.float(), size=(img_size, img_size), mode="bilinear",
                       align_corners=False, antialias=True)
    return (wam.detect(xr, **kw)["preds"][:, 1:1 + nbits] > 0).to(torch.uint8)


def charge_diffusion(chemin_conditionneur, nbits, garder_origine=False):
    from deps.efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
    from scripts.ciphermark.chaine_distseal_omega_variable import charge_conditionneur
    mod = DCAE_Diffusion_HF.from_pretrained(GENERATEUR).to(DEVICE).eval()
    ae = mod.autoencoder
    ae_origine = copy.deepcopy(ae).eval() if garder_origine else None
    with torch.no_grad():
        sonde = ae.encoder(torch.zeros(1, 3, 512, 512, device=DEVICE))
    # Le checkpoint de la phase D512 pese 6 Go, dont 3,3 d'etat d'optimiseur.
    # charge_conditionneur le charge directement sur le GPU : a cote du U-ViT
    # et de la copie du decodeur d'origine, un T4 (15 Go) deborde. On en
    # extrait d'abord, sur le CPU, les seules cles utiles.
    leger = os.path.splitext(chemin_conditionneur)[0] + "_leger.pt"
    if not os.path.exists(leger):
        ck = torch.load(chemin_conditionneur, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        garde = {k: v for k, v in sd.items()
                 if k.startswith(("omega_conditioner.", "decoder."))}
        torch.save({"state_dict": garde,
                    "global_step": ck.get("global_step", ck.get("epoch", "?"))}, leger)
        del ck, sd, garde
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cond = charge_conditionneur(ae, ae.decoder, sonde, nbits, leger)
    return mod, ae, ae_origine, cond


def vers01(brut):
    return (brut.float() * 0.5 + 0.5).clamp(0, 1)          # le DC-AE vit en [-1,1]


def genere_latent(diff, etiq, nulle, g, echelle):
    """Echantillonne le U-ViT. JAMAIS en float16 : le U-ViT-H, entraine en
    bfloat16, deborde en float16 et rend des NaN -- sur T4 (pas de bfloat16)
    toutes les images sortaient noires et Omega etait relu au hasard. On
    genere donc en bfloat16 si la carte le permet (A100...), sinon en float32."""
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        latent = diff.generate(etiq, nulle, 4.0, g).float() / echelle
    if not torch.isfinite(latent).all():
        raise SystemExit("latent genere non fini (NaN/inf) : generation invalide")
    return latent


def verifie_images(x, quoi):
    """Une image uniforme (toute noire, par exemple) n'a rien a tatouer."""
    if not torch.isfinite(x).all() or (x.flatten(1).std(1) < 1e-3).any():
        raise SystemExit(f"{quoi} : image non finie ou uniforme -- generation invalide")


def ecrit(chemin, obj):
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
    tmp = chemin + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, chemin)


# ===========================================================================
# 2. FID relatif sur images generees
# ===========================================================================
def mode_fid(args):
    from torchmetrics.image.fid import FrechetInceptionDistance
    wam, kw, nbits, img_size = charge_extracteur(args.watermarker)
    mod, ae, ae_origine, cond = charge_diffusion(args.conditionneur, nbits, garder_origine=True)
    diff, echelle = mod.diffusion_model, mod.scaling_factor

    def fid():
        return FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    paires = {"origine_vs_tatoue": fid(), "origine_vs_affine_sans_omega": fid(),
              "affine_sans_omega_vs_tatoue": fid()}
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dtype = torch.float16
    err_total, n_fait, t0 = 0, 0, time.time()
    while n_fait < args.n_images:
        B = min(args.batch, args.n_images - n_fait)
        etiq = torch.as_tensor(rng.integers(0, 1000, B), device=DEVICE)
        nulle = torch.full((B,), 1000, device=DEVICE)
        omega = torch.as_tensor(rng.integers(0, 2, (B, nbits)), device=DEVICE)
        latent = genere_latent(diff, etiq, nulle, g, echelle)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            x_o = vers01(ae_origine.decode(latent))
            x_n = vers01(ae.decode(latent))
            with cond.omega(omega):
                x_w = vers01(ae.decode(latent))
        if n_fait == 0:
            verifie_images(x_o, "premier lot")
        paires["origine_vs_tatoue"].update(x_o, real=True); paires["origine_vs_tatoue"].update(x_w, real=False)
        paires["origine_vs_affine_sans_omega"].update(x_o, real=True); paires["origine_vs_affine_sans_omega"].update(x_n, real=False)
        paires["affine_sans_omega_vs_tatoue"].update(x_n, real=True); paires["affine_sans_omega_vs_tatoue"].update(x_w, real=False)
        bits = lit_bits(wam, kw, x_w, nbits, img_size)
        err_total += int((bits != omega.to(torch.uint8)).sum())
        n_fait += B
        if n_fait == B and err_total / (B * nbits) > 0.2:
            # Omega non inscrit (cf. figure d'exemples de l'etape 2, ou le
            # decodage avec et sans Omega etait identique) : le FID serait
            # mesure sur des images non tatouees. On s'arrete net.
            raise SystemExit(f"premier lot : {err_total}/{B * nbits} bits faux, "
                             "Omega n'est pas inscrit -- FID annule")
        if (n_fait // B) % 25 == 0 or n_fait == args.n_images:
            ec = time.time() - t0
            log(f"  {n_fait}/{args.n_images}  reste ~{ec / n_fait * (args.n_images - n_fait) / 60:.0f} min")
    res = {"n_images": args.n_images, "generateur": GENERATEUR, "cfg_scale": 4.0,
           "omega": "aleatoire, un par image", "reference": "FID relatif, memes latents",
           "fid": {k: float(m.compute()) for k, m in paires.items()},
           "bit_acc_tatoue": 1 - err_total / (args.n_images * nbits)}
    ecrit(args.out, res)
    log(json.dumps(res, indent=2))
    return 0


# ===========================================================================
# 2 bis. Images d'exemple pour la figure (remplace la section C de l'etape 2)
# ===========================================================================
def mode_exemples(args):
    """Genere quelques classes, decode avec et sans Omega par le MEME chemin
    que la mesure a grande echelle (autocast), et s'arrete si Omega n'est pas
    lu : a l'etape 2, chaine_distseal_omega_variable.py a rendu des images
    avec et sans Omega identiques au pixel pres (Omega lu au hasard)."""
    from torchvision.utils import save_image
    wam, kw, nbits, img_size = charge_extracteur(args.watermarker)
    mod, ae, _, cond = charge_diffusion(args.conditionneur, nbits)
    diff, echelle = mod.diffusion_model, mod.scaling_factor
    os.makedirs(args.out_dir, exist_ok=True)
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    res = []
    for cl in args.classes:
        etiq = torch.tensor([cl], device=DEVICE)
        nulle = torch.tensor([1000], device=DEVICE)
        omega = torch.as_tensor(rng.integers(0, 2, (1, nbits)), device=DEVICE)
        latent = genere_latent(diff, etiq, nulle, g, echelle)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            x_n = vers01(ae.decode(latent))
            with cond.omega(omega):
                x_w = vers01(ae.decode(latent))
        verifie_images(x_n, f"classe {cl}")
        err = int((lit_bits(wam, kw, x_w, nbits, img_size) != omega.to(torch.uint8)).sum())
        mse = float(((x_w - x_n) ** 2).mean())
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        base = os.path.join(args.out_dir, f"classe_{cl:03d}")
        save_image(x_n, base + "_sans_omega.png")
        save_image(x_w, base + "_avec_omega.png")
        log(f"classe {cl} : Omega {err}/{nbits} bits faux, PSNR avec/sans {psnr:.2f} dB")
        res.append({"classe": cl, "erreurs_omega": err, "nbits": nbits, "psnr_vs_sans_omega": psnr})
    ecrit(os.path.join(args.out_dir, "resultats.json"), {"omega": "aleatoire", "resultats": res})
    mauvais = [r for r in res if r["erreurs_omega"] > 13 or not np.isfinite(r["psnr_vs_sans_omega"])]
    if mauvais:
        raise SystemExit(f"{len(mauvais)} image(s) sans Omega lisible : figure non valable")
    return 0


# ===========================================================================
# 3. Affinage LoRA du decodeur conditionne
# ===========================================================================
def mode_lora(args):
    from peft import LoraConfig, inject_adapter_in_model
    from scripts.ciphermark.eval_grande_echelle_decodeur import charge_lot, liste_images
    from distseal.utils.metrics import psnr as mesure_psnr
    torch.manual_seed(args.seed)
    wam, kw, nbits, img_size = charge_extracteur(args.watermarker)
    mod, ae, _, cond = charge_diffusion(args.conditionneur, nbits)
    del mod.diffusion_model                                # inutile ici
    torch.cuda.empty_cache()

    fichiers, _ = liste_images(args.corpus, args.n_train + args.n_eval, args.seed)
    f_train, f_eval = fichiers[:args.n_train], fichiers[args.n_train:]

    @torch.no_grad()
    def latents(fs, bs):
        out = []
        for i in range(0, len(fs), bs):
            x = charge_lot(fs[i:i + bs], 512)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append((ae.encoder(x * 2 - 1).float(), x))
        return out
    lots_train, lots_eval = latents(f_train, args.batch_train), latents(f_eval, 4)
    rng = np.random.default_rng(args.seed + 1)
    omegas_eval = [torch.as_tensor(rng.integers(0, 2, (z.shape[0], nbits)), device=DEVICE)
                   for z, _ in lots_eval]

    @torch.no_grad()
    def evalue():
        ae.decoder.eval()
        err, det, ok, n, ps = 0, 0, 0, 0, []
        for (z, x), om in zip(lots_eval, omegas_eval):
            with torch.autocast("cuda", dtype=torch.float16), cond.omega(om):
                xw = vers01(ae.decode(z))
            e = (lit_bits(wam, kw, xw, nbits, img_size) != om.to(torch.uint8)).sum(1)
            err += int(e.sum()); det += int((e <= 13).sum()); ok += int((e <= 6).sum())
            n += z.shape[0]; ps += mesure_psnr(xw, x).detach().cpu().tolist()
        ae.decoder.train()
        return {"bit_acc": 1 - err / (n * nbits), "detection": det / n,
                "omega_accepte": ok / n, "psnr_vs_image_reelle": float(np.mean(ps))}

    avant = evalue()
    log(f"avant LoRA : {avant}")

    # LoRA sur les convolutions et couches lineaires ordinaires du decodeur.
    # Exclus : les convolutions groupees (non prises en charge) et les modules
    # portant un crochet du conditionneur, que l'injection remplacerait.
    accroches = set(cond._index_par_module.keys())
    cibles = [nom for nom, m in ae.decoder.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "groups", 1) == 1
              and m not in accroches]
    inject_adapter_in_model(LoraConfig(r=args.rang, lora_alpha=2 * args.rang,
                                       target_modules=cibles, lora_dropout=0.0), ae.decoder)
    for p in list(ae.parameters()) + list(cond.parameters()):
        p.requires_grad_(False)
    params = [p for n, p in ae.decoder.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad_(True)
    log(f"LoRA rang {args.rang} sur {len(cibles)} couches, "
        f"{sum(p.numel() for p in params) / 1e6:.2f} M parametres entraines")
    verif = evalue()                                       # B=0 : identite exacte
    log(f"apres injection, avant entrainement : {verif}")

    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    courbe = [{"pas": 0, **verif}]
    res = {"lr": args.lr, "rang": args.rang, "n_train": args.n_train, "n_eval": args.n_eval,
           "pas_total": args.pas, "avant_lora": avant, "courbe": courbe,
           "lecture": "stabilite du canal sous reaffinage du decodeur ; Omega reste fourni"}
    ecrit(args.out, res)
    pas, t0 = 0, time.time()
    while pas < args.pas:
        for z, x in lots_train:
            if pas >= args.pas:
                break
            om = torch.as_tensor(rng.integers(0, 2, (z.shape[0], nbits)), device=DEVICE)
            with torch.autocast("cuda", dtype=torch.float16), cond.omega(om):
                perte = F.mse_loss(ae.decode(z).float(), x * 2 - 1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perte).backward()
            scaler.step(opt); scaler.update()
            pas += 1
            if pas % args.eval_tous == 0 or pas == args.pas:
                r = {"pas": pas, "perte": float(perte), **evalue()}
                courbe.append(r)
                ecrit(args.out, res)                       # progression conservee
                log(f"pas {pas}/{args.pas}  {r}  ({(time.time() - t0) / 60:.1f} min)")
    return 0


def main() -> int:
    # Premier passage : seulement le mode et le canal. En mode transfo, TOUT le
    # reste est transmis tel quel au script de campagne (--corpus, --n-images,
    # --out...), qu'il ne faut donc pas consommer ici.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", choices=("transfo", "fid", "lora", "exemples"), required=True)
    pre.add_argument("--canal", choices=("pixel", "diffusion", "autoregressif"), default="pixel")
    base, reste = pre.parse_known_args()
    if base.mode == "transfo":
        return mode_transfo(base, reste)

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--conditionneur")
    ap.add_argument("--watermarker")
    ap.add_argument("--corpus")
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--batch-train", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--pas", type=int, default=2500)
    ap.add_argument("--eval-tous", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rang", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--out-dir")
    ap.add_argument("--classes", type=int, nargs="+", default=[207, 388, 417, 979])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis pour les modes fid, lora et exemples")
    return {"fid": mode_fid, "lora": mode_lora, "exemples": mode_exemples}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())

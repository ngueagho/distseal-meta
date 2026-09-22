#!/usr/bin/env python3
"""
Volet B de la cloture : le DECODEUR CONDITIONNE a grande echelle.

Ce que ce script mesure, et qui n'etait mesure nulle part ailleurs :

    latent -> decodage avec modulation des poids par Omega (WOUAF/FiLM)
           -> ecriture dans la base de tracabilite
           -> les 17 attaques du protocole DistSeal
           -> recuperation d'Omega et verdict, par le chemin du verifieur

La chaine `chaine_distseal_omega_variable.py` fait la meme chose, mais UNE
image a la fois (2,4 s chacune) et avec deux attaques JPEG seulement. A 100 000
images ce serait 66 heures pour la seule generation. Ce script-ci travaille par
lots et couvre les 17 conditions.

DEUX SOURCES DE LATENT
----------------------
  --source encode   le latent vient de l'encodeur applique a une image reelle.
                    C'est la voie de DistSeal pour l'autoregressif (son depot
                    n'a pas de generateur), et la seule voie tenable a 100 000
                    images pour la diffusion.
  --source genere   le latent vient de l'echantillonnage du U-ViT (diffusion
                    seulement). Fidele a la chaine generative complete, mais
                    ~2,4 s par image : a reserver a un sous-ensemble.

Ce que la source change : d'ou vient le latent. Ce qu'elle ne change pas : le
decodeur voit un latent, le module par Omega, et produit une image marquee.
L'insertion, le stockage et la recuperation sont identiques dans les deux cas.

LES DEUX FAMILLES
-----------------
  diffusion       DC-AE f64c128, 512 px, conditionneur de la phase D-512
  autoregressif   MaskGIT-VQGAN, 256 px, conditionneur de la phase F

Les conventions d'intensite different : le DC-AE travaille en [-1,1], le
MaskGIT en [0,1]. Le script s'en occupe par famille -- s'y tromper produit des
images grises et des mesures qui n'ont aucun sens.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(RACINE, "deps"), RACINE]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from distseal.ciphermark.equation import (                        # noqa: E402
    CipherMarkThresholds, binomial_pvalue)
from distseal.ciphermark.witness import WitnessField              # noqa: E402
from distseal.utils.cfg import (get_config_from_checkpoint,        # noqa: E402
                                setup_model_from_checkpoint)
from scripts.ciphermark.chaine_distseal_omega_variable import (    # noqa: E402
    ProtocoleCrypto, charge_conditionneur)
from scripts.ciphermark.eval_recovery_robustness import build_attacks  # noqa: E402
from distseal.utils.metrics import psnr as mesure_psnr, ssim as mesure_ssim  # noqa: E402


def log(m):
    print(f"[volet-B] {time.strftime('%H:%M:%S')} {m}", flush=True)


# ---------------------------------------------------------------------------
# Le protocole, mais par lots. La version de la chaine hache une image a la
# fois : a 100 000 images x 18 hachages (l'image marquee + les 17 attaquees)
# cela ferait 1,8 million de passes DINOv2 en serie.
# ---------------------------------------------------------------------------
class ProtocoleLot(ProtocoleCrypto):

    def h_lot(self, imgs) -> list:
        """Hache un lot d'images. Retourne une liste de bytes."""
        with torch.no_grad():
            bits = self.phash(imgs).cpu().numpy()
        return [self._c.bits_to_bytes(b) for b in bits]

    def emettre_lot(self, imgs_nu, nonces):
        """h_base -> Omega, pour tout un lot. AVANT le decodage conditionne."""
        hs = self.h_lot(imgs_nu)
        om = np.stack([self.wf.build_omega(h, image_id=n)
                       for h, n in zip(hs, nonces)])
        return torch.tensor(om.astype("int64"), device=DEVICE), hs

    def enregistrer_lot(self, nonces, hs_base, imgs_w) -> None:
        hs_ref = self.h_lot(imgs_w)
        for n, hb, hr in zip(nonces, hs_base, hs_ref):
            self.registre.put(nonce=n, parity=hb, n_bits=self.cfg.n_bits,
                              rs_nsym=0, h_ref=hr, user=self.user_id,
                              session="volet-B")

    def attendus_lot(self, nonces):
        """Le chemin du verifieur : registre -> rederivation -> bits attendus.

        La cle est rederivee une fois par lot et non par image : l'utilisateur
        enregistre est le meme, et rederiver 100 000 fois la meme cle ne
        mesurerait que la vitesse de HKDF. La lecture du registre, elle, a bien
        lieu par image -- c'est elle qui porte le lien nonce -> empreinte.
        """
        n = self.cfg.n_bits
        entrees = [self.registre.get(x) for x in nonces]
        k = WitnessField.derive_user_key(self.k_master, entrees[0].user, self.cfg)
        wf = type(self.wf)(s_master=self.s_master, k_secret=k, cfg=self.cfg)
        att = np.stack([wf.expected_tag_bits(bytes(e.parity))[:n]
                        ^ wf.keystream_bits(x)[:n]
                        for e, x in zip(entrees, nonces)])
        refs = [self.registre.h_ref_for(x) for x in nonces]
        return att.astype(np.uint8), refs, entrees[0].user


def distance_hash(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def charge_lot(fichiers, taille) -> torch.Tensor:
    """Charge un lot d'images en [0,1], format [B,3,taille,taille]."""
    t = []
    for f in fichiers:
        im = Image.open(f).convert("RGB").resize((taille, taille), Image.BICUBIC)
        t.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.)
                 .permute(2, 0, 1))
    return torch.stack(t).to(DEVICE)


def liste_images(corpus, n, seed):
    import random
    fichiers = []
    for r, _, ns in os.walk(corpus):
        fichiers += [os.path.join(r, x) for x in sorted(ns)
                     if x.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not fichiers:
        raise SystemExit(f"aucune image dans {corpus}")
    random.Random(seed).shuffle(fichiers)
    # A 100 000 demandes sur un corpus plus petit, on repasse sur les memes
    # images -- mais avec un nonce et donc un Omega DIFFERENTS a chaque fois.
    # Ce qui est teste ici est le canal (image, Omega), pas la diversite du
    # contenu : le script dit combien d'images distinctes il a vraiment vues.
    if n > len(fichiers):
        k = (n + len(fichiers) - 1) // len(fichiers)
        fichiers = (fichiers * k)[:n]
    return fichiers[:n], len(set(fichiers))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--famille", choices=("diffusion", "autoregressif"),
                    required=True)
    ap.add_argument("--source", choices=("encode", "genere"), default="encode")
    ap.add_argument("--conditionneur", required=True)
    ap.add_argument("--watermarker", required=True)
    ap.add_argument("--generateur",
                    default="mit-han-lab/dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--user-id", default="createur-0042")
    ap.add_argument("--hash-bits", type=int, default=256)
    ap.add_argument("--seuil-hamming", type=float, default=0.27)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=None,
                    help="sous-ensemble d'attaques, separees par des virgules")
    ap.add_argument("--out-dir", default="runs/volet_b")
    ap.add_argument("--journal-tous", type=int, default=0,
                    help="si >0, ecrit le detail des N premieres images")
    ap.add_argument("--lpips", action="store_true",
                    help="mesure aussi LPIPS (plus lent, peut charger VGG)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis : le DC-AE utilise TritonRMSNorm.")
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------ l'extracteur ----
    cfg_w = get_config_from_checkpoint(args.watermarker)
    wam = setup_model_from_checkpoint(args.watermarker).to(DEVICE).eval()
    nbits, img_size = int(cfg_w.args.nbits), int(cfg_w.args.img_size)
    kw = {}
    try:
        from distseal.models.videoseal import VideoWam
        if isinstance(wam, VideoWam):
            kw = {"is_video": False}
    except Exception:
        pass
    log(f"extracteur : {nbits} bits, entraine en {img_size} px")

    # ------------------------------------------------- l'autoencodeur -------
    diff = echelle = None
    if args.famille == "diffusion":
        from deps.efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
        mod = DCAE_Diffusion_HF.from_pretrained(args.generateur).to(DEVICE).eval()
        ae, diff, echelle = mod.autoencoder, mod.diffusion_model, mod.scaling_factor
        taille = 512
        with torch.no_grad():
            sonde = ae.encoder(torch.zeros(1, 3, taille, taille, device=DEVICE))
    else:
        from deps.efficientvit.ae_model_zoo import MaskgitVqgan
        ae = MaskgitVqgan().to(DEVICE).eval()
        taille = 256
        with torch.no_grad():
            l0, _ = ae.encode_pre_quant(torch.zeros(1, 3, taille, taille,
                                                    device=DEVICE))
            sonde = ae.quantize(l0)
    log(f"{args.famille} : {taille} px, latent {tuple(sonde.shape)}")

    cond = charge_conditionneur(ae, ae.decoder, sonde, nbits, args.conditionneur)
    n_params_conditionneur = cond.n_parametres()
    n_params_decodeur = sum(p.numel() for p in ae.decoder.parameters())

    lpips_modele = None
    if args.lpips:
        try:
            import lpips
            lpips_modele = lpips.LPIPS(net="vgg").to(DEVICE).eval()
        except Exception as exc:
            raise SystemExit(f"LPIPS demande mais indisponible : {exc}") from exc

    if args.source == "genere" and args.famille != "diffusion":
        raise SystemExit("--source genere n'existe que pour la diffusion : le "
                         "depot de DistSeal n'a pas de generateur autoregressif.")

    # ---------------------------------------------------- les attaques ------
    attaques = build_attacks()
    if args.conditions:
        garde = {c.strip() for c in args.conditions.split(",")}
        attaques = [a for a in attaques if a[0] in garde]
        if not attaques:
            raise SystemExit(f"aucune attaque ne correspond a {args.conditions}")
    log(f"{len(attaques)} conditions : {', '.join(a[0] for a in attaques)}")

    proto = ProtocoleLot(args.user_id, args.hash_bits, args.seuil_hamming,
                         os.path.join(args.out_dir, "registre.sqlite"))

    fichiers, distinctes = liste_images(args.corpus, args.n_images, args.seed)
    log(f"{len(fichiers)} traitements sur {distinctes} images distinctes")

    # ----------------------------------------------------- accumulateurs ----
    # Une entree par attaque. Trois decisions DIFFERENTES sont conservees :
    #   detection_p    : p-valeur binomiale < 1e-6 (jusqu'a 13 erreurs / 64)
    #   omega_ok       : seuil AUTHENTIC du verifieur (BER <= 10 %, 6/64)
    #   authentic      : omega_ok ET hash_ok (le verdict complet CipherMark)
    # L'ancien champ JSON `verdicts` est maintenu comme alias de detection_p
    # pour ne pas casser les notebooks historiques, mais ne doit plus etre
    # presente comme un verdict AUTHENTIC.
    acc = {nom: {"err": 0, "n": 0, "detection_p": 0, "omega_ok": 0,
                 "authentic": 0, "d_hash": 0, "hash_ok": 0, "err_max": 0}
           for nom, _ in attaques}
    seuil_authentic = CipherMarkThresholds().authentic
    detail = []
    qualite = {"psnr": [], "ssim": [], "lpips": []}
    temps_decode_nu = 0.0
    temps_decode_waf = 0.0
    nonce = 0
    t_debut = time.time()
    torch.cuda.reset_peak_memory_stats()
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
             else torch.float16)
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)

    for d in range(0, len(fichiers), args.batch):
        lot = fichiers[d:d + args.batch]
        B = len(lot)

        # ---- 1. le latent -------------------------------------------------
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            if args.source == "genere":
                etiq = torch.randint(0, 1000, (B,), device=DEVICE)
                nulle = torch.full((B,), 1000, device=DEVICE)
                latent = diff.generate(etiq, nulle, 4.0, g).float() / echelle
                infos_dec = None
            elif args.famille == "diffusion":
                x0 = charge_lot(lot, taille)
                latent = ae.encoder(x0 * 2 - 1)      # le DC-AE vit en [-1,1]
                infos_dec = None
            else:
                x0 = charge_lot(lot, taille)
                l, tl = ae.encode_pre_quant(x0)      # le MaskGIT vit en [0,1]
                latent = ae.quantize(l)
                infos_dec = tl

        # ---- 2. l'image NUE : c'est de son hash que nait Omega -------------
        torch.cuda.synchronize()
        t_decode = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            brut = ae.decode(latent) if infos_dec is None \
                else ae.decode(latent, infos_dec)
        torch.cuda.synchronize()
        temps_decode_nu += time.perf_counter() - t_decode
        img_nu = ((brut.float() * 0.5 + 0.5) if args.famille == "diffusion"
                  else brut.float()).clamp(0, 1)

        nonces = list(range(nonce, nonce + B)); nonce += B
        omega, hs_base = proto.emettre_lot(img_nu, nonces)

        # ---- 3. le decodage CONDITIONNE : Omega module les poids -----------
        torch.cuda.synchronize()
        t_decode = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            with cond.omega(omega):
                brut_w = ae.decode(latent) if infos_dec is None \
                    else ae.decode(latent, infos_dec)
        torch.cuda.synchronize()
        temps_decode_waf += time.perf_counter() - t_decode
        img_w = ((brut_w.float() * 0.5 + 0.5) if args.famille == "diffusion"
                 else brut_w.float()).clamp(0, 1)

        # Qualite appariee : meme latent, avec et sans conditionnement WAF.
        # Ces mesures isolent l'effet du temoin de la variabilite du generateur.
        with torch.no_grad():
            qualite["psnr"].extend(
                mesure_psnr(img_w, img_nu).detach().cpu().tolist())
            qualite["ssim"].extend(
                mesure_ssim(img_w, img_nu).detach().cpu().tolist())
            if lpips_modele is not None:
                d_lpips = lpips_modele(img_w * 2 - 1, img_nu * 2 - 1)
                qualite["lpips"].extend(
                    d_lpips.detach().float().cpu().reshape(B).tolist())

        # ---- 4. la base de tracabilite ------------------------------------
        proto.enregistrer_lot(nonces, hs_base, img_w)
        attendus, hs_ref, utilisateur = proto.attendus_lot(nonces)

        # ---- 5. les attaques, puis la recuperation -------------------------
        for nom, f_att in attaques:
            try:
                xa = f_att(img_w.clone()).clamp(0, 1)
            except Exception as e:
                log(f"  attaque {nom} en echec : {type(e).__name__}: {e}")
                continue
            if xa.shape[-2:] != img_w.shape[-2:]:
                # recadrage et reechantillonnage changent la taille : on revient
                # a celle du modele, comme le ferait un verifieur.
                xa = F.interpolate(xa, size=img_w.shape[-2:], mode="bilinear",
                                   align_corners=False, antialias=True)
            xr = F.interpolate(xa, size=(img_size, img_size), mode="bilinear",
                               align_corners=False, antialias=True)
            with torch.no_grad():
                bits = (wam.detect(xr, **kw)["preds"][:, 1:1 + nbits] > 0) \
                    .to(torch.uint8).cpu().numpy()
            err = (bits != attendus).sum(axis=1)
            hs_obs = proto.h_lot(xa)
            a = acc[nom]
            a["n"] += B
            a["err"] += int(err.sum())
            a["err_max"] = max(a["err_max"], int(err.max()))
            dh = [distance_hash(o, r) for o, r in zip(hs_obs, hs_ref)]
            detection_p = np.asarray([
                binomial_pvalue(int(e), nbits) < 1e-6 for e in err
            ], dtype=bool)
            omega_ok = (err / nbits) <= seuil_authentic
            hash_ok = np.asarray([x <= proto.tau for x in dh], dtype=bool)
            authentic = omega_ok & hash_ok
            a["detection_p"] += int(detection_p.sum())
            a["omega_ok"] += int(omega_ok.sum())
            a["authentic"] += int(authentic.sum())
            a["d_hash"] += int(sum(dh))
            a["hash_ok"] += int(hash_ok.sum())
            if args.journal_tous and len(detail) < args.journal_tous:
                for i in range(min(B, args.journal_tous - len(detail))):
                    detail.append({"nonce": nonces[i], "attaque": nom,
                                   "erreurs": int(err[i]),
                                   "p_valeur": binomial_pvalue(int(err[i]), nbits),
                                   "d_hash": dh[i],
                                   "detection_p": bool(detection_p[i]),
                                   "omega_accepte": bool(omega_ok[i]),
                                   "hash_accepte": bool(hash_ok[i]),
                                   "authentic": bool(authentic[i]),
                                   "utilisateur": utilisateur})

        fait = min(d + args.batch, len(fichiers))
        if (d // args.batch) % 20 == 0 or fait == len(fichiers):
            ecoule = time.time() - t_debut
            reste = ecoule / max(fait, 1) * (len(fichiers) - fait)
            log(f"  {fait}/{len(fichiers)}  "
                f"{fait / max(ecoule, 1e-9):.1f} img/s  "
                f"reste ~{reste / 60:.0f} min")

    # ---------------------------------------------------------- le bilan ----
    duree = time.time() - t_debut
    lignes = []
    for nom, _ in attaques:
        a = acc[nom]
        if a["n"] == 0:
            continue
        lignes.append({
            "attaque": nom, "n": a["n"],
            "bit_acc": 1 - a["err"] / (a["n"] * nbits),
            "err_moy": a["err"] / a["n"], "err_max": a["err_max"],
            # Alias historiques : ils signifient detection statistique, pas
            # verdict AUTHENTIC. Les nouveaux consommateurs doivent utiliser
            # les champs explicites ci-dessous.
            "verdicts": a["detection_p"],
            "taux_verdict": a["detection_p"] / a["n"],
            "detections_p": a["detection_p"],
            "taux_detection_p": a["detection_p"] / a["n"],
            "omega_acceptes": a["omega_ok"],
            "taux_omega_accepte": a["omega_ok"] / a["n"],
            "d_hash_moy": a["d_hash"] / a["n"],
            "hash_dans_tau": a["hash_ok"] / a["n"],
            "authentic": a["authentic"],
            "taux_authentic": a["authentic"] / a["n"],
        })

    print()
    print("=" * 86)
    print(f"VOLET B -- decodeur conditionne, famille {args.famille}, "
          f"source {args.source}")
    print("=" * 86)
    print(f"{'attaque':<22}{'n':>8}{'bit_acc':>10}{'err moy':>9}"
          f"{'p<1e-6':>10}{'Omega OK':>10}{'hash OK':>10}{'AUTH.':>10}")
    print("-" * 86)
    for r in lignes:
        print(f"{r['attaque']:<22}{r['n']:>8}{r['bit_acc']:>10.4f}"
              f"{r['err_moy']:>9.2f}{r['taux_detection_p']:>10.1%}"
              f"{r['taux_omega_accepte']:>10.1%}"
              f"{r['hash_dans_tau']:>10.1%}{r['taux_authentic']:>10.1%}")
    print("-" * 86)
    print(f"  {len(fichiers)} traitements sur {distinctes} images distinctes, "
          f"{len(attaques)} conditions, {duree / 60:.1f} min")
    print(f"  registre : {os.path.join(args.out_dir, 'registre.sqlite')}")
    print(f"  qualite pairee : PSNR={np.mean(qualite['psnr']):.2f} dB, "
          f"SSIM={np.mean(qualite['ssim']):.4f}" +
          (f", LPIPS={np.mean(qualite['lpips']):.4f}"
           if qualite["lpips"] else ""))
    print(f"  decodage nu : {1000 * temps_decode_nu / len(fichiers):.2f} ms/image")
    print(f"  decodage WAF: {1000 * temps_decode_waf / len(fichiers):.2f} ms/image")
    print("=" * 86)

    chemin = os.path.join(args.out_dir,
                          f"volet_b_{args.famille}_{args.source}.json")
    def resume_distribution(valeurs):
        if not valeurs:
            return None
        a = np.asarray(valeurs, dtype=np.float64)
        return {"moyenne": float(a.mean()), "mediane": float(np.median(a)),
                "ecart_type": float(a.std()), "minimum": float(a.min()),
                "maximum": float(a.max()), "n": int(a.size)}

    bilan_qualite = {
        "reference": "meme latent decode sans Omega",
        "psnr_db": resume_distribution(qualite["psnr"]),
        "ssim": resume_distribution(qualite["ssim"]),
        "lpips_vgg": resume_distribution(qualite["lpips"]),
    }
    performance = {
        "decodeur_parametres": n_params_decodeur,
        "conditionneur_parametres": n_params_conditionneur,
        "decode_nu_secondes": temps_decode_nu,
        "decode_waf_secondes": temps_decode_waf,
        "decode_nu_ms_par_image": 1000 * temps_decode_nu / len(fichiers),
        "decode_waf_ms_par_image": 1000 * temps_decode_waf / len(fichiers),
        "surcout_waf_ms_par_image":
            1000 * (temps_decode_waf - temps_decode_nu) / len(fichiers),
        "pic_memoire_gpu_octets": int(torch.cuda.max_memory_allocated()),
    }

    with open(chemin, "w") as f:
        json.dump({"famille": args.famille, "source": args.source,
                   "n_traitements": len(fichiers), "n_distinctes": distinctes,
                   "nbits": nbits, "tau": proto.tau, "duree_s": duree,
                   "utilisateur": args.user_id,
                   "qualite_pairee": bilan_qualite,
                   "performance": performance,
                   "conditions": lignes, "detail": detail}, f, indent=1)
    log(f"bilan ecrit dans {chemin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
La chaine de DistSeal, telle qu'il l'a concue, plus un champ temoin variable.

Le principe
-----------
DistSeal ne genere pas depuis un texte. Sa voie diffusion est un U-ViT
conditionne par CLASSE ImageNet : on lui passe l'entier 208 et il produit un
labrador. Sa voie autoregressive n'a pas de generateur du tout -- le depot ne
contient que le tokeniseur MaskGIT-VQGAN, qui encode et decode.

Ce script reprend ces deux voies sans rien changer a leur protocole, et
n'ajoute qu'une chose : le decodeur latent est CONDITIONNE sur Omega. Le
champ temoin devient donc different a chaque generation, la ou DistSeal grave
un message fixe dans les poids.

    classe ImageNet --> U-ViT --> latent --> decodeur(Omega) --> image tatouee
                                                                      |
                                             extracteur <-------------+
                                                  |
                                             verdict + p-valeur

Un seul facteur change par rapport a DistSeal : la modulation du decodeur.
Tout le reste -- le generateur, l'autoencodeur, le facteur d'echelle, le
guidage -- est celui de son depot.

Sur la modulation
-----------------
WOUAF module les POIDS du decodeur a partir du message. Ce script module les
ACTIVATIONS par FiLM :

    a <- a * (1 + gamma(Omega)) + beta(Omega)

Les deux sont equivalents pour la mise a l'echelle des canaux de sortie d'une
convolution -- multiplier le filtre ou multiplier sa sortie donne le meme
resultat. L'avantage pratique : rien a regenerer a chaque image, donc
l'inference reste rapide, et le decodeur pre-entraine est bit a bit intact
tant que le conditionneur n'a pas appris.

Lancement :
    PYTHONPATH=deps:. python3 scripts/ciphermark/chaine_distseal_omega_variable.py \
        --famille diffusion --conditionneur /workspace/ckpt/phaseD.pt \
        --watermarker /workspace/ckpt/base_64bits.pth --classes 208 279 388
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distseal.ciphermark.conditioner import OmegaConditioner, decouvre_etages
from distseal.ciphermark.equation import binomial_pvalue
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quelques classes ImageNet lisibles, pour que les figures parlent d'elles-memes.
NOMS_CLASSES = {
    208: "labrador", 279: "renard arctique", 388: "panda geant",
    417: "montgolfiere", 933: "cheeseburger", 980: "volcan",
    973: "recif corallien", 992: "agaric", 1: "poisson rouge",
}


def log(m): print(f"[distseal] {time.strftime('%H:%M:%S')} {m}", flush=True)


def omega_hex(bits) -> str:
    """Serialise un Omega binaire en hexadecimal, bit de poids fort en tete.

    Enregistrer le temoin est ce qui permet a une mesure faite plus tard --
    robustesse, transplantation, contre-expertise -- de comparer a Omega
    LUI-MEME plutot qu'a la relecture de l'image intacte. Sans lui, ces
    mesures ne portent que sur une degradation relative.
    """
    v = "".join(str(int(b)) for b in bits.tolist())
    return f"{int(v, 2):0{(len(v) + 3) // 4}x}"


def charge_conditionneur(modele, decodeur, sonde_latente, nbits, chemin,
                         gamma=0.3, beta=0.1):
    """Restaure la phase D : le conditionneur ET le decodeur qu'elle a affine.

    Le chargement du conditionneur est STRICT. Si les etages decouverts ici
    differaient de ceux de l'entrainement, un chargement permissif laisserait
    une partie du conditionneur a son initialisation nulle : la modulation
    deviendrait l'identite sur ces etages, et le systeme rendrait des verdicts
    faux sans que rien ne le signale.

    Le decodeur compte tout autant. La phase D tournait avec
    freeze_decoder: false -- ses 224 tenseurs de decodeur ont ete affines
    CONJOINTEMENT au conditionneur. Charger le conditionneur seul par-dessus le
    decodeur pre-entraine donne un systeme qui module sans rien inscrire : le
    PSNR entre le decodage avec et sans temoin chute a 25 dB, ce qui donne
    toutes les apparences d'un tatouage fort, mais Omega ressort au hasard
    (mesure : 34 erreurs sur 64). L'encodeur, lui, est reste gele.
    """
    with torch.no_grad():
        etages = decouvre_etages(decodeur, sonde_latente)
    canaux = [c for _, _, c in etages]
    log(f"etages conditionnables : {canaux}")
    cond = OmegaConditioner(nbits=nbits, channels=canaux,
                            gamma_max=gamma, beta_max=beta).to(DEVICE)
    cond.attach([m for _, m, _ in etages])
    ck = torch.load(chemin, map_location=DEVICE, weights_only=False)
    sd = ck.get("state_dict", ck)
    pref = {k.split("omega_conditioner.", 1)[1]: v
            for k, v in sd.items() if k.startswith("omega_conditioner.")}
    if not pref:
        raise SystemExit(f"{chemin} ne contient aucun poids de conditionneur")
    try:
        cond.load_state_dict(pref, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"Les poids ne correspondent pas au decodeur.\n"
                         f"Etages ici : {canaux}\n{e}")
    log(f"conditionneur charge : {len(pref)} tenseurs, "
        f"pas {ck.get('global_step', ck.get('epoch', '?'))}")

    # --- le decodeur affine, sans quoi le conditionneur module dans le vide --
    avant = {k: v.detach().clone() for k, v in modele.state_dict().items()
             if k.startswith("decoder.")}
    if not any(k.startswith("decoder.") for k in sd):
        raise SystemExit(
            f"{chemin} ne contient aucun poids de decodeur. La phase D affine "
            f"le decodeur en meme temps que le conditionneur ; sans lui la "
            f"modulation n'inscrit rien et Omega sort au hasard.")
    modele.load_state_dict({k: v for k, v in sd.items()
                            if k.startswith("decoder.")}, strict=False)
    apres = modele.state_dict()
    bouges = sum(1 for k, v in avant.items()
                 if k in apres and not torch.equal(v, apres[k]))
    log(f"decodeur affine restaure : {bouges}/{len(avant)} tenseurs modifies")
    if bouges == 0:
        raise SystemExit(
            "aucun tenseur du decodeur n'a change au chargement : le "
            "checkpoint ne porte pas le decodeur de la phase D.")
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--famille", choices=("diffusion", "autoregressif"),
                    default="diffusion")
    ap.add_argument("--conditionneur", required=True,
                    help="checkpoint du decodeur conditionne (phase D pour la "
                         "diffusion, phase F pour l'autoregressif)")
    ap.add_argument("--watermarker", required=True,
                    help="couple embedder/extracteur qui relit Omega")
    ap.add_argument("--generateur",
                    default="mit-han-lab/dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k",
                    help="U-ViT de DistSeal, conditionne par classe ImageNet")
    ap.add_argument("--classes", type=int, nargs="+", default=[208, 279, 388, 417],
                    help="identifiants de classes ImageNet a generer")
    ap.add_argument("--corpus", default=None,
                    help="voie autoregressive : images a encoder puis redecoder")
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--taille", type=int, default=512)
    ap.add_argument("--attaques", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/chaine_distseal")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis : le DC-AE utilise TritonRMSNorm.")
    os.makedirs(args.out_dir, exist_ok=True)
    from torchvision.utils import save_image

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

    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
             else torch.float16)

    resultats = []
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)

    if args.famille == "diffusion":
        # ------------------- le generateur de DistSeal, tel quel ------------
        from deps.efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
        log(f"chargement de {args.generateur}")
        mod = DCAE_Diffusion_HF.from_pretrained(args.generateur).to(DEVICE).eval()
        ae, diff, echelle = mod.autoencoder, mod.diffusion_model, mod.scaling_factor
        log(f"U-ViT charge, facteur d'echelle {echelle}")

        with torch.no_grad():
            sonde = ae.encoder(torch.zeros(1, 3, args.taille, args.taille,
                                           device=DEVICE))
        cond = charge_conditionneur(ae, ae.decoder, sonde, nbits,
                                    args.conditionneur)

        for cl in args.classes:
            nom = NOMS_CLASSES.get(cl, f"classe {cl}")
            log(f"generation de la classe {cl} ({nom})")
            t0 = time.time()
            etiq = torch.tensor([cl], device=DEVICE)
            nulle = torch.tensor([1000], device=DEVICE)   # classe nulle du CFG
            omega = torch.randint(0, 2, (1, nbits), device=DEVICE)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                latent = diff.generate(etiq, nulle, args.cfg_scale, g)
            latent = latent.float() / echelle
            with torch.no_grad():
                # LE point : le meme latent, decode avec et sans temoin.
                with cond.omega(omega):
                    img_w = ae.decode(latent)
                img_nu = ae.decode(latent)
            t_gen = time.time() - t0
            img_w = (img_w.float() * 0.5 + 0.5).clamp(0, 1)
            img_nu = (img_nu.float() * 0.5 + 0.5).clamp(0, 1)

            x = F.interpolate(img_w, size=(img_size, img_size), mode="bilinear",
                              align_corners=False, antialias=True)
            with torch.no_grad():
                bits = (wam.detect(x, **kw)["preds"][:, 1:1 + nbits] > 0).to(torch.uint8)
            err = int((bits[0].cpu().numpy() != omega[0].cpu().numpy().astype(np.uint8)).sum())
            pv = binomial_pvalue(err, nbits)
            mse = float(((img_w - img_nu) ** 2).mean())
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            base = os.path.join(args.out_dir, f"classe_{cl:03d}_{nom.replace(' ', '_')}")
            save_image(img_nu, base + "_sans_omega.png")
            save_image(img_w, base + "_avec_omega.png")
            # Omega est enregistre en hexadecimal. Sans lui, toute mesure
            # ulterieure sur ces images doit prendre l'extraction propre pour
            # reference, et ne mesure donc qu'une degradation RELATIVE : on ne
            # peut plus verifier la distance au temoin reellement tire.
            ligne = {"famille": "diffusion", "classe": cl, "nom": nom,
                     "omega_hex": omega_hex(omega[0]),
                     "erreurs_omega": err, "nbits": nbits, "p_valeur": pv,
                     "psnr_vs_sans_omega": psnr, "secondes": round(t_gen, 1)}
            if args.attaques:
                from distseal.augmentation import valuemetric
                ligne["apres_attaque"] = {}
                for na, q in (("jpeg_q50", 50), ("jpeg_q30", 30)):
                    xa = valuemetric.JPEG()(x.clone(), None, quality=q)[0].clamp(0, 1)
                    with torch.no_grad():
                        b2 = (wam.detect(xa, **kw)["preds"][:, 1:1 + nbits] > 0).to(torch.uint8)
                    e2 = int((b2[0].cpu().numpy() != omega[0].cpu().numpy().astype(np.uint8)).sum())
                    ligne["apres_attaque"][na] = {"erreurs": e2,
                                                  "p_valeur": binomial_pvalue(e2, nbits)}
            resultats.append(ligne)
            log(f"    {t_gen:.1f}s | Omega {err}/{nbits} | p = {pv:.3g} | "
                f"PSNR vs sans temoin {psnr:.2f} dB")

    else:
        # --------------- la voie autoregressive, comme DistSeal la fait -----
        # Son depot ne contient aucun generateur autoregressif : maskgit_utils
        # n'expose qu'Encoder, Decoder et VectorQuantizer. On reproduit donc
        # exactement son protocole -- encoder une image, redecoder avec le
        # temoin -- sans pretendre generer depuis une intention.
        from deps.efficientvit.ae_model_zoo import MaskgitVqgan
        from PIL import Image
        import random
        log("chargement de MaskGIT-VQGAN (tokeniseur, sans generateur)")
        ae = MaskgitVqgan().to(DEVICE).eval()
        if not args.corpus:
            raise SystemExit("la voie autoregressive exige --corpus : DistSeal "
                             "n'ayant pas de generateur, on encode des images "
                             "reelles avant de les redecoder.")
        fichiers = []
        for r, _, ns in os.walk(args.corpus):
            fichiers += [os.path.join(r, n) for n in sorted(ns)
                         if n.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.Random(args.seed).shuffle(fichiers)
        fichiers = fichiers[:len(args.classes)]

        with torch.no_grad():
            lat, _ = ae.encode_pre_quant(torch.zeros(1, 3, 256, 256, device=DEVICE))
            lat = ae.quantize(lat)
        cond = charge_conditionneur(ae, ae.decoder, lat, nbits,
                                    args.conditionneur)

        for i, f in enumerate(fichiers):
            im = Image.open(f).convert("RGB").resize((256, 256))
            x0 = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.) \
                     .permute(2, 0, 1)[None].to(DEVICE)
            omega = torch.randint(0, 2, (1, nbits), device=DEVICE)
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                l, taille = ae.encode_pre_quant(x0)
                q = ae.quantize(l)
                with cond.omega(omega):
                    img_w = ae.decode(q, taille)
                img_nu = ae.decode(q, taille)
            img_w, img_nu = img_w.float().clamp(0, 1), img_nu.float().clamp(0, 1)
            xin = F.interpolate(img_w, size=(img_size, img_size), mode="bilinear",
                                align_corners=False, antialias=True)
            with torch.no_grad():
                bits = (wam.detect(xin, **kw)["preds"][:, 1:1 + nbits] > 0).to(torch.uint8)
            err = int((bits[0].cpu().numpy() != omega[0].cpu().numpy().astype(np.uint8)).sum())
            mse = float(((img_w - img_nu) ** 2).mean())
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
            base = os.path.join(args.out_dir, f"autoreg_{i:03d}")
            save_image(img_nu, base + "_sans_omega.png")
            save_image(img_w, base + "_avec_omega.png")
            resultats.append({"famille": "autoregressif", "source": os.path.basename(f),
                              "omega_hex": omega_hex(omega[0]),
                              "erreurs_omega": err, "nbits": nbits,
                              "p_valeur": binomial_pvalue(err, nbits),
                              "psnr_vs_sans_omega": psnr,
                              "secondes": round(time.time() - t0, 1)})
            log(f"    {os.path.basename(f)} | Omega {err}/{nbits} | "
                f"PSNR vs sans temoin {psnr:.2f} dB")

    # ------------------------------------------------------------- bilan ----
    print()
    print("=" * 86)
    print(f"CHAINE DISTSEAL + OMEGA VARIABLE -- famille {args.famille}")
    print("=" * 86)
    cle = "nom" if args.famille == "diffusion" else "source"
    print(f"{'sujet':<28}{'Omega':>10}{'p-valeur':>14}{'PSNR temoin':>14}{'secondes':>10}")
    print("-" * 86)
    for r in resultats:
        print(f"{str(r.get(cle, '?'))[:26]:<28}"
              f"{str(r['erreurs_omega']) + '/' + str(r['nbits']):>10}"
              f"{r['p_valeur']:>14.3g}{r['psnr_vs_sans_omega']:>14.2f}"
              f"{r['secondes']:>10.1f}")
    print("-" * 86)
    n_sur = sum(1 for r in resultats if r["p_valeur"] < 1e-6)
    parfaits = sum(1 for r in resultats if r["erreurs_omega"] == 0)
    print(f"  Omega relu sans erreur : {parfaits}/{len(resultats)}")
    print(f"  verdict rendu (p < 1e-6) : {n_sur}/{len(resultats)}")
    print()
    print("  Chaque image porte un Omega DIFFERENT, tire a la generation.")
    print("  DistSeal grave un message fixe dans les poids : toutes ses")
    print("  generations portent le meme filigrane.")
    print("=" * 86)

    with open(os.path.join(args.out_dir, "resultats.json"), "w") as f:
        json.dump({"famille": args.famille, "generateur": args.generateur,
                   "conditionneur": args.conditionneur, "resultats": resultats},
                  f, indent=2, ensure_ascii=False)
    log(f"resultats et images dans {args.out_dir}/")
    return 0 if n_sur == len(resultats) else 1


if __name__ == "__main__":
    raise SystemExit(main())

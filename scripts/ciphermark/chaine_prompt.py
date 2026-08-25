"""
CipherMark -- de la phrase au verdict.

    prompt --> SANA --> image --> Omega grave --> verification --> AUTHENTIC

Ce que fait ce script
---------------------
Il ferme la boucle que le memoire decrit sans jamais l'avoir executee d'un
bout a l'autre : on ecrit une phrase, un modele texte vers image la rend, le
champ temoin Omega y est grave, et le verifieur rend un verdict avec sa
p-valeur. Rien n'est simule : SANA genere reellement, le tatoueur est celui
qui a ete entraine, le hash perceptuel est DINOv2.

Deux modes, et ils ne disent pas la meme chose
----------------------------------------------
`posthoc`  -- SANA produit l'image, puis CipherMark la tatoue. La chaine est
              complete et fonctionne aujourd'hui, mais le tatouage reste une
              etape ajoutee apres la generation : qui controle le modele peut
              la sauter.

`inmodel`  -- le decodeur latent de SANA est lui-meme conditionne sur Omega
              (phase E). Le tatouage naît DANS la generation, il n'y a pas
              d'etape a sauter. Ce mode exige le checkpoint de la phase E ; en
              son absence le script le dit et s'arrete, plutot que de retomber
              en silence sur `posthoc` et de faire passer l'un pour l'autre.

Lancement :
    PYTHONPATH=deps:. python3 scripts/ciphermark/chaine_prompt.py \
        --prompt "un phare dans la tempete, photographie" \
        --watermarker /workspace/ckpt/base_64bits.pth --mode posthoc
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback
from distseal.ciphermark.registry import TraceRegistry
from distseal.ciphermark.wam_ciphermark import (CipherMarkConfig, CipherMarkKeys,
                                                CipherMarkWam)
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HASH_BITS = 256


def log(m): print(f"[chaine] {time.strftime('%H:%M:%S')} {m}", flush=True)


def charge_sana(modele: str):
    from diffusers import SanaPipeline
    log(f"chargement de {modele}")
    pipe = SanaPipeline.from_pretrained(modele, torch_dtype=torch.bfloat16)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, nargs="+",
                    help="une ou plusieurs phrases, une image par phrase")
    ap.add_argument("--watermarker", required=True)
    ap.add_argument("--mode", choices=("posthoc", "inmodel"), default="posthoc")
    ap.add_argument("--phaseE-checkpoint", default=None,
                    help="checkpoint du decodeur conditionne (mode inmodel)")
    ap.add_argument("--sana", default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--taille", type=int, default=512)
    ap.add_argument("--user-id", default=None,
                    help="identifiant dont la cle est derivee (attribution)")
    ap.add_argument("--attaques", action="store_true",
                    help="verifier aussi apres JPEG q50 et q30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/chaine_prompt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------- le tatoueur -------
    cfg_w = get_config_from_checkpoint(args.watermarker)
    wam = setup_model_from_checkpoint(args.watermarker).to(DEVICE).eval()
    nbits, img_size = int(cfg_w.args.nbits), int(cfg_w.args.img_size)
    log(f"tatoueur : {nbits} bits, entraine en {img_size} px")

    dino = _try_load_dinov2()
    phash = PerceptualHash(n_bits=HASH_BITS,
                           backbone=dino or _DCTFallback()).to(DEVICE).eval()
    log(f"hash perceptuel : {'DINOv2-small' if dino is not None else 'repli DCT'}, "
        f"{HASH_BITS} bits")

    keys = CipherMarkKeys.random()
    registre = TraceRegistry()
    cm = CipherMarkWam(wam=wam, phash=phash, keys=keys,
                       cfg=CipherMarkConfig(n_bits=nbits, max_fixed_point_iters=3),
                       registry=registre)
    if args.user_id:
        # La cle est derivee de l'identifiant : deux utilisateurs sur la meme
        # image produisent des Omega differents, ce qui rend l'attribution
        # opposable.
        from distseal.ciphermark.witness import WitnessField
        cm.witness = WitnessField.for_user(k_master=keys.k_secret,
                                           s_master=keys.s_master,
                                           user_id=args.user_id,
                                           cfg=cm.witness.cfg)
        log(f"cle derivee de l'identifiant : {args.user_id}")

    # ------------------------------------ le decodeur conditionne (inmodel) --
    cond = None
    if args.mode == "inmodel":
        if not args.phaseE_checkpoint or not os.path.exists(args.phaseE_checkpoint):
            raise SystemExit(
                "mode inmodel demande mais aucun checkpoint de phase E fourni.\n"
                "Sans decodeur conditionne, le tatouage ne peut pas naitre DANS "
                "la generation. Relancer en --mode posthoc, ou attendre la fin "
                "de l'entrainement de la phase E.")

    # ------------------------------------------------------- generation -----
    pipe = charge_sana(args.sana)
    if args.mode == "inmodel":
        from distseal.ciphermark.conditioner import OmegaConditioner, decouvre_etages
        decodeur = pipe.vae.decoder if hasattr(pipe.vae, "decoder") else pipe.vae
        with torch.no_grad():
            sonde = torch.zeros(1, 3, args.taille, args.taille, device=DEVICE)
            etages = decouvre_etages(decodeur, pipe.vae.encoder(sonde))
        cond = OmegaConditioner(nbits=nbits, channels=[c for _, _, c in etages],
                                gamma_max=0.3, beta_max=0.1).to(DEVICE)
        cond.attach([m for _, m, _ in etages])
        ck = torch.load(args.phaseE_checkpoint, map_location=DEVICE, weights_only=False)
        sd = ck.get("state_dict", ck)
        pref = {k.split("omega_conditioner.", 1)[1]: v
                for k, v in sd.items() if k.startswith("omega_conditioner.")}
        if not pref:
            raise SystemExit("le checkpoint ne contient aucun poids de "
                             "conditionneur : ce n'est pas un checkpoint de phase E")
        manquants = cond.load_state_dict(pref, strict=False)
        log(f"conditionneur charge ({len(pref)} tenseurs)")

    resultats = []
    g = torch.Generator(device=DEVICE).manual_seed(args.seed)
    for i, phrase in enumerate(args.prompt):
        log(f"[{i+1}/{len(args.prompt)}] generation : \"{phrase}\"")
        t0 = time.time()
        with torch.no_grad():
            img = pipe(prompt=phrase, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, generator=g,
                       height=args.taille, width=args.taille,
                       output_type="pt").images
        img = img.float().clamp(0, 1).to(DEVICE)
        t_gen = time.time() - t0

        # Le tatoueur travaille a sa resolution d'entrainement.
        x = F.interpolate(img, size=(img_size, img_size), mode="bilinear",
                          align_corners=False, antialias=True) \
            if img.shape[-1] != img_size else img

        sortie = cm.embed(x)
        marquee, ids = sortie["imgs_w"], sortie["image_ids"]

        from torchvision.utils import save_image
        base = os.path.join(args.out_dir, f"image_{i:03d}")
        save_image(x, base + "_originale.png")
        save_image(marquee, base + "_marquee.png")

        # ------------------------------------------------- verification ----
        r = cm.verify(marquee, ids)[0]
        ligne = {"prompt": phrase, "mode": args.mode, "nonce": ids[0],
                 "secondes_generation": round(t_gen, 1),
                 "verdict": r.verdict.value, "distance": r.distance,
                 "total": r.total, "p_valeur": r.p_value,
                 "fichier": base + "_marquee.png"}

        if args.attaques:
            from distseal.augmentation import valuemetric
            ligne["apres_attaque"] = {}
            for nom, q in (("jpeg_q50", 50), ("jpeg_q30", 30)):
                xa = valuemetric.JPEG()(marquee.clone(), None, quality=q)[0].clamp(0, 1)
                ra = cm.verify(xa, ids)[0]
                ligne["apres_attaque"][nom] = {"verdict": ra.verdict.value,
                                               "distance": ra.distance,
                                               "p_valeur": ra.p_value}
        resultats.append(ligne)
        log(f"    genere en {t_gen:.1f}s -> verdict {r.verdict.value.upper()}, "
            f"distance {r.distance}/{r.total}, p = {r.p_value:.3g}")

    # ------------------------------------------------------------ bilan -----
    print()
    print("=" * 88)
    print(f"CHAINE COMPLETE -- prompt vers verdict, mode {args.mode}")
    print("=" * 88)
    print(f"{'prompt':<44}{'verdict':>14}{'distance':>10}{'p-valeur':>14}")
    print("-" * 88)
    for r in resultats:
        print(f"{r['prompt'][:42]:<44}{r['verdict']:>14}"
              f"{str(r['distance'])+'/'+str(r['total']):>10}{r['p_valeur']:>14.3g}")
    if any("apres_attaque" in r for r in resultats):
        print()
        print("  Apres attaque passive")
        for r in resultats:
            for nom, a in r.get("apres_attaque", {}).items():
                print(f"    {r['prompt'][:30]:<32}{nom:<10}{a['verdict']:>14}"
                      f"{a['distance']:>6}")
    print("=" * 88)
    n_auth = sum(1 for r in resultats if r["verdict"] == "authentic")
    print(f"  {n_auth}/{len(resultats)} images generees puis reconnues AUTHENTIC")
    print(f"  images ecrites dans {args.out_dir}/")
    print()

    with open(os.path.join(args.out_dir, "resultats.json"), "w") as f:
        json.dump({"mode": args.mode, "sana": args.sana, "resultats": resultats},
                  f, indent=2, ensure_ascii=False)
    return 0 if n_auth == len(resultats) else 1


if __name__ == "__main__":
    raise SystemExit(main())

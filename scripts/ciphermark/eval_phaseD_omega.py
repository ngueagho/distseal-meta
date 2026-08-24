"""
CipherMark phase D -- evaluation du decodeur conditionne sur Omega, a grande
echelle.

Ce que la phase D a produit
---------------------------
Un decodeur DC-AE dont chaque etage est module par FiLM a partir d'un champ
temoin Omega. Le modele genere donc lui-meme des images porteuses d'un Omega
VARIABLE -- ce qu'aucune methode par distillation ne permet, puisqu'elles
gravent un message fixe dans les poids.

Ce qui n'avait pas ete mesure
-----------------------------
La bit_acc de 0,989 vient de la boucle de validation de l'entrainement : 10
lots, environ 40 images, avec un Omega tire deterministiquement. C'est assez
pour piloter un entrainement, pas pour affirmer que le decodeur "porte Omega".
Ce script rejoue la mesure sur des milliers d'images et, surtout, ajoute ce que
la boucle d'entrainement ne regardait pas :

  1. la DISTRIBUTION de la bit_acc, pas seulement sa mediane -- l'avalanche
     HMAC exige que chaque image individuellement depasse le seuil ; une bonne
     mediane avec une queue basse ne vaut rien ;
  2. le verdict du verifieur reel, distance de Hamming et p-valeur comprises ;
  3. la liaison au contenu sur les images GENEREES : derive du hash sous
     attaque passive, et transplantation du residu sur une autre image.

Piege evite
-----------
Le state_dict est charge avec strict=False, comme a l'entrainement. Si les cles
du conditionneur ne correspondaient pas, il resterait a son initialisation
nulle -- tanh(0) = 0, modulation identite -- et le script mesurerait 0,5 de
bit_acc en concluant a tort a un echec de la phase D. Le nombre de tenseurs du
conditionneur effectivement charges est donc verifie et le script s'arrete si
un seul manque.

Lancement :
    PYTHONPATH=deps:. python3 scripts/ciphermark/eval_phaseD_omega.py \
        --checkpoint /workspace/runs/phaseD_long/checkpoint/checkpoint.pt \
        --watermarker /workspace/ckpt/base_64bits.pth \
        --corpus /workspace/corpus-eval-12k --n-images 5000
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distseal.ciphermark.conditioner import OmegaConditioner, decouvre_etages
from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback
from distseal.ciphermark.equation import binomial_pvalue
from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint

DEVICE = "cuda"


def log(m): print(f"[phaseD] {time.strftime('%H:%M:%S')} {m}", flush=True)


def charge_images(corpus, size, n, seed=0):
    import random
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for root, _, names in os.walk(corpus):
        for nm in sorted(names):
            if nm.lower().endswith(exts):
                files.append(os.path.join(root, nm))
    if not files:
        raise SystemExit(f"aucune image dans {corpus}")
    random.Random(seed).shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f).convert("RGB").resize((size, size))
        out.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.)
                   .permute(2, 0, 1))
    return torch.stack(out)


def stats(nom, v, unite=""):
    v = np.asarray(v, dtype=np.float64)
    return {"cas": nom, "n": int(v.size), "moyenne": float(v.mean()),
            "median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "min": float(v.min()),
            "max": float(v.max()), "unite": unite}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint de la phase D (contient omega_conditioner.*)")
    ap.add_argument("--watermarker", required=True,
                    help="checkpoint du couple embedder/extracteur enseignant")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="dc-ae-f64c128-in-1.0")
    ap.add_argument("--n-images", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--omega-nbits", type=int, default=64)
    ap.add_argument("--gamma-max", type=float, default=0.3)
    ap.add_argument("--beta-max", type=float, default=0.1)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--hash-bits", type=int, default=256)
    ap.add_argument("--seuil-hamming", type=float, default=0.27,
                    help="seuil de la verification par comparaison de hash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval_phaseD_omega.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("le DC-AE exige CUDA (TritonRMSNorm ne tourne pas sur CPU)")

    # ------------------------------------------------------- le decodeur ----
    from deps.efficientvit.ae_model_zoo import DCAE_HF
    log(f"chargement de {args.model}")
    model = DCAE_HF.from_pretrained(f"mit-han-lab/{args.model}").cuda().eval()

    # Le decodeur non conditionne sert de reference : c'est lui qui donne
    # l'image "sans temoin" dont on soustrait le residu pour la transplantation.
    import copy
    reference = copy.deepcopy(model).cuda().eval()
    for p in reference.parameters():
        p.requires_grad = False

    with torch.no_grad():
        sonde = torch.zeros(1, 3, args.resolution, args.resolution).cuda()
        etages = decouvre_etages(model.decoder, model.encoder(sonde))
    if not etages:
        raise SystemExit("aucun etage conditionnable trouve dans le decodeur")
    cond = OmegaConditioner(nbits=args.omega_nbits,
                            channels=[c for _, _, c in etages],
                            gamma_max=args.gamma_max,
                            beta_max=args.beta_max).cuda()
    cond.attach([m for _, m, _ in etages])
    model.omega_conditioner = cond
    log(f"conditionneur : {len(etages)} etages {[c for _, _, c in etages]}, "
        f"{cond.n_parametres()/1e6:.2f} M parametres")

    # ------------------------------------- chargement, et sa verification ---
    ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    sd = ck.get("state_dict", ck)
    attendus = [k for k in model.state_dict() if k.startswith("omega_conditioner.")]
    presents = [k for k in attendus if k in sd]
    manquants = sorted(set(attendus) - set(presents))
    if manquants:
        raise SystemExit(
            f"{len(manquants)} tenseur(s) du conditionneur absents du checkpoint "
            f"(ex. {manquants[:3]}). Charge tel quel, le conditionneur resterait "
            f"a son initialisation nulle : la modulation serait l'identite et la "
            f"bit_acc mesuree, 0,5. Ce serait un faux echec de la phase D.")
    avant = model.state_dict()[attendus[0]].clone()
    model.load_state_dict(sd, strict=False)
    apres = model.state_dict()[attendus[0]]
    log(f"conditionneur : {len(presents)}/{len(attendus)} tenseurs charges, "
        f"epoque={ck.get('epoch', '?')} pas={ck.get('global_step', '?')}")
    if torch.equal(avant, apres) and float(avant.abs().sum()) == 0.0:
        raise SystemExit("les poids du conditionneur sont restes nuls apres "
                         "chargement : le checkpoint ne contient pas la phase D")
    model.eval()

    # ---------------------------------------------------- le watermarker ----
    cfg_w = get_config_from_checkpoint(args.watermarker)
    wam = setup_model_from_checkpoint(args.watermarker)
    if getattr(cfg_w.args, "scaling_w", None) is not None:
        try:
            wam.blender.scaling_w = float(cfg_w.args.scaling_w)
        except Exception:
            pass
    wam = wam.cuda().eval()
    nbits_wam = int(cfg_w.args.nbits)
    if nbits_wam != args.omega_nbits:
        raise SystemExit(f"le watermarker porte {nbits_wam} bits, "
                         f"Omega en demande {args.omega_nbits}")
    kw = {}
    try:
        from distseal.models.videoseal import VideoWam
        if isinstance(wam, VideoWam):
            kw = {"is_video": False}
    except Exception:
        pass

    dino = _try_load_dinov2()
    log(f"PHash : {'DINOv2-small REEL' if dino is not None else 'repli DCT'}, "
        f"{args.hash_bits} bits")
    phash = PerceptualHash(n_bits=args.hash_bits,
                           backbone=dino or _DCTFallback()).cuda().eval()

    def hash_bits(x01):
        with torch.no_grad():
            return phash(x01).cpu().numpy().astype(np.uint8)

    def extrait(x01):
        with torch.no_grad():
            preds = wam.detect(x01, **kw)["preds"]
        return (preds[:, 1:1 + args.omega_nbits] > 0).to(torch.uint8).cpu().numpy()

    # ------------------------------------------------------------ corpus ----
    imgs_all = charge_images(args.corpus, args.resolution, args.n_images, args.seed)
    n = imgs_all.shape[0]
    log(f"{n} images chargees")

    g = torch.Generator(device="cuda"); g.manual_seed(args.seed)
    attaques = [("jpeg_q50", 50), ("jpeg_q30", 30)]
    from distseal.augmentation import valuemetric

    bit_accs, d_omega = [], []
    d_hash = {"legitime": [], "jpeg_q50": [], "jpeg_q30": [], "transplantation": []}
    tau = int(round(args.seuil_hamming * args.hash_bits))
    nb = (n + args.batch - 1) // args.batch
    log(f"{nb} lots de {args.batch}, seuil de comparaison tau = {tau}/{args.hash_bits} bits")

    for bi in range(nb):
        x = imgs_all[bi * args.batch:(bi + 1) * args.batch].cuda()
        if x.shape[0] < 2:
            break
        omega = torch.randint(0, 2, (x.shape[0], args.omega_nbits),
                              device="cuda", generator=g)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # le modele attend des images dans [-1, 1]
            xn = x * 2 - 1
            with cond.omega(omega):
                y_wm, _, _ = model(xn, global_step=0)
            y_ref, _, _ = reference(xn, global_step=0)
        y_wm = (y_wm.float() * 0.5 + 0.5).clamp(0, 1)
        y_ref = (y_ref.float() * 0.5 + 0.5).clamp(0, 1)

        # 1. Omega survit-il a la generation ?
        bits = extrait(y_wm)
        om = omega.cpu().numpy().astype(np.uint8)
        diff = (bits != om).sum(axis=1)
        d_omega.extend(int(v) for v in diff)
        bit_accs.extend(1.0 - diff / args.omega_nbits)

        # 2. liaison au contenu sur les images GENEREES
        h_ref = hash_bits(y_wm)
        d_hash["legitime"].extend([0] * y_wm.shape[0])   # par construction
        for nom, q in attaques:
            xa = valuemetric.JPEG()(y_wm.clone(), None, quality=q)[0].clamp(0, 1)
            if xa.shape[-2:] != y_wm.shape[-2:]:
                xa = F.interpolate(xa, size=y_wm.shape[-2:], mode="bilinear",
                                   align_corners=False, antialias=True)
            d_hash[nom].extend(int(v) for v in (hash_bits(xa) != h_ref).sum(axis=1))

        # 3. transplantation : le residu genere par le conditionnement est
        #    greffe sur l'image DECODEE SANS temoin d'une autre scene.
        residu = y_wm - y_ref
        forgee = (torch.roll(y_ref, shifts=-1, dims=0) + residu).clamp(0, 1)
        d_hash["transplantation"].extend(
            int(v) for v in (hash_bits(forgee) != h_ref).sum(axis=1))

        if (bi + 1) % 20 == 0 or bi + 1 == nb:
            log(f"  lot {bi+1}/{nb} -- {len(bit_accs)} images, "
                f"bit_acc mediane {np.median(bit_accs):.4f}")

    # ------------------------------------------------------------ bilan -----
    bit_accs = np.asarray(bit_accs); d_omega = np.asarray(d_omega)
    pv = np.array([binomial_pvalue(int(d), args.omega_nbits) for d in d_omega])
    res = {
        "n_images": int(bit_accs.size), "omega_nbits": args.omega_nbits,
        "hash_bits": args.hash_bits, "seuil_hamming": args.seuil_hamming,
        "tau_bits": tau, "checkpoint": args.checkpoint,
        "bit_acc": stats("bit_acc", bit_accs),
        "hamming_omega": stats("hamming_omega", d_omega, "bits"),
        "fraction_parfaite": float((d_omega == 0).mean()),
        "fraction_sup_099": float((bit_accs >= 0.99).mean()),
        "fraction_p_inf_1e6": float((pv < 1e-6).mean()),
        "hash": {k: stats(k, v, "bits") for k, v in d_hash.items() if v},
    }
    for k, v in d_hash.items():
        if v:
            res["hash"][k]["fraction_sous_seuil"] = float((np.asarray(v) <= tau).mean())

    print()
    print("=" * 84)
    print(f"PHASE D -- decodeur conditionne, {res['n_images']} images generees")
    print("=" * 84)
    b = res["bit_acc"]
    print(f"  bit_acc      mediane {b['median']:.4f}   p05 {b['p05']:.4f}  "
          f"p95 {b['p95']:.4f}   min {b['min']:.4f}")
    print(f"  Omega exact (0 bit d'erreur)      : {res['fraction_parfaite']:.1%}")
    print(f"  bit_acc >= 0,99                   : {res['fraction_sup_099']:.1%}")
    print(f"  p-valeur < 1e-6 (verdict rendu)   : {res['fraction_p_inf_1e6']:.1%}")
    print("-" * 84)
    print(f"  liaison au contenu, seuil tau = {tau}/{args.hash_bits} bits")
    print(f"  {'cas':<18}{'mediane':>10}{'p95':>9}{'sous seuil':>13}")
    for k in ("legitime", "jpeg_q50", "jpeg_q30", "transplantation"):
        if k in res["hash"]:
            h = res["hash"][k]
            print(f"  {k:<18}{h['median']:>10.1f}{h['p95']:>9.1f}"
                  f"{h['fraction_sous_seuil']:>12.1%}")
    print("=" * 84)
    tr = res["hash"].get("transplantation", {}).get("fraction_sous_seuil", 1.0)
    print(f"  transplantation acceptee : {tr:.2%} "
          + ("-- liaison OPERANTE" if tr == 0 else "-- liaison EN DEFAUT"))
    print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

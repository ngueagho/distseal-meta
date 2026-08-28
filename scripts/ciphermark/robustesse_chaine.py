"""Robustesse d'Omega sur les images GENEREES par la chaine, a 512 px.

CE QUE MESURE CE SCRIPT, ET CE QU'IL NE MESURE PAS
--------------------------------------------------
Le vrai Omega n'a pas ete enregistre par la chaine ; on prend pour reference
les bits extraits de l'image INTACTE, qui en sont a moins de 8 erreurs sur 64.
La mesure est donc une DEGRADATION relative a l'etat propre : de combien une
attaque eloigne la lecture. C'est la grandeur qui nous interesse, mais elle
n'est pas la distance a Omega lui-meme -- la ligne "intacte" vaut 0 par
construction et ne doit pas etre lue comme un resultat.

RESULTAT DU 2026-08-28 (300 images, phase D-512 degelee)
--------------------------------------------------------
    attaque            erreurs med.   bit_acc    verdicts
    jpeg q50                    1,0    0,9879      100,0 %
    jpeg q30                    1,0    0,9841      100,0 %
    bruit std 0,05              0,0    0,9920      100,0 %
    flou k7                     6,0    0,9068      100,0 %
    recadrage 70 %             30,0    0,5315        0,3 %

Omega resiste a la compression, au bruit et au flou sans perdre un verdict.
Il ne resiste PAS au recadrage : c'est un message global de 64 bits reparti
sur toute l'image, non un motif repete localement. Limite a declarer telle
quelle dans le memoire.

Lancement :
    PYTHONPATH=deps:. python3 scripts/ciphermark/robustesse_chaine.py \
        --images /workspace/runs/chaine_512_grande_echelle \
        --watermarker /workspace/ckpt/base_64bits.pth
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distseal.utils.cfg import get_config_from_checkpoint, setup_model_from_checkpoint
from distseal.ciphermark.equation import binomial_pvalue
from distseal.augmentation import valuemetric, geometric


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--watermarker", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seuil-p", type=float, default=1e-6)
    args = ap.parse_args()

    cfg = get_config_from_checkpoint(args.watermarker)
    wam = setup_model_from_checkpoint(args.watermarker).cuda().eval()
    nbits, taille = int(cfg.args.nbits), int(cfg.args.img_size)
    kw = {}
    try:
        from distseal.models.videoseal import VideoWam
        if isinstance(wam, VideoWam):
            kw = {"is_video": False}
    except Exception:
        pass

    def lire(x):
        if x.shape[-1] != taille:
            x = F.interpolate(x, size=(taille, taille), mode="bilinear",
                              align_corners=False, antialias=True)
        with torch.no_grad():
            return (wam.detect(x, **kw)["preds"][:, 1:1 + nbits] > 0) \
                .to(torch.uint8).cpu().numpy()

    fics = sorted(glob.glob(os.path.join(args.images, "*_avec_omega.png")))
    if not fics:
        raise SystemExit(f"aucune image *_avec_omega.png dans {args.images}")

    jpeg, flou, bruit = (valuemetric.JPEG(), valuemetric.GaussianBlur(),
                         valuemetric.GaussianNoise())
    attaques = [
        ("intacte",        lambda x: x),
        ("jpeg q50",       lambda x: jpeg(x, None, quality=50)[0]),
        ("jpeg q30",       lambda x: jpeg(x, None, quality=30)[0]),
        ("flou k7",        lambda x: flou(x, None, kernel_size=7)[0]),
        ("bruit std 0,05", lambda x: bruit(x, None, std=0.05)[0]),
        ("recadrage 70 %", lambda x: geometric.Crop()(x, None, size=0.7)[0]),
    ]

    res: dict[str, list] = {n: [] for n, _ in attaques}
    for i in range(0, len(fics), args.batch):
        lot = torch.stack([
            torch.from_numpy(np.asarray(Image.open(c).convert("RGB"))
                             .astype(np.float32) / 255.).permute(2, 0, 1)
            for c in fics[i:i + args.batch]]).cuda()
        b0 = lire(lot)
        for nom, f in attaques:
            try:
                xa = f(lot.clone()).clamp(0, 1)
            except Exception:
                res[nom].append(None)
                continue
            res[nom].append((lire(xa) != b0).sum(1))

    print(f"\n  {len(fics)} images generees, {nbits} bits\n")
    print(f"  {'attaque':<18}{'erreurs med.':>13}{'bit_acc':>10}{'verdicts':>12}")
    print(f"  {'-' * 55}")
    for nom, _ in attaques:
        v = [a for a in res[nom] if a is not None]
        if not v:
            print(f"  {nom:<18}{'echec':>13}")
            continue
        d = np.concatenate(v)
        pv = np.array([binomial_pvalue(int(x), nbits) for x in d])
        print(f"  {nom:<18}{np.median(d):>13.1f}{1 - d.mean() / nbits:>10.4f}"
              f"{100 * (pv < args.seuil_p).mean():>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

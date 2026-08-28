"""
Construit le corpus d'evaluation du canal hash.

Trois sources, cumulables :
  * scikit-image : 7 photos embarquees (toujours disponible, hors ligne)
  * Kodak24      : 24 photos de reference (telechargement ~15 Mo)
  * BSDS300      : 300 photos naturelles Berkeley (telechargement ~22 Mo)

Chaque scene donne 5 recadrages carres (centre + 4 coins) redimensionnes
en 256x256, pour decorreler un peu le cadrage sans dupliquer le contenu.

Usage:
    python -m scripts.ciphermark.build_corpus --out corpus-test
    python -m scripts.ciphermark.build_corpus --out corpus-test --kodak --bsds
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.request

import numpy as np
from PIL import Image


KODAK_URL = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
BSDS_URL = ("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
            "grouping/segbench/BSDS300-images.tgz")
COCO_URL = "http://images.cocodataset.org/zips/{}.zip"


def five_crops(img: np.ndarray, size: int = 256):
    """Centre + 4 coins, carres au petit cote, redimensionnes.

    Pour les images (quasi) carrees les coins coincident avec le centre :
    on prend alors des sous-recadrages a 75% du cote pour garder 5 vues.
    """
    H, W = img.shape[:2]
    s = min(H, W)
    cy, cx = (H - s) // 2, (W - s) // 2
    boxes = [(s, cy, cx)]                       # centre plein cadre
    if H - s < s // 8 and W - s < s // 8:       # image ~carree
        s2 = (3 * s) // 4
        boxes += [(s2, 0, 0), (s2, 0, W - s2),
                  (s2, H - s2, 0), (s2, H - s2, W - s2)]
    else:
        boxes += [(s, 0, 0), (s, 0, W - s),
                  (s, H - s, 0), (s, H - s, W - s)]
    seen = set()
    for side, top, left in boxes:
        if (side, top, left) in seen:
            continue
        seen.add((side, top, left))
        crop = img[top:top + side, left:left + side]
        yield Image.fromarray(crop).resize((size, size), Image.BILINEAR)


def save_scene(img: np.ndarray, name: str, out: str, size: int) -> int:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    n = 0
    for i, im in enumerate(five_crops(img, size)):
        im.save(os.path.join(out, f"{name}_{i}.png"))
        n += 1
    return n


def add_skimage(out: str, size: int) -> int:
    from skimage import data
    total = 0
    for name in ["astronaut", "chelsea", "coffee", "rocket",
                 "camera", "hubble_deep_field", "cat"]:
        try:
            img = getattr(data, name)()
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        total += save_scene(np.asarray(img), f"ski_{name}", out, size)
    return total


def add_kodak(out: str, size: int) -> int:
    total = 0
    for k in range(1, 25):
        url = KODAK_URL.format(k)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                img = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
            total += save_scene(img, f"kodim{k:02d}", out, size)
            print(f"  kodim{k:02d} ok")
        except Exception as e:
            print(f"  skip kodim{k:02d}: {e}")
    return total


def add_bsds(out: str, size: int, max_scenes: int = 300) -> int:
    tmp = os.path.join(out, "_bsds.tgz")
    if not os.path.exists(tmp):
        print("  telechargement BSDS300 (~22 Mo)...")
        urllib.request.urlretrieve(BSDS_URL, tmp)
    total = 0
    scenes = 0
    with tarfile.open(tmp) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".jpg") or "/images/" not in m.name:
                continue
            if scenes >= max_scenes:
                break
            f = tf.extractfile(m)
            if f is None:
                continue
            img = np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"))
            stem = os.path.splitext(os.path.basename(m.name))[0]
            total += save_scene(img, f"bsds{stem}", out, size)
            scenes += 1
    os.remove(tmp)
    return total


def add_coco(out: str, size: int, split: str, max_scenes: int,
             tmp_dir: str | None = None, skip: int = 0) -> int:
    """
    Telecharge un split COCO (test2017: 40k, unlabeled2017: 123k) et en
    convertit max_scenes en 256x256 (recadrage carre central), format jpg.
    Une image par scene -- pas de multi-crops, on veut des scenes
    independantes pour les statistiques. Prevoir la place du zip
    (unlabeled2017 ~19 Go, uniquement raisonnable sur Colab).

    tmp_dir place le zip ailleurs que dans le corpus : sur un pod loue, le
    volume persistant est petit et le disque du conteneur large, or le zip est
    justement le fichier qu'on peut se permettre de perdre.

    skip saute les premieres scenes du zip. C'est ce qui garantit qu'un corpus
    d'ENTRAINEMENT construit ici ne recoupe pas un corpus d'EVALUATION deja tire
    du meme split : l'ordre de parcours du zip etant stable, sauter les n
    premieres scenes suffit a rendre les deux ensembles disjoints.
    """
    import zipfile

    tmp = os.path.join(tmp_dir or out, f"_{split}.zip")
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    if not os.path.exists(tmp):
        print(f"  telechargement {split}.zip ...")
        urllib.request.urlretrieve(COCO_URL.format(split), tmp)
    total, vues = 0, 0
    with zipfile.ZipFile(tmp) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".jpg"):
                continue
            vues += 1
            if vues <= skip:
                continue
            if total >= max_scenes:
                break
            with zf.open(info) as f:
                try:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                except Exception:
                    continue
            W, H = img.size
            s = min(W, H)
            box = ((W - s) // 2, (H - s) // 2,
                   (W - s) // 2 + s, (H - s) // 2 + s)
            img = img.crop(box).resize((size, size), Image.BILINEAR)
            stem = os.path.splitext(os.path.basename(info.filename))[0]
            img.save(os.path.join(out, f"coco_{stem}.jpg"), quality=95)
            total += 1
            if total % 5000 == 0:
                print(f"  {total} images converties...")
    os.remove(tmp)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus-test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--kodak", action="store_true", help="ajouter Kodak24")
    ap.add_argument("--bsds", action="store_true", help="ajouter BSDS300")
    ap.add_argument("--bsds-max", type=int, default=300,
                    help="nb max de scenes BSDS")
    ap.add_argument("--coco", choices=("val2017", "test2017", "unlabeled2017"),
                    default=None, help="ajouter un split COCO")
    ap.add_argument("--coco-max", type=int, default=50000,
                    help="nb max de scenes COCO")
    ap.add_argument("--coco-skip", type=int, default=0,
                    help="sauter les n premieres scenes du split -- sert a "
                         "construire un corpus disjoint d'un corpus deja tire "
                         "du meme split")
    ap.add_argument("--tmp-dir", default=None,
                    help="ou telecharger le zip (defaut : dans --out)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total = add_skimage(args.out, args.size)
    print(f"skimage: {total} images")
    if args.kodak:
        n = add_kodak(args.out, args.size)
        print(f"kodak: {n} images")
        total += n
    if args.bsds:
        n = add_bsds(args.out, args.size, args.bsds_max)
        print(f"bsds: {n} images")
        total += n
    if args.coco:
        n = add_coco(args.out, args.size, args.coco, args.coco_max,
                     tmp_dir=args.tmp_dir, skip=args.coco_skip)
        print(f"coco {args.coco}: {n} images")
        total += n
    print(f"total: {total} images dans {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

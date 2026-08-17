# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Genere les figures qualitatives du chapitre resultats du memoire CipherMark.

Objectif : produire, a partir d'un VRAI checkpoint DistSeal entraine
(nbits=256, embedder unet_small2_yuv_quant, extracteur convnext_tiny), les
trois preuves visuelles/numeriques attendues par le chapitre :

  1. original vs filigrane vs carte de difference amplifiee, sur quelques
     images reelles diversifiees, avec le bit_acc/PSNR/SSIM par image ;
  2. la propriete "filigrane variable par utilisateur" : DEUX messages
     aleatoires differents embarques dans la MEME image, avec verification
     numerique qu'ils different et se decodent chacun correctement avec son
     propre message (pas celui de l'autre).

IMPORTANT (a rappeler dans la legende des figures) : au stade actuel de
l'entrainement (cf. memoire-3-points-ouverts.md), les configs "phase 0"/
"phase 2" n'appliquent quasiment aucune pression de fidelite (lambda_i=0
puis 0.01 seulement) -- le filigrane est donc FORTEMENT visible (PSNR bas,
de l'ordre de 15-25 dB) et bit_acc plafonne autour de 0.55-0.65. Ce script
NE DOIT PAS enjoliver ces chiffres : il les mesure et les rapporte tels
quels.

Usage:
    python -m scripts.ciphermark.generate_qualitative_figures \
        --checkpoint ./runs/qualitative_checkpoint.pth \
        --data-dir ./corpus-colab/val \
        --n-images 6

Sorties (sous runs/qualitative_figures/, jamais commitees) :
    runs/qualitative_figures/grid_original_vs_watermarked.png
    runs/qualitative_figures/two_messages_same_image.png
    runs/qualitative_figures/summary.txt
"""

from __future__ import annotations

# Machine CPU-only pour ce script (le GPU, si present, est reserve a
# l'entrainement en cours sur un autre process) -- coupe CUDA avant meme
# d'importer torch, par coherence avec les autres scripts de ce dossier.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import random

import matplotlib
matplotlib.use("Agg")  # environnement sans affichage
import matplotlib.pyplot as plt
import numpy as np
import torch

from distseal.data.datasets import ImageFolder
from distseal.data.transforms import get_resize_transform
from distseal.utils import optim as uoptim
from distseal.utils.cfg import get_config_from_checkpoint, setup_model
from distseal.utils.metrics import bit_accuracy, psnr, ssim

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Chargement modele + donnees
# ---------------------------------------------------------------------------

def build_wam(checkpoint_path: str):
    """Reconstruit le VideoWam d'entrainement (meme chemin que
    `setup_model_from_checkpoint`) et le force sur CPU, en mode eval.

    IMPORTANT (bug trouve en ecrivant ce script) : `wam.blender.scaling_w`
    est un simple flottant Python, PAS un buffer -- il n'est donc PAS
    present dans le state_dict du checkpoint. `setup_model` le reconstruit
    a la valeur STATIQUE `args.scaling_w` (0.5 dans les deux configs
    phase0/phase2), qui n'est que le point de depart du schedule
    `scaling_w_schedule` (Cosine vers scaling_min=0.1 entre les epoques
    150 et 1050, cf. train.py:374-379 et
    `distseal.utils.optim.ScalingScheduler`). Le checkpoint reel a ete
    entraine bien au-dela de l'epoque 1050 (donc scaling_w=0.1 en
    pratique) -- utiliser 0.5 tel quel revient a appliquer un filigrane
    5x plus fort que celui reellement optimise/evalue pendant
    l'entrainement, ce qui a d'abord produit ici un PSNR ~9dB / SSIM
    negatif (image detruite) au lieu des ~22dB attendus (cf. log.txt de
    l'entrainement). On rejoue donc ici le meme schedule, au meme epoch
    que le checkpoint, pour retrouver le scaling_w REELLEMENT utilise.
    """
    cfg = get_config_from_checkpoint(checkpoint_path)
    wam = setup_model(cfg, checkpoint_path)
    wam = wam.to(DEVICE)
    wam.eval()

    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epoch = raw_ckpt.get("epoch", None)
    args = cfg.args
    if epoch is not None and args.get("scaling_w_schedule", None):
        schedule_params = uoptim.parse_params(args.scaling_w_schedule)
        scheduler = uoptim.ScalingScheduler(
            obj=wam.blender, attribute="scaling_w", scaling_o=args.scaling_w,
            **schedule_params,
        )
        old_scaling_w = wam.blender.scaling_w
        new_scaling_w = scheduler.step(epoch)
        print(f"[qualitative] checkpoint epoch={epoch} -- scaling_w schedule "
              f"'{args.scaling_w_schedule}' : {old_scaling_w:g} (config statique) "
              f"-> {new_scaling_w:g} (valeur reellement utilisee a cet epoch)")
    else:
        print("[qualitative] pas de scaling_w_schedule dans le checkpoint -- "
              f"scaling_w={wam.blender.scaling_w:g} (valeur de config)")

    return wam, cfg


def pick_images(data_dir: str, img_size: int, n: int, seed: int):
    """Choisit n images distinctes (scenes differentes quand possible) du
    dossier de validation, redimensionnees a img_size x img_size."""
    transform, _ = get_resize_transform(img_size, resize_only=True)
    ds = ImageFolder(data_dir, transform=transform)
    if len(ds) == 0:
        raise RuntimeError(f"{data_dir} ne contient aucune image exploitable")
    n = min(n, len(ds))

    # les images de ce corpus sont nommees "<scene>_<crop>.png" (cf.
    # build_corpus.py) : on privilegie des scenes differentes pour que la
    # figure montre une vraie diversite de contenu plutot que 5 recadrages
    # de la meme photo.
    names = [os.path.basename(p) for p in ds.samples]
    scenes = [name.split("_")[0] for name in names]
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    seen_scenes = set()
    chosen = []
    for i in order:
        if scenes[i] in seen_scenes:
            continue
        seen_scenes.add(scenes[i])
        chosen.append(i)
        if len(chosen) == n:
            break
    # dataset trop peu diversifie (moins de n scenes uniques) : complete
    # avec les images restantes plutot que d'echouer.
    if len(chosen) < n:
        for i in order:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == n:
                break

    imgs = torch.stack([ds[i][0] for i in chosen], dim=0)
    picked_names = [names[i] for i in chosen]
    return imgs, picked_names


# ---------------------------------------------------------------------------
# Metriques par image
# ---------------------------------------------------------------------------

def decode_bit_acc(wam, imgs_w: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
    """Detecte le message dans imgs_w et retourne le bit_acc par image face
    au message de reference msgs (b k)."""
    with torch.no_grad():
        preds = wam.detect(imgs_w, is_video=False)["preds"]  # b (1+k) h w
    bit_preds = preds[:, 1:]  # b k h w -- canal 0 = masque de detection, inutilise ici
    return bit_accuracy(bit_preds, msgs)


def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """CxHxW dans [0,1] -> HxWxC numpy pour matplotlib."""
    return t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# Figure 1 : original / filigrane / difference amplifiee, par image
# ---------------------------------------------------------------------------

def make_main_grid(wam, imgs: torch.Tensor, names: list, gain: float, out_path: str):
    """Une ligne par image, colonnes = [original, filigrane, |diff|*gain].

    La carte de difference est normalisee ainsi : diff = |imgs_w - imgs|,
    valeurs dans [0,1] par construction (images elles-memes dans [0,1]),
    puis multipliee par `gain` et reclipee a [0,1] pour l'affichage --
    c'est une AMPLIFICATION VISUELLE uniquement (facteur `gain`, documente
    dans le titre de la colonne), pas une mesure physique : sans ce
    facteur, le filigrane residuel serait a peine visible a l'oeil sur un
    ecran meme quand il degrade fortement le PSNR.
    """
    n = imgs.shape[0]
    results = []

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = imgs[i:i + 1]
        msg = wam.get_random_msg(1).to(DEVICE)
        with torch.no_grad():
            out = wam.embed(img, msg, is_video=False)
        img_w = out["imgs_w"]

        bit_acc = decode_bit_acc(wam, img_w, msg).item()
        val_psnr = psnr(img_w, img).item()
        val_ssim = ssim(img_w, img).item()
        results.append((names[i], bit_acc, val_psnr, val_ssim))

        diff = (img_w - img).abs() * gain
        diff = diff.clamp(0, 1)

        axes[i, 0].imshow(to_numpy_img(img[0]))
        axes[i, 1].imshow(to_numpy_img(img_w[0]))
        axes[i, 2].imshow(to_numpy_img(diff[0]))

        axes[i, 0].set_ylabel(names[i], fontsize=8)
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 1].set_title(
            f"bit_acc={bit_acc:.3f}  PSNR={val_psnr:.1f}dB  SSIM={val_ssim:.3f}",
            fontsize=8,
        )

    axes[0, 0].set_title("original", fontsize=10)
    axes[0, 1].set_title("filigrane (watermarked)", fontsize=10)
    axes[0, 2].set_title(f"|diff| x{gain:g} (amplifie, clip [0,1])", fontsize=10)

    fig.suptitle(
        "CipherMark / DistSeal -- nbits=256 -- checkpoint courant "
        "(filigrane fortement visible, pas encore de pression de fidelite)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# Figure 2 : deux messages differents dans la MEME image
# ---------------------------------------------------------------------------

def make_two_messages_demo(wam, img: torch.Tensor, name: str, gain: float, out_path: str):
    """Preuve concrete de la propriete "filigrane variable par
    utilisateur" : deux messages aleatoires distincts embarques dans la
    MEME image source, chacun decode correctement avec SON message et
    incorrectement (proche du hasard) si on le confronte a l'autre."""
    img = img[0:1]  # une seule image, garde la dim batch
    msg_a = wam.get_random_msg(1).to(DEVICE)
    msg_b = wam.get_random_msg(1).to(DEVICE)

    with torch.no_grad():
        out_a = wam.embed(img, msg_a, is_video=False)
        out_b = wam.embed(img, msg_b, is_video=False)
    img_wa = out_a["imgs_w"]
    img_wb = out_b["imgs_w"]

    n_bits = msg_a.shape[1]
    n_diff_msgs = int((msg_a != msg_b).sum().item())
    frac_diff_msgs = n_diff_msgs / n_bits

    acc_a_with_a = decode_bit_acc(wam, img_wa, msg_a).item()
    acc_b_with_b = decode_bit_acc(wam, img_wb, msg_b).item()
    # controle croise : decoder l'image A avec le message B (et vice versa)
    # doit donner un bit_acc proche du hasard (~0.5), preuve que le
    # filigrane porte bien SON message et pas un signal generique.
    acc_a_with_b = decode_bit_acc(wam, img_wa, msg_b).item()
    acc_b_with_a = decode_bit_acc(wam, img_wb, msg_a).item()

    n_diff_pixels = int((img_wa != img_wb).any(dim=1).sum().item())
    frac_diff_pixels = n_diff_pixels / (img_wa.shape[-2] * img_wa.shape[-1])

    print("\n[deux messages / meme image] -- preuve du filigrane variable par utilisateur")
    print(f"  image source                 : {name}")
    print(f"  messages differents (bits)   : {n_diff_msgs}/{n_bits} ({frac_diff_msgs:.1%})")
    print(f"  pixels de sortie differents   : {frac_diff_pixels:.1%} de l'image")
    print(f"  decodage A avec message A (attendu haut)  : bit_acc = {acc_a_with_a:.3f}")
    print(f"  decodage B avec message B (attendu haut)  : bit_acc = {acc_b_with_b:.3f}")
    print(f"  decodage A avec message B (attendu ~0.5)  : bit_acc = {acc_a_with_b:.3f}")
    print(f"  decodage B avec message A (attendu ~0.5)  : bit_acc = {acc_b_with_a:.3f}")

    diff_ab = (img_wa - img_wb).abs() * gain
    diff_ab = diff_ab.clamp(0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(to_numpy_img(img[0]))
    axes[0].set_title("original")
    axes[1].imshow(to_numpy_img(img_wa[0]))
    axes[1].set_title(f"filigrane msg A\nbit_acc(A|A)={acc_a_with_a:.3f}")
    axes[2].imshow(to_numpy_img(img_wb[0]))
    axes[2].set_title(f"filigrane msg B\nbit_acc(B|B)={acc_b_with_b:.3f}")
    axes[3].imshow(to_numpy_img(diff_ab[0]))
    axes[3].set_title(f"|A-B| x{gain:g}\nmsgs differents a {frac_diff_msgs:.0%}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Meme image ({name}), deux messages aleatoires distincts -- "
        f"controle croise bit_acc(A|B)={acc_a_with_b:.3f}, bit_acc(B|A)={acc_b_with_a:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "frac_diff_msgs": frac_diff_msgs,
        "frac_diff_pixels": frac_diff_pixels,
        "acc_a_with_a": acc_a_with_a,
        "acc_b_with_b": acc_b_with_b,
        "acc_a_with_b": acc_a_with_b,
        "acc_b_with_a": acc_b_with_a,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_data_dir(explicit: str | None) -> str:
    """Determine le dossier d'images a utiliser: argument explicite, sinon
    corpus-colab/val (plus diversifie), sinon corpus-smoketest/val, sinon
    echec (ce script ne telecharge jamais d'images depuis internet)."""
    if explicit is not None:
        return explicit
    candidates = ["./corpus-colab/val", "./corpus-smoketest/val"]
    for c in candidates:
        if os.path.isdir(c) and len(os.listdir(c)) > 0:
            return c
    raise RuntimeError(
        "Aucun dossier d'images local trouve (corpus-colab/val, "
        "corpus-smoketest/val) -- fournir --data-dir explicitement."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./runs/qualitative_checkpoint.pth")
    ap.add_argument("--data-dir", default=None,
                    help="dossier d'images reelles (defaut: auto-detection "
                         "corpus-colab/val puis corpus-smoketest/val)")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--gain", type=float, default=10.0,
                    help="facteur d'amplification visuelle des cartes de difference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/qualitative_figures")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = find_data_dir(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[qualitative] checkpoint = {args.checkpoint}")
    print(f"[qualitative] data_dir   = {data_dir}")

    wam, cfg = build_wam(args.checkpoint)
    img_size = cfg.args.img_size_proc if "img_size_proc" in cfg.args else cfg.args.img_size_extractor
    print(f"[qualitative] embedder={cfg.embedder.model} extractor={cfg.extractor.model} "
          f"nbits={cfg.args.nbits} img_size={img_size} "
          f"scaling_w_effectif={wam.blender.scaling_w:g} (cf. build_wam pour le schedule)")

    imgs, names = pick_images(data_dir, img_size, args.n_images, args.seed)
    print(f"[qualitative] {len(names)} images retenues: {names}")

    # --- figure 1 : grille original / filigrane / diff amplifiee -----------
    grid_path = os.path.join(args.out_dir, "grid_original_vs_watermarked.png")
    results = make_main_grid(wam, imgs, names, args.gain, grid_path)

    # --- figure 2 : deux messages differents, meme image --------------------
    two_msg_path = os.path.join(args.out_dir, "two_messages_same_image.png")
    two_msg_stats = make_two_messages_demo(wam, imgs[0:1], names[0], args.gain, two_msg_path)

    # --- tableau recap --------------------------------------------------
    lines = []
    lines.append("CipherMark / DistSeal -- figures qualitatives -- resume")
    lines.append(f"checkpoint : {args.checkpoint}")
    lines.append(f"data_dir   : {data_dir}")
    lines.append(f"embedder={cfg.embedder.model} extractor={cfg.extractor.model} "
                 f"nbits={cfg.args.nbits} scaling_w_effectif={wam.blender.scaling_w:g}")
    lines.append("")
    lines.append(f"{'image':30s}  {'bit_acc':>8s}  {'PSNR(dB)':>9s}  {'SSIM':>6s}")
    lines.append("-" * 60)
    accs, psnrs, ssims = [], [], []
    for name, bit_acc, val_psnr, val_ssim in results:
        lines.append(f"{name:30s}  {bit_acc:8.3f}  {val_psnr:9.2f}  {val_ssim:6.3f}")
        accs.append(bit_acc)
        psnrs.append(val_psnr)
        ssims.append(val_ssim)
    lines.append("-" * 60)
    lines.append(f"{'moyenne':30s}  {np.mean(accs):8.3f}  {np.mean(psnrs):9.2f}  {np.mean(ssims):6.3f}")
    lines.append("")
    lines.append("Demo deux messages / meme image (image: {}):".format(names[0]))
    lines.append(f"  messages differents a {two_msg_stats['frac_diff_msgs']:.1%} des bits")
    lines.append(f"  pixels de sortie differents: {two_msg_stats['frac_diff_pixels']:.1%}")
    lines.append(f"  bit_acc(A|msg A) = {two_msg_stats['acc_a_with_a']:.3f}  (attendu eleve)")
    lines.append(f"  bit_acc(B|msg B) = {two_msg_stats['acc_b_with_b']:.3f}  (attendu eleve)")
    lines.append(f"  bit_acc(A|msg B) = {two_msg_stats['acc_a_with_b']:.3f}  (attendu ~0.5, controle croise)")
    lines.append(f"  bit_acc(B|msg A) = {two_msg_stats['acc_b_with_a']:.3f}  (attendu ~0.5, controle croise)")
    summary = "\n".join(lines)

    print("\n" + summary)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")

    print(f"\n[qualitative] figures ecrites:")
    print(f"  {grid_path}")
    print(f"  {two_msg_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Premier test de bout en bout de CipherMark avec le VRAI Wam entraine.

Contexte : `tests/ciphermark/test_pipeline.py` valide toute la mecanique
(boucle de point fixe, registre, verifieur) mais avec un `FakeWam` -- un
tatoueur jouet qui ecrit Omega en clair dans les pixels. Ce script fait
transiter un Omega reel par un embedder/extracteur DistSeal REELLEMENT
entraines.

Ce que ce script mesure :
  1. La chaine complete (embed -> registre -> verify) de bout en bout, avec
     des poids reels, sur CPU.
  2. Propriete centrale de la these -- variabilite par utilisateur : la
     MEME image, sous deux jeux de cles differents, produit deux
     Omega/watermarks differents, chacun verifiable sous ses propres cles et
     rejete sous les mauvaises.
  3. Le registre : un nonce inconnu doit lever KeyError.

Le verdict depend ENTIEREMENT de la qualite du canal, jamais des seuils --
`CipherMarkThresholds` n'est pas modifie. Deux resultats de reference :

  checkpoint 256 bits (bit_acc 0.61, aout 2026) -> 0/5 AUTHENTIC,
      BER 39-55 %, point fixe plafonne a 3 iterations sans converger.
  checkpoint phaseA2_64bits_stable (bit_acc 0.9998) -> 5/5 AUTHENTIC,
      d = 0 bit d'erreur, p = 5.42e-20, point fixe converge en 1 iteration.

L'ecart tient a l'avalanche HMAC : un seul bit errone sur Omega fait diverger
le tag recalcule d'environ 50 %. La chaine exige donc un canal quasi parfait,
la ou un tatouage classique tolere des erreurs. Voir docs/a-faire-memoire.md.

Usage (CPU uniquement) :
    python -m scripts.ciphermark.full_chain_real_weights_test \
        --checkpoint runs/checkpoint.pth --val-dir corpus-colab/val
"""

from __future__ import annotations

# Machine CPU-only : on l'impose avant meme l'import de torch, par securite
# (cf. sse_hessian_experiment.py, meme discipline).
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import random
import time

import numpy as np
import torch

from distseal.ciphermark.equation import CipherMarkVerifier
from distseal.ciphermark.phash import PerceptualHash, _try_load_dinov2, _DCTFallback
from distseal.ciphermark.registry import TraceRegistry
from distseal.ciphermark.wam_ciphermark import (
    CipherMarkConfig,
    CipherMarkKeys,
    CipherMarkWam,
)
from distseal.data.datasets import ImageFolder
from distseal.data.transforms import get_resize_transform
from distseal.utils import optim as uoptim
from distseal.utils.cfg import get_config_from_checkpoint, setup_model

DEVICE = torch.device("cpu")


def log(msg: str) -> None:
    print(f"[full-chain] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Chargement du Wam reel (meme motif que sse_hessian_experiment.py /
# distseal.utils.cfg.setup_model_from_checkpoint)
# ---------------------------------------------------------------------------

def load_checkpoint_with_retry(path: str, retries: int = 5, delay: float = 5.0):
    """torch.load() avec re-essais : le checkpoint peut etre en cours de
    reecriture par un job d'entrainement concurrent (cf. meme motif que
    sse_hessian_experiment.py)."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log(f"lecture checkpoint echouee (tentative {attempt}/{retries}): {exc}, "
                f"nouvel essai dans {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"impossible de charger {path} apres {retries} tentatives") from last_exc


def _replay_scaling_schedule(wam, cfg, checkpoint_path: str) -> None:
    """
    BUG CORRIGE : `wam.blender.scaling_w` est un simple float Python (pas un
    buffer du state_dict), donc `setup_model()` le reconstruit a la valeur
    statique de la config (`args.scaling_w`, ex 0.5) au lieu de la valeur
    reellement atteinte a l'epoque du checkpoint via `scaling_w_schedule`
    (ex Cosine vers 0.1 des l'epoque 1050). Sans ce correctif, on embarque
    a une amplitude jusqu'a 5x trop forte par rapport a ce que l'extracteur
    a ete entraine a lire a ce stade -- PSNR artificiellement degrade et
    bit_acc biaise. On rejoue le meme ScalingScheduler que train.py
    (meme obj/attribute/parametres) jusqu'a l'epoque sauvegardee.
    Trouve/corrige initialement dans generate_qualitative_figures.py.
    """
    if cfg.args.scaling_w_schedule is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epoch = checkpoint.get("epoch")
    if epoch is None:
        return
    schedule_params = uoptim.parse_params(cfg.args.scaling_w_schedule)
    scheduler = uoptim.ScalingScheduler(
        obj=wam.blender, attribute="scaling_w", scaling_o=cfg.args.scaling_w,
        **schedule_params,
    )
    effective = scheduler.step(epoch)
    log(f"scaling_w rejoue depuis le schedule a l'epoque {epoch}: "
        f"{cfg.args.scaling_w} (statique) -> {effective:.4f} (effectif)")


def build_wam(checkpoint_path: str, retries: int = 5, delay: float = 5.0):
    """Reconstruit le VideoWam d'entrainement depuis le checkpoint, force CPU/eval."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            cfg = get_config_from_checkpoint(checkpoint_path)
            wam = setup_model(cfg, checkpoint_path)
            _replay_scaling_schedule(wam, cfg, checkpoint_path)
            wam = wam.to(DEVICE)
            wam.eval()
            return wam, cfg
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log(f"construction du Wam echouee (tentative {attempt}/{retries}): {exc}, "
                f"nouvel essai dans {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"impossible de construire le Wam depuis {checkpoint_path} apres {retries} tentatives"
    ) from last_exc


def load_val_images(val_dir: str, img_size: int, n_total: int, seed: int) -> torch.Tensor:
    """Charge n_total images reelles de validation (memes conventions que
    sse_hessian_experiment.py)."""
    transform, _ = get_resize_transform(img_size, resize_only=True)
    ds = ImageFolder(val_dir, transform=transform)
    if len(ds) < n_total:
        raise RuntimeError(f"{val_dir} ne contient que {len(ds)} images, il en faut {n_total}")
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n_total)
    imgs = torch.stack([ds[i][0] for i in idx], dim=0)
    return imgs


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def fmt_row(label: str, verdict: str, distance: int, total: int, p: float, conf: float) -> str:
    ber = distance / max(1, total)
    return (f"{label:38s} {verdict:16s} d={distance:4d}/{total:<4d} "
            f"ber={ber:6.3f} p={p:9.2e} conf={conf:6.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default="runs/checkpoint.pth")
    ap.add_argument("--val-dir", type=str, default="corpus-colab/val")
    ap.add_argument("--n-bits", type=int, default=64,
                    help="largeur d Omega. 64 et non 256 : mesure du 2026-08-21, "
                         "cf. docs/a-faire-memoire.md -- a 256 bits le canal "
                         "plafonne (bit_acc 0.61, 0/5 AUTHENTIC), a 64 il rend "
                         "5/5 a zero bit d erreur.")
    ap.add_argument("--n-images", type=int, default=5,
                     help="nb d'images distinctes pour le test multi-utilisateurs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--summary-path", type=str, default="runs/full_chain_test_summary.txt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    log(f"demarrage. checkpoint={args.checkpoint}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

    ckpt_epoch = load_checkpoint_with_retry(args.checkpoint).get("epoch")
    log(f"checkpoint epoch={ckpt_epoch}")

    t0 = time.time()
    wam, cfg = build_wam(args.checkpoint)
    nbits = int(cfg.args.nbits)
    img_size = int(cfg.args.img_size_proc)
    log(f"Wam charge en {time.time() - t0:.1f}s (nbits={nbits}, img_size={img_size}, "
        f"embedder={cfg.embedder.model}, extractor={cfg.extractor.model}, "
        f"scaling_w={cfg.args.scaling_w})")
    if nbits != args.n_bits:
        log(f"ATTENTION: --n-bits={args.n_bits} != nbits du checkpoint ({nbits}), "
            f"on utilise celui du checkpoint")
        args.n_bits = nbits

    # ------------------------------------------------------------- PHash ----
    # On tente DINOv2 reel en premier (pas le fallback DCT jouet). S'il n'est
    # pas installable/pas d'internet, _try_load_dinov2 renvoie None et on
    # bascule sur le fallback deja prevu par phash.py -- on le dit clairement.
    dino = _try_load_dinov2()
    if dino is not None:
        log("PHash: backbone DINOv2-small REEL charge")
        phash_backbone = dino
    else:
        log("PHash: DINOv2 indisponible dans cet environnement -> fallback "
            "_DCTFallback (deja prevu par phash.py). Resultats toujours valides "
            "mais avec un hash perceptuel plus fragile.")
        phash_backbone = _DCTFallback()

    log("construction du PerceptualHash (peut prendre un peu de temps: "
        "inference dummy pour inferer la dimension de features)")
    phash = PerceptualHash(n_bits=args.n_bits, backbone=phash_backbone).to(DEVICE)
    phash.eval()

    # ------------------------------------------------------------- images ---
    n_needed = args.n_images
    log(f"chargement de {n_needed} images reelles depuis {args.val_dir}")
    imgs = load_val_images(args.val_dir, img_size, n_needed, args.seed)

    cm_cfg = CipherMarkConfig(n_bits=args.n_bits, max_fixed_point_iters=3)

    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("CipherMark -- test de bout en bout avec le VRAI Wam entraine")
    summary_lines.append(f"checkpoint={args.checkpoint} (epoch={ckpt_epoch})")
    summary_lines.append(f"nbits={nbits} embedder={cfg.embedder.model} extractor={cfg.extractor.model} "
                          f"scaling_w={cfg.args.scaling_w}")
    summary_lines.append(f"phash backbone: {'DINOv2-small reel' if dino is not None else 'DCT fallback (DINOv2 indispo)'}")
    summary_lines.append("=" * 100)

    # =======================================================================
    # Partie A -- round-trip honnete multi-images (un seul "utilisateur")
    # =======================================================================
    log("=== Partie A: round-trip honnete sur plusieurs images ===")
    keys_main = CipherMarkKeys.random()
    registry_main = TraceRegistry()
    cm_main = CipherMarkWam(wam=wam, phash=phash, keys=keys_main,
                             cfg=cm_cfg, registry=registry_main)

    t0 = time.time()
    out = cm_main.embed(imgs)
    log(f"embed() de {n_needed} images en {time.time() - t0:.1f}s "
        f"(converged={out['converged']}, n_iters={out['n_iters']})")

    for i in range(n_needed):
        p = psnr(imgs[i], out["imgs_w"][i])
        log(f"  image {i}: PSNR = {p:.2f} dB")

    t0 = time.time()
    reports = cm_main.verify(out["imgs_w"], out["image_ids"])
    log(f"verify() en {time.time() - t0:.1f}s")

    summary_lines.append("")
    summary_lines.append("-- Partie A: round-trip honnete (memes cles, meme registre) --")
    n_authentic = 0
    for i, r in enumerate(reports):
        label = f"image_{i} (id={out['image_ids'][i]}) honnete"
        summary_lines.append(fmt_row(label, r.verdict.value, r.distance, r.total, r.p_value, r.confidence))
        log(fmt_row(label, r.verdict.value, r.distance, r.total, r.p_value, r.confidence))
        if r.verdict.value == "authentic":
            n_authentic += 1
    log(f"bilan Partie A: {n_authentic}/{n_needed} verdicts AUTHENTIC "
        f"(converged={out['converged']}, n_iters={out['n_iters']})")
    summary_lines.append(f"bilan: {n_authentic}/{n_needed} AUTHENTIC, "
                          f"fixed_point_converged={out['converged']}, n_iters={out['n_iters']}")

    # =======================================================================
    # Partie B -- variabilite par utilisateur (coeur de la these) :
    # meme image, deux jeux de cles distincts ("deux utilisateurs").
    # =======================================================================
    log("=== Partie B: variabilite par utilisateur (meme image, cles differentes) ===")
    shared_image = imgs[0:1]  # UNE seule image, partagee entre les deux "utilisateurs"

    keys_A = CipherMarkKeys.random()
    keys_B = CipherMarkKeys.random()
    registry_A = TraceRegistry()
    registry_B = TraceRegistry()
    cm_A = CipherMarkWam(wam=wam, phash=phash, keys=keys_A, cfg=cm_cfg, registry=registry_A)
    cm_B = CipherMarkWam(wam=wam, phash=phash, keys=keys_B, cfg=cm_cfg, registry=registry_B)

    out_A = cm_A.embed(shared_image)
    out_B = cm_B.embed(shared_image)

    omega_A = out_A["omega"][0].cpu().numpy()
    omega_B = out_B["omega"][0].cpu().numpy()
    n_diff_bits = int(np.sum(omega_A != omega_B))
    imgs_w_A, imgs_w_B = out_A["imgs_w"], out_B["imgs_w"]
    n_diff_pixels = float(torch.mean((imgs_w_A != imgs_w_B).float()).item())

    log(f"Omega_A vs Omega_B: {n_diff_bits}/{args.n_bits} bits differents "
        f"({n_diff_bits / args.n_bits:.1%})")
    log(f"pixels differents entre imgs_w_A et imgs_w_B: {n_diff_pixels:.1%}")

    summary_lines.append("")
    summary_lines.append("-- Partie B: variabilite par utilisateur (meme image source) --")
    summary_lines.append(f"Omega_A != Omega_B sur {n_diff_bits}/{args.n_bits} bits "
                          f"({n_diff_bits / args.n_bits:.1%})")
    summary_lines.append(f"fraction de pixels differents entre imgs_w_A et imgs_w_B: {n_diff_pixels:.1%}")

    # A verifie sous ses propres cles/registre (honnete)
    report_A_own = cm_A.verify(imgs_w_A, out_A["image_ids"])[0]
    # B verifie sous ses propres cles/registre (honnete)
    report_B_own = cm_B.verify(imgs_w_B, out_B["image_ids"])[0]
    # A verifie sous les cles/registre de B (mauvaises cles -- attaque/erreur)
    report_A_under_B = cm_B.verify(imgs_w_A, out_B["image_ids"])[0]
    # B verifie sous les cles/registre de A (mauvaises cles -- attaque/erreur)
    report_B_under_A = cm_A.verify(imgs_w_B, out_A["image_ids"])[0]

    for label, r in [
        ("watermark_A sous cles_A (honnete)", report_A_own),
        ("watermark_B sous cles_B (honnete)", report_B_own),
        ("watermark_A sous cles_B (MAUVAISES cles)", report_A_under_B),
        ("watermark_B sous cles_A (MAUVAISES cles)", report_B_under_A),
    ]:
        row = fmt_row(label, r.verdict.value, r.distance, r.total, r.p_value, r.confidence)
        log(row)
        summary_lines.append(row)

    security_ok = (
        report_A_under_B.verdict.value != "authentic"
        and report_B_under_A.verdict.value != "authentic"
    )
    log(f"propriete de securite (mauvaises cles => pas AUTHENTIC): "
        f"{'OK' if security_ok else 'ECHEC -- FAUX POSITIF'}")
    summary_lines.append(f"propriete de securite (mauvaises cles rejetees): "
                          f"{'OK' if security_ok else 'ECHEC -- FAUX POSITIF'}")

    # =======================================================================
    # Partie C -- sanity check supplementaire : nonce inconnu => KeyError
    # =======================================================================
    log("=== Partie C: nonce absent du registre => KeyError attendu ===")
    try:
        cm_main.verify(out["imgs_w"][:1], [999999])
        c_ok = False
        log("ECHEC: un nonce inconnu aurait du lever KeyError")
    except KeyError as e:
        c_ok = True
        log(f"OK: KeyError levee comme attendu ({e})")
    summary_lines.append("")
    summary_lines.append(f"-- Partie C: nonce inconnu => KeyError -- {'OK' if c_ok else 'ECHEC'}")

    # ------------------------------------------------------------ ecriture --
    summary_lines.append("")
    summary_lines.append("=" * 100)
    summary_lines.append(
        "NOTE: le verdict depend entierement de la qualite du canal, pas de ce "
        "script -- les seuils du verifieur (CipherMarkThresholds) ne sont JAMAIS "
        "modifies. Avec un checkpoint dont bit_acc plafonne (~0.60, cas des runs "
        "a 256 bits d'aout 2026), la Partie A rend 0/5 AUTHENTIC : l'avalanche "
        "HMAC fait diverger le tag des le premier bit errone. Avec un canal sain "
        "(bit_acc ~0.999, cf. phaseA2_64bits_stable), elle rend 5/5 AUTHENTIC a "
        "d=0 bit d'erreur. Un echec ici n'est donc pas un bug : c'est la mesure "
        "fidele de l'etat du canal."
    )
    os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
    with open(args.summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    log(f"resume ecrit dans {args.summary_path}")

    log("termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

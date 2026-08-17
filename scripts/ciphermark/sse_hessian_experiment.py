# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Experience SSE-Hessien sur un VRAI checkpoint entraine (jamais fait avant).

Contexte (cf. memoire, "3 points ouverts") : `stable_subspace.py` sait
calculer un projecteur base sur les vecteurs propres dominants du Hessien
du loss (Lanczos + produits Hessien-vecteur par double retropropagation),
mais ce projecteur n'a JAMAIS ete branche sur `CipherMarkMsgProcessor` --
celui-ci utilise en pratique un projecteur ALEATOIRE fixe (`LatentProjector`
avec seed=0xC0FFEE). Ce script fait la toute premiere validation empirique
de l'hypothese SSE sur un modele reel :

  1. On charge le Wam DistSeal reel entraine (nbits=256), pas encore
     enrobe par CipherMark -- on travaille directement sur les poids de la
     table d'embedding binaire de son MsgProcessor
     (`wam.embedder.msg_processor.msg_embeddings.weight`, forme
     (2*nbits, hidden_size)) : c'est la surface d'attaque naturelle pour un
     "LoRA fine-tuning" (mise a jour additive de faible rang sur cette
     matrice).
  2. On construit/charge (cache disque) un projecteur Hessien sur cette
     table via `build_projector_from_loss`, en utilisant comme loss le
     BCEWithLogits de decodage (embed -> detect) sur un petit batch fixe
     d'images reelles de `corpus-colab/val` et un message fixe.
  3. On simule un fine-tuning adverse (quelques dizaines de pas de descente
     de gradient normalisee sur une tache proxy -- minimiser la distortion
     imgs_w vs imgs, PAS le decodage) dont la mise a jour est contrainte a
     un sous-espace de dimension k, pour 3 conditions :
       (i)   aucune contrainte    -- attaque "pire cas" sur la table brute
       (ii)  sous-espace ALEATOIRE  -- ce que fait reellement CipherMark
             aujourd'hui
       (iii) sous-espace HESSIEN    -- l'hypothese SSE jamais testee
     et on mesure la derive du bit_acc de decodage avant/apres.

IMPORTANT (limite documentee, a mentionner dans le memoire) : le Hessien
est celui du loss de DECODAGE (ce qu'on protege), pas de la tache proxy de
l'attaquant. L'hypothese SSE complete suppose que les directions de plus
grande courbure du decodage sont aussi des directions "rigides" pour des
taches non liees (le gradient naturel d'un fine-tuning generique les
evite). Ce script mesure la retombee sur bit_acc dans les 3 conditions ET
la perte proxy atteinte par pas, ce qui permet de voir si le sous-espace
Hessien est effectivement moins "utile" a la tache de l'attaquant (preuve
indirecte de rigidite) -- mais ne prouve pas directement l'hypothese
generale.

Usage (CPU uniquement, ne touche jamais au GPU -- a lancer a cote d'un job
torchrun sur le meme pod) :

    python -m scripts.ciphermark.sse_hessian_experiment \
        --checkpoint /workspace/runs/runpod_256bits_phase0_puredecode/checkpoint.pth

Sorties (uniquement sous runs/, jamais dans le checkpoint source) :
    runs/sse_hessian_projector_phase0.pt        (cache du projecteur Hessien)
    runs/sse_hessian_experiment_results.json    (table de resultats)
"""

from __future__ import annotations

# Le GPU est dedie a un entrainement en cours sur ce pod : on interdit CUDA
# AVANT meme d'importer torch, pour ne jamais risquer d'y toucher.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from distseal.ciphermark.stable_subspace import (
    LanczosConfig,
    StableProjector,
    build_projector_from_loss,
    estimate_top_eigenspace,
    random_stable_projector,
)
from distseal.data.datasets import ImageFolder
from distseal.data.transforms import get_resize_transform
from distseal.utils import optim as uoptim
from distseal.utils.cfg import get_config_from_checkpoint, setup_model
from distseal.utils.metrics import bit_accuracy

DEVICE = torch.device("cpu")


def log(msg: str) -> None:
    print(f"[sse-hessian] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def patch_grn_for_double_backward() -> None:
    """Rustine (au runtime, sans toucher au fichier source) de la couche GRN
    du ConvNeXt extractor.

    Constat empirique (diagnostic sur ce checkpoint) : `distseal/modules/
    common.py::GRN.forward` utilise `torch.norm(x, p=2, dim=(1,2))`. La
    derivee SECONDE de `torch.norm` est singuliere quand la norme est nulle
    ou quasi nulle (ce qui arrive : certains canaux ConvNeXt ont une carte
    spatiale quasi nulle) -- `torch.autograd.grad(..., create_graph=True)`
    (necessaire pour les produits Hessien-vecteur de Lanczos) fait alors
    remonter des NaN via `DivBackward0`, confirme par
    `torch.autograd.set_detect_anomaly(True, check_nan=True)`.

    On ne peut pas editer `common.py` (script auto-contenu, perimetre de
    cette experience). On monkeypatch donc `GRN.forward` avec un calcul
    STRICTEMENT equivalent numeriquement (meme sortie a 1e-6 pres, le meme
    epsilon que celui deja utilise par les autres couches de norme de ce
    fichier), mais avec l'epsilon SOUS la racine plutot qu'apres coup, ce
    qui rend la derivee seconde partout bien definie.
    """
    from distseal.modules import common as _common_mod

    def _stable_grn_forward(self, x):
        Gx = torch.sqrt((x ** 2).sum(dim=(1, 2), keepdim=True) + 1e-6)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

    _common_mod.GRN.forward = _stable_grn_forward
    log("GRN.forward monkeypatche (norme stabilisee pour le double-backward Lanczos)")


def load_checkpoint_with_retry(path: str, retries: int = 5, delay: float = 5.0):
    """torch.load() avec re-essais.

    Le checkpoint est reecrit periodiquement par le job d'entrainement GPU
    qui tourne en parallele sur ce pod (cf. saveckpt_freq) -- une lecture
    peut donc tomber en plein milieu d'une ecriture et echouer
    (`PytorchStreamReader failed reading file ...`), observe empiriquement.
    On ne modifie jamais le fichier, on relit juste apres un court delai.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - on veut vraiment tout capter ici
            last_exc = exc
            log(f"lecture checkpoint echouee (tentative {attempt}/{retries}): {exc} "
                f"-- probablement une ecriture concurrente du job d'entrainement, "
                f"nouvel essai dans {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"impossible de charger {path} apres {retries} tentatives"
    ) from last_exc


# ---------------------------------------------------------------------------
# Chargement modele + donnees
# ---------------------------------------------------------------------------

def _replay_scaling_schedule(wam, cfg, checkpoint_path: str) -> None:
    """
    BUG CORRIGE : `wam.blender.scaling_w` est un simple float Python (pas un
    buffer du state_dict), donc `setup_model()` le reconstruit a la valeur
    statique de la config (`args.scaling_w`, ex 0.5) au lieu de la valeur
    reellement atteinte a l'epoque du checkpoint via `scaling_w_schedule`
    (ex Cosine vers 0.1 des l'epoque 1050). Sans ce correctif, on attaque
    une table d'embedding utilisee a une amplitude jusqu'a 5x trop forte par
    rapport a ce que l'extracteur a ete entraine a lire a ce stade -- biaise
    aussi bien le Hessien (loss de decodage evalue au mauvais point de
    fonctionnement) que le bit_acc de reference. On rejoue le meme
    ScalingScheduler que train.py (meme obj/attribute/parametres) jusqu'a
    l'epoque sauvegardee. Trouve/corrige initialement dans
    generate_qualitative_figures.py, propage dans full_chain_real_weights_test.py
    (cf. sa fonction `_replay_scaling_schedule`, meme motif ici).
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
    """Reconstruit exactement le VideoWam d'entrainement (meme chemin que
    setup_model_from_checkpoint), force sur CPU.

    `get_config_from_checkpoint`/`setup_model` font leur propre `torch.load`
    en interne (fichier hors perimetre de ce script, on ne le modifie pas) --
    on encapsule donc l'ENSEMBLE dans une boucle de re-essai, pour la meme
    raison que `load_checkpoint_with_retry` (lecture concurrente du job
    d'entrainement GPU qui reecrit ce checkpoint)."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            cfg = get_config_from_checkpoint(checkpoint_path)
            wam = setup_model(cfg, checkpoint_path)
            _replay_scaling_schedule(wam, cfg, checkpoint_path)
            wam = wam.to(DEVICE)
            wam.eval()  # batchnorm sur stats roulantes -> loss_fn deterministe
            return wam, cfg
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log(f"construction du Wam echouee (tentative {attempt}/{retries}): {exc} "
                f"-- nouvel essai dans {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"impossible de construire le Wam depuis {checkpoint_path} apres {retries} tentatives"
    ) from last_exc


def load_val_images(val_dir: str, img_size: int, n_total: int, seed: int) -> torch.Tensor:
    """Charge n_total images de validation, resize a img_size, ordre
    melange de facon deterministe (seed) pour pouvoir decouper en batches
    disjoints (hessien / fine-tune / eval)."""
    transform, _ = get_resize_transform(img_size, resize_only=True)
    ds = ImageFolder(val_dir, transform=transform)
    if len(ds) < n_total:
        raise RuntimeError(
            f"{val_dir} ne contient que {len(ds)} images, il en faut {n_total}"
        )
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n_total)
    imgs = torch.stack([ds[i][0] for i in idx], dim=0)
    return imgs


def get_msg_embedding_weight(wam) -> torch.nn.Parameter:
    """Recupere la table d'embedding binaire du MsgProcessor DistSeal brut
    (2*nbits, hidden_size). C'est la surface d'attaque LoRA naturelle."""
    try:
        return wam.embedder.msg_processor.msg_embeddings.weight
    except AttributeError as exc:
        raise RuntimeError(
            "Impossible de trouver wam.embedder.msg_processor.msg_embeddings "
            "-- l'architecture de l'embedder a peut-etre change."
        ) from exc


# ---------------------------------------------------------------------------
# Loss de decodage (pour Lanczos ET pour l'evaluation bit_acc)
# ---------------------------------------------------------------------------

def make_masks(imgs: torch.Tensor) -> torch.Tensor:
    return torch.ones(imgs.shape[0], 1, imgs.shape[-2], imgs.shape[-1], device=imgs.device)


def decode_forward(wam, imgs: torch.Tensor, msg: torch.Tensor):
    """Un forward embed+detect complet (differentiable), meme chemin que
    train.py (`wam(imgs, masks, msgs=..., is_video=False)`)."""
    masks = make_masks(imgs)
    out = wam(imgs, masks, msgs=msg, is_video=False)
    preds = out["preds"]  # b (1+nbits) [h w]
    return out, preds


def decode_bce_loss(preds: torch.Tensor, msg: torch.Tensor, nbits: int) -> torch.Tensor:
    logits = preds[:, 1:1 + nbits]
    if logits.dim() > 2:
        logits = logits.mean(dim=tuple(range(2, logits.dim())))
    return F.binary_cross_entropy_with_logits(logits, msg.float())


@torch.no_grad()
def eval_bit_acc(wam, imgs: torch.Tensor, msg: torch.Tensor, nbits: int) -> float:
    _, preds = decode_forward(wam, imgs, msg)
    acc = bit_accuracy(preds[:, 1:1 + nbits], msg).nanmean().item()
    return acc


# ---------------------------------------------------------------------------
# Estimation du sous-espace Hessien sur la table d'embedding reelle
# ---------------------------------------------------------------------------

def build_hessian_projector(
    wam, embed_weight: torch.nn.Parameter, imgs_hess: torch.Tensor,
    msg_hess: torch.Tensor, nbits: int, k: int, n_iter: int, n_samples: int,
    seed: int, cache_path: str,
) -> StableProjector:
    def loss_fn():
        _, preds = decode_forward(wam, imgs_hess, msg_hess)
        return decode_bce_loss(preds, msg_hess, nbits)

    cfg = LanczosConfig(
        k=k, n_iter=n_iter, n_samples=n_samples, seed=seed, verbose=True,
    )

    # build_projector_from_loss() (appele plus bas) ne conserve dans son
    # cache que la base propre, pas les valeurs propres -- on les veut
    # pour verifier "a l'oeil" que le sous-espace n'est pas degenere. On
    # calcule donc nous-memes une seule fois si le cache n'existe pas
    # encore, on sauve un cache ETENDU (basis + vals), puis on appelle
    # build_projector_from_loss qui se contente alors de le RELIRE (pas de
    # calcul en double).
    if os.path.exists(cache_path):
        log(f"cache {cache_path} deja present -- pas de recalcul Lanczos")
    else:
        log(f"lancement Lanczos: d={embed_weight.numel()}, k={k}, n_iter={n_iter}, "
            f"n_samples={n_samples} (peut prendre plusieurs minutes sur CPU)")
        t0 = time.time()
        vals, basis = estimate_top_eigenspace(loss_fn, [embed_weight], cfg)
        log(f"Lanczos termine en {time.time() - t0:.1f}s")
        log(f"valeurs propres top-{len(vals)} du Hessien (loss de decodage): "
            f"{[round(v, 6) for v in vals.tolist()]}")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({"basis": basis, "vals": vals}, cache_path)
        log(f"cache (basis + vals) sauve dans {cache_path}")

    proj = build_projector_from_loss(
        loss_fn, params=[embed_weight], k=k, cache_path=cache_path, cfg=cfg,
    )
    return proj


# ---------------------------------------------------------------------------
# Simulation LoRA-style : descente de gradient normalisee, contrainte a un
# sous-espace de dimension k, sur une tache PROXY (pas le decodage).
# ---------------------------------------------------------------------------

def proxy_loss(wam, imgs: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    """Loss proxy = distortion imgs_w vs imgs. Represente une taches de
    fine-tuning generique (ex: reduire l'empreinte visuelle du watermark)
    qui n'a rien a voir avec le decodage mais retombe sur la meme table de
    poids -- exactement le scenario documente en phase1 du memoire
    (lambda_i qui ecrase le signal)."""
    out, _ = decode_forward(wam, imgs, msg)
    return F.mse_loss(out["imgs_w"], imgs)


def run_condition(
    name: str, wam, embed_weight: torch.nn.Parameter, w0: torch.Tensor,
    imgs_ft: torch.Tensor, msg_ft: torch.Tensor,
    imgs_eval: torch.Tensor, msg_eval: torch.Tensor,
    nbits: int, projector: StableProjector | None,
    n_steps: int, step_size: float,
) -> dict:
    with torch.no_grad():
        embed_weight.data.copy_(w0)

    bit_acc_before = eval_bit_acc(wam, imgs_eval, msg_eval, nbits)
    with torch.no_grad():
        proxy_before = proxy_loss(wam, imgs_ft, msg_ft).item()

    log(f"[{name}] avant fine-tune: bit_acc={bit_acc_before:.4f}, "
        f"proxy_loss={proxy_before:.6f}")

    embed_weight.requires_grad_(True)
    t0 = time.time()
    for step in range(n_steps):
        if embed_weight.grad is not None:
            embed_weight.grad = None
        loss = proxy_loss(wam, imgs_ft, msg_ft)
        loss.backward()
        g = embed_weight.grad.detach().reshape(-1)

        if projector is not None:
            g = projector.project(g.unsqueeze(0)).squeeze(0)

        g_norm = g.norm()
        if g_norm > 1e-12:
            direction = (g / g_norm).view_as(embed_weight)
            with torch.no_grad():
                embed_weight -= step_size * direction

        if step % 10 == 0 or step == n_steps - 1:
            log(f"  [{name}] pas {step:2d}/{n_steps}: proxy_loss={loss.item():.6f}, "
                f"||grad||={g_norm.item():.4e}, t={time.time() - t0:.1f}s")

    bit_acc_after = eval_bit_acc(wam, imgs_eval, msg_eval, nbits)
    with torch.no_grad():
        proxy_after = proxy_loss(wam, imgs_ft, msg_ft).item()

    log(f"[{name}] apres fine-tune: bit_acc={bit_acc_after:.4f}, "
        f"proxy_loss={proxy_after:.6f}")

    with torch.no_grad():
        embed_weight.data.copy_(w0)

    return {
        "condition": name,
        "bit_acc_before": bit_acc_before,
        "bit_acc_after": bit_acc_after,
        "bit_acc_delta": bit_acc_after - bit_acc_before,
        "proxy_loss_before": proxy_before,
        "proxy_loss_after": proxy_after,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str,
                     default="/workspace/runs/runpod_256bits_phase0_puredecode/checkpoint.pth")
    ap.add_argument("--val-dir", type=str, default="corpus-colab/val")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-hessian-images", type=int, default=12)
    ap.add_argument("--n-ft-images", type=int, default=8)
    ap.add_argument("--n-eval-images", type=int, default=8)
    ap.add_argument("--k", type=int, default=16,
                     help="dimension du sous-espace (doit etre << 2*nbits*hidden_size)")
    ap.add_argument("--lanczos-n-iter", type=int, default=24)
    ap.add_argument("--lanczos-n-samples", type=int, default=1,
                     help="batch fixe -> pas de stochasticite a moyenner, 1 suffit")
    ap.add_argument("--lora-steps", type=int, default=30)
    ap.add_argument("--lora-step-size", type=float, default=0.05,
                     help="deplacement L2 par pas (descente de gradient normalisee)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16,
                     help="threads CPU intra-op -- laisse de la marge au job GPU/dataloaders")
    ap.add_argument("--cache-path", type=str, default="runs/sse_hessian_projector_phase0.pt")
    ap.add_argument("--results-path", type=str, default="runs/sse_hessian_experiment_results.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    log(f"demarrage. checkpoint={args.checkpoint}")
    log(f"CPU uniquement, threads={args.threads}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

    patch_grn_for_double_backward()

    ckpt_epoch = load_checkpoint_with_retry(args.checkpoint).get("epoch")
    log(f"checkpoint epoch={ckpt_epoch}")

    t0 = time.time()
    wam, cfg = build_wam(args.checkpoint)
    nbits = int(cfg.args.nbits)
    log(f"Wam charge en {time.time() - t0:.1f}s (nbits={nbits})")

    embed_weight = get_msg_embedding_weight(wam)
    d = embed_weight.numel()
    log(f"table d'embedding attaquee: shape={tuple(embed_weight.shape)}, d={d}")
    if args.k >= d:
        raise ValueError(f"k={args.k} doit etre < d={d}")

    # gel de tout sauf la table d'embedding : c'est la seule surface qu'on
    # attaque/analyse ici.
    for p in wam.parameters():
        p.requires_grad_(False)

    n_total = args.n_hessian_images + args.n_ft_images + args.n_eval_images
    log(f"chargement de {n_total} images depuis {args.val_dir}")
    imgs_all = load_val_images(args.val_dir, args.img_size, n_total, args.seed)
    imgs_hess = imgs_all[:args.n_hessian_images]
    imgs_ft = imgs_all[args.n_hessian_images:args.n_hessian_images + args.n_ft_images]
    imgs_eval = imgs_all[args.n_hessian_images + args.n_ft_images:]

    # message FIXE, meme convention que get_random_msg (torch.randint(0,2,...))
    torch.manual_seed(args.seed)
    fixed_msg = torch.randint(0, 2, (1, nbits))
    msg_hess = fixed_msg.repeat(imgs_hess.shape[0], 1)
    msg_ft = fixed_msg.repeat(imgs_ft.shape[0], 1)
    msg_eval = fixed_msg.repeat(imgs_eval.shape[0], 1)

    # ---- sanity check: bit_acc de depart (avant toute attaque) ----
    embed_weight.requires_grad_(False)
    baseline_acc = eval_bit_acc(wam, imgs_eval, msg_eval, nbits)
    log(f"bit_acc de depart (checkpoint intact, avant experience) = {baseline_acc:.4f}")

    # ---- projecteur Hessien (calcule + cache sur disque) ----
    w0 = embed_weight.detach().clone()
    embed_weight.requires_grad_(True)
    hess_proj = build_hessian_projector(
        wam, embed_weight, imgs_hess, msg_hess, nbits,
        k=args.k, n_iter=args.lanczos_n_iter, n_samples=args.lanczos_n_samples,
        seed=args.seed, cache_path=args.cache_path,
    )
    embed_weight.requires_grad_(False)
    with torch.no_grad():
        embed_weight.data.copy_(w0)  # Lanczos ne doit pas avoir bouge les poids, sanity restore

    ckpt = torch.load(args.cache_path, map_location="cpu")
    if "vals" in ckpt:
        top_vals = ckpt["vals"]
    else:
        # recompute non stocke dans le cache (build_projector_from_loss ne
        # sauve que basis) -> on les reprojette depuis basis pour affichage
        top_vals = None
    log(f"projecteur Hessien: U shape={tuple(hess_proj.U.shape)}")

    # ---- projecteur aleatoire (= ce que CipherMark fait reellement) ----
    random_proj = random_stable_projector(d=d, k=args.k, seed=args.seed + 777)

    # ---- 3 conditions ----
    results = []
    conditions = [
        ("aucune_projection_raw_finetune", None),
        ("projection_aleatoire_actuel_ciphermark", random_proj),
        ("projection_hessien_sse_jamais_teste", hess_proj),
    ]
    for name, proj in conditions:
        log(f"=== condition: {name} ===")
        res = run_condition(
            name, wam, embed_weight, w0, imgs_ft, msg_ft, imgs_eval, msg_eval,
            nbits, proj, n_steps=args.lora_steps, step_size=args.lora_step_size,
        )
        results.append(res)

    # ---- rapport ----
    log("=" * 78)
    log(f"{'condition':38s} {'acc avant':>10s} {'acc apres':>10s} {'delta':>10s}")
    for r in results:
        log(f"{r['condition']:38s} {r['bit_acc_before']:10.4f} "
            f"{r['bit_acc_after']:10.4f} {r['bit_acc_delta']:+10.4f}")
    log("=" * 78)

    out = {
        "__doc__": (
            "Resultats de l'experience SSE-Hessien vs projection aleatoire "
            "vs pas de projection, sur checkpoint reel. Genere par "
            "scripts/ciphermark/sse_hessian_experiment.py. Lecture seule "
            "recommandee, ne pas editer a la main."
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt_epoch,
        "nbits": nbits,
        "embedding_shape": list(embed_weight.shape),
        "d": d,
        "k": args.k,
        "lanczos_n_iter": args.lanczos_n_iter,
        "lanczos_n_samples": args.lanczos_n_samples,
        "lora_steps": args.lora_steps,
        "lora_step_size": args.lora_step_size,
        "n_hessian_images": args.n_hessian_images,
        "n_ft_images": args.n_ft_images,
        "n_eval_images": args.n_eval_images,
        "seed": args.seed,
        "baseline_bit_acc_before_any_attack": baseline_acc,
        "hessian_top_eigenvalues": top_vals.tolist() if top_vals is not None else None,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"resultats sauves dans {args.results_path}")
    log("termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

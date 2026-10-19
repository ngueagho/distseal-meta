"""Le recadrage casse-t-il le canal, ou seulement l'alignement ?

detect() reetire le morceau recadre a 256 px : le motif est agrandi et decale.
On compare cette lecture a une lecture qui replace le morceau a sa taille
d'origine sur un canevas de 256 px, a la bonne position (oracle) ou a la
meilleure position d'une grille (recherche aveugle, seuil corrige en p/K).
"""
import json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEPOT = os.environ.get("DEPOT", "/home/roberto/Documents/M2/dr haminou/code-memoire")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, DEPOT)
sys.path.insert(0, os.path.join(DEPOT, "scripts", "ciphermark"))
from distseal.utils.cfg import setup_model_from_checkpoint, get_config_from_checkpoint  # noqa
from eval_recovery_robustness import replay_scaling  # noqa

CKPT = os.environ.get("CKPT", os.path.join(DEPOT, "runs/ciphermark_64bits_checkpoint.pth"))
CORPUS = os.environ.get("CORPUS", os.path.join(DEPOT, "corpus-coco-5000", "toutes"))
N, LOT, SEED, PAS = int(os.environ.get("N", 100)), 10, 0, int(os.environ.get("PAS", 4))
SEUIL = 1e-6
# tailles ou faire la recherche aveugle ("" = etape rapide seulement)
GRILLE = [float(x) for x in os.environ.get("GRILLE", "").split(",") if x]
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() - 1))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def queue_binomiale(n):
    """p(d) = P(au plus d erreurs sur n bits | hasard)."""
    c = [math.comb(n, k) for k in range(n + 1)]
    cum, out = 0, []
    for d in range(n + 1):
        cum += c[d]
        out.append(cum / 2 ** n)
    return out


wam = setup_model_from_checkpoint(CKPT)
cfg = get_config_from_checkpoint(CKPT)
replay_scaling(wam, cfg, CKPT)
wam = wam.to(DEVICE).eval()
NB, T = int(cfg.args.nbits), int(cfg.args.img_size)
KW = {"is_video": False} if type(wam).__name__ == "VideoWam" else {}
P = queue_binomiale(NB)
d_max = lambda seuil: max(d for d in range(NB + 1) if P[d] < seuil)  # noqa


@torch.no_grad()
def lire(x):
    pr = wam.detect(x.to(DEVICE), **KW)["preds"][:, 1:1 + NB]
    if pr.dim() > 2:
        pr = pr.mean(dim=tuple(range(2, pr.dim())))
    return (pr > 0).to(torch.uint8).cpu()


def canevas(morceau, i, j):
    """Replace le morceau a sa taille d'origine ; fond = couleur moyenne."""
    c = morceau.mean(dim=(1, 2), keepdim=True).expand(3, T, T).clone()
    h, w = morceau.shape[-2:]
    c[:, i:i + h, j:j + w] = morceau
    return c


def positions(c):
    ax = sorted(set(list(range(0, T - c + 1, PAS)) + [T - c]))
    return [(i, j) for i in ax for j in ax]


fichiers = sorted(f for f in os.listdir(CORPUS) if f.lower().endswith((".jpg", ".png")))
random.Random(SEED).shuffle(fichiers)
fichiers = fichiers[:N]

TAILLES = (0.9, 0.7, 0.5)
res = {s: {k: [] for k in ("reetire", "centre", "oracle", "oracle+2", "oracle+4", "oracle+8",
                            "grille", "grille_vierge")} for s in TAILLES}
res_sans = []
K_par_taille = {}
t0 = time.time()

for b in range(0, N, LOT):
    ims = torch.stack([torch.from_numpy(np.asarray(
        Image.open(os.path.join(CORPUS, f)).convert("RGB").resize((T, T)),
        dtype=np.float32) / 255.).permute(2, 0, 1) for f in fichiers[b:b + LOT]])
    msgs = torch.randint(0, 2, (len(ims), NB)).float()
    with torch.no_grad():
        imw = wam.embed(ims.to(DEVICE), msgs=msgs.to(DEVICE), **KW)["imgs_w"].clamp(0, 1).cpu()
    m8 = msgs.to(torch.uint8)
    res_sans += (lire(imw) == m8).float().mean(1).tolist()

    for s in TAILLES:
        c = int(s * T)
        grille = positions(c)
        K_par_taille[s] = len(grille)
        for k in range(len(ims)):
            i, j = random.randint(0, T - c), random.randint(0, T - c)
            mo = imw[k, :, i:i + c, j:j + c]
            vi = ims[k, :, i:i + c, j:j + c]        # meme morceau, image NON tatouee
            ref = m8[k:k + 1]

            x = F.interpolate(mo[None], size=(T, T), mode="bilinear",
                              align_corners=False, antialias=True)
            res[s]["reetire"].append(int((lire(x) != ref).sum()))
            for nom, dlt in (("oracle", 0), ("oracle+2", 2), ("oracle+4", 4), ("oracle+8", 8)):
                ii, jj = min(i + dlt, T - c), min(j + dlt, T - c)
                res[s][nom].append(int((lire(canevas(mo, ii, jj)[None]) != ref).sum()))
            x = canevas(mo, (T - c) // 2, (T - c) // 2)[None]
            res[s]["centre"].append(int((lire(x) != ref).sum()))
            if s not in GRILLE:
                continue
            for nom, src in (("grille", mo), ("grille_vierge", vi)):
                lot = torch.stack([canevas(src, a, bb) for a, bb in grille])
                d = (lire(lot) != ref).sum(1)
                res[s][nom].append(int(d.min()))
    log(f"{b + len(ims)}/{N} images  ({time.time() - t0:.0f} s)")

print()
print("=" * 92)
print(f"Resynchronisation du recadrage -- {N} images COCO, Omega {NB} bits, pas de grille {PAS} px")
print(f"sans attaque : bit_acc {np.mean(res_sans):.4f}")
print("seuil : p < 1e-6 (soit au plus %d erreurs) ; grille : p*K < 1e-6" % d_max(SEUIL))
print("=" * 92)
print(f"{'recadrage':<11}{'lecture':<15}{'bit_acc':>9}{'err. med':>10}{'verdicts':>10}{'seuil d<=':>11}")
print("-" * 92)
sortie = {"n": N, "nbits": NB, "pas": PAS, "bit_acc_sans_attaque": float(np.mean(res_sans)),
          "conditions": {}}
for s in TAILLES:
    K = K_par_taille[s]
    sortie["conditions"][str(s)] = {"K_grille": K}
    for nom, ds in res[s].items():
        if not ds:
            continue
        ds = np.array(ds)
        dm = d_max(SEUIL / K) if nom.startswith("grille") else d_max(SEUIL)
        v = float((ds <= dm).mean())
        print(f"{s:<11}{nom:<15}{1 - ds.mean() / NB:>9.4f}{np.median(ds):>10.0f}{v:>9.0%}{dm:>11}"
              + (f"   (K={K})" if nom.startswith("grille") else ""))
        sortie["conditions"][str(s)][nom] = {"bit_acc": float(1 - ds.mean() / NB),
                                             "erreurs_mediane": float(np.median(ds)),
                                             "taux_verdict": v, "d_max": dm}
    print("-" * 92)
print("grille_vierge = meme recherche sur l'image NON tatouee : tout verdict y est un faux positif")
suffixe = f"_grille{'-'.join(map(str, GRILLE))}_pas{PAS}" if GRILLE else ""
out = os.path.join(DEPOT, "runs", f"diag_resync_recadrage_N{N}{suffixe}.json")
json.dump(sortie, open(out, "w"), indent=2)
log(f"ecrit : {out}")

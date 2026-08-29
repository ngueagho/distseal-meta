"""CipherMark -- l'attribution par utilisateur devient-elle une preuve ?

CE QUE CE SCRIPT DEMONTRE
-------------------------
L'argument central face a WOUAF : leur code utilisateur est arbitraire, donc
forgeable par quiconque comprend le mecanisme. Chez nous la cle de chaque
utilisateur est DERIVEE de la cle maitresse du fournisseur :

    k_user = HKDF-Expand(k_master, info = etiquette || user_id)
    Omega  = HMAC(k_user, h) XOR PRG(s_master, nonce)

et le registre associe nonce -> user_id. Quatre proprietes a etablir, chacune
avec son taux mesure :

  1. ATTRIBUTION : l'image d'un utilisateur est attribuee a LUI, au niveau de
     bruit reellement mesure sur les chaines validees (2/64 en diffusion,
     8/64 en autoregressif) ;
  2. SEPARATION : la cle d'un AUTRE utilisateur ne valide jamais l'image --
     c'est elle qui rend la forge impossible sans k_master ;
  3. NON-TATOUE : une image sans temoin n'est attribuee a personne ;
  4. IDENTIFICATION AVEUGLE : sans connaitre le nonce a l'avance, le verifieur
     retrouve l'entree du registre par le hash de reference, puis rederive la
     cle du user enregistre -- le protocole complet, de l'image au nom.

Le canal est simule par bruit binaire symetrique AUX TAUX MESURES sur les
vraies chaines -- ce script ne genere pas d'images, il etablit les proprietes
cryptographiques du protocole au-dessus des canaux deja valides.

Identites synthetiques uniquement (createur-0042) : ces sorties finissent dans
le memoire.

Lancement (CPU seul) :
    PYTHONPATH=. python3 scripts/ciphermark/eval_attribution_utilisateur.py \
        --users 100 --images-par-user 50
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from distseal.ciphermark import crypto
from distseal.ciphermark.witness import WitnessField, WitnessConfig
from distseal.ciphermark.equation import binomial_pvalue
from distseal.ciphermark.registry import TraceRegistry


def log(m): print(f"[attribution] {time.strftime('%H:%M:%S')} {m}", flush=True)


def canal(bits: np.ndarray, ber: float, rng: np.random.Generator) -> np.ndarray:
    """Canal binaire symetrique au taux d'erreur mesure sur la vraie chaine."""
    if ber <= 0:
        return bits.copy()
    flips = rng.random(bits.shape) < ber
    return (bits ^ flips.astype(np.uint8)).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--images-par-user", type=int, default=50)
    ap.add_argument("--nbits", type=int, default=64)
    ap.add_argument("--seuil-p", type=float, default=1e-6)
    # 0.031 = 2/64, la mediane mesuree de la chaine diffusion validee ;
    # 0.125 = 8/64, la mediane de la chaine autoregressive ;
    # 0.20 et 0.28 : marges de stress au-dela du mesure.
    ap.add_argument("--bers", type=float, nargs="+",
                    default=[0.0, 0.031, 0.125, 0.20, 0.28])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval_attribution.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = WitnessConfig(n_bits=args.nbits)
    k_master = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
    s_master = bytes(rng.integers(0, 256, 32, dtype=np.uint8))

    users = [f"createur-{i:04d}" for i in range(args.users)]
    wfs = {u: WitnessField.for_user(s_master=s_master, k_master=k_master,
                                    user_id=u, cfg=cfg) for u in users}
    log(f"{args.users} utilisateurs, cles derivees d'une seule cle maitresse")

    # ------------------------------------------------ emission + registre ---
    reg = TraceRegistry()          # :memory:
    N = args.users * args.images_par_user
    h_tous = rng.integers(0, 256, (N, 32), dtype=np.uint8)
    omegas = np.empty((N, args.nbits), dtype=np.uint8)
    proprietaire = []
    h_index = {}                   # h_ref -> nonce : l'identification aveugle
    for i in range(N):
        u = users[i % args.users]
        h = bytes(h_tous[i])
        omegas[i] = wfs[u].build_omega(h, image_id=i)
        reg.put(nonce=i, parity=b"", n_bits=args.nbits, rs_nsym=0,
                user=u, h_ref=h, session="eval-attribution")
        h_index[h] = i
        proprietaire.append(u)
    log(f"{N} images emises et enregistrees (nonce -> user + h_ref)")

    # tags attendus par (image, cle candidate) -- precomputes par cle
    log("precalcul des tags HMAC pour chaque cle candidate...")
    tags = np.empty((args.users, N, args.nbits), dtype=np.uint8)
    for j, u in enumerate(users):
        wf = wfs[u]
        for i in range(N):
            tags[j, i] = wf.expected_tag_bits(bytes(h_tous[i]))
    keystreams = np.empty((N, args.nbits), dtype=np.uint8)
    for i in range(N):
        keystreams[i] = wfs[users[0]].keystream_bits(i)   # s_master commun

    resultats = {}
    for ber in args.bers:
        rng_c = np.random.default_rng(args.seed + int(ber * 1e4) + 1)
        obs = canal(omegas, ber, rng_c)
        # cote verifieur : Omega_obs XOR keystream, compare au tag de CHAQUE cle
        detag = obs ^ keystreams                       # (N, nbits)
        d = (tags != detag[None, :, :]).sum(axis=2)    # (users, N)

        vrai_j = np.array([j % args.users for j in range(N)])
        d_vrai = d[vrai_j, np.arange(N)]
        d_sans_vrai = d.copy()
        d_sans_vrai[vrai_j, np.arange(N)] = args.nbits + 1
        d_meilleur_autre = d_sans_vrai.min(axis=0)

        # attribution : meilleur candidat sous le seuil
        j_min = d.argmin(axis=0)
        d_min = d.min(axis=0)
        pv = np.array([binomial_pvalue(int(x), args.nbits) for x in d_min])
        attribue = pv < args.seuil_p
        correct = attribue & (j_min == vrai_j)
        faux = attribue & (j_min != vrai_j)

        # image non tatouee : Omega aleatoire, personne ne doit valider
        alea = rng_c.integers(0, 2, (N, args.nbits), dtype=np.uint8)
        d_alea = (tags != (alea ^ keystreams)[None, :, :]).sum(axis=2).min(axis=0)
        pv_alea = np.array([binomial_pvalue(int(x), args.nbits) for x in d_alea])

        resultats[f"ber_{ber:.3f}"] = {
            "attribution_correcte": float(correct.mean()),
            "fausse_attribution": float(faux.mean()),
            "non_attribue": float((~attribue).mean()),
            "d_vraie_cle": {"mediane": float(np.median(d_vrai)),
                            "p95": float(np.percentile(d_vrai, 95))},
            "d_meilleure_autre_cle": {"mediane": float(np.median(d_meilleur_autre)),
                                      "min": int(d_meilleur_autre.min())},
            "marge_min": int((d_meilleur_autre - d_vrai).min()),
            "non_tatoue_attribue_a_tort": float((pv_alea < args.seuil_p).mean()),
        }
        r = resultats[f"ber_{ber:.3f}"]
        log(f"BER {ber:.3f} : attribution {100*r['attribution_correcte']:6.2f} %  "
            f"fausse {100*r['fausse_attribution']:.4f} %  "
            f"marge min {r['marge_min']:+d} bits")

    # --------------------------------- identification aveugle, protocole ----
    # Sans nonce fourni : h_obs -> registre -> user enregistre -> rederivation.
    ok_aveugle = 0
    for i in rng.choice(N, size=min(500, N), replace=False):
        h = bytes(h_tous[i])
        nonce = h_index.get(h)                    # recherche par hash
        if nonce is None:
            continue
        u_reg = reg.user_for(nonce)
        k = WitnessField.derive_user_key(k_master, u_reg, cfg)
        wf = WitnessField(s_master=s_master, k_secret=k, cfg=cfg)
        d = int((wf.build_omega(h, nonce) != omegas[i]).sum())
        if d == 0 and u_reg == proprietaire[i]:
            ok_aveugle += 1
    n_teste = min(500, N)
    log(f"identification aveugle (h -> registre -> rederivation) : "
        f"{ok_aveugle}/{n_teste}")

    bilan = {"users": args.users, "images": N, "nbits": args.nbits,
             "seuil_p": args.seuil_p, "canaux": resultats,
             "identification_aveugle": f"{ok_aveugle}/{n_teste}"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bilan, f, indent=2)

    print()
    print("=" * 86)
    print(f"ATTRIBUTION PAR UTILISATEUR -- {args.users} createurs, "
          f"{N} images, {args.nbits} bits, seuil p < {args.seuil_p:g}")
    print("=" * 86)
    print(f"  {'canal':<26}{'attribuee au bon':>18}{'au mauvais':>12}"
          f"{'refusee':>9}{'marge min':>11}")
    print(f"  {'-'*80}")
    noms = {0.0: "parfait", 0.031: "diffusion (mesure)",
            0.125: "autoregressif (mesure)", 0.20: "stress 20 %",
            0.28: "stress 28 %"}
    for ber in args.bers:
        r = resultats[f"ber_{ber:.3f}"]
        print(f"  {noms.get(ber, f'BER {ber}'):<26}"
              f"{100*r['attribution_correcte']:>17.2f} %"
              f"{100*r['fausse_attribution']:>11.4f} %"
              f"{100*r['non_attribue']:>8.2f} %"
              f"{r['marge_min']:>+10d} b")
    print(f"\n  non-tatouees attribuees a tort (pire canal) : "
          f"{100*max(r['non_tatoue_attribue_a_tort'] for r in resultats.values()):.4f} %")
    print(f"  identification aveugle par le registre       : {ok_aveugle}/{n_teste}")
    print("=" * 86)
    log(f"resultats ecrits dans {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
CipherMark -- validation de la couche cryptographique a grande echelle.

Pourquoi ce script
------------------
Les proprietes cryptographiques de CipherMark (unicite d'Omega, avalanche,
uniformite, attribution par identite) n'avaient ete verifiees que sur une
poignee de tirages -- 5 images pour la chaine complete, quelques dizaines pour
le reste. Un comportement correct sur 5 tirages ne dit rien : un biais de 1 %
sur les bits d'Omega, ou une collision tous les 10 000 nonces, y sont
strictement invisibles.

Cette couche est la seule du systeme qui ne coute rien a tester en grand : elle
n'appelle ni reseau de neurones ni GPU. Il n'y a donc aucune raison de la
mesurer sur un petit echantillon. Ce script la pousse a 100 000 tirages par
defaut, sur CPU, en quelques minutes.

Ce qui est mesure
-----------------
  1. unicite      -- N generations, aucune collision d'Omega ni de nonce
  2. separation   -- distance de Hamming entre Omega de generations distinctes
  3. avalanche    -- 1 bit change dans h, dans le nonce, dans la cle
  4. uniformite   -- frequence de 1 par position, chi2, monobit
  5. identite     -- N utilisateurs derives d'un identifiant : cles et Omega
                     distincts, rejet croise
  6. determinisme -- meme graine + meme nonce => meme keystream, N fois
  7. independance -- correlation entre les bits de h et ceux d'Omega

Lancement :
    PYTHONPATH=. python3 scripts/ciphermark/eval_crypto_grande_echelle.py --n 100000
"""
from __future__ import annotations
import argparse, json, os, secrets, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from distseal.ciphermark import crypto
from distseal.ciphermark.witness import WitnessField, WitnessConfig
from distseal.ciphermark.registry import TraceRegistry, NonceReuseError


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bits_matrice(wf, hs, ids):
    """Empile les Omega de (hash, nonce) successifs en une matrice N x n_bits."""
    return np.stack([wf.build_omega(h, i) for h, i in zip(hs, ids)]).astype(np.uint8)


def hamming_paires(M, echantillon, rng):
    """Distance de Hamming moyenne/min entre paires distinctes.

    Sur N = 100 000 il y a 5 milliards de paires : on tire un sous-echantillon.
    Le calcul passe par un produit matriciel sur l'encodage +-1, ou
    <a,b> = b_tot - 2d, donc d = (b_tot - <a,b>) / 2."""
    n, b_tot = M.shape
    if n > echantillon:
        M = M[rng.choice(n, echantillon, replace=False)]
    X = M.astype(np.float32) * 2 - 1
    G = (b_tot - X @ X.T) / 2
    np.fill_diagonal(G, -1.0)
    v = G[G >= 0]
    return float(v.mean()), int(v.min()), int(v.max()), M.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000,
                    help="nombre de generations (hash + nonce distincts)")
    ap.add_argument("--n-bits", type=int, default=64)
    ap.add_argument("--n-users", type=int, default=1000,
                    help="nombre d'identites derivees pour le test d'attribution")
    ap.add_argument("--echantillon-paires", type=int, default=4000,
                    help="taille du sous-echantillon pour les distances par paires")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval_crypto_grande_echelle.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    N, B = args.n, args.n_bits
    res = {"n": N, "n_bits": B, "prg_backend": crypto.prg_backend()}
    log(f"N={N} generations, Omega de {B} bits, PRG = {res['prg_backend']}")
    if res["prg_backend"] != "chacha20":
        log("ATTENTION : le PRG n'est pas ChaCha20. Les resultats ne valent que "
            "pour le repli utilise.")

    cfg = WitnessConfig(n_bits=B)
    wf = WitnessField(s_master=crypto.random_seed(32),
                      k_secret=crypto.random_seed(32), cfg=cfg)

    taille_hash = 32                       # hash perceptuel 256 bits
    hs = [bytes(rng.integers(0, 256, taille_hash, dtype=np.uint8)) for _ in range(N)]
    ids = list(range(1, N + 1))

    # ------------------------------------------------------- 1. unicite ----
    log("1/7 unicite d'Omega et des nonces")
    t0 = time.time()
    M = bits_matrice(wf, hs, ids)
    log(f"    {N} Omega construits en {time.time() - t0:.1f}s "
        f"({N / max(1e-9, time.time() - t0):.0f}/s)")
    paquets = {np.packbits(M[i]).tobytes() for i in range(N)}
    n_collisions = N - len(paquets)
    nonces = {crypto.make_nonce(i) for i in ids}
    res["unicite"] = {"n_omega_distincts": len(paquets), "collisions_omega": n_collisions,
                      "n_nonces_distincts": len(nonces),
                      "collisions_nonce": N - len(nonces)}
    # Reference : sur 64 bits, l'anniversaire donne ~N^2/2^{B+1} collisions
    # attendues meme pour une source parfaite. On l'affiche pour que zero
    # collision ne soit pas lu comme une preuve plus forte qu'il ne l'est.
    res["unicite"]["collisions_attendues_hasard"] = N * (N - 1) / 2 ** (B + 1)
    log(f"    collisions Omega : {n_collisions} "
        f"(attendu par hasard : {res['unicite']['collisions_attendues_hasard']:.2e})")

    # ---------------------------------------------------- 2. separation ----
    log("2/7 distance de Hamming entre generations distinctes")
    moy, mini, maxi, n_ech = hamming_paires(M, args.echantillon_paires, rng)
    res["separation"] = {"hamming_moyen": moy, "hamming_min": mini,
                         "hamming_max": maxi, "n_echantillon": n_ech,
                         "attendu": B / 2}
    log(f"    moyenne {moy:.2f}/{B} (attendu {B/2}), min {mini}, max {maxi}, "
        f"sur {n_ech} tirages")

    # ----------------------------------------------------- 3. avalanche ----
    log("3/7 avalanche : un seul bit change en entree")
    n_av = min(N, 20_000)
    av = {}

    #  a) un bit du hash
    d = []
    for k in range(n_av):
        h = bytearray(hs[k])
        bit = int(rng.integers(0, taille_hash * 8))
        h[bit // 8] ^= 1 << (7 - bit % 8)
        d.append(int((wf.build_omega(bytes(h), ids[k]) != M[k]).sum()))
    av["hash_1bit"] = d

    #  b) le nonce voisin, hash inchange
    d = [int((wf.build_omega(hs[k], ids[k] + 1) != M[k]).sum()) for k in range(n_av)]
    av["nonce_voisin"] = d

    #  c) un bit de la cle secrete
    d = []
    for k in range(min(n_av, 5000)):
        kb = bytearray(wf.k_secret)
        bit = int(rng.integers(0, 256))
        kb[bit // 8] ^= 1 << (7 - bit % 8)
        wf2 = WitnessField(s_master=wf.s_master, k_secret=bytes(kb), cfg=cfg)
        d.append(int((wf2.build_omega(hs[k], ids[k]) != M[k]).sum()))
    av["cle_1bit"] = d

    res["avalanche"] = {}
    for nom, d in av.items():
        d = np.asarray(d)
        res["avalanche"][nom] = {"n": len(d), "moyenne": float(d.mean()),
                                 "ecart_type": float(d.std()),
                                 "min": int(d.min()), "max": int(d.max()),
                                 "attendu_moyenne": B / 2,
                                 "attendu_ecart_type": float(np.sqrt(B) / 2)}
        log(f"    {nom:<14} {d.mean():.2f} +- {d.std():.2f} bits changes "
            f"(attendu {B/2} +- {np.sqrt(B)/2:.2f})")

    # ---------------------------------------------------- 4. uniformite ----
    log("4/7 uniformite des bits d'Omega")
    freq = M.mean(axis=0)                       # frequence de 1 par position
    total_uns = int(M.sum())
    total_bits = M.size
    # chi2 d'ajustement a la loi uniforme, position par position
    obs1 = M.sum(axis=0).astype(np.float64)
    obs0 = N - obs1
    chi2 = float((((obs1 - N / 2) ** 2) / (N / 2) + ((obs0 - N / 2) ** 2) / (N / 2)).sum())
    # monobit NIST : |S_n| / sqrt(n) doit rester petit
    s_n = abs(2.0 * total_uns - total_bits) / np.sqrt(total_bits)
    res["uniformite"] = {
        "frequence_min": float(freq.min()), "frequence_max": float(freq.max()),
        "frequence_moyenne": float(freq.mean()),
        "chi2": chi2, "ddl": B, "chi2_par_ddl": chi2 / B,
        "monobit_s_n": float(s_n),
        "monobit_p_valeur": float(np.math.erfc(s_n / np.sqrt(2))) if hasattr(np, "math")
                            else float(__import__("math").erfc(s_n / np.sqrt(2)))}
    log(f"    frequence de 1 : [{freq.min():.4f}, {freq.max():.4f}] "
        f"moyenne {freq.mean():.5f}")
    log(f"    chi2 = {chi2:.1f} pour {B} ddl ({chi2/B:.2f} par ddl), "
        f"monobit p = {res['uniformite']['monobit_p_valeur']:.3f}")

    # ------------------------------------------------------ 5. identite ----
    log(f"5/7 attribution : {args.n_users} identites derivees d'un identifiant")
    k_master = crypto.random_seed(32)
    users = [f"utilisateur-{i:06d}@upb.cm" for i in range(args.n_users)]
    cles = [WitnessField.derive_user_key(k_master, u, cfg) for u in users]
    res["identite"] = {"n_users": args.n_users,
                       "cles_distinctes": len(set(cles)),
                       "collisions_cles": args.n_users - len(set(cles))}
    # Meme image, meme nonce, identites differentes : les Omega doivent differer
    h0, id0 = hs[0], ids[0]
    wfs = [WitnessField(s_master=wf.s_master, k_secret=c, cfg=cfg) for c in cles]
    Mu = np.stack([w.build_omega(h0, id0) for w in wfs]).astype(np.uint8)
    moy_u, min_u, max_u, n_u = hamming_paires(Mu, args.echantillon_paires, rng)
    res["identite"].update({"omega_distincts": len({np.packbits(r).tobytes() for r in Mu}),
                            "hamming_moyen": moy_u, "hamming_min": min_u,
                            "hamming_max": max_u})
    log(f"    cles distinctes : {res['identite']['cles_distinctes']}/{args.n_users}, "
        f"Omega distincts : {res['identite']['omega_distincts']}/{args.n_users}")
    log(f"    Hamming entre identites : {moy_u:.2f}/{B} (min {min_u})")

    # Rejet croise : l'identite j verifie-t-elle le tatouage de l'identite i ?
    # Sans bruit de canal, le seul cas AUTHENTIC possible serait une collision.
    seuil = int(round(0.25 * B))           # seuil de decision usuel du verifieur
    n_croise = min(args.n_users, 500)
    faux = 0
    for i in range(n_croise):
        j = (i + 1) % n_croise
        faux += int((Mu[i] != Mu[j]).sum() <= seuil)
    res["identite"].update({"n_croises": n_croise, "faux_acceptes": faux,
                            "seuil_bits": seuil})
    log(f"    rejet croise : {faux}/{n_croise} faussement acceptes "
        f"(seuil {seuil} bits)")

    # -------------------------------------------------- 6. determinisme ----
    log("6/7 determinisme du PRG et du HMAC")
    n_det = min(N, 20_000)
    identiques = sum(1 for k in range(n_det)
                     if np.array_equal(wf.build_omega(hs[k], ids[k]), M[k]))
    res["determinisme"] = {"n": n_det, "reproductions_exactes": identiques}
    log(f"    {identiques}/{n_det} reconstructions bit a bit identiques")

    # ------------------------------------------------- 7. independance ----
    log("7/7 independance entre les bits du hash et ceux d'Omega")
    n_ind = min(N, 20_000)
    H = np.unpackbits(np.frombuffer(b"".join(hs[:n_ind]), dtype=np.uint8)) \
          .reshape(n_ind, taille_hash * 8)[:, :B].astype(np.float64)
    O = M[:n_ind].astype(np.float64)
    Hc = H - H.mean(0); Oc = O - O.mean(0)
    den = (np.sqrt((Hc ** 2).sum(0))[:, None] * np.sqrt((Oc ** 2).sum(0))[None, :])
    corr = np.abs((Hc.T @ Oc) / np.maximum(den, 1e-12))
    res["independance"] = {"n": n_ind, "correlation_max": float(corr.max()),
                           "correlation_moyenne": float(corr.mean()),
                           "seuil_bruit_2sigma": float(2 / np.sqrt(n_ind))}
    log(f"    |correlation| max {corr.max():.4f}, moyenne {corr.mean():.4f} "
        f"(bruit a 2 sigma : {2/np.sqrt(n_ind):.4f})")

    # ---------------------------------------------------- 8. registre ------
    log("bonus : rejeu de nonce dans le registre")
    reg = TraceRegistry()
    n_reg = min(N, 20_000)
    for k in range(n_reg):
        reg.put(nonce=ids[k], parity=b"", n_bits=B, rs_nsym=0, h_ref=hs[k])
    try:
        reg.put(nonce=ids[0], parity=b"", n_bits=B, rs_nsym=0, h_ref=hs[0])
        rejeu_detecte = False
    except NonceReuseError:
        rejeu_detecte = True
    # Le hash de reference doit ressortir intact : c'est lui que la
    # verification par distance de Hamming compare au hash observe.
    h_ref_ok = sum(1 for k in range(min(n_reg, 5000))
                   if reg.h_ref_for(ids[k]) == hs[k])
    res["registre"] = {"n_entrees": len(reg), "rejeu_detecte": rejeu_detecte,
                       "h_ref_relus_intacts": h_ref_ok,
                       "h_ref_verifies": min(n_reg, 5000)}
    log(f"    {len(reg)} entrees, rejeu detecte : {rejeu_detecte}, "
        f"h_ref relus intacts : {h_ref_ok}/{min(n_reg, 5000)}")
    reg.close()

    # --------------------------------------------------------- verdict -----
    echecs = []
    if res["unicite"]["collisions_omega"] > 5 * max(1.0, res["unicite"]["collisions_attendues_hasard"]):
        echecs.append("collisions d'Omega au-dela du hasard")
    if abs(res["separation"]["hamming_moyen"] - B / 2) > 0.5:
        echecs.append("separation moyenne eloignee de n/2")
    for nom, a in res["avalanche"].items():
        if abs(a["moyenne"] - B / 2) > 0.5:
            echecs.append(f"avalanche {nom} hors tolerance")
    if res["uniformite"]["monobit_p_valeur"] < 0.01:
        echecs.append("test monobit rejete")
    if res["identite"]["collisions_cles"] or res["identite"]["faux_acceptes"]:
        echecs.append("attribution par identite en defaut")
    if res["determinisme"]["reproductions_exactes"] != res["determinisme"]["n"]:
        echecs.append("PRG non deterministe")
    if res["independance"]["correlation_max"] > 10 * res["independance"]["seuil_bruit_2sigma"]:
        echecs.append("correlation hash/Omega au-dela du bruit")
    if not res["registre"]["rejeu_detecte"]:
        echecs.append("rejeu de nonce non detecte")
    if res["registre"]["h_ref_relus_intacts"] != res["registre"]["h_ref_verifies"]:
        echecs.append("hash de reference altere en base")
    res["echecs"] = echecs

    print()
    print("=" * 78)
    print(f"COUCHE CRYPTOGRAPHIQUE -- {N} generations, Omega de {B} bits")
    print("=" * 78)
    print(f"  collisions d'Omega            {res['unicite']['collisions_omega']}")
    print(f"  Hamming entre generations     {res['separation']['hamming_moyen']:.2f} / {B}")
    for nom, a in res["avalanche"].items():
        print(f"  avalanche {nom:<19} {a['moyenne']:.2f} +- {a['ecart_type']:.2f}")
    print(f"  frequence de 1                {res['uniformite']['frequence_moyenne']:.5f}")
    print(f"  monobit (p-valeur)            {res['uniformite']['monobit_p_valeur']:.3f}")
    print(f"  identites : Omega distincts   {res['identite']['omega_distincts']} / {args.n_users}")
    print(f"  rejet croise (faux acceptes)  {res['identite']['faux_acceptes']} / {res['identite']['n_croises']}")
    print(f"  correlation hash/Omega max    {res['independance']['correlation_max']:.4f}")
    print(f"  rejeu de nonce detecte        {res['registre']['rejeu_detecte']}")
    print(f"  h_ref relus intacts           {res['registre']['h_ref_relus_intacts']} / {res['registre']['h_ref_verifies']}")
    print("=" * 78)
    if echecs:
        print("VERDICT : " + str(len(echecs)) + " propriete(s) EN DEFAUT")
        for e in echecs:
            print("  - " + e)
    else:
        print(f"VERDICT : les {8} proprietes tiennent sur {N} tirages.")
    print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"resultats ecrits dans {args.out}")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())

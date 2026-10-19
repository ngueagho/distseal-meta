"""
Test de charge du registre de tracabilite, lance par la CI a chaque push.

Le memoire (chapitre 2, « Volume du registre ») affirme : 173 octets par
entree sur disque, consultation en O(log N) grace a l'index sur le nonce.
Ce script le remesure sur un vrai fichier SQLite, aux tailles N1 et N2, et
echoue si l'une des deux affirmations ne tient plus.

  - debit d'ecriture : put() un par un (un commit par entree), puis
    put_many() par lots (une transaction par lot) ;
  - consultation : get() sur des nonces tires au hasard, mediane et p99 ;
  - rejeu : un nonce deja vu doit lever NonceReuseError, meme sous charge ;
  - taille : octets du fichier / nombre d'entrees.

Les champs ont la taille reelle : parite 32 octets, h_ref 32 octets (256
bits), identifiant utilisateur et session comme dans la campagne.

Usage :
    PYTHONPATH=deps:. python3 tests/charge/charge_registre.py \
        --n 10000 100000 --out runs/charge_registre.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from distseal.ciphermark.registry import NonceReuseError, TraceRegistry  # noqa: E402

N_UNITAIRE = 300   # put() un par un : assez pour un debit, pas plus (lent)
LOT = 256


def mesure(n: int, n_lectures: int, rng: random.Random) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        chemin = os.path.join(tmp, "registre.sqlite")
        reg = TraceRegistry(chemin)
        # nonces non contigus : prefixe serveur sur 16 bits, compteur local
        nonces = [(7 << 48) | k for k in range(n)]

        # put() un par un (un commit par entree) sur les premieres entrees,
        # puis put_many() par lots de LOT (une transaction par lot)
        n_unit = min(n, N_UNITAIRE)
        t0 = time.perf_counter()
        for nonce in nonces[:n_unit]:
            reg.put(nonce=nonce, parity=rng.randbytes(32), n_bits=64, rs_nsym=32,
                    session="charge", h_ref=rng.randbytes(32), user="createur-0042")
        unitaire_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        reste = nonces[n_unit:]
        for k in range(0, len(reste), LOT):
            reg.put_many([(nonce, rng.randbytes(32), 64, 32, rng.randbytes(32), "createur-0042")
                          for nonce in reste[k:k + LOT]], session="charge")
        lots_s = time.perf_counter() - t0

        lectures = []
        for nonce in rng.sample(nonces, min(n_lectures, n)):
            t = time.perf_counter()
            entree = reg.get(nonce)
            lectures.append(time.perf_counter() - t)
            assert entree is not None
        lectures.sort()

        try:
            reg.put(nonce=nonces[n // 2], parity=bytes(32), n_bits=64, rs_nsym=32)
            rejeu_refuse = False
        except NonceReuseError:
            rejeu_refuse = True

        taille = os.path.getsize(chemin)
        return {
            "n": n,
            "ecritures_par_s_unitaire": round(n_unit / unitaire_s, 1),
            "ecritures_par_s_lots": round(len(reste) / lots_s, 1) if reste else None,
            "lecture_mediane_us": round(1e6 * statistics.median(lectures), 2),
            "lecture_p99_us": round(1e6 * lectures[int(0.99 * (len(lectures) - 1))], 2),
            "octets_par_entree": round(taille / n, 1),
            "rejeu_refuse": rejeu_refuse,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs=2, default=[10_000, 100_000])
    ap.add_argument("--lectures", type=int, default=5_000)
    ap.add_argument("--out", default="runs/charge_registre.json")
    args = ap.parse_args()

    rng = random.Random(0)
    res = [mesure(n, args.lectures, rng) for n in args.n]
    for r in res:
        print(f"N={r['n']:>7}  {r['ecritures_par_s_unitaire']:>7.0f} ecr/s unitaire, "
              f"{r['ecritures_par_s_lots'] or 0:>8.0f} par lots de {LOT}  "
              f"lecture {r['lecture_mediane_us']:.1f} us (p99 {r['lecture_p99_us']:.1f})  "
              f"{r['octets_par_entree']:.0f} o/entree  rejeu refuse={r['rejeu_refuse']}")

    petit, grand = res
    rapport_lecture = grand["lecture_mediane_us"] / petit["lecture_mediane_us"]
    echecs = []
    if not all(r["rejeu_refuse"] for r in res):
        echecs.append("un nonce reutilise a ete accepte")
    # O(log N) : passer de N1 a N2 = 10 N1 ne doit pas multiplier la lecture
    # par 10 (O(N)). Seuil large pour absorber le bruit des machines de CI.
    if rapport_lecture > 3.0:
        echecs.append(f"lecture x{rapport_lecture:.1f} pour N x{grand['n'] // petit['n']}")
    # memoire : 173 o/entree mesures sur le registre de la campagne
    if not 120 <= grand["octets_par_entree"] <= 250:
        echecs.append(f"{grand['octets_par_entree']} o/entree, hors [120, 250]")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"mesures": res, "rapport_lecture": round(rapport_lecture, 2),
                   "echecs": echecs}, f, indent=2)
    print(f"rapport lecture N2/N1 : x{rapport_lecture:.2f}  ->  {args.out}")
    for e in echecs:
        print("ECHEC :", e)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())

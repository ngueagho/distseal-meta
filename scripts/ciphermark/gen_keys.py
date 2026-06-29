"""
Genere une paire (s_master, k_secret) et l'ecrit sur le disque.

Usage:
    python -m scripts.crystal.gen_keys --out-dir ./keys

ATTENTION: ne pas committer les fichiers generes.
"""

import argparse
import os

from distseal.crystal.crypto import random_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    s_path = os.path.join(args.out_dir, "s_master.bin")
    k_path = os.path.join(args.out_dir, "k_secret.bin")

    for p in (s_path, k_path):
        if os.path.exists(p) and not args.force:
            print(f"refus: {p} existe deja (utiliser --force)")
            return 1

    with open(s_path, "wb") as f:
        f.write(random_seed(32))
    with open(k_path, "wb") as f:
        f.write(random_seed(32))

    # restreindre les droits (best-effort, ne marche pas sur certains FS)
    try:
        os.chmod(s_path, 0o600)
        os.chmod(k_path, 0o600)
    except Exception:
        pass

    print(f"OK -- ecrit {s_path} et {k_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

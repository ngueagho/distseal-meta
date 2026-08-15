#!/usr/bin/env bash
# CipherMark -- config 3/4 : run paye RunPod, budget de temps ~5h.
# A executer dans le terminal du pod (Community Cloud, A100 recommandee).
#
# Usage:
#   bash runpod/run_5h.sh
#
# Le "timeout 5h" arrete proprement le training a l'heure, quel que soit le
# nombre d'epoques atteint -- le dernier checkpoint sauvegarde (checkpoint.pth
# ou le dernier checkpointNNN.pth, cf. saveckpt_freq=10) est celui a recuperer.
set -euo pipefail

REPO_URL="https://github.com/ngueagho/distseal-meta.git"
REPO_DIR="code-memoire"
COCO_MAX="${COCO_MAX:-5000}"

if [ ! -d "$REPO_DIR" ]; then
    git clone -b ciphermark "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull

pip install -q -r requirements.txt

# Corpus : reseau datacenter RunPod, pas de souci de bande passante attendu
# (cf. l'echec du telechargement COCO en local).
if [ ! -d corpus-colab/train ]; then
    python -m scripts.ciphermark.build_corpus --out corpus-colab-raw \
        --coco unlabeled2017 --coco-max "$COCO_MAX" --kodak --bsds
    python - <<'EOF'
import os, shutil
src = "corpus-colab-raw"
files = sorted(os.listdir(src))
os.makedirs("corpus-colab/train", exist_ok=True)
os.makedirs("corpus-colab/val", exist_ok=True)
for i, f in enumerate(files):
    dst = "corpus-colab/val" if i < len(files) // 10 else "corpus-colab/train"
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
print("train:", len(os.listdir("corpus-colab/train")))
print("val:  ", len(os.listdir("corpus-colab/val")))
EOF
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Budget de temps : 5h pile, ajuster ici si vous payez pour plus/moins.
timeout 5h torchrun --nproc_per_node=1 --standalone train.py \
    --config configs/training/ciphermark/posthoc_pixel_256bits_runpod_5h.yaml \
    --output_dir /workspace/runs/runpod_256bits_5h

echo "Arret (timeout ou fin naturelle). Dernier checkpoint :"
ls -lat /workspace/runs/runpod_256bits_5h/checkpoint*.pth 2>/dev/null | head -5

# Push automatique vers Drive -- SEULEMENT si rclone est deja configure sur ce
# pod (fichier ~/.config/rclone/rclone.conf, mis en place une fois via
# `rclone config`, cf. runpod/README.md). Ce fichier n'est JAMAIS dans le repo
# git -- aucun identifiant Drive n'est commite. Si absent, on ne fait rien de
# plus que ce qui est deja affiche ci-dessus (recuperation manuelle via
# runpodctl reste possible).
if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^gdrive:"; then
    echo "rclone configure -- push vers gdrive:ciphermark/runs/runpod_256bits_5h/"
    rclone copy /workspace/runs/runpod_256bits_5h gdrive:ciphermark/runs/runpod_256bits_5h/ \
        --include "*.pth" --include "log.txt" --progress
else
    echo "rclone non configure sur ce pod -- push manuel necessaire (voir runpod/README.md)."
fi

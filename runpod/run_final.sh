#!/usr/bin/env bash
# CipherMark -- config 3 finale : depense le reste du budget RunPod (~10$) en
# un seul run le plus long possible, sur le pod SECURE RTX 4090 deja existant
# (21ah4heywiwjrx, 0.74$/h -> ~12h30 avant d'epuiser le budget avec marge).
#
# Usage (sur le pod, apres redemarrage) :
#   bash runpod/run_final.sh
set -euo pipefail

REPO_URL="https://github.com/ngueagho/distseal-meta.git"
REPO_DIR="code-memoire"
COCO_MAX="${COCO_MAX:-5000}"
# Marge de securite sous les ~13.5h theoriques (10$ / 0.74$/h) : laisse du
# temps pour le push Drive final et une marge sur le solde du compte.
BUDGET_HOURS="${BUDGET_HOURS:-12.5h}"

if [ ! -d "$REPO_DIR" ]; then
    git clone -b ciphermark "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull

pip install -q -r requirements.txt

# Corpus : reutilise celui deja construit sur le volume persistant si present.
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

echo "=== Calibration rapide (5 min, verifie vitesse + pas d'OOM/effondrement) ==="
timeout 5m torchrun --nproc_per_node=1 --standalone train.py \
    --config configs/training/ciphermark/posthoc_pixel_256bits_runpod_final.yaml \
    --output_dir /workspace/runs/runpod_256bits_final_calib \
    --epochs 5 --iter_per_epoch 20 || true
echo "=== Fin calibration -- verifier s/it, psnr, bit_acc ci-dessus avant de continuer ==="
echo ""

echo "=== Run final : budget ${BUDGET_HOURS} ==="
timeout "$BUDGET_HOURS" torchrun --nproc_per_node=1 --standalone train.py \
    --config configs/training/ciphermark/posthoc_pixel_256bits_runpod_final.yaml \
    --output_dir /workspace/runs/runpod_256bits_final

echo "Arret (timeout ou fin naturelle). Dernier checkpoint :"
ls -lat /workspace/runs/runpod_256bits_final/checkpoint*.pth 2>/dev/null | head -5

# Push automatique vers Drive -- rclone.conf est deja configure sur ce pod
# (~/.config/rclone/rclone.conf, jamais dans git) depuis la session precedente.
if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^gdrive:"; then
    echo "rclone configure -- push vers gdrive:ciphermark/runs/runpod_256bits_final/"
    rclone copy /workspace/runs/runpod_256bits_final gdrive:ciphermark/runs/runpod_256bits_final/ \
        --include "*.pth" --include "log.txt" --progress
else
    echo "rclone non configure sur ce pod -- push manuel necessaire (voir runpod/README.md)."
fi

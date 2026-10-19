#!/bin/bash
# Phase K -- test court : le recadrage fort est-il apprenable ?
#
# A lancer sur le pod (volume persistant monte sur /workspace) :
#   bash /workspace/test_recadrage.sh
#
# Le code du volume peut dater d'une campagne precedente : on repart donc d'un
# depot frais, tire de l'archive du Drive, sur lequel on pose les fichiers du
# test (configs + scripts non encore dans l'archive). Rien n'est pousse sur
# GitHub.
#
# Duree visee : ~2 h 30 d'entrainement (40 epoques) + ~15 min de mesures.
# Les produits partent sur le Drive au fil de l'eau (regle du depot).
set -uo pipefail

REMOTE=${REMOTE:-gdrive_local}
DRIVE=$REMOTE:ciphermark
DEST=$DRIVE/15-test-recadrage-phaseK
W=/workspace
D=$W/code-memoire-k
LOGS=$W/logs; mkdir -p "$LOGS" "$W/ckpt"
J=$LOGS/phaseK.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$J"; }

# --- 1. depot frais + fichiers du test --------------------------------------
if [ ! -d "$D/distseal/utils" ]; then
  rm -rf "$W/_k" && mkdir -p "$W/_k"
  rclone copy "$DRIVE/00-code-source-a-jour.tar.gz" "$W/_k/" || exit 1
  tar xzf "$W/_k/00-code-source-a-jour.tar.gz" -C "$W/_k" || exit 1
  R=$(dirname "$(find "$W/_k" -type d -path '*distseal/utils' | head -1)")
  mv "$(dirname "$R")" "$D"
fi
rclone copy "$DEST/fichiers_test.tar.gz" "$W/_k/" || exit 1
tar xzf "$W/_k/fichiers_test.tar.gz" -C "$D" || exit 1
cd "$D" || exit 1
log "depot pret : $D"

PF=$W/preflight.py; [ -f "$PF" ] || PF=runpod/preflight.py
python3 "$PF" >> "$J" 2>&1 || { log "preflight en echec"; exit 1; }
export PYTHONPATH=deps:. WANDB_MODE=offline

# --- 2. poids de depart et corpus d'entrainement ----------------------------
B=$W/ckpt/phaseB_64bits_robuste.pth
[ -f "$B" ] || rclone copyto "$DRIVE/03-tatoueur-robuste-aux-attaques-passives/checkpoint.pth" "$B"
REF=$W/ckpt/base_64bits.pth
[ -f "$REF" ] || rclone copyto "$DRIVE/01-tatoueur-modeles-de-reference-64-et-128-bits/phaseA2_64bits_stable_checkpoint.pth" "$REF"
if [ ! -d corpus-colab/train ]; then
  if [ -d "$W/code-memoire/corpus-colab/train" ]; then
    ln -s "$W/code-memoire/corpus-colab" corpus-colab
  else
    rclone copy "$DRIVE/11-archive-corpus-et-checkpoints-intermediaires/corpus-colab.tar.gz" "$W/_k/"
    tar xzf "$W/_k/corpus-colab.tar.gz" -C .
  fi
fi
log "train : $(ls corpus-colab/train | wc -l) images, val : $(ls corpus-colab/val | wc -l)"

# --- 3. l'entrainement, borne a 3 h -----------------------------------------
C=configs/training/ciphermark/phaseK_test_recadrage.yaml
OUT=$W/runs/phaseK_test_recadrage
log "entrainement : $C depuis $B"
timeout 3h torchrun --nproc_per_node=1 train.py --config "$C" --resume_from "$B" \
  >> "$J" 2>&1
log "entrainement termine (code $?)"
rclone copy "$OUT/log.txt" "$DEST/" ; rclone copy "$J" "$DEST/"
rclone copy "$OUT/checkpoint.pth" "$DEST/" || log "checkpoint non pousse"

# --- 4. la mesure qui tranche -----------------------------------------------
# Lecture reetiree ET a la position exacte, recadrages 0,9 / 0,7 / 0,5,
# 100 images COCO val2017 jamais vues. Le modele de reference est remesure
# dans les memes conditions, pour une comparaison a corpus identique.
if [ ! -d "$W/corpus-coco-5000/toutes" ]; then
  rclone copy "$DRIVE/13-colab-diagnostics-c2-c3/corpus-coco-5000.tar.gz" "$W/_k/"
  tar xzf "$W/_k/corpus-coco-5000.tar.gz" -C "$W"
fi
for nom in phaseK:"$OUT/checkpoint.pth" phaseB:"$B" reference:"$REF"; do
  etiquette=${nom%%:*}; ck=${nom#*:}
  CKPT=$ck DEPOT=$D CORPUS=$W/corpus-coco-5000/toutes N=100 \
    python3 scripts/ciphermark/diag_resync_recadrage.py > "$LOGS/oracle_$etiquette.log" 2>&1
  mv runs/diag_resync_recadrage_N100.json "runs/oracle_$etiquette.json" 2>/dev/null
  rclone copy "$LOGS/oracle_$etiquette.log" "$DEST/"
  rclone copy "runs/oracle_$etiquette.json" "$DEST/"
  log "mesure $etiquette faite"
done
log "fin du test : resultats dans $DEST"

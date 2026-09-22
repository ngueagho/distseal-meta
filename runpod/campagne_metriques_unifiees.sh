#!/usr/bin/env bash
# Recalcule les metriques WAF avec les definitions uniformisees du memoire.
#
# Lancement court de validation :
#   N=100 N_GEN=20 bash runpod/campagne_metriques_unifiees.sh
#
# Campagne du memoire :
#   N=5000 N_GEN=1000 bash runpod/campagne_metriques_unifiees.sh
#
# Chaque condition produit quatre decisions distinctes :
#   p < 1e-6 ; Omega accepte ; hash accepte ; verdict AUTHENTIC complet.
set -euo pipefail

RACINE=${RACINE:-/workspace/code-memoire}
CKPT=${CKPT:-/workspace/ckpt/base_64bits.pth}
COND_DIFF=${COND_DIFF:-/workspace/runs/phaseD512_degele/checkpoint.pt}
COND_AR=${COND_AR:-/workspace/runs/phaseF_maskgit/checkpoint.pt}
CORPUS=${CORPUS:-/workspace/corpus-eval-12k}
SORTIE=${SORTIE:-/workspace/runs/metriques_unifiees}
LOGS=${LOGS:-$SORTIE/logs}
USER_ID=${USER_ID:-createur-0042}

N=${N:-5000}
N_GEN=${N_GEN:-1000}
B_DIFF=${B_DIFF:-8}
B_AR=${B_AR:-16}
WITH_LPIPS=${WITH_LPIPS:-1}

mkdir -p "$SORTIE" "$LOGS"
cd "$RACINE"
export PYTHONPATH=deps:.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}

for chemin in "$CKPT" "$COND_DIFF" "$COND_AR"; do
  if [[ ! -f "$chemin" ]]; then
    echo "Checkpoint introuvable : $chemin" >&2
    exit 1
  fi
done
if [[ ! -d "$CORPUS" ]]; then
  echo "Corpus introuvable : $CORPUS" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU CUDA requis : nvidia-smi est introuvable." >&2
  exit 1
fi

extras=()
if [[ "$WITH_LPIPS" == "1" ]]; then
  extras+=(--lpips)
fi

lance() {
  local nom=$1
  shift
  local bilan="$SORTIE/$nom/volet_b_${nom}.json"
  if [[ -f "$bilan" ]]; then
    echo "[$nom] résultat déjà présent, étape ignorée : $bilan"
    return
  fi
  echo "[$nom] démarrage"
  "$@" 2>&1 | tee "$LOGS/$nom.log"
  echo "[$nom] terminé"
}

lance diffusion_encode \
  python3 scripts/ciphermark/eval_grande_echelle_decodeur.py \
    --famille diffusion --source encode \
    --conditionneur "$COND_DIFF" --watermarker "$CKPT" \
    --corpus "$CORPUS" --n-images "$N" --batch "$B_DIFF" \
    --user-id "$USER_ID" --out-dir "$SORTIE/diffusion_encode" \
    "${extras[@]}"

lance autoregressif_encode \
  python3 scripts/ciphermark/eval_grande_echelle_decodeur.py \
    --famille autoregressif --source encode \
    --conditionneur "$COND_AR" --watermarker "$CKPT" \
    --corpus "$CORPUS" --n-images "$N" --batch "$B_AR" \
    --user-id "$USER_ID" --out-dir "$SORTIE/autoregressif_encode" \
    "${extras[@]}"

lance diffusion_genere \
  python3 scripts/ciphermark/eval_grande_echelle_decodeur.py \
    --famille diffusion --source genere \
    --conditionneur "$COND_DIFF" --watermarker "$CKPT" \
    --corpus "$CORPUS" --n-images "$N_GEN" --batch "$B_DIFF" \
    --user-id "$USER_ID" --out-dir "$SORTIE/diffusion_genere" \
    "${extras[@]}"

echo
echo "Campagne terminée. Résultats : $SORTIE"
find "$SORTIE" -name 'volet_b_*.json' -print

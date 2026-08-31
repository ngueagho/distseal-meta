#!/usr/bin/env bash
# ============================================================================
# CAMPAGNE DE CLOTURE -- les deux volets du memoire, a grande echelle
#
# VOLET A  couple embedder/extracteur en espace PIXEL
#          eval_recovery_robustness.py -- tatouage, registre, 17 attaques,
#          recuperation d'Omega
#
# VOLET B  decodeur conditionne, insertion d'Omega par MODULATION DES POIDS
#          eval_grande_echelle_decodeur.py -- pour les deux familles :
#          diffusion (DC-AE 512 px) et autoregressif (MaskGIT 256 px)
#
# Chaque volet ecrit son registre SQLite : c'est la base de tracabilite du
# schema, et le verifieur la relit pour rederiver la cle de l'utilisateur.
#
# REPRISE APRES EVICTION
# ----------------------
# Le pod est un spot : il peut disparaitre a tout moment. Chaque etape pose un
# jalon dans $ETAT ; au redemarrage, celles qui sont finies sont sautees. Une
# etape interrompue en cours redemarre depuis zero -- les scripts d'evaluation
# n'ont pas de reprise interne, et un bilan partiel ne vaut rien.
# ============================================================================
set -u

RACINE=/workspace/code-memoire
CKPT=/workspace/ckpt/base_64bits.pth
COND_DIFF=/workspace/runs/phaseD512_degele/checkpoint.pt
COND_AR=/workspace/runs/phaseF_maskgit/checkpoint.pt
CORPUS=${CORPUS:-/workspace/corpus-eval-12k}
SORTIE=/workspace/runs/cloture
LOGS=/workspace/logs
ETAT=$SORTIE/etapes_faites
USER_ID=${USER_ID:-createur-0042}

# Taille de la campagne. Surchargeable : N=5000 bash campagne_cloture.sh
N=${N:-100000}
# Lots : le DC-AE travaille en 512 px et tient moins d'images en memoire que
# le MaskGIT en 256 px. Le volet pixel est domine par les attaques, sur CPU.
B_PIXEL=${B_PIXEL:-100}
B_DIFF=${B_DIFF:-8}
B_AR=${B_AR:-16}

mkdir -p "$SORTIE" "$LOGS"
touch "$ETAT"
cd "$RACINE" || exit 1
export PYTHONPATH=deps:.
# 761 fils Python pour une boucle sequentielle : la sursouscription faisait
# tomber le chargement a 9 images/s quand un processus seul en fait 7500. On
# borne, sans etouffer les attaques qui, elles, profitent du parallelisme.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

journal() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOGS/cloture.log"; }

fait()    { grep -qx "$1" "$ETAT"; }
marque()  { echo "$1" >> "$ETAT"; }

# --- une etape : sautee si deja faite, sinon lancee et jalonnee -------------
etape() {
  local nom="$1"; shift
  if fait "$nom"; then
    journal "$nom deja fait -- saute"
    return 0
  fi
  journal "$nom demarre"
  local d=$(date +%s)
  if "$@" > "$LOGS/cloture_$nom.log" 2>&1; then
    marque "$nom"
    journal "$nom termine en $(( ($(date +%s) - d) / 60 )) min"
  else
    # Un echec n'est PAS jalonne : la prochaine execution reprendra l'etape.
    journal "$nom ECHEC (code $?) -- voir $LOGS/cloture_$nom.log"
    tail -5 "$LOGS/cloture_$nom.log" | sed 's/^/    /' | tee -a "$LOGS/cloture.log"
    return 1
  fi
}

# --------------------------------------------------------------- preflight --
python3 /workspace/preflight.py >> "$LOGS/cloture.log" 2>&1 \
  || journal "preflight en echec -- on continue, les imports peuvent tenir"

for f in "$CKPT" "$COND_DIFF" "$COND_AR"; do
  [ -f "$f" ] || { journal "MANQUE : $f"; exit 1; }
done
[ -d "$CORPUS" ] || { journal "MANQUE le corpus : $CORPUS"; exit 1; }

journal "campagne de cloture : N=$N, corpus $CORPUS, utilisateur $USER_ID"

# ===================================================================== A =====
etape volet_a_pixel \
  python3 scripts/ciphermark/eval_recovery_robustness.py \
    --checkpoint "$CKPT" \
    --corpus "$CORPUS" \
    --n-images "$N" --batch "$B_PIXEL" \
    --out "$SORTIE/volet_a_pixel.json"

# ===================================================================== B =====
etape volet_b_diffusion \
  python3 scripts/ciphermark/eval_grande_echelle_decodeur.py \
    --famille diffusion --source encode \
    --conditionneur "$COND_DIFF" --watermarker "$CKPT" \
    --corpus "$CORPUS" --n-images "$N" --batch "$B_DIFF" \
    --user-id "$USER_ID" --out-dir "$SORTIE/volet_b_diffusion"

etape volet_b_autoregressif \
  python3 scripts/ciphermark/eval_grande_echelle_decodeur.py \
    --famille autoregressif --source encode \
    --conditionneur "$COND_AR" --watermarker "$CKPT" \
    --corpus "$CORPUS" --n-images "$N" --batch "$B_AR" \
    --user-id "$USER_ID" --out-dir "$SORTIE/volet_b_autoregressif"

# --- la chaine generative complete, sur un sous-ensemble --------------------
# L'echantillonnage du U-ViT coute ~2,4 s par image : impossible a $N, mais le
# memoire doit montrer que le tatouage naît dans une GENERATION et pas
# seulement dans un reencodage. D'ou ce temoin a petite echelle.
etape volet_b_diffusion_generee \
  python3 scripts/ciphermark/eval_grande_echelle_decodeur.py \
    --famille diffusion --source genere \
    --conditionneur "$COND_DIFF" --watermarker "$CKPT" \
    --corpus "$CORPUS" --n-images "${N_GEN:-1000}" --batch "$B_DIFF" \
    --user-id "$USER_ID" --out-dir "$SORTIE/volet_b_diffusion_generee"

journal "CAMPAGNE TERMINEE -- bilans dans $SORTIE"
ls -1 "$SORTIE"/*/*.json "$SORTIE"/*.json 2>/dev/null | sed 's/^/  /' \
  | tee -a "$LOGS/cloture.log"

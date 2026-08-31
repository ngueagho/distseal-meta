#!/bin/bash
# ============================================================================
# SUPERVISEUR -- enchaine les etapes sans intervention
#
# POURQUOI
# --------
# Le pod est evince environ toutes les douze heures et perd alors /root, /tmp
# et tous ses paquets pip. Chaque incident coutait jusqu'ici un temps mort egal
# au delai avant que je m'en apercoive. Ce script vit sur /workspace, qui
# survit, et fait trois choses :
#
#   1. il relance l'etape courante si son processus a disparu ;
#   2. il passe a l'etape suivante quand la courante est terminee ;
#   3. il appelle le preflight avant chaque lancement, les paquets pip ayant
#      disparu avec le conteneur.
#
# IL S'ARRETE PLUTOT QUE DE S'ENTETER
# -----------------------------------
# Si une etape echoue trois fois d'affilee sans progresser, il s'arrete et
# l'ecrit. Relancer indefiniment un run qui plante masque le probleme et brule
# du GPU. De meme, il ne franchit une etape que si le critere de reussite est
# atteint -- une phase 0 qui n'a pas quitte ln 2 ne doit pas enchainer sur la
# phase 1, qui ne ferait qu'empirer.
#
# ETAT
# ----
# /workspace/etat_superviseur : nom de l'etape courante, une ligne. Permet de
# reprendre au bon endroit apres une eviction du superviseur lui-meme.
# ============================================================================
set -u
J=/workspace/logs/superviseur.log
ETAT=/workspace/etat_superviseur
R=/workspace/code-memoire
C=configs/training/ciphermark

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$J"; }

# --- description des etapes : nom | config | journal | lanceur -------------
etape_config()  { case "$1" in
  phaseF)   echo "$C/phaseF_reprise_corpus_disjoint.yaml";;
  e3a_p0)   echo "$C/etape3_latent_phase0_solo.yaml";;
  # v2 : lambda_i 0,5 et scaling_w 0,8. La v1, qui restaurait la recette de
  # DistSeal telle quelle (lambda_i 0), detruisait l'image -- 8,21 dB, 95,7 %
  # des pixels modifies. Elle est archivee sous etape3_p1_v1_distseal_psnr8.
  e3a_p1)   echo "$C/etape3_latent_phase1_fidelite.yaml";;
  e3b_p0)   echo "$C/etape3b_latent_diffusion_phase0.yaml";;
  e3b_p1)   echo "$C/etape3b_latent_phase1.yaml";;
esac; }
etape_journal() { case "$1" in
  phaseF)   echo "/workspace/logs/phaseF_reprise.log";;
  e3a_p0)   echo "/workspace/logs/etape3_p0_solo.log";;
  e3a_p1)   echo "/workspace/logs/etape3_p1_fidelite.log";;
  e3b_p0)   echo "/workspace/logs/etape3b_p0.log";;
  e3b_p1)   echo "/workspace/logs/etape3b_p1.log";;
esac; }
etape_suivante() { case "$1" in
  phaseF)   echo "e3a_p0";;
  e3a_p0)   echo "e3a_p1";;
  e3a_p1)   echo "e3b_p0";;
  e3b_p0)   echo "e3b_p1";;
  e3b_p1)   echo "FINI";;
esac; }

vivant() { pgrep -f "config=$(etape_config "$1")" >/dev/null && return 0
           pgrep -f "config $(etape_config "$1")" >/dev/null && return 0
           return 1; }

lancer() {
  local e=$1 cfg jrn
  cfg=$(etape_config "$e"); jrn=$(etape_journal "$e")
  cd "$R" || return 1
  python3 /workspace/preflight.py >>"$J" 2>&1 || { log "preflight en echec pour $e"; return 1; }
  export PYTHONPATH=deps:. WANDB_MODE=offline
  if [ "$e" = "phaseF" ]; then
      nohup python3 distill.py config="$cfg" >>"$jrn" 2>&1 &
  else
      nohup torchrun --nproc_per_node=1 --standalone train.py --config "$cfg" >>"$jrn" 2>&1 &
  fi
  log "$e lance (pid $!)"
}

# --- critere de fin, propre a chaque etape ---------------------------------
terminee() {
  local e=$1 jrn; jrn=$(etape_journal "$e")
  [ -f "$jrn" ] || return 1
  case "$e" in
    phaseF) grep -aq "max steps 60000 reached" "$jrn";;
    # Motif generique : derniere epoque atteinte, quel que soit le total.
    # Un motif code en dur (199/200, 399/400) casse des qu'on change le
    # nombre d'epoques -- l'etape ne serait jamais declaree terminee et le
    # superviseur resterait bloque dessus.
    # Motif generique : derniere epoque atteinte, quel que soit le total.
    # Avec -F'[][/]', les champs sont 2 (courante) et 3 (total) -- une erreur
    # d'indice ici declarerait toute etape terminee des la premiere epoque.
    # END{} et non une action de ligne : sans cela, un journal VIDE ne declenche
    # aucune action, awk sort avec 0, et une etape jamais commencee serait
    # declaree terminee -- le superviseur la sauterait.
    *)      grep -aoE "Epoch: \[[0-9]+/[0-9]+\]" "$jrn" 2>/dev/null | tail -1 \
            | awk -F'[][/]' 'NF>=3 && $3+0>0 {c=$2+0; t=$3+0}
                             END {exit !(t>0 && c >= t-1)}';;
  esac
}

# --- critere de REUSSITE : la phase 0 doit avoir quitte ln 2 ---------------
reussie() {
  local e=$1 jrn p; jrn=$(etape_journal "$e")
  case "$e" in
    e3a_p0|e3b_p0)
      p=$(grep -ao "loss_decode: [0-9.]* ([0-9.]*)" "$jrn" 2>/dev/null \
          | tail -1 | grep -o "([0-9.]*)" | tr -d "()")
      [ -n "$p" ] && awk "BEGIN{exit !($p < 0.680)}";;
    *) return 0;;
  esac
}

E=$(cat "$ETAT" 2>/dev/null || echo phaseF)
log "superviseur demarre, etape courante : $E"
ECHECS=0

while [ "$E" != "FINI" ]; do
  if vivant "$E"; then ECHECS=0; sleep 300; continue; fi

  if terminee "$E"; then
    if reussie "$E"; then
      S=$(etape_suivante "$E")
      log "$E TERMINEE et reussie -> passage a $S"
      E=$S; echo "$E" > "$ETAT"; ECHECS=0
      [ "$E" = "FINI" ] && break
      lancer "$E"; sleep 300; continue
    else
      log "$E terminee mais le critere de reussite n'est PAS atteint (loss_decode toujours a ln 2). ARRET du superviseur : enchainer serait inutile."
      exit 2
    fi
  fi

  ECHECS=$((ECHECS+1))
  if [ "$ECHECS" -ge 4 ]; then
    log "$E a echoue $ECHECS fois sans progresser. ARRET du superviseur."
    exit 3
  fi
  log "$E absente et non terminee (echec $ECHECS/4) -- relance"
  lancer "$E"
  sleep 300
done
log "TOUTES LES ETAPES SONT TERMINEES"

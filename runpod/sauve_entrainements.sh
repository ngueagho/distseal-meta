#!/bin/bash
# Sauvegarde periodique des entrainements vers le Drive.
#
# --min-age 3m est OBLIGATOIRE : rclone qui copie pendant un torch.save
# capture une archive TRONQUEE, de bonne apparence et de bonne taille, que
# torch.load refuse d'ouvrir ("failed finding central directory"). Le
# checkpoint de la phase G a l'epoque 242 a ete perdu ainsi. Comparer les
# tailles ne detecte rien : la copie est fidele, c'est la source qui etait
# incomplete.
#
# Le remote qui fonctionne est gdrive_local ; la section gdrive_remote du
# rclone.conf n'a aucun jeton.
#
# Deploiement : /workspace/sauve_entrainements.sh (survit aux evictions).
export RCLONE_CONFIG=/workspace/.rclone.conf
J=/workspace/logs/nuit.log
declare -A CIBLE=(
  [phaseF_maskgit]="phaseF_maskgit"
  [phaseD512_degele]="phaseD512-degele-decodeur-a-512px"
  [etape3_latent_p0_v3]="etape3a-embedder-latent-phase0"
  [etape3_latent_phase1]="etape3a-embedder-latent-phase1-fidelite"
  [etape3b_latent_diffusion_phase0]="etape3b-embedder-latent-diffusion-phase0"
  [etape3b_latent_p1]="etape3b-embedder-latent-diffusion-phase1"
)
while true; do
    for d in "${!CIBLE[@]}"; do
        [ -d "/workspace/runs/$d" ] || continue
        rclone copy "/workspace/runs/$d" \
            "gdrive_local:ciphermark/06-entrainements-en-cours-non-termines/${CIBLE[$d]}" \
            --include "checkpoint.pt*" --include "checkpoint.pth" \
            --include "log.txt" --include "results.json" --include "config.yaml" \
            --min-age 3m --stats-one-line >> "$J" 2>&1 \
        && echo "[sauve $(date +%H:%M)] ${CIBLE[$d]}" >> "$J"
    done
    sleep 1200
done

#!/bin/bash
# ============================================================================
# CipherMark -- campagne de validation a grande echelle (5000 images)
#
# Pourquoi
# --------
# Les resultats du memoire ont ete obtenus sur des echantillons de 5 a 100
# images. Un systeme qui rend 5/5 verdicts corrects n'a rien prouve : le meme
# tirage sur 5000 images revele les cas limites, les queues de distribution et
# les seuils mal calibres. Cette campagne rejoue TOUS les tests images du
# projet a 5000 scenes independantes (COCO), sans changer un seul seuil.
#
# Discipline : aucun parametre de decision n'est modifie entre le petit et le
# grand echantillon. Seul --n-images change. Toute difference de resultat est
# donc imputable a la taille de l'echantillon, et a rien d'autre.
#
# Usage sur le pod :
#   bash scripts/ciphermark/campagne_5000.sh 2>&1 | tee /workspace/runs/campagne.log
# ============================================================================
set -u

RACINE=${RACINE:-/workspace/code-memoire}
CORPUS=${CORPUS:-/workspace/corpus-eval-12k}
OUT=${OUT:-/workspace/runs/campagne5000}
N=${N:-5000}
BATCH=${BATCH:-25}

cd "$RACINE" || exit 1
export PYTHONPATH=deps:.
mkdir -p "$OUT"
BILAN="$OUT/bilan.txt"

etape () {                       # etape <nom> <commande...>
    nom=$1; shift
    if [ -f "$OUT/$nom.fait" ]; then
        echo "== $nom : deja fait, ignore"
        return
    fi
    echo "== $nom : demarrage $(date +%H:%M:%S)"
    debut=$(date +%s)
    if "$@" > "$OUT/$nom.log" 2>&1; then
        duree=$(( $(date +%s) - debut ))
        echo "OK      $nom  (${duree}s)" | tee -a "$BILAN"
        touch "$OUT/$nom.fait"
    else
        code=$?
        duree=$(( $(date +%s) - debut ))
        echo "ECHEC   $nom  (code $code, ${duree}s)" | tee -a "$BILAN"
        tail -20 "$OUT/$nom.log" | sed 's/^/          /'
    fi
}

# --------------------------------------------------------------- 0. corpus --
# COCO test2017 : une image par scene, recadrage carre central. Des scenes
# independantes sont indispensables -- des recadrages multiples d'une meme
# photo fausseraient toute statistique de collision ou de separation.
if [ "$(find "$CORPUS" -type f -name '*.jpg' 2>/dev/null | wc -l)" -lt 10000 ]; then
    echo "== construction du corpus (12 000 scenes COCO)"
    mkdir -p "$CORPUS/toutes"
    python3 scripts/ciphermark/build_corpus.py \
        --out "$CORPUS/toutes" --coco test2017 --coco-max 12000 \
        2>&1 | tail -5
fi
DISPO=$(find "$CORPUS" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
echo "== corpus : $DISPO images dans $CORPUS"
[ "$DISPO" -lt 200 ] && { echo "corpus trop petit, abandon"; exit 1; }

# n pour les attaques actives : le script charge 2n images (marquees + vierges)
N_ATT=$(( DISPO / 2 )); [ "$N_ATT" -gt "$N" ] && N_ATT=$N

# --------------------------------------------- checkpoints a evaluer --------
CK64=""
for c in /workspace/ckpt/base_64bits.pth "$RACINE/runs/ciphermark_64bits_checkpoint.pth" \
         /workspace/runs/*64*/checkpoint.pth; do
    [ -f "$c" ] && { CK64=$c; break; }
done
[ -z "$CK64" ] && { echo "aucun checkpoint 64 bits trouve, abandon"; exit 1; }
echo "== checkpoint 64 bits : $CK64"

: > "$BILAN"
echo "campagne demarree $(date)" >> "$BILAN"
echo "corpus=$DISPO images  N=$N  N_ATT=$N_ATT  checkpoint=$CK64" >> "$BILAN"

# ================================================== volet 1 : le canal =======
etape robustesse_5000 python3 scripts/ciphermark/eval_recovery_robustness.py \
    --checkpoint "$CK64" --corpus "$CORPUS" --n-images "$N" --batch "$BATCH" \
    --out "$OUT/robustesse_5000.json"

etape attaques_actives_5000 python3 scripts/ciphermark/eval_attaques_actives.py \
    --checkpoint "$CK64" --corpus "$CORPUS" --n-images "$N_ATT" --batch "$BATCH" \
    --out "$OUT/attaques_actives_5000.json"

etape chaine_complete_5000 python3 scripts/ciphermark/full_chain_real_weights_test.py \
    --checkpoint "$CK64" --val-dir "$CORPUS" --n-images "$N" --batch "$BATCH" \
    --summary-path "$OUT/chaine_complete_5000.txt"

# ================================================== volet 2 : le hash ========
etape derive_hash_5000 python3 scripts/ciphermark/mesure_derive_hash.py \
    --checkpoint "$CK64" --corpus "$CORPUS" --n-images "$N" --batch "$BATCH" \
    --out "$OUT/derive_hash_5000.json"

etape collisions_hash_5000 python3 scripts/ciphermark/mesure_collisions_hash.py \
    --corpus "$CORPUS" --n-images "$N" --batch "$BATCH" \
    --out "$OUT/collisions_hash_5000.json"

etape hash256_omega64_5000 python3 scripts/ciphermark/test_hash256_omega64.py \
    --checkpoint "$CK64" --corpus "$CORPUS" --n-images "$N" --batch "$BATCH" \
    --out "$OUT/hash256_omega64_5000.json"

etape reparer_hash_5000 python3 scripts/ciphermark/reparer_hash.py \
    --checkpoint "$CK64" --corpus "$CORPUS" --n-images "$N" --batch "$BATCH" \
    --out "$OUT/reparer_hash_5000.json"

etape phash_robustesse_5000 python3 scripts/ciphermark/eval_phash_robustness.py \
    --data-dir "$CORPUS" --n "$N" --n-bits 256 --device cuda --chunk "$BATCH" \
    --csv "$OUT/phash_robustesse_5000.csv"

# ============================ volet 3 : les checkpoints jamais evalues =======
# Phase B (robustesse) et 96 bits ont ete entraines puis laisses de cote. Les
# passer dans la meme chaine dit enfin s'ils apportent quelque chose.
for etiquette in phaseB 96bits; do
    CK=""
    for c in /workspace/ckpt/*${etiquette}*.pth /workspace/runs/*${etiquette}*/checkpoint.pth \
             "$RACINE/runs/"*${etiquette}*.pth; do
        [ -f "$c" ] && { CK=$c; break; }
    done
    if [ -z "$CK" ]; then
        echo "SAUTE   checkpoint_$etiquette : introuvable sur ce pod" | tee -a "$BILAN"
        continue
    fi
    echo "== checkpoint $etiquette : $CK"
    etape "robustesse_${etiquette}_5000" python3 scripts/ciphermark/eval_recovery_robustness.py \
        --checkpoint "$CK" --corpus "$CORPUS" --n-images "$N" --batch "$BATCH" \
        --out "$OUT/robustesse_${etiquette}_5000.json"
done

# ==================================== volet 4 : la phase D ==================
# Le decodeur conditionne n'avait ete mesure que par la boucle de validation de
# l'entrainement : 10 lots, un Omega deterministe. Ici, des milliers d'images
# generees, la distribution complete de la bit_acc, et la liaison au contenu
# sur les images GENEREES et non plus marquees a posteriori.
CKD=""
for c in /workspace/runs/phaseD_long/checkpoint/checkpoint.pt \
         /workspace/runs/phaseD*/checkpoint/checkpoint.pt \
         /workspace/ckpt/phaseD*.pt /workspace/ckpt/phaseD*.pth; do
    [ -f "$c" ] && { CKD=$c; break; }
done
if [ -z "$CKD" ]; then
    echo "SAUTE   phaseD_5000 : aucun checkpoint de phase D trouve" | tee -a "$BILAN"
else
    echo "== checkpoint phase D : $CKD"
    # Fumee d'abord : le chargement du conditionneur et la forme des sorties
    # se verifient en une minute, pas apres trois heures de calcul.
    etape phaseD_fumee python3 scripts/ciphermark/eval_phaseD_omega.py \
        --checkpoint "$CKD" --watermarker "$CK64" --corpus "$CORPUS" \
        --n-images 16 --batch 8 --out "$OUT/phaseD_fumee.json"
    if [ -f "$OUT/phaseD_fumee.fait" ]; then
        etape phaseD_5000 python3 scripts/ciphermark/eval_phaseD_omega.py \
            --checkpoint "$CKD" --watermarker "$CK64" --corpus "$CORPUS" \
            --n-images "$N" --batch 8 --out "$OUT/phaseD_5000.json"
    else
        echo "SAUTE   phaseD_5000 : le test de fumee a echoue" | tee -a "$BILAN"
    fi
fi

echo
echo "================================ BILAN ================================"
cat "$BILAN"
echo "======================================================================"

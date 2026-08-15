# Notebooks Colab -- CipherMark

Ce dossier regroupe tous les notebooks Colab utilises pour entrainer/evaluer
CipherMark a des echelles qui depassent ce qui est raisonnable en local (pas de
GPU, bande passante limitee pour les gros telechargements comme COCO).

## Les 4 configs d'entrainement (nbits=256)

| # | Nom | Ou | Budget | Statut |
| --- | --- | --- | --- | --- |
| 1 | smoke test | local (CPU) | 10 epoques, 16 images | fait, code valide |
| 2 | Colab gratuit | Colab (T4 gratuit) | 100 epoques / 20 000 pas | fix OOM + PSNR en cours de verification |
| 3 | RunPod paye | RunPod Community Cloud (A100) | ~5h de temps GPU paye | notebook/script pret, `runpod/run_5h.sh` |
| 4 | echelle DistSeal complete ("total parametre") | cluster multi-GPU | ~500h-GPU A100 | pas encore lance -- cf. discussion cout (55-75k EUR achat, ~700-1000$ location) |

Resultats de repli en cascade : 3 sert de repli si 4 n'est pas financable, 2
sert de repli si 3 ne l'est pas non plus.

## Contenu de ce dossier

| Notebook | Config associee | But |
| --- | --- | --- |
| `config2_posthoc_pixel_256bits.ipynb` | `configs/training/ciphermark/posthoc_pixel_256bits_colab.yaml` | Config 2 : entrainement 256 bits, resolution reelle (256), corpus elargi (COCO+Kodak+BSDS+scikit-image). |

Le script de la config 3 (RunPod, pas un notebook Colab) est dans `runpod/run_5h.sh`.

## Usage (Colab)

1. Ouvrir le notebook sur Colab (`Fichier > Ouvrir un notebook > GitHub`, ou
   uploader le `.ipynb` directement).
2. `Runtime > Change runtime type > GPU`.
3. Executer les cellules dans l'ordre -- chacune est commentee.
4. Le checkpoint est sauve sur Google Drive (`output_dir`) : une session qui se
   coupe peut reprendre en relancant la meme cellule d'entrainement, sans rien
   changer (`train.py` reprend automatiquement depuis `checkpoint.pth`).

## Config 4 (echelle DistSeal complete)

Pas encore de script dedie -- a ecrire si le budget (cf. discussion cout
GPU/cluster) est valide. Necessite un vrai multi-GPU (8 cartes pour le
post-hoc, 32 pour la distillation), hors de portee de Colab et de RunPod en
mono-GPU.

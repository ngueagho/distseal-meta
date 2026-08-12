# Notebooks Colab -- CipherMark

Ce dossier regroupe tous les notebooks Colab utilises pour entrainer/evaluer
CipherMark a des echelles qui depassent ce qui est raisonnable en local (pas de
GPU, bande passante limitee pour les gros telechargements comme COCO).

## Contenu

| Notebook | Config associee | But |
| --- | --- | --- |
| `config2_posthoc_pixel_256bits.ipynb` | `configs/training/ciphermark/posthoc_pixel_256bits_colab.yaml` | Config 2/3 : entrainement 256 bits, resolution reelle (256), corpus elargi (COCO+Kodak+BSDS+scikit-image). Resultats de repli si la config 3 (echelle DistSeal complete) n'est pas financable. |

## Usage

1. Ouvrir le notebook sur Colab (`Fichier > Ouvrir un notebook > GitHub`, ou
   uploader le `.ipynb` directement).
2. `Runtime > Change runtime type > GPU`.
3. Executer les cellules dans l'ordre -- chacune est commentee.
4. Le checkpoint est sauve sur Google Drive (`output_dir`) : une session qui se
   coupe peut reprendre en relancant la meme cellule d'entrainement, sans rien
   changer (`train.py` reprend automatiquement depuis `checkpoint.pth`).

## Config 1 (smoke test) et config 3 (echelle DistSeal complete)

- La config 1 (`configs/training/ciphermark/posthoc_pixel_256bits_smoketest.yaml`)
  tourne en local (CPU), pas besoin de notebook Colab.
- La config 3 n'a pas encore de notebook dedie ici -- a ajouter si le budget
  (cf. discussion cout GPU/Colab) est valide.

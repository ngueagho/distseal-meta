# Diapositives de soutenance

`CipherMark_soutenance.pptx` — 17 diapositives, format 16:9, notes du
presentateur sur chacune. `CipherMark_soutenance.pdf` en est l'export, utile si
la machine de la salle n'a pas PowerPoint.

## Regenerer apres un nouveau resultat

Les chiffres ne sont **jamais saisis a la main** : ils sont lus depuis
`resultats/campagne-cloture/5000/*.json`. Si une mesure change, on relance :

    python3 -m venv /tmp/venv-pptx && /tmp/venv-pptx/bin/pip install python-pptx
    /tmp/venv-pptx/bin/python scripts/ciphermark/genere_diapositives.py

## A completer avant la soutenance

- `ENCADREUR` dans le script : « Dr HAMINOU » est une supposition tiree du
  chemin du projet, a confirmer avec le nom complet et le titre exact.
- La date de soutenance n'apparait pas sur les diapositives ; l'ajouter si
  l'usage de la faculte le demande.
- Le meme champ est vide dans le memoire (`UPBMATH_def.tex` : `\defensedate`
  vaut « A definir », et les auteurs valent encore « Prenom NOM »).

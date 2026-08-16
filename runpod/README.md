# RunPod -- CipherMark config 3

## Run final (depense le reste du budget, ~10$ / ~12h30 sur RTX 4090 secure)

```bash
bash runpod/run_final.sh
```

Reprend le pod existant (21ah4heywiwjrx, volume 16rikircio deja attache) --
meme recette validee que le run 5h (scaling_w=0.5, lambda_i=0.1, augmentations
reelles), juste un budget de temps plus long. Calibration de 5 min d'abord,
puis le run long (`timeout ${BUDGET_HOURS:-12h30m}`), quel que soit le nombre
d'epoques atteint. Checkpoints numerotes toutes les 10 epoques
(`saveckpt_freq: 10`) en plus du `checkpoint.pth` courant, pour ne rien perdre
si l'arret tombe pendant une ecriture.

## Run precedent (~5h payees, deja execute)

```bash
bash runpod/run_5h.sh
```

Arrete proprement apres 5h (`timeout 5h`), quel que soit le nombre d'epoques
atteint. Checkpoints numerotes toutes les 10 epoques (`saveckpt_freq: 10`) en
plus du `checkpoint.pth` courant, pour ne rien perdre si l'arret tombe pendant
une ecriture.

## Push automatique vers Google Drive -- a configurer UNE FOIS par pod

**Important : ne jamais mettre le fichier `rclone.conf` (ou tout jeton Drive)
dans le depot git.** Il reste sur le pod uniquement, hors du dossier du repo.

1. Sur le pod, installer et configurer rclone :
   ```bash
   curl https://rclone.org/install.sh | bash
   rclone config
   ```
   `n` (new remote) -> nom `gdrive` -> type `drive` -> client_id/secret vides
   -> scope `1` -> **`Use auto config? -> n`** (pas de navigateur sur le pod).

2. Une commande `rclone authorize "drive"` s'affiche : la lancer **sur votre
   PC** (qui a un navigateur), autoriser l'acces, copier le code retourne et
   le coller dans le prompt reste ouvert sur le pod.

3. Verifier : `rclone listremotes` doit afficher `gdrive:`. A partir de la,
   `run_5h.sh` detecte automatiquement `gdrive:` et pousse le checkpoint en
   fin de run, sans rien faire de plus.

## Nouveau pod = a refaire ?

Le fichier `~/.config/rclone/rclone.conf` vit sur le disque du pod, pas dans
le repo -- il est donc perdu si le pod est supprime (pas juste stoppe). Deux
options pour eviter de refaire l'etape ci-dessus a chaque pod :
- Attacher un **Network Volume** RunPod (persistant entre pods) et y stocker
  `~/.config/rclone/` ;
- Ou coller le contenu de `rclone.conf` dans une variable d'environnement du
  template de pod RunPod (`RCLONE_CONFIG_CONTENT`), puis au demarrage :
  ```bash
  mkdir -p ~/.config/rclone
  echo "$RCLONE_CONFIG_CONTENT" > ~/.config/rclone/rclone.conf
  ```
  Cette variable est stockee cote RunPod (chiffree dans leurs "Environment
  Variables" / "Secrets"), jamais dans git.

## Recuperation manuelle (si rclone non configure)

```bash
runpodctl send /workspace/runs/runpod_256bits_5h/checkpoint.pth
# puis sur votre PC : runpodctl receive <code>
```

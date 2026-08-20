# Phase A -- Ou casse reellement le canal de tatouage

Resultats bruts des experiences du 20 aout 2026, toutes sur CPU local, sans GPU.
Les logs vivent dans `runs/` (non versionne) ; ce fichier fige les chiffres.

## Point de depart : trois mesures anterieures, toutes invalides

| Run | Reglages | Corpus | Iterations | bit_acc |
| --- | --- | --- | --- | --- |
| RunPod 256 bits | pure decode, OK | 997 img, 256x256 | 180 000 | **0.61** |
| Local 64 bits | pure decode, OK | 16 img, 64x64 | 2 400 | 0.53 |
| `overfit_check` 256 bits | `lambda_i=0.1` -- collapse | 16 img, 64x64 | 1 600 | 0.50 |
| `overfit_check` 64 bits | `lambda_i=0.1` -- collapse | 16 img, 64x64 | 1 600 | 0.51 |

Les deux `overfit_check` avaient une pression de fidelite active : leur PSNR final
(43.9 et 42.3 dB) montre que l'embedder avait appris a **ne rien ecrire**. Ce sont
des mesures de collapse, pas de capacite. Aucun test de surapprentissage valide
n'existait donc avant cette phase.

## Piege a connaitre : `iter_per_epoch` est un plafond

`train.py:594` fait `if it >= params.iter_per_epoch: break` en bouclant sur le
dataloader. Avec 16 images et `batch_size=2` le dataloader ne contient que 8 lots :
une epoque vaut **toujours** 8 iterations, quelle que soit la valeur du parametre.
Le seul levier pour allonger l'entrainement sur un petit corpus est le nombre
d'epoques. C'est ce piege qui rendait `overfit_check` six fois plus court qu'il
n'y paraissait.

## A1 -- Le pipeline fonctionne-t-il ?

`scripts/ciphermark/floor_single_pair_test.py`, une image, un message fixe,
aucune augmentation, aucune pression de fidelite, `scaling_w=0.1`.

```
step   0 | loss 0.7574 | bit_acc 0.438 | psnr 28.8 | satures 0.0%
step  20 | loss 0.0024 | bit_acc 1.000 | psnr 25.2 | satures 0.0%
step 299 | loss 0.0001 | bit_acc 1.000 | psnr 25.0 | satures 0.0%
```

**bit_acc = 1.000 en 20 pas.** Le gradient circule, l'appariement message/lecture
est correct, l'architecture apprend. Toute hypothese de bug structurel est
eliminee.

## A2 -- Ou est le mur ? La diversite des messages

Meme image, meme `scaling_w=0.1`, seule la taille du vivier de messages varie.

| Messages distincts | bit_acc a 400 pas | bit_acc a 6 000 pas | gain |
| --- | --- | --- | --- |
| 1 | **1.000** | -- | -- |
| 2 | 0.844 | -- | -- |
| 4 | 0.750 | -- | -- |
| 16 | 0.629 | **0.637** | +0.008 |
| 64 | 0.551 | -- | -- |
| aleatoire (regime `train.py`) | 0.551 | **0.473** | aucun |

Deux lectures :

1. La degradation est monotone et brutale. Le modele sait memoriser un motif, il
   ne sait pas apprendre une fonction d'encodage generique a cette echelle.
2. **Quinze fois plus de calcul n'achete rien** (16 messages : +0.008). Ce n'est
   pas de la lenteur, c'est un mur.

Attention a la lecture des petits viviers : un modele qui sort une prediction
constante obtient deja ~75 % d'accord binaire quand le vivier ne contient que 2 ou
4 messages. Seule la ligne a 1 message (loss 0.0001) est un apprentissage reel.

## A3 -- La force du filigrane

Deux runs `train.py` complets, 1250 epoques, 16 bits, 64x64, 16 images,
`lambda_i=0`, identiques sauf `scaling_w`.

| `scaling_w` | meilleur bit_acc | epoque du pic | loss finale | PSNR |
| --- | --- | --- | --- | --- |
| **0.5** | **0.7148** | 1064 | 0.6119 | 8.1 dB |
| 0.1 | 0.6094 | 410 | 0.6726 | 21.6 dB |

Le filigrane fort decode nettement mieux, **malgre** 17 a 33 % de pixels satures
par le clamp `imgs_w = torch.clamp(imgs_w, 0, 1)` (`wam.py:186`). L'hypothese
inverse -- que la saturation tuait le gradient -- est refutee : le gain en force
de signal l'emporte largement sur la perte de gradient.

Mais le compromis est inexploitable : pour gagner 0.11 de bit_acc on tombe de
21.6 a 8.1 dB, c'est-a-dire une image visiblement detruite. **Aucun point de
fonctionnement acceptable n'existe a cette echelle** -- il faudrait ~0.99 de
bit_acc pour un verdict HMAC fiable.

## Precaution methodologique

Le pic du run `scaling_w=0.5` arrive a l'**epoque 1064** sur 1250. Aux epoques 105
et 208 ce meme run paraissait mort (loss collee a ln(2) = 0.6931). **Aucune
lecture avant l'epoque ~1000 n'est interpretable sur ce projet.** Deux runs ont
failli etre supprimes sur la foi d'une lecture a 8 % du parcours.

## Conclusion

Le plateau a 0.61 n'est pas un echec inexplique : c'est un **mur de capacite**,
reproduit ici independamment sur WAM. MetaSeal (TMLR 02/2026, arXiv 2509.10766)
documente le meme phenomene sur HiDDeN -- precision qui chute de 0.987 a 32 bits
a ~0.50 a 512 bits -- avec deux points de mesure. La courbe complete sur WAM ne
semble pas publiee.

Le seul levier restant est le **nombre de pixels par bit** :

| Configuration | pixels | bits | px/bit | bit_acc |
| --- | --- | --- | --- | --- |
| RunPod | 65 536 | 256 | 256 | 0.61 |
| A3 local | 4 096 | 16 | 256 | 0.61-0.71 |

Le meme ratio donne le meme plafond a deux echelles tres differentes, ce qui
suggere fortement que c'est la bonne variable. Tester au-dela de 256 px/bit
demande du GPU : resolution superieure, ou beaucoup moins de bits a 256x256.

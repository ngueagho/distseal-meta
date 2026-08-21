# Phase A2 -- resultats, nuit du 20 au 21 aout 2026

Tous les runs : 256x256, corpus-colab (5398 train / 599 val), lambda_i=0,
augmentation identite, RTX 4090.

## Le resultat

| nbits | warmup | lr | bit_acc final | PSNR | SSIM | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 5 | 5e-4 | 0.9747 (ep 79) | 7.7 | -- | arrete, montait encore |
| 64 | 5 | 5e-4 | 0.536 puis COLLAPSE ep 42 | -- | -- | instabilite |
| 128 | 5 | 5e-4 | 0.523 puis COLLAPSE ep 7 | -- | -- | instabilite |
| **64** | **50** | **2e-4** | **0.9996** (max 0.9998) | **22.5** | **0.582** | **reussi** |
| 128 | 50 | 2e-4 | en cours | | | |

Reference historique : `runpod_256bits_phase0_puredecode`, 2288 epoques,
bit_acc 0.6076 au mieux (epoque 830), PSNR 22.0.

## Ce que ca etablit

**Le blocage n'etait pas un mur de capacite mais une instabilite
d'entrainement.** Avec des messages uniformement aleatoires, un extracteur qui
sort des logits nuls a un gradient d'esperance NULLE (E[sigmoid(0)-y] = 0). La
solution triviale -- repondre 0.5 partout, loss = ln 2 = 0.6931 -- est donc un
vrai point stationnaire. Un modele qui y tombe n'en sort jamais : le run a
128 bits a tenu 200 epoques avec UNE SEULE valeur distincte de loss sur 67 000
iterations, la ou un run sain en montre 26 sur 50 epoques.

Deux hyperparametres suffisent a l'eviter : `warmup_t` 5 -> 50 et `lr`
5e-4 -> 2e-4. Les deux runs effondres avaient leur pic juste avant que le LR
n'atteigne son sommet avec l'ancien warmup.

**A qualite d'image identique, le canal passe de 0.61 a 0.9996** (PSNR 22.0
contre 22.5, SSIM 0.572 contre 0.582).

## Detecteurs Meta pre-entrainees (cards/)

Testes le meme jour, sur messages ARBITRAIRES (pas seulement leur signature) :

| fiche | nbits | espace | bit_acc | PSNR |
| --- | --- | --- | --- | --- |
| **detector_dc-ae** | 64 | latent | **0.9992** | **30.9** |
| detector_maskgit_prequant | 64 | latent | 0.9977 | 20.0 |
| detector_rar | 64 | latent | 1.0000 | 19.1 |
| detector_maskgit_postquant | 64 | latent | 0.7625 | 21.0 |

Les quatre sont des canaux multi-bits reels, utilisables pour Omega. Tous en
espace LATENT (l'embedder attend le latent du generateur, pas des pixels RGB --
il faut brancher `neuralcompression.DCAEf64c128` ou `MaskgitVqgan`).
`detector_dc-ae` domine : 0.999 de lecture a 30.9 dB, soit ~8 dB de mieux que
notre run pixel a qualite de lecture equivalente.

## Profil materiel

Modele : 39.7M parametres (embedder U-Net 6.5M + extracteur ConvNeXt 33.2M),
159 Mo de poids, 636 Mo avec gradients et etats Adam.

Debit mesure sur RTX 4090 :

| lot | img/s | pic VRAM |
| --- | --- | --- |
| 8 | 135.8 | 4.2 Go |
| 16 | 134.2 | 7.8 Go |
| 32 | 129.3 | 14.8 Go |
| 64 | OOM | |

**Le debit est plat** : le GPU est sature en calcul des le lot 8. La VRAM
inutilisee (36 %) n'est pas du gaspillage de performance -- elle ne se
convertit pas en vitesse sur une charge limitee par le calcul. Le chargement
des donnees est negligeable (0.00006 s/iteration), donc le CPU n'est pas non
plus un goulot.

Consequence : rester sur RTX 4090 mais **passer en Community Cloud**
(0.34 $/h contre 0.74 $/h en Secure) -- facture divisee par 2.2 sans rien
perdre. Une carte de 12 Go suffirait aussi (7.8 Go au lot 16).

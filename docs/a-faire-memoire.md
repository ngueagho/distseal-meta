# A faire sur le memoire -- etat au 2026-08-22

Ce fichier est autoportant : tous les chiffres necessaires a la redaction y
sont, il n'y a rien a recalculer ni a relancer.

Sources brutes :
- `docs/phaseA-mur-de-capacite.md` -- diagnostic local (phase A)
- `runpod-results/phaseA2/` -- logs des 5 runs + `RESULTATS.md`
- `runs/eval_robustness_64bits.json` -- les 17 conditions d'attaque
- `runs/full_chain_64bits_summary.txt` -- la chaine de bout en bout
- Drive : `gdrive_local:ciphermark/runpod_runs/phaseA2_2026-08-21/`

---

## URGENT -- le chapitre 3 contient des resultats devenus FAUX

Redige le 2026-08-20, il affirme que la chaine complete rend **0/5 AUTHENTIC**
avec un BER de 39-55 % et un point fixe non converge (plafonne a 3 iterations).
C'etait honnete avec le checkpoint de l'epoque (bit_acc 0.61).

**Ce n'est plus vrai.** Avec le checkpoint `phaseA2_64bits_stable` :
5/5 AUTHENTIC, **d = 0 bit d'erreur**, point fixe converge en 1 iteration.

Le memoire decrit donc aujourd'hui un echec qui n'existe plus. C'est la
premiere chose a corriger.

Fichier : `/home/roberto/Downloads/UPB_DMCS_BS_MATH_Thesis_Template/chap3_resultats.tex`
Section concernee : `sec:chaine-reelle`, table `tab:chaine-reelle`.

---

## Les 4 blocs a ecrire dans le chapitre 3

### Bloc 1 -- Le diagnostic du collapse (contribution methodologique)

**Le mecanisme.** Avec des messages uniformement aleatoires, un extracteur qui
sort des logits nuls a un gradient d'**esperance nulle** :

    E[dL/dlogit] = E[sigmoid(0) - y] = 0.5 - E[y] = 0   pour y bit uniforme

La solution triviale (repondre 0.5 partout, loss = ln 2 = 0.6931) n'est donc
pas un plateau dont on finit par sortir : c'est un **vrai point stationnaire**.

**La signature empirique.** Le run 128 bits effondre a tenu 200 epoques avec
**une seule valeur distincte de loss** (0.6932) sur 67 000 iterations, la ou un
run sain en montre 26 sur 50 epoques.

**Preuve que ce n'est pas un mur de capacite** : l'ordre n'est pas monotone en
nbits. 16 bits reussit (0.9747), 64 et 128 s'effondrent avec les anciens
reglages, 256 apprend lentement (0.6076). Une limite de capacite serait
monotone.

**Le correctif** : `warmup_t` 5 -> 50 et `lr` 5e-4 -> 2e-4. Les deux runs
effondres avaient leur pic juste avant que le LR n'atteigne son sommet.

| nbits | warmup | lr | resultat |
| --- | --- | --- | --- |
| 64 | 5 | 5e-4 | pic 0.536 ep 30, COLLAPSE ep 42 |
| 128 | 5 | 5e-4 | pic 0.523 ep 7, COLLAPSE |
| 64 | 50 | 2e-4 | **0.9998** |
| 128 | 50 | 2e-4 | 0.9616 |

**Preuve complementaire (phase A, local)** : sur UNE image et UN message fixe,
bit_acc atteint 1.000 en 20 pas -- le pipeline n'a aucun bug. La difficulte est
la generalisation a des messages arbitraires : 1 message -> 1.000,
16 -> 0.637, aleatoire -> 0.473, et 15x plus de pas n'apportent que +0.008.

**Remede de principe implemente mais non teste** : `--msg_curriculum_epochs`
dans `train.py` (commit 33e491e) -- fait croitre l'entropie du message de 1 a
2^nbits, pour briser la symetrie du point stationnaire.

### Bloc 2 -- La courbe de capacite sur WAM (inedite)

Toutes conditions identiques : 256x256, corpus-colab (5398 train / 599 val),
lambda_i=0, augmentation identite, 1200 epoques, warmup 50, lr 2e-4.

| nbits | px/bit | bit_acc final | PSNR | SSIM | verdict CipherMark |
| --- | --- | --- | --- | --- | --- |
| **64** | 1024 | **0.9998** | 22.5 | 0.582 | **utilisable** |
| 128 | 512 | 0.9616 | 22.4 | 0.552 | insuffisant |
| 256 | 256 | 0.6076 | 22.0 | 0.572 | inutilisable |

**L'argument central** : la qualite d'image est *identique* aux trois largeurs.
Ce n'est donc pas un compromis mal regle -- c'est la **capacite de lecture** qui
sature. La frontiere utilisable est entre 64 et 128 bits.

**Consequence pour Omega** : `WitnessConfig.n_bits` doit valoir **64**, pas 256.
2^-64 reste le minimum accepte en cryptographie moderne, et c'est exactement la
largeur pour laquelle DistSeal a ete concu (`train.py:109`, et les 5 configs
d'origine sous `configs/training/{diffusion,autoregressive}/`).

A citer : MetaSeal (TMLR 02/2026, arXiv 2509.10766) documente le meme mur sur
HiDDeN -- precision 0.987 a 32 bits, ~0.50 a 512 bits -- mais avec **deux points
de mesure seulement**. La courbe complete sur WAM ne semble pas publiee.

### Bloc 3 -- La chaine CipherMark verifie de bout en bout

Checkpoint `phaseA2_64bits_stable` (epoch 1200), 5 images reelles de
`corpus-colab/val`, PHash DINOv2-small **reel** (pas le repli DCT).

**Partie A -- aller-retour honnete**

    5/5 AUTHENTIC   d = 0/64   ber = 0.000   p = 5.42e-20   conf = 1.000
    point fixe converge en 1 iteration

**Partie B -- securite**

    Omega_A vs Omega_B : 38/64 bits differents (59.4 %)
    pixels differents entre les deux images marquees : 99.6 %
    watermark_A sous cles_B : no_wm, ber 0.594 -- aucun faux AUTHENTIC
    watermark_B sous cles_A : no_wm, ber 0.594

**Partie C** : nonce absent du registre -> `KeyError`, comme attendu.

**Ce que ca valide** : tout l'appareil cryptographique -- phash DINOv2, HMAC,
PRG ChaCha20, XOR, boucle de point fixe, registre de traces, verifieur -- etait
correct depuis le debut. Il lui manquait uniquement un canal capable de
transporter Omega. Aucun seuil de `CipherMarkThresholds` n'a ete modifie.

### Bloc 4 -- La robustesse (50 images, 17 conditions)

PSNR 22.62 dB, SSIM 0.919, point fixe 1.00 iteration, 100 % converges.

| attaque | bit_acc | hash ok | BER | AUTHENTIC |
| --- | --- | --- | --- | --- |
| aucune | 0.9997 | 100 % | 0.0003 | 100 % |
| jpeg q90 | 0.9997 | 100 % | 0.0003 | 100 % |
| jpeg q70 | 1.0000 | 100 % | 0.0000 | 100 % |
| jpeg q50 | 0.9997 | 100 % | 0.0003 | 100 % |
| **jpeg q30** | 0.9997 | 100 % | 0.0003 | **100 %** |
| flou k3 | 0.9987 | 100 % | 0.0013 | 100 % |
| flou k5 | 0.9766 | 100 % | 0.0234 | 100 % |
| flou k7 | 0.9200 | 100 % | 0.0800 | 72 % |
| resize 0.75 | 0.9966 | 100 % | 0.0034 | 100 % |
| resize 0.5 | 0.9438 | 100 % | 0.0563 | 92 % |
| **crop 0.9** | 0.6259 | 100 % | 0.3741 | **0 %** |
| crop 0.7 | 0.5216 | 100 % | 0.4784 | 0 % |
| crop 0.5 | 0.4947 | 100 % | 0.5053 | 0 % |
| lumin 0.8 | 0.9994 | 100 % | 0.0006 | 100 % |
| lumin 1.2 | 0.9825 | 100 % | 0.0175 | 92 % |
| contraste 0.8 | 0.9994 | 100 % | 0.0006 | 100 % |
| contraste 1.2 | 0.9991 | 100 % | 0.0009 | 100 % |

**Deux resultats a mettre en avant.**

*Le JPEG* : 100 % AUTHENTIC jusqu'a la qualite **30**. MetaSeal ne survit
qu'au-dessus de la qualite **84** (leur Fig. 7 : seuils blur sigma < 0.7,
bruit < 0.05, JPEG > 84). Comparaison directe et favorable, sur le terrain ou
le papier concurrent admet sa faiblesse.

*La decomposition temoin / hash* : la colonne `hash ok` vaut **100 % partout**,
y compris sous recadrage. Le hash perceptuel DINOv2 corrige par Reed-Solomon
est donc parfaitement stable, et **la construction cryptographique n'est jamais
en cause**. L'echec au recadrage est entierement imputable au canal.

Cette decomposition est propre a CipherMark et n'existe pas dans un tatouage
classique : comme Omega = HMAC(K, h), le verifieur RECALCULE h sur l'image
recue. Un verdict exige donc DEUX conditions -- le temoin survit ET le hash
reste stable. Les separer est ce qui rend les echecs interpretables.

*La cause de l'echec au recadrage est identifiee* : `augmentation_config:
configs/augmentation/identity_only.yaml`. Le modele n'a **jamais vu une seule
attaque** pendant son entrainement -- choix volontaire pour isoler le collapse.
DistSeal entraine normalement avec `all_augs_v3.yaml` (recadrages, rotations,
perspectives). C'est un reglage a changer, pas une limite de la methode.

---

## Dette du chapitre 2

Deux renvois vers le chapitre 3 annoncent des mesures qui n'ont jamais ete
faites (cf. memoire `renvois-chap2-chap3-a-corriger`). A verifier et corriger.

Deja corrige le 2026-08-20 : la fausse affirmation « hypothese validee au
chapitre 3 » a propos du projecteur Hessien, remplacee par « testee
experimentalement [...] elle ne s'y confirme pas ».

---

## Chantiers techniques (apres la redaction)

### 1. Entrainement avec augmentations -- transforme la seule faiblesse mesuree

Relancer `phaseA2_64bits_stable` avec `augmentation_config:
configs/augmentation/all_augs_v3.yaml`. ~13 h de GPU. Devrait donner la
robustesse au recadrage qui manque aujourd'hui.

### 2. Deriver les cles d'un identifiant utilisateur

Aujourd'hui l'identite n'est **nulle part dans Omega** -- l'utilisateur n'est
identifie qu'implicitement par la paire de cles employee. Il faut :

    k_user = HKDF(master_secret, info = identifiant_utilisateur)
    Omega  = HMAC(k_user, h) XOR PRG(s_master, nonce)

`hkdf_expand` existe deja dans `distseal/ciphermark/crypto.py`. ~15 lignes.
C'est ce qui rend vrai l'argument « l'attribution devient une preuve » face a
WOUAF. Utiliser des identites **synthetiques mais realistes** pour la demo
(`creator_id=4417`), jamais de donnees personnelles reelles.

### 3. Extension generative (phases D-F) -- redevient possible

Le verrou etait l'absence d'extracteur gele fonctionnel. Il existe maintenant.
Voir l'artefact « Protocole CipherMark » pour le plan detaille : conditionnement
du decodeur facon WOUAF, strategie d'entrainement facon WMAdapter, cadre unifie
DistSeal couvrant diffusion et autoregressif. Point de branchement identifie :
`deps/efficientvit/aecore/trainer.py`, lignes 127 (creation du message) et 446
(`msg_repeated`).

---

## Detecteurs Meta pre-entraines -- alternative possible

Testes le 2026-08-21 sur messages **arbitraires** (pas seulement leur signature
de distillation) :

| fiche | nbits | espace | bit_acc | PSNR |
| --- | --- | --- | --- | --- |
| **detector_dc-ae** | 64 | latent | **0.9992** | **30.9** |
| detector_maskgit_prequant | 64 | latent | 0.9977 | 20.0 |
| detector_rar | 64 | latent | 1.0000 | 19.1 |
| detector_maskgit_postquant | 64 | latent | 0.7625 | 21.0 |

Les quatre sont de vrais canaux multi-bits, donc utilisables pour Omega. Tous
en espace **latent** : leur embedder attend le latent du generateur (128 canaux
pour le DC-AE), pas des pixels RGB -- il faut brancher
`neuralcompression.DCAEf64c128` ou `MaskgitVqgan`, le modele HuggingFace brut
n'expose pas `encode_pre_quant`.

`detector_dc-ae` atteint **30.9 dB** contre 22.5 pour notre run pixel a qualite
de lecture equivalente, soit ~8 dB de mieux. A considerer comme point de
comparaison dans le memoire, voire comme canal alternatif.

---

## Profil materiel (pour justifier les choix de calcul)

Modele : **39.7 M parametres** (embedder U-Net 6.5 M + extracteur ConvNeXt
33.2 M), 159 Mo de poids, 636 Mo avec gradients et etats Adam.

Debit mesure sur RTX 4090 :

| lot | img/s | pic VRAM |
| --- | --- | --- |
| 8 | 135.8 | 4.2 Go |
| 16 | 134.2 | 7.8 Go |
| 32 | 129.3 | 14.8 Go |
| 64 | OOM | |

**Le debit est plat** : le GPU est sature en calcul des le lot 8. La VRAM
inutilisee (36 % sur une 24 Go) n'est pas du gaspillage de performance. Le
chargement des donnees est negligeable (0.00006 s/iteration), donc le CPU n'est
pas non plus un goulot.

# A faire sur le memoire -- etat au 2026-08-23

Ce fichier est autoportant : tous les chiffres necessaires a la redaction y
sont, il n'y a rien a recalculer ni a relancer.

Sources brutes :
- `docs/phaseA-mur-de-capacite.md` -- diagnostic local (phase A)
- `runpod-results/phaseA2/` -- logs des 5 runs + `RESULTATS.md`
- `runs/eval_robustness_64bits.json` -- les 17 conditions d'attaque passives
- `runs/eval_attaques_actives.json` -- les 4 attaques actives
- `runs/mesure_derive_hash.json` -- derive du hash perceptuel
- `runs/full_chain_64bits_summary.txt` -- la chaine de bout en bout
- Drive : `gdrive_local:ciphermark/runpod_runs/phaseA2_2026-08-21/`
- Drive : `gdrive_local:ciphermark/phaseB_64bits_robuste/` (phase B, 2026-08-23)
- `/workspace/runs/phaseD_*` sur le pod -- les trois essais de la phase D

---

## LE PLUS IMPORTANT -- la liaison au contenu ne fonctionne pas

Trouve le 2026-08-22 en testant les attaques ACTIVES. C'est le resultat le plus
lourd de consequences du projet, et il touche l'argument central du memoire.

### Le symptome

`scripts/ciphermark/eval_attaques_actives.py`, 8 images :

| attaque | faux AUTHENTIC | BER moyen | verdicts |
| --- | --- | --- | --- |
| reference honnete | 8/8 (normal) | 0.0000 | authentic=8 |
| **A. transplantation** | **8/8** | **0.0078** | **authentic=8** |
| B. rejeu (autre nonce) | 0/8 | 0.4805 | heavy_edit=2, no_wm=6 |
| C. mixup | 0/8 | 0.4434 | heavy_edit=3, no_wm=5 |
| D. collage vu par A | 0/8 | 0.2871 | heavy_edit=5, light_edit=3 |
| D. collage vu par B | 0/8 | 0.2949 | heavy_edit=4, light_edit=3, no_wm=1 |

Coller le filigrane d'une image sur une AUTRE image produit 8 verdicts
AUTHENTIC sur 8. C'est precisement l'attaque que la dependance au contenu
devait rendre impossible -- et c'est l'argument oppose a WOUAF et WMAdapter,
dont le code utilisateur est arbitraire donc transplantable.

Les trois autres attaques echouent correctement : le reste de la construction
tient.

### La cause immediate : Reed-Solomon sur-corrige

`RSCodec(nsym=16)` corrige jusqu'a `nsym//2 = 8` octets. Le hash de 64 bits en
fait exactement **8**. La capacite de correction egale la taille de la donnee,
donc la parite stockee au registre suffit a reconstituer le hash de reference
SANS aucune information de l'image. Verifie experimentalement :

    hash etranger 70c127684ee31842 -> corrige == reference ? True
    hash etranger d72a38edb03d4271 -> corrige == reference ? True
    hash etranger 5f619907d0b7fa72 -> corrige == reference ? True

Le verifieur calcule `h_hat = RS_corrige(PHash(image recue), parite)`, mais
cette correction **ignore completement** `PHash(image recue)`.

### La cause profonde : le hash perceptuel ne separe pas les images

Mesure sur 30 images (`scripts/ciphermark/mesure_derive_hash.py`), en bits :

| cas | bits differents / 64 | soit | plage |
| --- | --- | --- | --- |
| marquage de la MEME image | 19.07 | 29.8 % | 8 a 35 |
| une AUTRE image | 28.47 | 44.5 % | 18 a 35 |

**Marquer une image change 30 % des bits de son hash ; une image totalement
differente n'en change que 44 %.** Les plages se recouvrent de 18 a 35 bits des
deux cotes : aucun seuil ne peut separer les deux cas.

Aucune valeur de `nsym` ne peut donc sauver la construction. Il faudrait
corriger 30 % d'erreurs tout en refusant d'en corriger 44 % -- c'est
mathematiquement impossible avec ces distributions.

### Pourquoi la chaine semblait fonctionner

Le point fixe convergeait en 1 iteration et le test rendait 5/5 AUTHENTIC,
uniquement parce que Reed-Solomon reconstruisait le hash de reference a partir
de la parite seule. **La liaison au contenu n'a jamais ete testee -- elle etait
court-circuitee.** Les 5/5 AUTHENTIC restent vrais pour l'aller-retour honnete,
mais ils ne prouvent rien sur l'anti-transplantation.

### Ce qu'il faut faire

Le probleme est le HASH, pas le code correcteur. Un hash utilisable doit etre
quasi invariant au marquage -- quelques bits de derive, pas vingt. Trois pistes,
par ordre de promesse :

1. **Stabiliser le hash.** `RandomHyperplaneLSH` binarise les features DINOv2
   par projections aleatoires ; les bits proches d'une frontiere de decision
   sont fragiles. Utiliser les composantes principales, ou quantifier avec une
   marge (zone morte), reduirait beaucoup la derive.
2. **Calculer le hash sur les basses frequences** de l'image, la ou le
   filigrane ecrit peu.
3. **Renoncer a la liaison par hash** et lier Omega au contenu autrement (par
   exemple un identifiant enregistre plutot que derive).

### Comment le rediger

Ne pas le cacher : c'est un resultat de recherche, et le trouver soi-meme vaut
mieux que de ne pas l'avoir teste. La formulation honnete est que la
construction reposait sur un artefact du code correcteur, que le test d'attaque
active l'a revele, et que la mesure de derive en explique la cause. MetaSeal
teste exactement ces attaques (section 3.1 : replay, mixup, PGD) -- ne pas les
tester aurait ete une lacune visible pour un jury.

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

**A LIRE AVEC LA SECTION SUR LA LIAISON AU CONTENU** (en tete de ce fichier) :
les 5/5 AUTHENTIC ci-dessous sont exacts pour l'aller-retour honnete, mais ils
ne prouvent RIEN sur l'anti-transplantation -- Reed-Solomon reconstruisait le
hash de reference a partir de la parite seule. La partie B ci-dessous montre
que deux cles differentes donnent des Omega differents, ce qui reste vrai ;
elle ne montre pas que le filigrane est intransplantable, ce qui est faux.

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

---

# AUDIT DU MEMOIRE, FICHIER PAR FICHIER (2026-08-23)

Lecture integrale de `~/Downloads/UPB_DMCS_BS_MATH_Thesis_Template/` :
`introduction.tex`, `chap1_cadre_theorique.tex`, `chap2_methodologie.tex`,
`chap3_resultats.tex`, `conclusion.tex`, `annexes.tex`, `abreviations.tex`,
`main.tex`, `bibliographie.bib`.

Les points sont classes par gravite. Un point CRITIQUE est une affirmation que
le code contredit : un jury qui execute le depot trouve l'inverse.

## CRITIQUE 1 -- la revendication centrale est contredite par le code

**Ou** : `chap3` tableau `tab:validation`, ligne "Rejeu" ; tableau
`tab:vsdistseal`, ligne "rejeu du message sur une autre image" ; et surtout
`conclusion.tex` l.42-44 et l.15-22.

Le memoire fait de la resistance au rejeu LA contribution differentielle :
"0/20 acceptes", "impossible (0/20)", et la conclusion la reprend comme reponse
a la question de recherche.

`scripts/ciphermark/eval_attaques_actives.py` mesure **8/8 faux AUTHENTIC sous
transplantation**, BER 0.0078.

Le chapitre 3 est techniquement honnete -- il precise que ces 0/20 valent "sous
un canal d'embedding idealise". La conclusion, elle, reprend le chiffre sans la
reserve. Et la cause est structurelle, pas accidentelle : `RSCodec(nsym=16)`
corrige `nsym//2 = 8` octets sur un hash de 64 bits qui en fait 8. La parite
seule reconstitue le hash de reference sans jamais consulter l'image.

**A faire** : soit corriger le dimensionnement de nsym et re-mesurer, soit
requalifier la revendication en "resistance au rejeu au niveau de la couche
cryptographique, non encore atteinte sur la chaine complete". La seconde option
est defendable et honnete ; la premiere demande de resoudre d'abord le probleme
du hash (voir la section LE PLUS IMPORTANT plus haut).

## CRITIQUE 2 -- Omega est a 256 bits dans tout le memoire, le code est a 64

**Ou** : `chap2` eq. `eq:ciphermark` (`Omega dans {0,1}^256`), `chap2`
§Formalisation (plusieurs occurrences de 256), `chap3` l.60-61.

Le memoire justifie 256 par "la largeur d'un tag HMAC-SHA256 imposant 256". Le
code est passe a 64 (`WitnessConfig.n_bits`, `CipherMarkConfig.n_bits`), parce
que 256 bits ne fonctionne pas : bit_acc 0.6076.

**Nouvelle justification a ecrire** : on tronque le tag HMAC a 64 bits. La
troncature d'un HMAC est une pratique normalisee (RFC 2104 §5) ; elle ramene la
resistance a la forge de 2^256 a 2^64, ce qui reste tres au-dela du besoin. Le
tableau du choix de 64 bits (section suivante) fournit les chiffres.

## CRITIQUE 3 -- la section 3.1.2 raconte un echec qui n'en est plus un

**Ou** : `chap3` §`sec:chaine-reelle`, tableau `tab:chaine-reelle`, et le
paragraphe "Le canal Omega reel n'atteint pas encore le regime Authentic" dans
§`sec:limites`.

Le texte decrit un checkpoint a bit_acc 0.60 et **0/5 AUTHENTIC**. Obsolete.
Etat reel : bit_acc **0.9998** a 64 bits, chaine complete **5/5 AUTHENTIC,
d = 0, p = 5.42e-20**.

A reecrire entierement. Y raconter aussi le diagnostic du palier, qui est un
resultat en soi (voir Bloc 1 plus haut) : ce n'etait pas un mur de capacite
mais un point stationnaire trivial, corrige par warmup_t 5->50 et lr
5e-4->2e-4.

## IMPORTANT 4 -- l'algorithme decrit n'est pas celui qui tourne

**Ou** : `chap2` Algorithme 1, ligne `z_w <- z + eps * P(W_theta.embed(z, Omega))`.

L'algorithme decrit une injection dans le LATENT avec projecteur hessien.
L'implementation reelle est post-hoc PIXEL (`latent_watermarker: false` -- le
WAM a ete entraine sur du 256x256 RGB et ne peut pas s'appliquer au latent
f64c128, qui fait 128 canaux en 4x4). Et le projecteur hessien est infirme au
chapitre 3 : c'est la projection aleatoire qui est deployee.

L'algorithme doit refleter le code, sinon il decrit un systeme qui n'existe pas.

## IMPORTANT 5 -- la phase D n'existe nulle part dans le memoire

Le memoire promet une "compatibilite architecturale couvrant diffusion et
autoregressif" (objectif principal, introduction l.74), mais l'architecture du
chapitre 2 est du tatouage post-hoc. Rien ne decrit comment le modele generatif
produit LUI-MEME des images marquees avec un identifiant variable.

C'est pourtant le travail de la phase D. A ajouter :
- **chapitre 2, nouvelle sous-section** : le conditionnement du decodeur sur
  Omega. Reseau de mapping (Omega -> code latent -> gamma/beta par canal),
  modulation affine facon FiLM, initialisation a zero (identite exacte au
  depart), bornes tanh sur gamma et beta, et le point de conception central :
  Omega est ALEATOIRE a l'entrainement, la liaison au contenu etant le CHOIX
  d'Omega decide a l'inference par la boucle de point fixe.
- **chapitre 2, positionnement** : c'est l'idee de WOUAF (modulation de poids)
  transposee aux activations. Pour une convolution, moduler le canal de sortie
  c est mathematiquement identique a moduler les poids du filtre c ; WOUAF
  ajoute une renormalisation. L'interet de la variante activations est
  l'absence totale de modification du decodeur (hooks PyTorch).
- **chapitre 3, nouvelle section** : les trois essais et leurs enseignements
  (voir la section PHASE D plus bas).

## IMPORTANT 6 -- positionnement WOUAF, et MetaSeal absent

**Ou** : `chap1` l.74-76 et `bibliographie.bib`.

WOUAF est expedie en deux lignes. Or la phase D fait de la modulation
conditionnee par l'identite, c'est-a-dire l'idee de WOUAF. Un jury demandera la
difference. Elle existe et il faut l'ecrire : **WOUAF module sur un
identifiant arbitraire ; CipherMark le derive cryptographiquement du contenu
(HMAC du hash perceptuel) et d'une cle utilisateur (HKDF).**

Verifie dans `bibliographie.bib` : **MetaSeal, UniMark, IndexMark, WMAdapter,
Safe-VAR sont tous absents**. MetaSeal est le plus risque : il fait du tatouage
cryptographique lie au contenu, c'est-a-dire le voisin direct de la
contribution revendiquee. Ne pas le citer expose a la question "en quoi vous
distinguez-vous de MetaSeal ?" sans preparation.

## MOYEN 7 -- environnement technique et annexes obsoletes

**Ou** : `chap2` §2.2.1, §2.2.2, et `annexes.tex` en entier.

- "Google Colab (GPU T4 ou A100)" -> RunPod, RTX 4090 (24.5 Go), Community Cloud
- "bibliotheque `cryptography` de Python" -> en realite **pycryptodome** pour
  ChaCha20, `hmac`/`hashlib` de la stdlib pour HMAC-SHA256 et HKDF
- Corpus annonces au chap2 (MS-COCO 1000, ImageNet 1000, synthetique 2000) : ne
  correspondent ni aux 50 997 du chap3, ni au corpus reel d'entrainement
  (5 398 train / 599 val). Trois chiffres differents pour la meme campagne.
- `annexes.tex` renvoie a des CSV Colab et a un carnet
  `notebooks/colab_ciphermark.ipynb` : verifier qu'ils existent encore, ajouter
  les scripts reellement utilises (`eval_recovery_robustness.py`,
  `eval_attaques_actives.py`, `mesure_derive_hash.py`,
  `mesure_collisions_hash.py`, `reparer_hash.py`, `profile_model.py`,
  `test_meta_detectors.py`).

## MOYEN 8 -- l'objectif principal promet la resistance LoRA

**Ou** : `introduction.tex` l.74.

"en vue de garantir [...] et une resistance au fine-tuning LoRA". Le chapitre 3
infirme proprement cette hypothese. Annoncer une garantie puis l'infirmer
affaiblit le texte ; formuler des l'introduction une **hypothese a tester** le
renforce, et rend le resultat negatif du chapitre 3 valorisant.

## A AJOUTER 9 -- deux resultats acquis et absents du memoire

**La derivation de l'identite utilisateur.** `WitnessField.for_user()` derive la
cle par HKDF depuis un identifiant (`distseal/ciphermark/witness.py`). Mesure :
Omega_A et Omega_B different sur 28/64 bits ; A verifie sous ses propres cles
donne ber 0.000, sous les mauvaises 0.438. C'est ce qui rend l'attribution
*prouvable* plutot que simplement *unique*. Absent du memoire.

**La robustesse reelle, 17 conditions** (`runs/eval_robustness_64bits.json`).
JPEG q30 a 100 % AUTHENTIC -- a comparer a la limite q84 de MetaSeal, c'est un
argument fort. Mais recadrage 0.9 / 0.7 / 0.5 -> **0 % AUTHENTIC** (bit_acc
0.63 / 0.52 / 0.49) : la falaise geometrique touche aussi le canal Omega, pas
seulement le hash. Le memoire ne l'attribue qu'au hash.

---

# JUSTIFIER LE CHOIX DE 64 BITS -- tableau et figure

## Le raisonnement

Deux forces opposees fixent la largeur du canal Omega.

**Vers le haut.** Plus de bits, plus il est difficile qu'une image non marquee
franchisse le seuil par hasard. C'est la securite : `2^n` possibilites, et
surtout la probabilite de faux positif au seuil de decision.

**Vers le bas.** Plus de bits a inscrire dans le meme nombre de pixels, moins
l'extracteur les lit correctement. C'est la capacite, et elle est mesuree.

Le point non evident, et c'est lui qui porte la demonstration : **la securite
theorique 2^n ne vaut rien si le canal ne delivre jamais de verdict.** A 256
bits, l'espace de recherche fait 2^256 -- inviolable -- mais la probabilite
qu'une image legitimement marquee soit reconnue vaut **0.0000**. Une securite
parfaite sur un systeme qui ne fonctionne jamais.

## Les chiffres

Toutes les valeurs de bit_acc sont mesurees dans des conditions identiques
(256x256, corpus-colab, `identity_only`, meme calendrier de scaling_w, configs
identiques a `nbits` pres -- verifie par diff). Seuil de decision : BER <= 10 %,
celui du verdict AUTHENTIC de l'Algorithme 2.

| n bits | bit_acc | espace 2^n | seuil tau | P(succes) | P(faux positif) | securite eff. | marge attaque |
|--------|---------|-----------|-----------|-----------|-----------------|---------------|---------------|
| 32     | non teste | 2^32    | 3         | --        | 1.28e-06        | 19.6 bits     | --            |
| **64** | **0.9998** | 2^64  | 6         | **1.0000**| 4.51e-12        | 37.7 bits     | **0.0998**    |
| 96     | 0.9236* | 2^96      | 9         | 0.8025    | 1.82e-17        | 55.6 bits     | 0.0236        |
| 128    | 0.9616  | 2^128     | 12        | 0.9986    | 7.76e-23        | 73.4 bits     | 0.0616        |
| 256    | 0.6076  | 2^256     | 25        | **0.0000**| 2.98e-43        | 141.3 bits    | negative      |

\* 96 bits : run encore en cours (ep 999/1199), valeur provisoire.

Definitions :
- **P(succes)** = P(Bin(n, 1-bit_acc) <= tau) : probabilite qu'une image
  reellement marquee soit reconnue AUTHENTIC, sur image propre.
- **P(faux positif)** = P(Bin(n, 0.5) <= tau) : probabilite qu'une image non
  marquee franchisse le seuil.
- **securite effective** = -log2 P(faux positif) : les bits de securite REELS,
  par opposition aux n bits nominaux.
- **marge attaque** = 0.10 - (1 - bit_acc) : accuracy que le canal peut encore
  perdre sous attaque avant de passer sous le seuil.

## Ce que le tableau demontre

**256 bits est elimine par la capacite.** P(succes) = 0.0000 sur image propre,
avant toute attaque. Ses 141 bits de securite effective sont sans objet.

**96 bits est elimine aussi** : une image sur cinq echoue deja sur image propre
(P(succes) = 0.80). A noter, ce point est anormal -- il fait moins bien que 128
bits alors qu'il devrait faire mieux (voir AVERTISSEMENT ci-dessous).

**32 bits est elimine par la securite, sans avoir besoin de l'experience** : la
securite effective est bornee par n, donc au mieux ~20 bits, soit un faux
positif toutes les 800 000 images. Insuffisant pour un usage forensique.
L'argument est structurel, pas empirique.

**Restent 64 et 128 bits**, et c'est un vrai choix, pas une evidence :

| | 64 bits | 128 bits |
|---|---|---|
| P(succes) sur image propre | 1.0000 | 0.9986 |
| securite effective | 37.7 bits | 73.4 bits |
| marge attaque | 0.0998 | 0.0616 |

**64 bits est retenu pour la marge d'attaque**, 1.6 fois celle de 128 bits.
C'est le critere qui compte ici, parce que les attaques mesurees entament
reellement la bit_acc : recadrage a 0.9 fait tomber le canal a 0.63, bien en
dessous du seuil dans les deux cas, mais JPEG et photometrie se jouent
precisement dans cette marge. Et 37.7 bits de securite effective signifient un
faux positif tous les 2.10^11 images -- amplement suffisant pour un usage
forensique ou la charge de la preuve pese sur la detection positive.

**Honnetete requise dans la redaction** : 128 bits est plus sur (73 contre 38
bits) et reste viable. Le choix de 64 est un arbitrage robustesse/securite, pas
un optimum unique. Le presenter comme "le point ideal" serait surinterpreter ;
le presenter comme "le meilleur compromis pour un canal dont les attaques
geometriques entament la precision" est exact et defendable.

## AVERTISSEMENT -- un confondant dans la courbe de capacite

`distseal/models/embedder.py:244` :

```python
hidden_size = int(nbits * hidden_size_multiplier)
```

**La taille du reseau qui traite le message est proportionnelle au nombre de
bits** : 64 -> 128 canaux caches, 96 -> 192, 128 -> 256. Et
`cfg.decoder.z_channels = hidden_size + cfg.encoder.z_channels` suit.

La "courbe de capacite" confond donc deux effets opposes : plus de bits a
encoder (plus dur) mais un reseau d'encodage plus gros (plus facile). Ce n'est
pas une experience de capacite propre.

C'est probablement ce qui explique l'anomalie du 96 bits, qui fait moins bien
que 128 a toutes les epoques et a PSNR comparable (verifie : configs identiques
a `nbits` pres). Deux lectures possibles, non tranchees :
1. le gain de taille du reseau l'emporte sur le cout des bits supplementaires
   dans cette plage, et la courbe n'est pas monotone ;
2. le run 96 bits a eu de la malchance (graine), auquel cas un rejeu avec une
   autre graine le remettrait dans le rang.

**A faire avant de publier la courbe** : soit rejouer 96 bits avec une autre
graine (une nuit de GPU), soit fixer `hidden_size` a une valeur constante
independante de `nbits` et rejouer les points -- c'est la seule facon d'obtenir
une vraie courbe de capacite. A defaut, mentionner explicitement le confondant.

## Figure pour le memoire (TikZ, style du chapitre 3)

Deux axes : securite effective (barres, echelle de gauche) et P(succes)
(courbe, echelle de droite). Le croisement materialise la fenetre viable.

```latex
\begin{figure}[h!]
\centering
\begin{tikzpicture}[x=1.9cm, y=0.055cm]
% --- axes
\draw[->] (-0.35,0) -- (4.6,0);
\draw[->] (-0.35,0) -- (-0.35,160);
\foreach \y in {0,25,50,75,100,125,150} {
  \draw (-0.45,\y) -- (-0.35,\y);
  \node[left, font=\scriptsize] at (-0.45,\y) {\y};
  \draw[gray!20, thin] (-0.35,\y) -- (4.5,\y);
}
\node[rotate=90, font=\scriptsize] at (-1.0,80)
     {securite effective (bits)};
% --- barres : securite effective
\foreach \i/\v/\lab in {0/19.6/32, 1/37.7/64, 2/55.6/96, 3/73.4/128, 4/141.3/256} {
  \fill[blue!25] (\i-0.17,0) rectangle (\i+0.17,\v);
  \node[below, font=\scriptsize] at (\i,-5) {\lab};
}
% 64 bits mis en evidence
\fill[blue!70] (1-0.17,0) rectangle (1+0.17,37.7);
% --- courbe : P(succes), echelle 0-1 mappee sur 0-150
\draw[red!75, very thick]
  (1,150.0) -- (2,120.4) -- (3,149.8) -- (4,0.0);
\foreach \i/\v in {1/150.0, 2/120.4, 3/149.8, 4/0.0}
  \fill[red!75] (\i,\v) circle (2.4pt);
\node[right, font=\scriptsize, text=red!75] at (4.05,6) {P(succes)};
% --- seuil de viabilite
\draw[dashed, gray!70] (-0.35,148.5) -- (4.5,148.5);
\node[right, font=\scriptsize, text=gray!70] at (2.6,143) {P(succes) = 0.99};
% --- fenetre viable
\fill[green!18, opacity=0.5] (0.6,0) rectangle (3.35,158);
\node[font=\scriptsize\bfseries, text=green!45!black] at (2.0,166)
     {fenetre viable};
\node[below, font=\scriptsize] at (2.0,-13) {largeur du canal $\Omega$ (bits)};
\end{tikzpicture}
\caption[Choix de la largeur du canal $\Omega$]{Securite effective (barres,
$-\log_2$ de la probabilite de faux positif au seuil AUTHENTIC) et probabilite
de reconnaissance d'une image reellement marquee (courbe rouge, echelle
0--1 sur la meme hauteur). A 256 bits la securite theorique est maximale mais
le canal ne delivre plus aucun verdict ; a 32 bits la securite effective
plafonne sous 20 bits. La fenetre viable se reduit a 64--128 bits ; 64 est
retenu pour sa marge de robustesse (0,0998 contre 0,0616).}
\label{fig:choix-nbits}
\end{figure}
```

Commande de reproduction des chiffres :

```bash
python3 - <<'PY'
from scipy.stats import binom; import math
for n, p in [(32,None),(64,0.9998),(96,0.9236),(128,0.9616),(256,0.6076)]:
    tau = math.floor(0.10*n); pfp = binom.cdf(tau, n, 0.5)
    ps  = binom.cdf(tau, n, 1-p) if p else float('nan')
    print(n, p, tau, round(ps,4), f"{pfp:.2e}", round(-math.log2(pfp),1))
PY
```

---

# PHASE D -- extension generative (etat au 2026-08-23)

## Ce qui est acquis

Le conditionnement du decodeur sur Omega est **implemente et valide
unitairement** : `distseal/ciphermark/conditioner.py`, plus des patches dans
`deps/efficientvit/aecore/{evaluator,trainer}.py` et
`deps/efficientvit/models/efficientvit/dc_ae.py`.

Verifications : identite exacte a l'initialisation (ecart 0.000e+00 sur le vrai
DCAE), independance par image exacte, 12/12 tests unitaires, bornes de
modulation effectives (|gamma| plafonne a gamma_max meme avec des poids x50).

## Les trois essais et ce qu'ils ont appris

| essai | reglage | bit_acc finale | lecture |
|-------|---------|----------------|---------|
| 1 -- `phaseD_smoke_64bits` | lr 1e-4, extractor_weight 1.0 | 0.51 (hasard) | decodeur detruit en 100 pas, perte de reconstruction en HAUSSE |
| 2 -- `phaseD_conditionneur_seul` | decodeur GELE, lr 5e-5 | 0.552 (plafond) | Omega atteint les pixels (~5 sigma) mais plafonne ; image degradee malgre le gel |
| 3 -- `phaseD_borne_degele` | bornes tanh + decodeur degele, 2 LR | 0.625 a mi-parcours, en hausse | premier run sain : bit_acc ET reconstruction s'ameliorent ensemble |

**Enseignement 1 (essai 2).** Avec un decodeur gele, la bit_acc monte a 0.552 et
y reste. Omega atteint donc reellement les pixels -- le mecanisme est valide --
mais la modulation par canal est spatialement UNIFORME, et les filtres spatiaux
etant figes, l'effet net ressemble a un decalage global de couleur et de
contraste. L'extracteur, entraine sur des motifs structures, y voit surtout du
bruit. **La structure spatiale ne peut venir que du decodeur.**

**Enseignement 2 (essai 2).** Le SSIM s'est effondre alors meme que le decodeur
ne pouvait pas bouger. Le coupable n'etait donc pas le fine-tuning mais le
conditionneur, dont rien ne bornait gamma ni beta : l'optimiseur les faisait
grossir pour satisfaire l'extracteur au prix de l'image. D'ou les bornes tanh.

**Enseignement 3 (essai 3).** Bornes + decodeur degele + LR separes : c'est le
premier run ou la bit_acc monte SANS que la reconstruction se degrade. Les deux
progressent ensemble.

## Trois bugs trouves, tous silencieux

Ils meritent une mention en annexe : chacun aurait produit un entrainement
d'apparence normale et un resultat faux.

1. **Ordre de creation avant DDP.** Le conditionneur cree dans le bloc
   watermarker du Trainer serait absent de l'optimiseur (construit plus tot) et
   non synchronise par DDP. Il ne se serait jamais entraine, sans erreur.
2. **`msg.repeat(x.shape[0], 1)` dans `dc_ae.py`.** Avec un Omega (B, nbits) au
   lieu de (1, nbits), produit (B*B, nbits).
3. **`setup_optimizer` jette ses groupes de parametres** quand `no_wd_keys` est
   vide (le defaut) : il repart de `network.parameters()`, un groupe plat. Tout
   LR par groupe etait silencieusement perdu.

## Bug crypto corrige le meme jour

`crypto.py` basculait sur un repli HMAC-CTR quand pycryptodome manquait. Le pod
l'avait, la machine locale non : **les deux calculaient des Omega differents
pour la meme cle**. Une image marquee sur le pod aurait echoue a la verification
en local, sans erreur, juste un mauvais BER qu'on aurait impute au canal.
Corrige : le repli est refuse par defaut, `prg_backend()` expose le generateur
reellement utilise. A journaliser dans toute mesure dependant d'Omega.

Le repli lui-meme n'etait pas faible -- c'est HMAC-DRBG (NIST SP 800-90A) -- le
probleme etait la divergence silencieuse.

## Ce qui reste ouvert sur la phase D

- La bit_acc n'a pas atteint 0.99. L'essai 3 monte encore ; a suivre.
- Si elle replafonne : le suspect est le point stationnaire trivial de la phase
  A, et le remede existe deja dans `train.py`
  (`--msg_curriculum_epochs`, ecrit mais **jamais teste**).
- **La convergence de la boucle de point fixe n'est pas verifiee** quand Omega
  module le decodeur. En post-hoc elle converge en 1 iteration parce que le
  filigrane est une petite perturbation ; ici le rendu entier change. Question
  empirique, bornee par `max_fixed_point_iters`.

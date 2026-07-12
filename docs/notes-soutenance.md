# Notes de compréhension — DistSeal & CipherMark

> Document de révision personnel pour la soutenance. Mis à jour au fil du travail.
> Dernière mise à jour : 12 juillet 2026.

---

## Sommaire

1. [Vocabulaire de base](#1-vocabulaire-de-base)
2. [DistSeal : comment ça marche](#2-distseal--comment-ça-marche) — dont diffusion vs autorégressif, et l'entraînement du duo embedder/extracteur
3. [Où est caché le tatouage ?](#3-où-est-caché-le-tatouage-)
4. [Les failles de DistSeal](#4-les-failles-de-distseal)
5. [CipherMark : les briques et ce qu'elles réparent](#5-ciphermark--les-briques-et-ce-quelles-réparent)
6. [Exemple complet : Alice et le chat roux](#6-exemple-complet--alice-et-le-chat-roux)
7. [Les quatre scénarios de vérification](#7-les-quatre-scénarios-de-vérification)
8. [Tableau de synthèse DistSeal vs CipherMark](#8-tableau-de-synthèse)
9. [Limites assumées et statut des hypothèses](#9-limites-assumées-et-statut-des-hypothèses)
   - 9bis. Genèse du design : les alternatives envisagées (P1–P5)
   - 9ter. Questions probables du jury, réponses courtes
10. [Phrases clés pour l'oral](#10-phrases-clés-pour-loral)
    - Annexe 0 : où trouver le reste (audio, docs, code)
    - Annexe : commandes de test et grille de lecture

---

## 1. Vocabulaire de base

| Terme | Définition simple |
|---|---|
| **Latent** | Résumé compressé d'une image. Image 256×256×3 = 196 608 nombres ; latent 8×8×128 = 8 192 nombres (24× moins). Contient des *concepts* (« poil roux », « cuir rouge »), pas des pixels. Analogie : le latent est la partition, l'image est le concert. |
| **Encodeur / Décodeur** | Réseaux qui font la navette : encodeur = image → latent ; décodeur = latent → image (l'« orchestre » qui joue la partition). |
| **Tatouage (watermark)** | Information invisible cachée dans une image, lisible par une machine. Filigrane statistique réparti sur toute l'image. |
| **Embedder (W_θ)** | Petit réseau (UNet) *appris* qui prend (latent, message) et produit une perturbation δ à ajouter au latent : `z_w = z + ε·δ`. |
| **Extracteur (E_θ)** | Réseau (ConvNeXt) *appris en duo* avec l'embedder, qui lit une image (pixels) et ressort les bits du message. Sort 256 « logits » ; le signe de chacun = un bit. |
| **JND** | *Just Noticeable Difference* : carte indiquant, zone par zone, combien de modification l'œil tolère (beaucoup dans les textures, très peu dans un ciel lisse). Sert à atténuer le tatouage là où il se verrait. |
| **Augmentations** | Attaques simulées pendant l'entraînement (JPEG, crop, flou, bruit) pour forcer la robustesse. |
| **HMAC-SHA256** | « Empreinte digitale à clé » : avale des données + une clé secrète, recrache 256 bits. Infalsifiable sans la clé (~2^128 essais). Standard RFC 2104, utilisé dans TLS/HTTPS. |
| **Effet avalanche** | Changer 1 bit d'entrée d'un hash change ~50 % des bits de sortie, sans aucun rapport avec l'ancien résultat. Force (anti-falsification) ET talon d'Achille (aucune tolérance d'erreur). |
| **PRG / ChaCha20** | Générateur pseudo-aléatoire : à partir d'une graine secrète + un numéro, déroule un flot de bits imprévisible. ChaCha20 = chiffrement par flot standard (TLS 1.3, Google). |
| **Nonce** | *Number used once* : numéro unique par image, garantit qu'un flot de bits n'est jamais réutilisé. C'est la discipline « one-time » du système. |
| **XOR (⊕)** | Addition bit à bit sans retenue (0⊕0=0, 0⊕1=1, 1⊕1=0). Propriété magique : `(a⊕b)⊕b = a`. Si b est aléatoire et à usage unique, `a⊕b` ne révèle RIEN sur a = principe du one-time pad (Shannon, sécurité parfaite prouvée). |
| **DINOv2** | Réseau de vision (Meta, auto-supervisé sur 142 M d'images) : image → vecteur de 384 nombres résumant la *sémantique*. Deux JPEG du même chat → vecteurs quasi identiques. |
| **LSH** | *Locality-Sensitive Hashing* : 256 hyperplans aléatoires fixes ; pour chacun, on note de quel côté tombe le vecteur → 256 bits stables. Un vecteur qui bouge peu reste du même côté de presque tous les plans. |
| **Reed-Solomon (RS)** | Code correcteur d'erreurs (CD, QR codes). 16 octets de parité permettent de réparer jusqu'à **8 octets abîmés** du hash. Attention : 8 *octets*, pas 8 bits — 9 bits dispersés dans 9 octets différents = échec. |
| **Φ⁻¹ (inverse CDF normale)** | Transforme un nombre uniforme [0,1] en valeur gaussienne « en cloche » (ex. 0,67 → +0,44). Sert à rendre la perturbation statistiquement identique au bruit naturel du latent. |
| **Hessienne** | Mesure la courbure du paysage d'entraînement. Direction à forte courbure = direction « rigide » que tout ré-entraînement évite de bouger. Idée : écrire dans le béton, pas dans le sable. |
| **Lanczos** | Algorithme qui trouve les directions propres principales de la Hessienne sans calculer la matrice géante entière. |
| **LoRA** | Technique de fine-tuning léger. Utilisée en attaque (annexe E de DistSeal) pour faire « oublier » le tatouage à un modèle. |
| **Distance de Hamming** | Nombre de positions où deux suites de bits diffèrent. |
| **BER** | *Bit Error Rate* = distance de Hamming / nombre total de bits. |
| **p-valeur** | Probabilité qu'une image NON tatouée obtienne un si bon score par hasard. À 25 bits d'écart sur 256 : ~10⁻⁴⁰. |
| **Point fixe** | Valeur qu'une fonction renvoie inchangée : f(x) = x. Ici : le hash qui ne bouge plus quand on ré-injecte le tatouage. |
| **Kerckhoffs (principe de)** | La sécurité doit reposer uniquement sur le secret des clés, jamais sur le secret du mécanisme. DistSeal le viole ; CipherMark le respecte. |

---

## 2. DistSeal : comment ça marche

Article de référence : Rebuffi et al., *Learning to Watermark in the Latent Space of Generative Models* (Meta, 2026).

### Le problème d'origine

Prouver qu'une image vient d'un modèle génératif donné. Deux familles avant DistSeal :

- **Post-hoc pixels** (WAM, TrustMark) : un réseau séparé signe l'image après génération. Lent (~63 ms CPU) et **contournable** : qui vole les poids du modèle saute l'étape de signature.
- **In-model** (Stable Signature) : le modèle lui-même est modifié pour tout signer. Plus sûr, mais limité à une architecture (diffusion latente).

### L'idée DistSeal en deux étapes

**Étape A — le tatoueur post-hoc latent.** Entraîner en duo un embedder (écrit dans le latent) et un extracteur (lit dans les pixels), sur des millions d'itérations avec attaques simulées + contraintes d'invisibilité (JND, discriminateur). Résultat : un « stylo invisible » appris, robuste, 20× plus rapide que le tatouage pixel.

**Étape B — la distillation.** Rééduquer le modèle par imitation (professeur = modèle+tatoueur, élève = copie du modèle) pour que la signature devienne un réflexe fondu dans les poids. Deux cibles possibles :
- le **décodeur** (traducteur) : plus facile, meilleure qualité, mais remplaçable par un décodeur standard ;
- le **générateur** (cerveau) : plus profond, mais plus difficile et plus coûteux en qualité.

Cas autorégressif (RAR) : tatouage **avant quantization** (dans les jetons → distillable dans le générateur, robustesse ~91 %) ou **après** (plus robuste ~94 %, mais décodeur seulement).

### Chiffres clés à retenir

| Mesure | Valeur |
|---|---|
| Vitesse tatouage latent vs pixel | ~3 ms vs ~63 ms (20×) |
| Robustesse moyenne DC-AE latent vs pixel | 95,18 % vs 97,78 % |
| Point faible : attaques combinées | 84,28 % vs 97,29 % |
| Oubli par LoRA (annexe E, 2 500 pas) | chute à ~0,70–0,82 |

### Diffusion vs autorégressif : les deux mondes couverts

| | Diffusion (DC-AE + UViT-H) | Autorégressif (RAR-XL + MaskGIT-VQGAN) |
|---|---|---|
| Comment il génère | part d'un pur bruit, le débruite par petites touches (le « sculpteur ») dans l'espace latent | écrit l'image comme une phrase : 256 jetons discrets choisis un par un dans un dictionnaire, puis décodés |
| Latent | continu, grille 8×8×128 (compression extrême : « écrire sur un timbre-poste ») | jetons discrets, mais chaque jeton correspond à un vecteur continu à l'intérieur du tokenizer (16×16×256) |
| Où insérer le tatouage | un seul point naturel : sur le latent débruité, juste avant décodage | deux options : **avant quantization** (dans le continu, juste avant l'arrondi → la signature s'écrit dans le choix des jetons) ou **après quantization** (sur les vecteurs sortis du dictionnaire) |
| Compromis | robustesse ~95 % vs ~98 % en pixels, contre 20× la vitesse + distillabilité | avant quant. = distillable dans le générateur mais ~91 % ; après quant. = ~94 % mais décodeur seulement |
| Piège spécifique | néant (continu partout) | « entre le jeton 42 et le jeton 43, il n'y a rien » : on ne peut pas ajouter 0,03 à un mot d'un dictionnaire — d'où le passage par le continu pré-quantization |

**Quantization** : l'arrondi du continu vers le jeton du dictionnaire le plus proche. C'est elle qui écrase les perturbations subtiles (limite A3 de l'article).

### Comment le duo embedder/extracteur est entraîné (Étape A)

Boucle répétée ~600 000 fois (601 époques × 1 000 itérations dans le papier) :
1. tirer un message au hasard → l'embedder écrit dans le latent → décodage en image ;
2. **maltraiter l'image exprès** (augmentations : JPEG, crop, flou, bruit, masques) ;
3. l'extracteur doit relire le message malgré tout (perte de décodage) ;
4. pendant ce temps, la JND atténue là où l'œil verrait, et un **discriminateur** (réseau juge qui tente de distinguer tatouée/non tatouée) force l'invisibilité ;
5. l'intensité ε suit un **curriculum** (`scaling_w`) : forte au début pour amorcer l'apprentissage, réduite ensuite pour l'invisibilité.

Résultat : personne n'a écrit la formule du tatouage — les deux réseaux ont co-inventé une écriture invisible, robuste et lisible. CipherMark hérite de tout cela sans y toucher.

---

## 3. Où est caché le tatouage ?

**Réponse à ne pas rater : le tatouage est *écrit* dans le latent, mais il *existe* dans l'image.**

1. Pendant la génération, on retouche les 8 192 nombres du latent (ex. `1.23 → 1.25`). On ne touche jamais un pixel directement.
2. Le décodeur transforme le latent retouché en image. La retouche se **matérialise** dans les pixels : écarts de ±1-2 sur des valeurs 0-255, répartis sur toute l'image selon un motif structuré, invisible à l'œil.
3. Le détecteur, lui, ne voit que des pixels (il n'a jamais accès au latent). Il lit ce micro-motif.

Analogie : on glisse une note discrète dans la partition ; l'orchestre la joue sans que le public la remarque ; un auditeur entraîné dans la salle sait l'entendre.

Ce mécanisme est **identique** dans DistSeal et CipherMark. Ce qui change : *ce qu'on écrit*, pas *où et comment*.

---

## 4. Les failles de DistSeal

Le « secret » de DistSeal = 64 bits **fixes pour toute la vie du modèle, identiques sur toutes les images, distribués en clair** (fichier .txt) avec le détecteur. Générés par un PRNG non cryptographique, graine par défaut 0. Aucune couche cryptographique dans tout le code (vérifié par grep).

Conséquences :

| Faille | Attaque rendue possible |
|---|---|
| Secret en clair avec le détecteur | quiconque a le détecteur a le secret |
| Même message partout (viol du principe OTP) | **moyennage** : accumuler des milliers d'images fait émerger le motif commun → effacement ou copie |
| Message indépendant du contenu | **rejeu/forge** : recopier le motif sur un faux → déclaré authentique |
| Pas d'attribution | impossible de savoir quel utilisateur a généré quoi |
| Pas de test statistique | seuil arbitraire, pas de p-valeur (le code a `pvalue()` mais ne l'utilise pas) |
| Oubli LoRA | fine-tuning adverse efface le tatouage (annexe E) |

---

## 5. CipherMark : les briques et ce qu'elles réparent

L'équation centrale :

```
Ω = HMAC(K_secret, h) ⊕ PRG(s_master, nonce)
```

C'est **Ω** (256 bits) qui est écrit dans le latent, à la place du message fixe.

| Brique | Rôle | Problème DistSeal réparé |
|---|---|---|
| Deux clés secrètes (s_master, K_secret) | tout le reste peut être public | secret en clair (Kerckhoffs restauré) |
| HMAC(K, h) | certificat infalsifiable lié au **contenu** | rejeu/forge |
| PRG + nonce unique par image | masque jetable, jamais réutilisé | moyennage + confidentialité + attribution |
| Hash perceptuel h (DINOv2+LSH+RS) | empreinte visuelle stable/sensible | liaison au contenu (n'existait pas) |
| Boucle de point fixe | résout la circularité Ω↔h | (problème propre à CipherMark) |
| Projection Π (Hessien/Lanczos) | écrire dans les directions « rigides » | oubli LoRA — **HYPOTHÈSE à valider** |
| Verdict gradué + p-valeur | décision statistique à 4 niveaux | seuil arbitraire, pas de diagnostic |

Ce qui est **réutilisé tel quel** de DistSeal : l'embedder, l'extracteur, la JND, toute la robustesse apprise. CipherMark change ce qu'on écrit, pas le stylo.

Corrections actées suite à la relecture critique (juillet 2026) :
- la vérification **recalcule le hash sur l'image observée** (sinon l'anti-rejeu est cassé — c'était le bug de l'algo 2 du chapitre 2) ;
- boucle d'embedding : latent tiré **une seule fois**, seule la **perturbation** est projetée, sortie quand le hash retombe sur le même mot de code RS ;
- la **parité RS est stockée en BD** (règle le transport génération → vérification) ;
- garanties requalifiées : **computationnelles** (PRF), pas « Shannon » ;
- évaluation : les deux canaux mesurés **séparément**, attaques combinées incluses, LoRA à 2 500 pas comme l'annexe E.

---

## 6. Exemple complet : Alice et le chat roux

Acteurs : PixGen (opérateur, modèle DC-AE+UViT), Alice (utilisatrice n° 17), équipe forensique (vérifieur, détient les clés). On n'affiche que les 16 premiers bits des valeurs de 256 bits.

**Étape 0 (une fois).** `s_master = a35f09…c2`, `K_secret = 7b41e8…9e` → HSM.

**1. Prompt.** Alice : « un chat roux assis sur une valise rouge, photo réaliste ». 384ᵉ génération → `image_id = 583201` → nonce (utilisé une seule fois).

**2. Latent.** Le modèle produit `z` (grille 8×8×128), **gelé** pour toute la suite. Ex : `z[0,0] = [1.23, -0.87, 0.45, …]`.

**3. Empreinte.** Image provisoire x₀ = D(z) → DINOv2 → 384 nombres → LSH → `h = 1011 0010 1101 0001 …` (256 bits). Parité `π = RS_parité(h)` (16 octets, répare ≤ 8 octets).

**4. Certificat.** `T = HMAC(K_secret, h) = 0110 1110 0010 1011 …`

**5. Masque.** `KS = ChaCha20(s_master, 583201) = 1100 0101 1001 1110 …`

**6. XOR.**
```
      T  = 0110 1110 0010 1011
  ⊕  KS  = 1100 0101 1001 1110
  ─────────────────────────────
      Ω  = 1010 1011 1011 0101
```
Sans les clés, Ω = pile-ou-face parfait. Et le moyennage ne donne rien :
`Ω₁ ⊕ Ω₂ = (T₁⊕T₂) ⊕ (KS₁⊕KS₂)` → encore du bruit, rien ne s'accumule.

**7. Injection.** Blocs de 16 bits → gaussiennes : `1010101110110101 = 43957 → 43957/65536 = 0.6707 → Φ⁻¹ ≈ +0.44`. L'embedder produit δ, projection, ajout : `z_w = z + ε·Π(δ)` (ex. `1.23 → 1.25`).

**8. Point fixe.** Décode x_w, re-hashe : h₁ diffère de h sur 3 bits dans 3 octets → RS répare (3 ≤ 8) → h retrouvé exactement. Convergence en 1 itération.

**9. Archivage.** `BD[583201] = (π, Alice, date, modèle)`. La base ne contient **ni h, ni Ω, ni aucun secret**.

---

## 7. Les quatre scénarios de vérification

Procédure du vérifieur (qui a les clés) :
```
1. Ω̂ ← extracteur(image suspecte)                       (bits lus, avec erreurs)
2. ĥ ← PHash(image suspecte), réparé avec π              (empreinte du contenu OBSERVÉ)
3. gauche : Ω̂ ⊕ ChaCha20(s_master, nonce)               (on retire le masque)
4. droite : HMAC(K_secret, ĥ)                            (certificat attendu pour CE contenu)
5. distance de Hamming → BER → verdict + p-valeur
```
Verdicts : ≤10 % AUTHENTIC ; ≤25 % LIGHT_EDIT ; ≤40 % HEAVY_EDIT ; sinon NOT_WATERMARKED.

| Scénario | Canal Ω (souple) | Canal h (dur) | Distance | Verdict |
|---|---|---|---|---|
| **A. JPEG q=80 (Twitter)** | 14 bits d'erreur | 5 bits/4 octets → réparé exactement | 5,5 % | **AUTHENTIC** (p ≈ 10⁻⁵⁴) |
| **B. Rejeu (Bob recopie Ω sur son faux)** | quasi intact | 121 bits/31 octets → irréparable → avalanche | ~51 % | **NOT_WATERMARKED** ✓ voulu |
| **C. Image innocente** | bruit (~50 %) | sans objet | ~50 % | **NOT_WATERMARKED** ✓ (faux positif : p ≈ 10⁻⁴⁰) |
| **D. Crop 70 % + JPEG** | 12 bits d'erreur (tient !) | 23 bits/12 octets → 12 > 8 → irréparable | ~50 % | **NOT_WATERMARKED** ✗ raté = LA FALAISE |

Points clés :
- Le rejeu échoue **par construction** : le certificat volé certifie l'empreinte du chat, pas celle du faux.
- Le système ne fait **jamais de faux positif** (p-valeur) ; son seul mode d'échec est le faux négatif du scénario D.
- Point opérationnel : retrouver le nonce si les métadonnées sont effacées → scan des nonces candidats en base (un XOR + un HMAC par essai, microsecondes), filtré par période/utilisateur.

---

## 8. Tableau de synthèse

| Étape | DistSeal | CipherMark |
|---|---|---|
| Contenu du message | 64 bits arbitraires, fixes, en clair | 256 bits chiffrés (HMAC⊕PRG), uniques/image, liés au contenu |
| Empreinte du contenu | n'existe pas | PHash DINOv2+LSH+RS |
| Où écrire | latent, partout | latent, projeté sur sous-espace stable (hypothèse) |
| Comment écrire | embedder UNet + JND appris | **réutilisé tel quel** (+ Φ⁻¹ en amont) |
| Circularité | sans objet | boucle de point fixe |
| Relire | extracteur ConvNeXt | **réutilisé tel quel** |
| Décider | comparaison au message en clair, seuil arbitraire | équation crypto recalculée + verdict gradué + p-valeur |
| Clés | aucune | 2 clés secrètes ; tout le reste peut être public |
| Distillation | oui (générateur ou décodeur) | **question ouverte** (post-hoc à ce stade) |
| Faiblesse principale | moyennage, rejeu, secret public | effet falaise sur le canal h |

---

## 9. Limites assumées et statut des hypothèses

À dire soi-même avant que le jury ne le demande :

1. **Effet falaise (deux canaux en série).** Ω tolère 10 % d'erreurs ; ĥ doit être réparé EXACTEMENT (RS ≤ 8 octets). Hash trop abîmé = détection morte quel que soit le reste. → L'expérience n° 1 du mémoire (`eval_phash_robustness.py`) mesure la « récupérabilité » du hash par distorsion.
2. **Garanties computationnelles, pas Shannon.** Le keystream vient d'un PRG, pas d'une source parfaite. Héritage OTP = la *discipline d'usage unique* (nonce), pas le secret parfait. Ne JAMAIS dire « inconditionnel ».
3. **L'inforgeabilité couvre le tag, pas l'extracteur.** L'extracteur reste un CNN attaquable adversarialement. Sécurité bout-en-bout = min(crypto, robustesse extracteur).
4. **Sous-espace stable = hypothèse.** Non démontré face à un attaquant adaptatif (Π se calcule depuis les poids publics). Ablation prévue : aléatoire vs hessien vs sans.
5. **Distillation = question ouverte.** CipherMark tel qu'implémenté est post-hoc à l'inférence. Comparaison loyale : vs Gaussian Shading/PRC (post-hoc chiffrés), pas frontalement vs DistSeal distillé.
6. **État de l'art à créditer honnêtement.** Gaussian Shading EST crypto (ChaCha20, preuve « performance-lossless ») ; PRC watermarks ont des preuves d'indétectabilité. Le créneau CipherMark : liaison au contenu par HMAC + vérification relationnelle + usage unique, dans l'écosystème d'un tatoueur latent distillable.
7. **Hashes perceptuels attaquables** (collisions type NeuralHash) : résistance adversariale de DINOv2+LSH à évaluer, pas à supposer.

---

## 9bis. Genèse du design : les alternatives envisagées

Question de jury quasi certaine : *« quelles autres approches avez-vous considérées, et pourquoi celle-ci ? »* Cinq pistes avaient été étudiées avant de converger vers CipherMark :

| Piste | Idée | Devenir |
|---|---|---|
| **P1 — Découplage porteur/identité** | graver un nonce sans valeur, reconstruire l'identité chez le vérifieur par `m = n ⊕ k` | **intégrée** : c'est le terme `⊕ PRG(s_master, nonce)` de l'équation |
| **P2 — Message dépendant du contenu** | message calculé depuis le latent via une fonction à clé | **intégrée** : c'est le terme `HMAC(K_secret, h)` |
| **P3 — Étalement par cellule + codes fontaine** | un masque différent par case du latent + code sans rendement (survit à la perte de cases) | non retenue en cœur (chantier d'entraînement trop lourd pour un M2) ; piste « perspectives » contre les attaques combinées |
| **P4 — Partage de Shamir** | découper la clé de vérification en fragments à seuil (3 sur 5) | complément protocolaire, mentionné comme variante de déploiement (section 8.3 du cours), pas une contribution centrale |
| **P5 — Allocation perceptuelle du pad** | placer les bits de clé selon la capacité JND locale | non retenue seule (gain difficile à isoler) ; la JND reste utilisée en atténuation comme dans DistSeal |

CipherMark = P1 + P2 fusionnées dans une seule équation, plus deux ajouts propres : la liaison à l'image par hash perceptuel (anti-rejeu) et le sous-espace stable (anti-LoRA, hypothèse).

À savoir aussi : les propositions intermédiaires de HI.md (HoloSeal, LieMark, TopoSeal, NLOT, SBOT, SOTNI…) sont des brouillons exploratoires antérieurs — ne pas les citer en soutenance, seul CipherMark est le design final.

---

## 9ter. Questions probables du jury — réponses courtes

- **« Pourquoi ne pas chiffrer simplement le message de DistSeal ? »** Chiffrer un message fixe le rend confidentiel mais toujours identique sur toutes les images : le moyennage et le rejeu restent possibles. Il faut l'usage unique (nonce) ET la liaison au contenu (hash), pas seulement le chiffrement.
- **« Pourquoi HMAC et pas une signature (RSA/EdDSA) ? »** HMAC est symétrique : seul l'opérateur vérifie — cohérent avec notre modèle (vérification forensique interne). La variante asymétrique (EdDSA, vérification publique) existe, au prix de tags plus longs (512 bits) ; c'est une extension documentée.
- **« Que se passe-t-il si la base de données est perdue ? »** On perd la parité RS et le lien nonce→utilisateur, donc l'attribution fine ; mais pas les clés. Une vérification dégradée reste possible si le hash observé est intact (0 octet à corriger).
- **« Et si l'attaquant connaît tout le mécanisme ? »** C'est notre hypothèse de travail (Kerckhoffs) : modèle, détecteur, équation publics ; seules les 2 clés sont secrètes. La seule chose non couverte par cette garantie : l'extracteur CNN lui-même (attaquable adversarialement) et le sous-espace Π (calculable depuis les poids publics — d'où son statut d'hypothèse).
- **« Votre système se distille-t-il comme DistSeal ? »** Non démontré : Ω change à chaque image, or la distillation fige un comportement. C'est LA question ouverte du travail, dite honnêtement ; à ce stade CipherMark est post-hoc côté serveur.
- **« Pourquoi 256 bits et pas 64 ? »** Il faut la largeur d'un tag HMAC-SHA256 ; et plus de bits = p-valeurs plus écrasantes (10⁻⁴⁰ au seuil), au prix d'une charge plus lourde pour l'embedder — compromis mesuré en éval.

---

## 10. Phrases clés pour l'oral

- **Le pitch** : « DistSeal a inventé un stylo invisible remarquable, mais écrit la même phrase publique sur toutes ses images. CipherMark garde le stylo et change ce qu'il écrit : un certificat chiffré, unique par image, scellé au contenu de l'image elle-même. »
- **Où est le tatouage** : « Écrit dans le latent, matérialisé dans les pixels, lu dans les pixels. »
- **Pourquoi OTP** : « DistSeal viole le principe fondateur du one-time pad — une clé, un usage. Le nonce de CipherMark le restaure : aucun keystream n'est jamais réutilisé. »
- **Pourquoi le rejeu échoue** : « Le certificat volé certifie l'empreinte de l'image d'origine, pas celle du faux ; l'avalanche HMAC fait le reste. »
- **La limite, dite en premier** : « Le système est à falaise : il ne se trompe jamais en accusant, mais peut rater un tatouage si le hash perceptuel est trop abîmé. Mesurer cette falaise est le cœur de notre protocole expérimental. »
- **HMAC pour un jury non crypto** : « La même primitive qui sécurise vos connexions bancaires. »

---

## Annexe 0 — Où trouver le reste

| Ressource | Contenu |
|---|---|
| `audio-explications/01-07*.mp3` (+ scripts .txt) | la série audio : 1-4 vue d'ensemble DistSeal, 5 autorégressif, 6 diffusion, 7 les pistes de solution (~35 min) |
| `docs/ciphermark.md` | doc technique : garanties requalifiées, tableau état de l'art honnête (Gaussian Shading, PRC, Tree-Ring, Stable Signature), limites A1–A8 de DistSeal avec sources exactes dans l'article, limites propres à CipherMark |
| `HI.md` | archives des explorations (brouillons antérieurs — ne pas citer tel quel) |
| chapitre 2 du mémoire (`chap2_methodologie.tex`) | version corrigée : algorithmes justes, garanties computationnelles, protocole en 5 phases |
| `distseal/ciphermark/` | l'implémentation : crypto.py, phash.py, witness.py, equation.py, stable_subspace.py, wam_ciphermark.py |

---

## Annexe — Commandes de test (à lancer, résultats à coller ici)

```bash
# 1) tests unitaires (30 s, aucun téléchargement)
PYTHONPATH=. python3 -m tests.ciphermark.run_all

# 2) chaîne crypto bout-à-bout (1 min)
python3 -m scripts.ciphermark.gen_keys --out-dir ./keys
PYTHONPATH=. python3 -m scripts.ciphermark.eval_ciphermark --n 20 --keys-dir ./keys

# 3) expérience n°1 : robustesse du canal hash
#    a. contrôle sans réseau (borne basse, fallback DCT)
PYTHONPATH=. python3 -m scripts.ciphermark.eval_phash_robustness --n 20 --backbone dct
#    b. le run qui compte (DINOv2, vraies images, ~90 Mo au 1er lancement)
PYTHONPATH=. python3 -m scripts.ciphermark.eval_phash_robustness \
    --n 50 --data-dir ./mes_images --csv phash_dino.csv
```

Grille de lecture du run 3b (colonne `recuperable`, phase 1) :
- ≳ 90 % sur jpeg-50, crop-90 et attaques combinées → le concept tient ;
- 50–90 % → augmenter `--rs-nsym` (32 → répare 16 octets) et re-mesurer ;
- ≲ 50 % → binarisation LSH à retravailler ; résultat négatif mais publiable.

*Résultats du run de contrôle DCT (synthétique, 12 images, 9 juillet 2026) : le hash s'effondre sous crop/rotation/luminosité (0 % récupérable) → falaise confirmée en borne basse. C'est attendu : le fallback DCT est faible par construction. La conclusion réelle attend DINOv2.*

### Résultats du 12 juillet 2026 (à jour)

**Tests unitaires : 14/14 OK.** Noyau crypto validé (HMAC déterministe, PRG reproductible, round-trip XOR, verdicts, p-values).

**Chaîne bout-à-bout (`eval_ciphermark`, n=20, clés persistantes `./keys/`) :**
- round-trip : **20/20** — toute image tatouée est reconnue ;
- rejeu : **0/20** — aucun Ω transplanté sur une autre image ne passe ;
- faux positifs sur bruit : **0/50**.

**Protocole canal h — DCT sur 28 images naturelles (`results/phash_dct_reelles.csv`) :**

| Distorsion | Récupérable (≤ 8 octets) |
|---|---|
| identity / noise-0.02 | 100 % |
| jpeg-80 / jpeg-50 | **100 %** (1,3 flip en moyenne !) |
| noise+jpeg-60 | 93 % |
| contrast-1.5 | 57 % |
| crop-90 | 14 % |
| crop-70, rot-5, combinées crop+jpeg | **0 %** |

Lecture : même le faible DCT tient parfaitement sur compression et bruit — le problème est **géométrique et photométrique** (crop, rotation, luminosité), exactement là où les features sémantiques de DINOv2 sont censées être invariantes. L'avalanche est confirmée expérimentalement (1 flip non corrigé → distance tag 50 %).

### Résultats DINOv2 (Colab, 12 juillet 2026) — LE verdict

Trois configurations croisées sur les mêmes 28 images (CSV dans `results/`) :

| Récupérabilité | DCT cap. 8 | DINOv2 cap. 8 | **DINOv2 cap. 16 (retenu)** |
|---|---|---|---|
| jpeg-80 / jpeg-50 | 100 % / 100 % | 36 % / 0 % | **100 % / 71 %** |
| luminosité 0,7 / 1,5 | 21 % / 14 % | 100 % / 36 % | **100 % / 100 %** |
| contraste 1,5 | 57 % | 43 % | **100 %** |
| bruit 0,02 | 100 % | 29 % | 86 % |
| crop-90 / crop-70 | 14 % / 0 % | 14 % / 0 % | 64 % / 29 % |
| rot-5 / rot-15 | 0 % / 7 % | 0 % / 0 % | 29 % / 29 % |
| combinées (crop/rot + jpeg) | 0 % | 0 % | 14–29 % |

**Les trois enseignements à réciter :**

1. **La parité 32 octets (capacité 16) est une nécessité, pas une option** : le hash DINOv2 dérive naturellement de 10-15 octets sous distorsion légère ; à capacité 8 il est inutilisable (jpeg-50 : 0 %). → Adopté comme défaut dans le code.
2. **La fenêtre de discrimination** (l'argument le plus fort du chapitre 3) : dérives légitimes = 3-16 octets ; attaques géométriques = 20-22 ; contenus distincts (rejeu) ≈ 31. La capacité doit couvrir [0-16] sans mordre sur ~31 : 16 est le point d'équilibre. On ne peut PAS monter à 24 pour rattraper la géométrie sans rogner la marge anti-rejeu. Conclusion : **une rotation déplace le hash presque autant qu'un changement de contenu** — limite intrinsèque du hash global, à traiter par resynchronisation géométrique (perspective), pas par plus de parité.
3. **Complémentarité DCT/DINOv2** : le DCT est imbattable sur compression/bruit (basses fréquences, insensibles au JPEG par construction), DINOv2 sur la photométrie (features sémantiques). Aucun ne couvre la géométrie. → Perspective : hash hybride, chaque vue avec sa parité.

**Périmètre de viabilité démontré** : compression (jpeg-80 100 %, jpeg-50 71 %), photométrie (100 %), bruit (86 %) — l'écrasante majorité des transformations qu'une image subit en ligne. **Falaise résiduelle** : crops sévères, rotations, combinées — même zone faible que DistSeal (84 % en combiné), mais en tout-ou-rien chez nous.

Reproductibilité : le run DCT Colab reproduit exactement les chiffres locaux (1,07 flip moyen sur jpeg-80 dans les deux environnements).

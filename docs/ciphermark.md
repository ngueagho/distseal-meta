# CipherMark

Extension cryptographique de DistSeal : le tatouage n'est plus un message
embarque mais une relation verifiable entre la perturbation latente et le
hash perceptuel de l'image :

```
Omega  XOR  PRG(s_master, nonce)  ==  HMAC(K_secret, h)
```

Voir le memoire pour la theorie complete.

## Positionnement (a lire avant de citer)

CipherMark tel qu'implemente ici est une couche **post-hoc** : Omega est
calcule a l'inference (nonce frais par image, hash de l'image generee).
La distillation dans les poids -- ce qui fait la valeur de DistSeal --
reste une **question ouverte** : une cible qui change avec chaque image
n'a pas encore ete demontree distillable. Le memoire doit soit assumer le
cadre post-hoc (et se comparer a Gaussian Shading / PRC, pas a DistSeal
distille), soit traiter la distillation contenu-dependante comme objectif
experimental. Comparer directement CipherMark post-hoc aux tableaux
in-model de DistSeal serait inequitable.

## Architecture (rapide)

```
                +---------------------+
   image x ---->|  PerceptualHash     |---> h (256 bits stables)
                |  DINOv2 + LSH + RS  |
                +---------------------+
                          |
                          v
           +-----------------------------+
           |  WitnessField               |
           |  Omega = HMAC(K, h)         |
           |          XOR PRG(s, nonce)  |
           +-----------------------------+
                          |
                          v
           +-----------------------------+
           |  CipherMarkMsgProcessor     |
           |  bits -> gaussien -> SSE    |
           +-----------------------------+
                          |
                          v
              latent ----> embedder (Wam) ----> imgs_w
```

## Modules

| fichier                                      | role                                            |
| -------------------------------------------- | ----------------------------------------------- |
| `distseal/ciphermark/crypto.py`                 | HMAC-SHA256, ChaCha20, HKDF, bits<->gaussien    |
| `distseal/ciphermark/phash.py`                  | hash perceptuel DINOv2 + LSH + Reed-Solomon     |
| `distseal/ciphermark/witness.py`                | construction et verification de Omega           |
| `distseal/ciphermark/stable_subspace.py`        | Hessien + Lanczos + projecteur Pi_stable        |
| `distseal/ciphermark/equation.py`               | equation CipherMark + verdict + p-value            |
| `distseal/ciphermark/msg_processor.py`          | drop-in pour `MsgProcessor`                     |
| `distseal/ciphermark/wam_ciphermark.py`            | orchestrateur + boucle de point fixe            |

## Quickstart

```bash
# 1) generer une paire de cles
python -m scripts.ciphermark.gen_keys --out-dir ./keys

# 2) test bout-a-bout (sans modele lourd)
python -m scripts.ciphermark.eval_ciphermark --n 20 --keys-dir ./keys

# 3) protocole experimental n.1 : robustesse du canal hash
python -m scripts.ciphermark.eval_phash_robustness --n 50 --csv results_phash.csv

# 4) tests unitaires
PYTHONPATH=. python -m tests.ciphermark.run_all
```

## Garanties (enonces honnetes)

Les garanties sont **computationnelles**, pas information-theoriques. Le
schema n'est pas un one-time pad au sens de Shannon : le keystream vient
d'un PRG (ChaCha20), pas d'une source uniforme parfaite. Ce que le schema
herite de l'OTP est sa **discipline d'usage** : grace au nonce, un
keystream n'est jamais reutilise entre deux images -- c'est precisement
la propriete que DistSeal violait (un meme message fixe sur toutes les
images).

1. **Indistinguabilite computationnelle** -- sous l'hypothese que ChaCha20
   est un PRG sur, Omega est indistinguable d'une suite uniforme pour qui
   ne connait pas `s_master`. Pas de borne de Shannon : un adversaire non
   borne casse le PRG par recherche exhaustive de la graine.

2. **Inforgeabilite du tag** -- produire un Omega valide pour une image
   choisie sans `K_secret` revient a forger HMAC-SHA256 (PRF-securite,
   ~2^128). Attention au perimetre : cette borne couvre la forge du tag,
   **pas** la chaine d'extraction. L'extracteur qui recupere Omega depuis
   les pixels reste un CNN appris, attaquable adversarialement ; la
   securite de bout en bout est min(securite crypto, robustesse de
   l'extracteur).

3. **Stabilite LoRA** -- statut : **hypothese experimentale, non
   demontree**. Le projecteur actuellement branche (`LatentProjector`)
   est aleatoire sur les canaux, pas issu du Hessien ; le mode `hessian`
   (Lanczos, implemente) n'a pas encore ete calcule sur un vrai modele.
   De plus, Pi_stable se derive des poids publics : un attaquant
   boite-blanche peut calculer le meme projecteur et cibler exactement ce
   sous-espace. La revendication defendable est une resistance au
   fine-tuning **non cible**, a valider par ablation (aleatoire vs
   hessien vs sans projection).

## Le point critique : deux canaux en serie

La detection reussit seulement si les deux canaux passent :

* **canal h (dur)** : le hash percu par le verifieur doit etre corrige
  EXACTEMENT vers le hash de generation. Un seul bit non corrige et
  l'avalanche de HMAC rend le tag attendu aleatoire (distance ~50%). Le
  Reed-Solomon `nsym=16` corrige au plus **8 octets** errones -- pas 16
  bits arbitraires : 9 bits disperses dans 9 octets differents suffisent
  a tout perdre. Par ailleurs la parite RS calculee a la generation doit
  etre transportee jusqu'au verifieur (embarquee dans le payload ou en
  metadonnee) : ce canal de transport reste a specifier.
* **canal Omega (souple)** : tolere ~10% d'erreurs binaires (seuil de
  Hamming + p-value).

Consequence : le systeme est "a falaise" (tout ou rien sur h). La
robustesse du phash sous distorsions, y compris combinees, est donc
l'experience n.1 du memoire -- voir `eval_phash_robustness.py` qui mesure
separement le taux de recuperation du hash, l'effet falaise, et le taux
de detection joint.

## Comparaison a l'etat de l'art (version honnete)

| Methode | Espace d'insertion | In-model / distillable | Crypto | Liaison au contenu | Preuves formelles |
|---|---|---|---|---|---|
| Tree-Ring (Wen et al. 2023) | bruit initial diffusion | non | non | non | non |
| Gaussian Shading (Yang et al. 2024) | bruit initial diffusion | non | **oui** (chiffrement par flot) | non | oui (performance-lossless) |
| PRC watermarks (Christ et al. 2024 ; Gunn et al.) | bruit initial | non | **oui** (codes pseudo-aleatoires) | non | oui (indetectabilite) |
| Stable Signature (2023) | poids du decodeur LDM | oui (decodeur) | non | non | non |
| DistSeal (Rebuffi et al. 2026) | latent + poids | oui (generateur ou decodeur) | non | non | non |
| CipherMark | latent (post-hoc a ce stade) | a demontrer | oui (HMAC + ChaCha20) | **oui** (phash) | computationnelles |

Le creneau de nouveaute reellement defendable : **liaison au contenu par
HMAC + verification relationnelle + keystream a usage unique, dans
l'ecosysteme d'un tatoueur latent distillable**. Ne pas pretendre que les
methodes ci-dessus n'ont "aucune crypto" ou "aucun theoreme" -- Gaussian
Shading et les PRC en ont, et un rapporteur le sait.

## Cles

Les cles sont 2 x 32 octets, generees via `os.urandom`. **Ne jamais
committer** les fichiers `s_master.bin` / `k_secret.bin`.

Rotation conseillee tous les 6 mois (a documenter dans le manifeste
operateur).

## Limites de DistSeal original (source : article Rebuffi et al. 2025)

Ces limites sont explicitement reconnues par les auteurs et motivent l'extension CipherMark.

| # | Limite | Source dans l'article |
|---|--------|-----------------------|
| A1 | Le tatouage reste supprimable ; la methode n'est « pas encore assez robuste pour resister a des attaques tres puissantes » (attaques de regeneration). | Sec. 6 « Conclusion et limites », p. 8 |
| A2 | Robustesse en retrait par rapport au post-hoc pixel : 84,28 % vs 97,29 % (DC-AE, Tab. 1) ; 82,35 % vs 93,93 % (RAR, Tab. 2). | Sec. 5.2, Tab. 1–2 |
| A3 | Cas autoregressif contraint par les jetons discrets : l'etape de quantification supprime les perturbations subtiles avant distillation dans le generateur. | Sec. 5.2, Sec. 3.2 |
| A4 | Oubli du tatouage sous fine-tuning LoRA : la precision binaire chute a ~0,70–0,82 apres 2500 etapes sur donnees non tatouees. | Annexe E, Fig. 7 |
| A5 | Distillation dans le decodeur latent vulnerable au message fige (« WM fixe ») : pas de flexibilite de message en cours d'inference. | Tab. 6, Sec. 5.5 |
| A6 | Degradation en multi-tatouage : le DC-AE distille perd 7 a 8 points de precision binaire lorsqu'il est combine a un tatouage post-hoc. | Annexe F, Tab. 11–12 |
| A7 | Compromis qualite/detection dans le decodeur : FID plus eleve, risque de decodage flou pour RAR si λp=0, sur-ajustement si le poids d'extracteur est trop grand. | Sec. 5.5, Tab. 6, Annexe D, Fig. 14 |
| A8 | Capacite limitee du decodeur latent (surtout RAR) : ne reproduit pas les motifs haute frequence du tatoueur pixel enseignant (precision 51,38 % sans perte d'extracteur). | Sec. 5.3.1, Fig. 3 |

**Reponse CipherMark :** A5 est adresse par le message flexible par image
via nonce+HMAC. A4 est *vise* par la projection Pi_stable, statut
hypothese (voir Garanties, point 3).

## Limites propres a CipherMark

* Systeme a falaise sur le canal h (voir section dediee) : la robustesse
  reelle est bornee par la stabilite exacte du phash, a mesurer avant
  toute autre experience.
* La projection SSE actuelle est factorisee sur les canaux du latent
  (`LatentProjector`), initialisee aleatoirement. Pour le vrai Hessien on
  a besoin du modele entraine -- voir `build_projector_from_loss`.
* Le transport de la parite Reed-Solomon du hash (generation ->
  verification) n'est pas encore specifie.
* Le fallback PHash (DCT) est moins robuste que DINOv2 -- a utiliser
  uniquement quand le reseau est indisponible.
* Le mode "no-cover" requiert une iteration de point fixe ; le taux de
  convergence n'a pas encore ete mesure sur un vrai decodeur.
* Les hashes perceptuels ont des attaques par collision connues
  (cf. litterature NeuralHash) : la resistance adversariale du couple
  DINOv2+LSH est a evaluer, pas a supposer.

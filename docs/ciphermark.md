# CipherMark

Extension cryptographique de DistSeal pour tatouer les modeles generatifs avec
garanties de type "one-time pad". Voir le memoire pour la theorie complete.

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
           |  CipherMarkMsgProcessor        |
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

# 3) tests unitaires
PYTHONPATH=. python -m tests.ciphermark.run_all
```

## Garanties

Voir le memoire pour les enonces formels. En resume:

1. **Indistinguabilite Shannon (computationnelle)** -- Omega est uniformement
   distribue, donc une image tatouee est indistinguable d'une image normale
   pour qui ne connait pas `s_master`.

2. **Stabilite LoRA** -- le watermark vit dans `Pi_stable`, sous-espace
   ortho au noyau des perturbations LoRA dominantes.

3. **Inforgeabilite** -- forger un Omega valide sans `K_secret` revient a
   casser HMAC-SHA256 (PRF-securite, ~ 2^128).

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

**Reponse CipherMark :** A4 est adresse par la projection dans le sous-espace Hessien (`Pi_stable`) ; A5 est adresse par le message flexible par image via OTP+HMAC (nonce unique).

## Limites propres a CipherMark

* La projection SSE actuelle est factorisee sur les canaux du latent
  (`LatentProjector`). Pour le vrai Hessien on a besoin du modele entraine
  -- voir `build_projector_from_loss`.
* Le fallback PHash (DCT) est moins robuste que DINOv2 -- a utiliser
  uniquement quand le reseau est indisponible.
* Le mode "no-cover" requiert une iteration de point fixe (1-2 passes
  generalement suffisantes).

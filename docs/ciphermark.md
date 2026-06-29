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

## Limites connues

* La projection SSE actuelle est factorisee sur les canaux du latent
  (`LatentProjector`). Pour le vrai Hessien on a besoin du modele entraine
  -- voir `build_projector_from_loss`.
* Le fallback PHash (DCT) est moins robuste que DINOv2 -- a utiliser
  uniquement quand le reseau est indisponible.
* Le mode "no-cover" requiert une iteration de point fixe (1-2 passes
  generalement suffisantes).

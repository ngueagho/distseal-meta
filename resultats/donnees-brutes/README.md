# Données brutes citées en annexe du mémoire

Ces fichiers sont produits par les scripts de `scripts/ciphermark/`. Le
dossier `runs/` étant exclu du dépôt, ils sont versionnés ici.

| Fichier | Mesure | Script |
|---|---|---|
| `crypto_5M_six_largeurs/crypto_{16,32,64,96,128,256}bits.json`, `bilan.txt` | Noyau cryptographique, 5 000 000 tirages par largeur | `eval_crypto_grande_echelle.py` |
| `attaques_actives_5000.json` | Quatre attaques actives, 5 000 images, modèle de référence | `eval_attaques_actives.py` |
| `mesure_derive_hash_5000.json` | Dérive du hash perceptuel sous distorsion, 5 000 images | `mesure_derive_hash.py` |
| `mesure_collisions_hash_5000.json` | Collisions et séparation du hash, 5 000 scènes COCO | `mesure_collisions_hash.py` |
| `chaine_crypto_100/resultats.json` | Chaîne générative, 100 générations, un couple (d_tag, d_hash) par image | `chaine_distseal_omega_variable.py` |
| `sse_64bits/resultats.json`, `run.log` | Ablation du projecteur hessien, modèle à 64 bits | `sse_hessian_experiment.py` |
| `complements_distseal/A_transfo_*` | Transformations supplémentaires (retournement, teinte, saturation, niveaux de gris, combinées), 1 000 images par canal | `eval_complements_distseal.py --mode transfo` |
| `complements_distseal/B_fid.json`, `.log` | FID relatif, 500 générations U-ViT, décodeur d'origine / affiné sans Ω / avec Ω | `eval_complements_distseal.py --mode fid` |
| `complements_distseal/C_lora_lr{1e-5,1e-4}.json`, `.log` | Réaffinage LoRA du décodeur conditionné, 2 500 pas | `eval_complements_distseal.py --mode lora` |
| `complements_distseal/D_exemples_v2/resultats.json`, `D_exemples_v2.log` | Quatre générations avec et sans Ω (figure d'exemples) | `eval_complements_distseal.py --mode exemples` |

Le verdict complet de la chaîne générative se recalcule ainsi :

```python
import json
r = json.load(open("chaine_crypto_100/resultats.json"))["resultats"]
print(sum(x["crypto"]["d_tag"] <= 6 and x["crypto"]["d_hash"] <= 69 for x in r))  # 97
```

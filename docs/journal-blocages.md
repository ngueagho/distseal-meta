# Journal des blocages -- CipherMark

Trace chronologique de chaque difficulte rencontree depuis le debut du projet
et de ce qui a ete fait pour la resoudre. Mis a jour a chaque nouveau blocage.

---

## Phase A -- diagnostic du plateau bit_acc

### Le plateau bit_acc ~0.50 (des mois de blocage)
**Symptome** : bit_acc bloque au niveau du hasard sur des dizaines d'epoques,
quelle que soit la config testee.
**Diagnostic** : avec des messages uniformement aleatoires, un extracteur qui
emet des logits nuls a un gradient d'esperance exactement nul --
`E[sigmoid(0) - y] = 0.5 - E[y] = 0`. La solution triviale (perte = ln 2 =
0.6931) est un vrai point stationnaire, pas un plateau d'optimisation.
Signature dans les logs : une seule valeur de perte distincte sur 200 epoques
(67 000 iterations), contre 26 valeurs distinctes en 50 epoques pour un
entrainement sain.
**Fix** : `warmup_t` 5->50 et `lr` 5e-4->2e-4. A porte bit_acc de 0.605 a
0.9998 (64 bits).

### `iter_per_epoch` mal compris
**Symptome** : mis a 50 en esperant 10 000 iterations, mais le dataloader
n'avait que 8 batches (16 images / batch 2).
**Cause** : `iter_per_epoch` est un PLAFOND (`train.py:594`,
`if it >= params.iter_per_epoch: break`), pas une cible. Avec 16 images et
batch 2, une epoque vaut toujours 8 iterations.
**Fix** : relever `epochs` a 1250 et aligner `t_initial`.

### Erreur de conception : 256 bits a 64x64
**Symptome** : condition presentee a tort comme "la plus favorable possible".
**Cause** : 256 bits sur une image 64x64 = 16 px/bit, 16x plus dur que la
reference. Erreur de conception, pas de bug.
**Fix** : bascule sur 16 bits a 64x64 (256 px/bit).

### Hypothese de saturation par clamp, DISPROUVEE
**Hypothese testee** : le clamp saturerait les pixels et tuerait le gradient.
**Resultat** : sw=0.5 (17-33% de pixels satures) donne un MEILLEUR resultat
(0.7148) que sw=0.1 (0.6094) -- l'inverse de l'hypothese. "Correction"
appliquee dans le mauvais sens, retiree.

### Runs tues prematurement
**Symptome** : runs declares morts aux epoques 105 et 208.
**Cause** : le run sw=0.5 a en realite culmine a l'epoque 1064. Un `kill` sur
le PID torchrun ne tue pas le processus enfant `train.py`, donc le run
continuait derriere le dos du monitoring.
**Lecon retenue** : aucune lecture avant l'epoque ~1000 n'est interpretable sur
ce corpus. Memorise pour ne plus refaire l'erreur.

### `pgrep -f <pattern>` s'auto-matchait
**Symptome** : faux "PROCESS_DEAD" et faux "phase B terminee".
**Cause** : les scripts de monitoring matchaient leur propre ligne de
commande.
**Fix** : sentinelles coupees en deux chaines (`"STA""TUS_RUN"`), puis
finalement bascule sur la date de modification du log plutot que sur `pgrep`.

---

## Environnement RunPod

### Premier pod Community avec CUDA casse
**Symptome** : `torch.cuda.is_available()` retourne False alors que
`nvidia-smi` fonctionne. "CUDA unknown error".
**Fix** : redemarrage inefficace. Pod supprime et recree
(`qurj4bhtkj2qq7` -> `mm5zidc4x0rfwy`), fonctionne.

### Dependances Python decouvertes une a une
**Symptome** : `wandb`, `onnx`, `onnxsim`, `torchmetrics`, `lpips` manquants
un par un sur un pod neuf -- ~15 min de GPU facture perdues.
**Lecon retenue** : installer `requirements.txt` PLUS la liste complete
d'emblee, jamais au coup par coup.

### `ScalingScheduler` appele avec le mauvais kwarg
**Symptome** : PSNR mesure a 10 dB au lieu de 22.4 dB dans
`eval_recovery_robustness.py`.
**Cause** : appel avec `scaling_sched=` au lieu du bon nom de parametre ->
repli silencieux sur la valeur statique 0.5 au lieu de la valeur effective
(0.1152 a l'epoque 1200).
**Fix** : copie du motif `uoptim.parse_params()` qui fonctionnait deja
ailleurs.

### `scaling_w` statique vs effectif
**Symptome generique** : `wam.blender.scaling_w` est un simple float Python,
pas un buffer de state_dict, donc `setup_model()` le reconstruit depuis la
valeur statique de la config a chaque chargement de checkpoint.
**Fix** : rejouer `uoptim.ScalingScheduler(...).step(epoch)` a chaque script
qui charge un checkpoint. Deja source d'au moins deux mesures faussees avant
d'etre systematise.

### Import chain EfficientViT
**Symptome** : `wandb`, `onnx`, `onnxsim`, `torchmetrics` manquants a l'import
de `deps.efficientvit.*`.
**Fix** : installation groupee. Egalement `pip` ciblait python3.10 alors que
les scripts tournent en 3.12 -> `python3 -m pip install --break-system-packages`.

### `from distseal.augmentation import Augmenter` echoue
**Fix** : le bon chemin est
`from distseal.augmentation.augmenter import Augmenter`.

### Detecteurs Meta : mauvais wrapper
**Symptome** : le modele HuggingFace brut n'a pas `encode_pre_quant`.
**Fix** : passer par `neuralcompression.DCAEf64c128`, pas `load_decoder`.

---

## Le bug crypto : repli silencieux (2026-08-23)

### Divergence PRG entre le pod et la machine locale
**Symptome potentiel, jamais materialise en incident concret mais detecte
avant qu'il ne cause de degats** : `crypto.py` bascule sur un repli
HMAC-SHA256-CTR quand `pycryptodome` est absent. Verifie : le pod avait
pycryptodome (-> ChaCha20), la machine locale non (-> repli). **Meme cle,
meme nonce, Omega DIFFERENT selon la machine.**
**Consequence potentielle si non detecte** : une image marquee sur le pod
aurait echoue a la verification en local, sans erreur -- juste un mauvais BER
qu'on aurait impute au canal ou au modele.
**Fix** : le repli est desormais REFUSE par defaut (`RuntimeError` explicite).
`CIPHERMARK_PRG_FALLBACK=1` le reactive pour les tests hors-ligne uniquement.
`prg_backend()` expose le generateur reellement utilise, a journaliser dans
toute mesure dependant d'Omega. pycryptodome installe partout.
**Verification** : keystream et Omega identiques au bit pres entre pod et
local apres correction (meme empreinte SHA256).
**Note** : le repli lui-meme n'etait pas cryptographiquement faible (c'est
HMAC-DRBG, NIST SP 800-90A) -- le danger etait la divergence silencieuse entre
deux implementations, pas la qualite de l'une d'elles.

---

## Phase D -- conditionnement du decodeur sur Omega

### Piege 1 : ordre de creation avant DDP (attrape avant execution)
**Ou** : creation du conditionneur dans le bloc watermarker du `Trainer`,
l'endroit le plus naturel pour ce code.
**Pourquoi c'est un piege** : ce bloc s'execute APRES `setup_optimizer()` et
apres l'enveloppe DDP, qui capturent toutes deux la liste des parametres a
leur construction. Un conditionneur cree la n'aurait jamais ete vu par
l'optimiseur : il ne se serait jamais entraine, SANS AUCUNE ERREUR pour le
signaler.
**Fix** : deplace dans `evaluator.py`, avant l'enveloppe DDP et avant
`setup_optimizer()`.

### Piege 2 : decouverte des etages du decodeur
**Tentation** : coder en dur la liste de canaux depuis `width_list` de la
config (`[128,256,512,512,1024,1024,2048]`, 7 valeurs).
**Realite mesuree sur le vrai DCAE** : 6 etages seulement s'executent (un a
profondeur nulle n'est jamais appele), et dans l'ordre INVERSE de la config
(qui decrit l'encodeur, pas le decodeur). Une liste en dur aurait mal cible la
plupart des etages, en tombant juste par coincidence sur deux d'entre eux
partageant le meme nombre de canaux -- echec PARTIEL et SILENCIEUX.
**Fix** : `decouvre_etages()` releve les etages par une passe a blanc
(observation des formes reellement produites), jamais par lecture de config.

### Piege 3 : `msg.repeat()` avec Omega par image
**Ou** : `deps/efficientvit/models/efficientvit/dc_ae.py`,
`msg_batch = msg.repeat(x.shape[0], 1)`.
**Cause** : suppose un message unique `(1, nbits)`. Avec un Omega par image
`(B, nbits)`, produit `(B*B, nbits)`.
**Fix** : rendu conditionnel selon `msg.shape[0] == 1`.

### Piege 4 : `setup_optimizer` jetait ses groupes de LR
**Ou** : `deps/efficientvit/aecore/trainer.py`.
**Cause** : les groupes de parametres (avec LR distincts) etaient construits
soigneusement puis JETES si `no_wd_keys` etait vide (le defaut), pour repartir
de `network.parameters()`, un groupe plat. Un `omega_lr` different du LR du
decodeur aurait ete silencieusement ignore.
**Fix** : la branche `net_params` est utilisee des qu'un `omega_lr > 0` est
demande, pas seulement quand `no_wd_keys` est non vide.

### Essai 1 -- decodeur libre, sans borne : ECHEC
**Reglages** : lr=1e-4, extractor_weight=1.0.
**Resultat** : bit_acc reste a 0.51 (hasard). Le decodeur pre-entraine est
detruit en 100 pas -- la perte de reconstruction MONTE (0.126 -> 0.268) au
lieu de descendre.
**Diagnostic initial (partiellement faux)** : attribue a un LR trop fort sur
le decodeur.

### Essai 2 -- decodeur GELE : enseignement partiel
**Objectif** : isoler la question "Omega atteint-il les pixels ?" de "le
decodeur survit-il au fine-tuning ?".
**Resultat** : bit_acc monte a 0.552 (~5 sigma au-dessus du hasard, mecanisme
valide) puis PLAFONNE. Mais le SSIM s'effondre QUAND MEME (0.652 -> 0.125)
alors que le decodeur ne pouvait pas bouger.
**Ce que ca revele** : le diagnostic de l'essai 1 etait incomplet. Le
coupable n'est pas seulement le fine-tuning du decodeur mais le
CONDITIONNEUR lui-meme, dont rien ne bornait gamma et beta -- l'optimiseur les
faisait grossir pour satisfaire l'extracteur au prix de l'image.
**Cause structurelle du plafond** : la modulation par canal est spatialement
UNIFORME ; decodeur gele, ses filtres ne peuvent pas convertir ca en structure
spatiale, et l'extracteur (entraine sur des motifs structures) n'y voit que du
bruit.
**Fix derive** : bornes tanh sur gamma (0.3) et beta (0.1) ; degeler le
decodeur pour qu'il fournisse la structure spatiale.

### Essai 3 -- bornes + decodeur degele, calendrier cosinus
**Resultat** : bit_acc monte a 0.698 (pas 3250) puis s'aplatit legerement a
0.693 (pas 4000). Reconstruction et bit_acc progressent ENFIN ensemble.
**Piege evite de justesse** : le calendrier etait un cosinus s'annulant en fin
de parcours -- impossible de distinguer un vrai plafond d'un simple arret du
calendrier de LR.
**Decision** : relancer a LR CONSTANT plutot que de conclure sur un plafond
peut-etre illusoire.

### Essai 4 -- LR constant, 20 000 pas : la question tranchee
**Resultat** : bit_acc continue de monter jusqu'a **0.9641** (SSIM 0.415).
L'aplatissement de l'essai 3 etait bien un artefact du calendrier, pas un
plafond reel.
**Toujours en dessous du seuil** : 0.99 requis par l'avalanche HMAC, mais
l'ecart s'est considerablement reduit (0.69 -> 0.96).

---

## Extinction de la machine locale et redemarrage du pod (2026-08-24)

### Scratchpad perdu au reboot
**Symptome** : `/tmp/claude-1000/...` vide apres redemarrage de la machine
locale (comportement normal de `/tmp`).
**Fix** : `pod4.sh` (le script d'execution SSH) recree a l'identique depuis
zero.

### `container not found`
**Symptome** : le proxy SSH RunPod authentifie la cle mais repond
`container not found` de facon constante sur trois tentatives.
**Diagnostic** : le pod avait ete arrete ou supprime cote RunPod (cause exacte
non determinee -- possiblement liee a l'extinction de la machine locale qui
gerait une session ouverte, possiblement une politique Community Cloud
independante). Le serveur MCP RunPod etait lui-meme deconnecte, empechant une
verification directe de l'etat du pod.
**Resolution** : l'utilisateur a redemarre le pod manuellement depuis le
tableau de bord RunPod.

### Faux departs pendant le redemarrage du pod
**Symptome** : premieres tentatives de connexion SSH juste apres le
redemarrage utilisateur pendaient indefiniment (0 sortie meme apres 90s) sans
message d'erreur exploitable.
**Cause** : le conteneur etait en cours d'initialisation (CUDA, montage des
volumes). Le comportement a change de "container not found" (refus net) a un
pendillement silencieux (conteneur qui demarre), signe distinctif utile pour
la prochaine fois.
**Fix** : attendre et reessayer avec un timeout genereux plutot que
diagnostiquer un probleme reseau ou de cle.

### `/root` reinitialise, `/workspace` intact
**Bonne nouvelle constatee** : les deux runs en cours (phase D long a 20 000
pas, 96 bits a 1199 epoques) etaient allees jusqu'a LEUR TERME avant la
coupure, et `/workspace` (code, checkpoints, corpus) a integralement survecu
au redemarrage.
**Mauvaise nouvelle** : `/root` a ete reinitialise -- `rclone.conf` disparu,
et TOUS les paquets Python installes en session (`omegaconf`, `wandb`,
`onnx`, `onnxsim`, `torchmetrics`, `lpips`, `pycryptodome`, `reedsolo`,
`diffusers`) ont disparu avec lui, y compris ceux ajoutes tres tot dans le
projet.
**Fix rclone** : config retransferee avec `stty -echo` cote distant et
verification par empreinte MD5 des deux cotes -- le jeton d'acces complet au
Drive n'a jamais transite en clair dans la conversation.
**Fix dependances** : reinstallation groupee de `requirements.txt` + la liste
connue, PLUS `diffusers`, decouvert manquant seulement a la relance (import
chain de `ae_model_zoo.py`). Verifie une par une avant de relancer
l'entrainement.
**Lecon** : sur ce pod, `/root` n'est PAS persistant a travers un redemarrage
complet (contrairement a un simple `restart` de conteneur). A prevoir : soit
documenter la liste complete des paquets a reinstaller a chaque reprise apres
redemarrage dur, soit committer un `requirements-full.txt` exhaustif dans le
depot pour eviter de redecouvrir les manques un par un.

### Reprise de l'entrainement phase D apres coupure
**Verification faite avant de faire confiance au mecanisme** : lecture du
code de `try_resume_from_checkpoint()` avant de relancer, pour confirmer qu'il
restaure bien reseau + optimiseur (les 2 groupes de LR) + scheduler +
global_step + RNG depuis `checkpoint.pt`, et que `max_steps` n'est qu'une
condition d'arret modifiable sans toucher au reste.
**Config de reprise** : `run_dir` volontairement IDENTIQUE (pointe vers
`phaseD_long`, pas un nouveau dossier) pour que la reprise automatique
s'accroche au bon checkpoint ; seul `max_steps` change (20000 -> 40000).
**Verifie apres relance** : la premiere validation post-reprise donne
EXACTEMENT bit_acc=0.9640625, identique au dernier point avant coupure --
reprise bit-exacte confirmee, aucune perte.

---

## Reperes methodologiques retenus (transverses)

- Ne jamais tuer un run avant l'epoque ~1000 sur ce corpus : rien n'y est
  interpretable avant.
- Toujours rejouer `ScalingScheduler.step(epoch)` apres chargement d'un
  checkpoint ; `scaling_w` statique de la config != valeur effective.
- Installer TOUTES les dependances connues d'un coup sur un pod neuf ou
  redemarre, jamais au coup par coup (perte de temps GPU facture).
- Avant tout script touchant a Omega ou au PRG, journaliser
  `crypto.prg_backend()` : deux machines qui divergent la produisent des
  Omega differents pour la meme cle, en silence.
- Avant de coder en dur une structure derivee d'une config (canaux d'un
  decodeur, ordre d'etages...), verifier par observation reelle (passe a
  blanc, hooks) plutot que de faire confiance a la config déclarative.
- Sur ce pod, `/root` ne survit pas a un redemarrage complet ; `/workspace`
  oui. Ne jamais laisser une config ou un secret non versionne unique dans
  `/root` sans sauvegarde.

---

## Test du hash a 256 bits avec Omega a 64 bits (2026-08-24)

### Erreur de conception du test : la mauvaise attaque mesuree
**Symptome** : mon script `test_hash256_omega64.py` concluait
"transplantation 0/40 -- liaison au contenu OPERANTE", alors que
`eval_attaques_actives.py` mesurait 8/8 faux AUTHENTIC dans la meme
configuration a 64 bits.
**Cause** : mon "juge de paix" verifiait une image marquee `j` en lui passant
le nonce de l'image `i`. C'est l'attaque B (rejeu par changement de nonce),
qui echouait deja. La VRAIE transplantation fait l'inverse : elle garde le
nonce et change l'image --
`forgees = (other + (marquees - src))`, verifiees avec `ids` d'origine.
**Ce qui a sauve la mesure** : avoir confronte le chiffre au script de
reference plutot que de le prendre pour argent comptant. Une incoherence
0/8 contre 8/8 dans la meme config ne pouvait pas etre ignoree.
**Lecon** : quand un test "repare" soudainement un bug connu, verifier
d'abord qu'il mesure bien la meme chose que le test qui l'avait revele.

### Metrique aveugle : compter en octets ce qui bascule en bits
**Constat** : la derive au marquage vaut 7.57/8 octets a 64 bits (94.6 %) et
29.57/32 octets a 256 bits (92.4 %) -- allonger le hash ne change presque
rien a la PROPORTION.
**Cause** : Reed-Solomon corrige des OCTETS, mais l'instabilite est au niveau
des BITS (~30 % de bascule au marquage). Si 30 % des bits basculent, un octet
reste intact avec probabilite 0.7^8 = 5.7 %, donc ~94 % des octets different,
QUELLE QUE SOIT la longueur du hash. La metrique en octets sature.
**Consequence** : au niveau des bits, 256 bits separe pourtant bien mieux
(marquage ~77 bits contre ~128 pour une autre image, soit ~6.6 sigma d'ecart
contre ~3.3 sigma a 64 bits). L'information de separation EXISTE, mais un
code correcteur par octets ne sait pas l'exploiter.
**Piste qui en decoule** : comparer les hash par distance de Hamming avec
seuil, au niveau du bit, plutot que de "corriger" par Reed-Solomon.

### La verification legitime s'effondre a 256 bits
**Mesure** : hash 256 bits, nsym=32 -> LEGITIME 2/8 AUTHENTIC (contre 8/8 a
64 bits).
**Cause** : la derive au marquage (29.57 octets) depasse largement la
capacite de correction (16 octets), donc une image honnetement marquee n'est
plus reconnue. Un systeme qui rejette tout est trivialement infalsifiable et
parfaitement inutile -- d'ou la necessite de TOUJOURS mesurer le legitime et
l'attaque cote a cote, jamais l'attaque seule.

### `pkill -f <motif>` s'auto-matche (RECIDIVE)
**Symptome** : deux taches de fond tuees avec le code 144, fichiers de sortie
vides.
**Cause** : `pkill -f verif_legit` place en tete de commande a matche la
ligne de commande du shell qui le portait, se tuant lui-meme avec le test.
**Note** : ce piege etait DEJA consigne dans ce journal (section Phase A,
"pgrep -f <pattern> s'auto-matchait") et a quand meme ete reproduit.
**Fix** : ne pas melanger pkill et lancement dans la meme commande, ou couper
le motif (`"verif""_legit"`).

---

## 2026-08-24/25 — La campagne à 5000 images

Consigne : refaire **tous** les tests, sur au moins 5000 images, parce qu'un
système validé sur 20 échantillons et déclaré « pleinement fonctionnel » ne
prouve rien.

### Le pod perdu, et reconstruit

`mm5zidc4x0rfwy` refuse de redémarrer : *This machine does not have the
resources to deploy your pod*. Pénurie de capacité sur la machine physique, pas
une perte de données. Nouveau pod `ejofaxn095ef8g` — mais **le volume réseau
`16rikircio` n'y était pas attaché** : `/workspace` était un disque local vide
de 30 Go. Tout a été reconstruit : dépôt cloné depuis GitHub puis 8 commits
locaux transférés par *git bundle* en base64 à travers le proxy SSH (tranches
de 3000 octets — le PTY limite une ligne à 4096 en mode canonique), rclone
réinstallé, 7,2 Go de checkpoints retirés du Drive.

**Leçon :** un volume réseau ne se rattache pas tout seul à un nouveau pod. Il
faut partir de *Storage → le volume → Deploy*, sinon la console propose des
machines d'autres datacenters qui ne peuvent pas le monter.

### Cinq scripts incapables de monter en charge

`mesure_derive_hash`, `eval_attaques_actives`, `test_hash256_omega64`,
`reparer_hash`, `full_chain_real_weights_test` chargeaient tout le corpus puis
appelaient `embed()` sur le tenseur entier. Correct à 20 images, OOM garanti à
5000. Re-batchés, en préservant la sémantique : le mixup exige la moyenne des
résidus de TOUTES les images, elle est donc accumulée sur une première passe et
appliquée sur une seconde, pour que l'attaquant reste aussi fort. Et `paires()`
dans `reparer_hash` faisait n(n-1)/2 appels numpy — 12,5 millions d'itérations
à 5000 images ; ramené à un produit matriciel sur l'encodage ±1.

### Trois défauts que seule l'échelle a révélés

**1. Le hash couplé à Ω.** `mesure_derive_hash`, `reparer_hash` et surtout
`full_chain_real_weights_test` construisaient le hash perceptuel à la largeur
d'Ω. Depuis que la vérification compare les hash au lieu de les corriger, le
garde-fou des 128 bits les faisait échouer à l'instanciation : **le test qui
produit le « 5/5 AUTHENTIC » du mémoire ne pouvait plus tourner.** Le hash ne
traverse pas l'image, sa largeur est indépendante. Corrigé par `--hash-bits`,
256 par défaut.

**2. `DEVICE = torch.device("cpu")` en dur** dans `full_chain`. Invisible à 5
images ; à 5000, un seul cœur, 320 images en 1 h 40, soit 29 h pour finir contre
vingt minutes sur GPU. Le petit échantillon ne cachait pas une erreur de
calcul — il cachait un coût.

**3. La métrique en octets est aveugle.** Dès que 30 % des bits basculent, 94 %
des octets diffèrent (0,7⁸ = 5,7 % d'octets intacts). À 5000 images toutes les
conditions se collent à 32/32 : JPEG q30 sort à 23,0 octets contre 31,7 pour une
image étrangère. En bits, ces deux cas sont séparés d'un facteur trois. Cette
métrique ne servait qu'au dimensionnement Reed-Solomon, abandonné.

### Un corpus qui mentait

Premier verdict sur `corpus-colab` : distance minimale nulle entre deux images.
Le corpus contient 997 fichiers pour **332 scènes** — trois recadrages chacune.
Passage à COCO : encore une collision, `ski_chelsea` contre `ski_cat` — deux
noms de la même photo de chat dans les images d'exemple de scikit-image, que
`build_corpus` ajoute systématiquement. Sur 5000 scènes COCO **pures** :
**0 collision, distance minimale 12 bits, entropie effective 251,9/256.**

**Leçon :** compter les collisions sans nommer les paires est inactionnable.
Le script rapporte désormais les vingt paires les plus proches avec l'écart-type
de leurs pixels, ce qui distingue un défaut du hachage d'un doublon de corpus.

### Le résultat qui change le mémoire

Attaques actives sur 5000 images marquées et 5000 vierges :

| attaque | faux AUTHENTIC |
|---|---|
| transplantation | **5 / 5000 (0,10 %)** |
| rejeu de nonce | 0 |
| mixup | 0 |
| collage (deux identités) | 2 et 7 |

**14 faux AUTHENTIC sur 25 000.** À 20 images, ces quatre attaques donnaient
toutes zéro. Le mémoire écrit « aucune des 100 transplantations testées n'est
acceptée » et « aucune attaque n'a produit de faux AUTHENTIC » : **les deux
phrases sont fausses à 5000.**

Le mécanisme est identifié et cohérent : la dérive en bits montre que **0,1 %
des images étrangères passent sous le seuil de 69/256** — exactement le taux de
transplantations acceptées. Ce n'est pas du bruit, c'est le taux d'erreur
intrinsèque du seuil à 27 %. La liaison au contenu n'est pas binaire : elle
laisse passer une image sur mille.

Second point caché par le petit échantillon : **15 images légitimes sur 5000 ne
sont pas reconnues** sur un aller-retour parfaitement honnête, soit 0,3 % de
faux négatifs sans aucune attaque.

### L'injection latente est hors de portée

Demande : entraîner le décodeur dans l'espace latent. Vérifié avant de lancer
plutôt que supposé — `wam.py` fait `preds_w = self.embedder(latent, msgs)`, donc
l'embedder doit accepter le latent. Or **notre embedder attend UN canal** : il
tatoue la luminance. Testé : il refuse un tenseur à 128 canaux
(`expected input[1, 128, 4, 4] to have 1 channels`). `latent_watermarker: true`
planterait à la première convolution.

Le faire tenir demanderait d'entraîner un couple embedder/extracteur *pour* le
latent — une phase A entière à refaire. Voie retenue à la place : conditionner
le décodeur de **SANA** (`dc-ae-f32c32-sana-1.0`), qui est un vrai décodeur
latent de modèle texte→image. La phase D avait montré le principe sur un
autoencodeur ImageNet qui ne se pilote pas.

### Divers

- `eval_phash_robustness` utilise `os.listdir` non récursif : il ne voyait pas
  le sous-dossier `toutes/`. Il avance à ~600 images/heure, ce qui en fait le
  point long de toute la campagne.
- Récidive du `pgrep` qui se trouve lui-même dans sa propre ligne de commande.
  Déjà consigné, refait deux fois. Filtrer sur le binaire, pas sur le motif.
- Un spécificateur de format fautif (`{x:.4f }`, espace parasite) n'aurait
  explosé qu'à l'impression finale, après des heures de calcul. Testé à part.

## 2026-08-28 — La chaîne DistSeal rend un Ω au hasard : deux causes, pas une

Premier essai de la chaîne complète (U-ViT génère le latent, le décodeur
conditionné rend l'image, le vérifieur relit Ω) : **34 erreurs sur 64**, soit
le hasard exact, sur les six classes. Aucun verdict rendu.

Le piège était que tout *avait l'air* de fonctionner : le conditionneur se
chargeait en `strict=True`, ses étages `[2048, 1024, 1024, 512, 512, 256]`
correspondaient, et le PSNR entre le décodage avec et sans témoin tombait à
23 dB — signe apparent d'un tatouage puissant. Le système modulait fort et
n'inscrivait rien.

### Cause 1 — le décodeur affiné n'était pas rechargé

La phase D tourne avec `freeze_decoder: false`. Vérification faite après coup :
**224 tenseurs de décodeur sur 224 diffèrent du DC-AE pré-entraîné**, tandis
que les 162 tenseurs d'encodeur sont inchangés. Le décodeur a donc été appris
*conjointement* au conditionneur — c'est lui qui traduit la modulation FiLM en
motif lisible par l'extracteur.

Mon script ne chargeait que les clés `omega_conditioner.*`. Le conditionneur
pilotait un décodeur qui n'avait jamais appris à l'écouter.

Ce qui m'a induit en erreur : `eval_phaseD_omega.py`, qui lui fonctionne, fait
`model.load_state_dict(sd, strict=False)` sur le modèle **entier**. Le décodeur
y était rechargé sans que ce soit dit nulle part. Un chargement large masquait
une dépendance essentielle.

*Correctif* : `charge_conditionneur()` recharge maintenant les clés
`decoder.*`, compte les tenseurs effectivement modifiés, et s'arrête si aucun
ne bouge ou si le checkpoint n'en porte aucun.

### Cause 2 — la résolution

Une fois la cause 1 corrigée, la mesure isole le reste :

| résolution | latent | erreurs Ω | bit_acc |
|---|---|---|---|
| 256 px | (128, 4, 4) | 2/64 | 0,9688 |
| 512 px | (128, 8, 8) | 35/64 | 0,4531 |

La phase D est entraînée à 256 px ; les générateurs de
`diffusion_model_zoo.py` sont **tous** en 512 px, sans variante 256. Aucune
stratégie de lecture ne rattrape le 512 — natif, redimensionné, recadrage
central, moyenne 2×2 : 39, 39, 33, 38 erreurs sur 64. Ce n'est pas la lecture
qui échoue, c'est l'inscription.

D'où la phase D-512 : repartir du checkpoint 256 px et porter le conditionneur
à 512. La capacité y est plus favorable, non moins — le latent passe de
2048 à 8192 valeurs, soit 128 valeurs par bit contre 32.

### Leçon de méthode

Mon premier test de résolution tournait sur le décodeur non rechargé : il
donnait 27 erreurs à 256 px comme 35 à 512, et **innocentait la résolution à
tort**. Tant qu'une cause connue reste active, aucune autre hypothèse ne peut
être testée. Corriger d'abord ce qu'on sait faux, mesurer ensuite.

### Deux effets de bord de l'étape 0

- Le nettoyage du disque a supprimé `corpus_if` et `corpus-distill`, les
  corpus d'**entraînement**, ainsi que `runs/phaseD_40k/` — la copie archivée
  dans `/workspace/ckpt/` a été vérifiée saine par réévaluation (médiane 1,0).
  Un corpus reconstruit doit rester disjoint de `corpus-eval-12k` : d'où
  l'option `--coco-skip` de `build_corpus`, qui saute les 12000 premières
  scènes de test2017, exactement celles de l'évaluation.
- La section `gdrive_remote` de `rclone.conf` n'a **aucun jeton**. Le remote
  qui fonctionne est `gdrive_local`. Mes commandes échouaient sur ce seul
  détail pendant que le script de sauvegarde, lui, poussait sans problème.

## 2026-08-28 — La phase F s'entraînait sur le corpus d'évaluation

Découvert en préparant l'étape 2, avant de la lancer.

`phaseF_maskgit.yaml` pointe `data_dir` sur `/workspace/corpus-eval-12k`. Le
journal de la phase F le confirme sans ambiguïté : « Train Epoch #29 : 1002 it »
à un lot de 12, soit **12 024 images — exactement les 12 025 du corpus
d'évaluation**.

La phase F s'est donc entraînée sur les images qui servent à la juger. Sa
bit_acc de 0,6462 n'est pas fausse : elle ne mesure simplement pas ce qu'on
croyait. C'est une performance sur données vues, muette sur la généralisation.

### Ce qui rend l'erreur facile à commettre

Le nom `corpus-eval-12k` dit pourtant sa fonction. Mais un `data_dir` se règle
une fois, en tête de config, et ne se relit plus ; et l'entraînement ne se
plaint de rien, puisque des images sont bien là. Rien dans la boucle ne
signale qu'on apprend sur son propre jeu de test.

Corollaire de méthode : un corpus d'évaluation devrait être en lecture seule,
et aucun `data_dir` d'entraînement ne devrait pouvoir le désigner.

### Réparation, et ce qu'elle ne répare pas

`phaseF_reprise_corpus_disjoint.yaml` reprend au pas 28000 sur
`corpus-train512`, disjoint par construction (test2017 en sautant les 12000
premières scènes). `phaseF_maskgit.yaml` est laissé intact : il documente ce
qui a réellement tourné.

Mais les 28000 premiers pas restent entraînés sur le corpus d'évaluation. Le
modèle a vu ces images. **La mesure de référence du mémoire doit être faite sur
un corpus jamais vu**, quoi qu'il arrive — et si l'on veut une phase F
pleinement propre, il faut la reprendre de zéro, soit une dizaine d'heures.
Arbitrage à faire.

### Au passage

`corpus-eval-12k` n'a plus que `toutes/`, sans `train/` ni `val/` :
`ImageFolder` cherche `data_dir/train` et la reprise aurait planté au
démarrage, comme la phase D-512 ce matin. Deuxième effet de bord de l'étape 0.

## 2026-08-28 — Éviction du pod en pleine phase D-512, et quatre relances ratées

Le pod a été évincé vers 13:18, au pas 6000 du run dégelé. Diagnostic par
élimination : disque conteneur retombé de 1,1 Go à 16 Mo, processus disparus,
`/workspace` intact. La charge de l'hôte affichait 780 — machine partagée du
Community Cloud, saturée.

**Ce qui a survécu :** le checkpoint du pas 6000 (6,0 Go, poids et optimiseur),
tout `/workspace`. **Ce qui est parti :** `/root`, `/tmp`, et *tous* les
paquets pip.

### La vraie perte de temps : quatre relances ratées

Chaque relance mourait sur un `ModuleNotFoundError` différent — `omegaconf`,
puis `transformers`/`diffusers`/`accelerate`, puis `torchmetrics`. J'ai deviné
la liste des paquets manquants au lieu de la faire déterminer par la machine.
Pire, `pip install -r requirements.txt` a *désinstallé* certains paquets que
je venais de poser.

Correctif : `runpod/preflight.py` tente l'import réel du trainer, lit le module
absent dans l'exception, l'installe, et recommence — jusqu'à quinze fois. La
liste n'a plus à être connue.

Deuxième piège, plus discret : mes scripts de relance vivaient dans `/tmp`. Ils
ont disparu au redémarrage suivant. Tout ce qui doit survivre à une éviction va
dans `/workspace`.

### Deux défauts de mon propre moniteur

- Il cherchait les signatures d'erreur dans **tout** le journal, qui est ouvert
  en ajout : il s'est déclenché sur le traceback d'une éviction déjà réparée.
  La mort du processus est le signal fiable — un plantage tue le processus.
- Son compteur de paliers n'avançait que d'un cran par sondage, donc
  l'étiquette annonçait « seuil 0,75 franchi » quand la bit_acc valait déjà
  0,898. Il rattrape maintenant en boucle.

### La reprise elle-même

`phaseD512_degele.yaml` passe `resume_schedule` et `resume_optimizer` à `true`.
Ils valaient `false` au démarrage, quand la source était le run gelé à un seul
groupe de paramètres ; désormais `run_dir/checkpoint.pt` vient de la même
phase et porte les deux groupes. Les laisser à `false` aurait remis le compteur
à zéro et jeté 6000 pas de moments d'Adam.

Reprise vérifiée : `global_step=6000`, optimiseur rechargé, première validation
à 0,9297 contre 0,9273 avant la coupure. Rien de perdu.

## 2026-08-28 — Étape 1 close : la chaîne DistSeal rend 6/6 verdicts

Clôture de l'entrée « la chaîne DistSeal rendait un Ω au hasard ».

Après la phase D-512 dégelée (20 000 pas, meilleures validations 0,989 /
0,984 / 0,983), le même test qu'au matin :

| sujet | Ω | p-valeur | PSNR témoin |
|---|---|---|---|
| classe 207 | 0/64 | 5,4e-20 | 26,60 dB |
| renard arctique | 0/64 | 5,4e-20 | 27,09 dB |
| panda géant | 0/64 | 5,4e-20 | 25,53 dB |
| montgolfière | 3/64 | 2,4e-15 | 24,12 dB |
| cheeseburger | 0/64 | 5,4e-20 | 24,21 dB |
| volcan | 3/64 | 2,4e-15 | 25,00 dB |

**0/6 verdicts le matin, 6/6 le soir**, sur la même commande et les mêmes six
classes. Ce qui séparait les deux : le décodeur affiné n'était pas rechargé
(cause 1), et le conditionneur n'avait jamais vu le 512 px qu'impose le
générateur de DistSeal (cause 2).

### Ce que le gel a coûté et appris

L'hypothèse « conditionneur seul » a consommé 5500 pas pour plafonner à 0,649.
Elle n'était pas gratuite : elle a produit le point de départ du run dégelé —
même décodeur, conditionneur déjà adapté à 512 — et surtout elle a établi que
des gains par canal ne peuvent pas compenser un changement d'échelle spatiale
quand le sous-échantillonnage 512→256 précède la détection. C'est un résultat
négatif utilisable dans le mémoire, pas du temps perdu.

### Reste à faire

Six images ne prouvent rien : validation à grande échelle lancée sur 300
classes tirées régulièrement (0, 3, 6, … 897) pour ne pas sélectionner des
sujets favorables.

## 2026-08-28 — L'étape 3a tombe au point stationnaire trivial

Lancé avec la recette complète de DistSeal, l'embedder latent est resté collé à
`ln 2` pendant huit époques, learning rate à pleine valeur :

| époque | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|
| loss_decode | 0,6952 | 0,6927 | 0,6927 | 0,6921 | 0,6928 | 0,6942 | 0,6945 | 0,6927 |

`ln 2 = 0,693147`, bit_acc entre 0,492 et 0,502. C'est le point stationnaire
trivial déjà rencontré : l'extracteur sort des logits nuls, σ(0) = 0,5, et le
gradient moyen vaut 0,5 − E[y] = 0 dès que les messages sont équilibrés. Rien
ne pousse le modèle à en sortir.

### Pourquoi DistSeal n'y tombe pas et nous si

Sa recette met `all_augs_v3` avec `num_augs: 2` et le discriminateur dès
l'époque 0. La tâche est trop dure d'emblée pour que la symétrie se brise — à
notre taille de lot. Lui dispose d'un lot bien plus grand, dont le gradient
moyen est assez peu bruité pour amorcer.

### Ce qui aurait dû m'alerter plus tôt

Toutes les phases qui ont appris sur ce projet — `phaseA1_floor`,
`phaseA1_overfit_clean`, `curriculum_phase1`, `local_phase0` — partagent le
même départ : `identity_only.yaml`, `lambda_d: 0.0`, `disc_start: 999999`. Le
curriculum était déjà établi ici, écrit dans quatre configs, et j'ai lancé
l'étape 3 sans le reprendre. « Reprendre le protocole de DistSeal tel quel »
vaut pour ses hyperparamètres, pas pour ignorer une contrainte d'amorçage que
nos propres runs avaient déjà documentée.

### Critère de décision, fixé d'avance

`etape3_latent_phase0.yaml` ne change qu'une chose : retirer la difficulté du
départ. Le signal à surveiller n'est pas la bit_acc mais `loss_decode`, qui
doit quitter 0,6931. Si elle y reste après 5000 itérations, le curriculum
n'est pas en cause et il faudra chercher ailleurs — amorçage de l'extracteur,
ou `scaling_w`. Le moniteur alerte dans les deux sens.

## 2026-08-29 — Deuxième éviction, et un préflight qui ne couvrait qu'une chaîne

Deuxième éviction en douze heures, vers 00:20. Même signature : disque
conteneur remis à zéro, aucun processus survivant, `/workspace` intact. Les
deux checkpoints ont survécu — phase F au pas 44 000, étape 3a à l'époque 18.

### Ce que le préflight a bien fait

Il a réinstallé **douze paquets** tout seul pour la chaîne `distill.py`. Le
même incident avait coûté quatre relances ratées la veille.

### Ce qu'il ne faisait pas

L'étape 3a est morte au démarrage sur `ModuleNotFoundError: pytorch_msssim`,
alors que le préflight venait d'annoncer douze installations réussies. Il ne
sondait que les imports de `distill.py`, via le module trainer. Or `train.py`
— qui porte les phases A, B, H et l'étape 3 — a une chaîne d'imports
différente.

Corrigé : le préflight éprouve maintenant les **deux** points d'entrée.
`train.py` n'étant pas importable comme un module, on l'exerce par `--help`,
qui exécute tous ses imports sans rien entraîner. La sonde a immédiatement
trouvé trois paquets de plus : `pytorch_msssim`, `lpips`, `tensorboard`.

### La leçon, qui est la même que la veille sous un autre angle

Un outil censé m'éviter de deviner ne vaut que s'il couvre tout ce qu'on lance.
J'avais écrit le préflight en pensant à l'entraînement en cours à ce
moment-là, pas à l'ensemble des points d'entrée du dépôt. Une vérification
partielle qui annonce « imports résolus » est plus trompeuse qu'aucune
vérification, parce qu'elle donne le sentiment d'avoir couvert le sujet.

## 2026-08-29 — Le parallélisme empêchait l'étape 3a de converger

Lancée en parallèle de la phase F, l'étape 3a est sortie du point stationnaire
grâce au curriculum, puis y est retombée :

| époque | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|
| loss_decode | 0,6740 | 0,6746 | 0,6738 | 0,6734 | 0,6848 | 0,6938 | 0,6930 |

`ln 2 = 0,693147`. Le PSNR a suivi la rechute : 17,26 puis 15,08.

### Ce que la chronologie exclut

La reprise après éviction n'y est pour rien. Elle s'est faite à l'époque 18,
proprement — « All keys matched », optimiseur, discriminateur et les deux
ordonnanceurs rechargés — et l'effondrement est survenu **trois époques après**,
aux époques 21 et 22.

### Le diagnostic

L'échappée n'avait creusé que 0,02 sous ln 2 en vingt époques. Le partage du
GPU imposait un lot de 8 au lieu des 16 de DistSeal — la phase F occupait
12,8 Go des 24,5. Un lot deux fois plus petit double le bruit de gradient, et
une échappée de point stationnaire est précisément ce qui y résiste le moins :
le modèle n'était pas assez loin du bassin pour ne pas y être renvoyé.

**Le parallélisme ne ralentissait donc pas seulement l'étape 3a : il
l'empêchait de converger.** Ce n'est pas un arbitrage débit contre latence,
c'est une condition de convergence.

### Décision

Sérialisation. L'étape 3a est arrêtée, la phase F retrouve le GPU entier et
finit plus tôt ; l'étape 3a repartira seule, avec le lot de 16 de DistSeal.
Le run effondré est archivé sous `runs/etape3_latent_phase0_effondre_lot8`
comme pièce du diagnostic, et non effacé.

### Ce qu'il faut en retenir pour la suite

« Lancer en parallèle » n'est pas neutre pour un entraînement : la taille de
lot est un hyperparamètre, pas un réglage d'infrastructure. Toute mise en
parallèle qui la réduit modifie la recette.

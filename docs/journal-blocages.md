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

"""Installe, par essais successifs, ce qu'il faut pour lancer les entrainements.

POURQUOI CE SCRIPT
------------------
Le conteneur du pod perd ses paquets pip a chaque redemarrage, et la liste de
ce qui manque n'est pas stable. La deviner a coute quatre relances ratees le
2026-08-28 (omegaconf, puis transformers et consorts, puis torchmetrics). On
tente donc l'import reel, on lit le module absent dans l'exception, on
l'installe, et on recommence.

POURQUOI IL EPROUVE DEUX CHAINES
--------------------------------
La premiere version ne resolvait que les imports de distill.py, via le module
trainer. Elle a laisse passer pytorch_msssim, dont seul train.py a besoin :
l'etape 3a est morte au demarrage apres une eviction, alors meme que le
preflight venait d'annoncer douze installations reussies.

Les deux points d'entree ont des chaines d'imports differentes ; il faut donc
eprouver les deux. train.py n'etant pas importable comme un module, on
l'exerce par `--help`, qui execute tous ses imports sans rien entrainer.

Deploiement sur le pod : /workspace/preflight.py -- /workspace survit aux
evictions, /root et /tmp non.
"""
import importlib
import re
import subprocess
import sys

RACINE = "/workspace/code-memoire"
# quelques noms pip qui different du nom de module
PIP = {"cv2": "opencv-python-headless", "PIL": "pillow",
       "sklearn": "scikit-learn", "skimage": "scikit-image"}
MAX_ESSAIS = 20


def installe(mod: str) -> bool:
    paquet = PIP.get(mod, mod)
    print(f"  manque {mod} -> pip install {paquet}", flush=True)
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                           paquet]).returncode == 0


def sonde_module(chemin: str):
    """Renvoie le nom du module absent, ou None si l'import passe."""
    try:
        importlib.import_module(chemin)
        return None
    except ModuleNotFoundError as e:
        return e.name.split(".")[0]


def sonde_script(argv: list[str]):
    """Idem, en executant un script : ses imports ne sont pas ceux d'un module."""
    r = subprocess.run([sys.executable] + argv, cwd=RACINE,
                       capture_output=True, text=True)
    if r.returncode == 0:
        return None
    m = re.search(r"No module named '([^']+)'", r.stderr + r.stdout)
    if m:
        return m.group(1).split(".")[0]
    # echec pour une autre raison : ce n'est plus l'affaire du preflight
    print(f"  {argv[0]} echoue sans ModuleNotFoundError :\n{r.stderr[-600:]}")
    sys.exit(1)


def resous(nom: str, sonde) -> None:
    for essai in range(MAX_ESSAIS):
        mod = sonde()
        if mod is None:
            print(f"  {nom} : imports resolus apres {essai} installation(s)")
            return
        if not installe(mod):
            print(f"  ECHEC d'installation pour {nom}")
            sys.exit(1)
    print(f"  {nom} : imports toujours incomplets apres {MAX_ESSAIS} essais")
    sys.exit(1)


def main() -> int:
    sys.path[:0] = [f"{RACINE}/deps", RACINE]
    # distill.py : phases D, E, F, G, I -- passe par le trainer
    resous("distill.py", lambda: sonde_module("deps.efficientvit.aecore.trainer"))
    # train.py : phases A, B, H et etape 3 -- chaine d'imports differente
    resous("train.py", lambda: sonde_script(["train.py", "--help"]))
    # Le zoo de diffusion : encore une troisieme chaine. Il tire
    # torch_fidelity via inception_score, ce que ni distill.py ni train.py ne
    # font -- la chaine crypto a echoue dessus apres une eviction, le
    # 2026-08-31. Les scripts d'evaluation en dependent autant que les
    # entrainements.
    resous("diffusion_model_zoo",
           lambda: sonde_module("deps.efficientvit.diffusion_model_zoo"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

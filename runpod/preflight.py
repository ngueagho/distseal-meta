"""Installe, par essais successifs, ce qu'il faut pour importer le trainer.

Le conteneur du pod perd ses paquets pip a chaque redemarrage, et la liste de
ce qui manque n'est pas stable : la deviner paquet par paquet a coute quatre
relances ratees le 2026-08-28 (omegaconf, puis transformers et consorts, puis
torchmetrics). On tente donc l'import reel, on lit le module absent dans
l'exception, on l'installe, et on recommence.

Deploiement sur le pod : /workspace/preflight.py -- /workspace survit aux
evictions, /root et /tmp non.
"""
import subprocess, sys, importlib

# quelques noms pip qui different du nom de module
PIP = {"cv2": "opencv-python-headless", "PIL": "pillow",
       "sklearn": "scikit-learn", "skimage": "scikit-image"}

sys.path[:0] = ["/workspace/code-memoire/deps", "/workspace/code-memoire"]
for essai in range(15):
    try:
        importlib.import_module("deps.efficientvit.aecore.trainer")
        print(f"imports resolus apres {essai} installation(s)")
        break
    except ModuleNotFoundError as e:
        mod = e.name.split(".")[0]
        paquet = PIP.get(mod, mod)
        print(f"  manque {mod} -> pip install {paquet}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", paquet])
        if r.returncode != 0:
            print(f"  ECHEC d'installation de {paquet}"); sys.exit(1)
else:
    print("toujours des imports manquants apres 15 essais"); sys.exit(1)

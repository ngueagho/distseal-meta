"""
Conditionnement d'un decodeur generatif sur Omega -- phase D de CipherMark.

Le probleme
-----------
DistSeal distille un filigrane dans un decodeur generatif, mais avec un message
FIXE : `deps/efficientvit/aecore/trainer.py:124-127` tire un unique `self.msg`
depuis une graine, et le decodeur entraine ne recoit jamais ce message -- il
doit apprendre a le produire depuis ses seuls poids. C'est le schema Stable
Signature : un identifiant grave dans le reseau, le meme pour tout le monde.

Pour que le message devienne variable par utilisateur -- ce que WOUAF (Kim et
al., CVPR 2024) et WMAdapter (Ci et al., ICML 2025) obtiennent sur des modeles
de diffusion -- il faut CONDITIONNER le decodeur sur le message.

Le mecanisme retenu
-------------------
Modulation affine par canal, facon FiLM, pilotee par un reseau de mapping :

    Omega (B, nbits)  ->  mapping  ->  code w (B, hidden)
    w  ->  tete_i  ->  (gamma_i, beta_i) par canal de l'etage i
    activation  <-  activation * (1 + gamma_i) + beta_i

C'est l'idee de WOUAF (qui module les poids) transposee aux activations : plus
simple, non invasif, et suffisant pour porter l'information du message.

Deux choix de conception qui comptent
-------------------------------------
1. INITIALISATION A L'IDENTITE. La derniere couche de chaque tete est mise a
   zero, donc gamma = beta = 0 au depart et le decodeur conditionne se comporte
   EXACTEMENT comme l'original. Sans cela on detruirait le decodeur pre-entraine
   des la premiere iteration, et l'entrainement partirait d'un point bien pire
   que le point de depart.

2. OMEGA EN {-1, +1} et non {0, 1}. Une entree centree donne des gradients
   mieux conditionnes et evite que la moitie des bits n'apporte aucun signal
   au reseau de mapping.

Usage
-----
    cond = OmegaConditioner(nbits=64, channels=[128,256,512,512,1024,1024,2048])
    cond.attach(decoder, cibles)      # pose les hooks une fois
    with cond.omega(omega_bits):      # (B, nbits), uint8 ou float
        images = decoder(latents)
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["OmegaConditioner"]


class OmegaConditioner(nn.Module):
    def __init__(
        self,
        nbits: int,
        channels: Sequence[int],
        hidden: int = 256,
        n_couches_mapping: int = 3,
        gamma_max: float = 0.0,
        beta_max: float = 0.0,
    ):
        """
        nbits      : largeur d'Omega (64 pour CipherMark, cf. WitnessConfig)
        channels   : nombre de canaux de chaque etage conditionne du decodeur
        hidden     : largeur du code latent produit par le reseau de mapping
        gamma_max  : borne sur la modulation multiplicative. 0 = non bornee.
        beta_max   : borne sur la modulation additive. 0 = non bornee.

        Pourquoi borner
        ---------------
        Mesure du run phaseD_cond_seul : avec un decodeur GELE, la bit_acc est
        montee de 0.501 a 0.536 -- Omega atteint donc bien les pixels -- mais
        le SSIM s'est effondre de 0.652 a 0.181 et la perte de reconstruction
        a MONTE (0.126 -> 0.303). Le decodeur ne pouvant pas bouger, la
        degradation ne peut venir que du conditionneur : rien ne bornait gamma
        ni beta, et l'optimiseur les a fait grossir pour satisfaire
        l'extracteur au prix de l'image.

        Avec des bornes, gamma vit dans [-gamma_max, +gamma_max] (facteur
        multiplicatif dans [1-gamma_max, 1+gamma_max]) et beta dans
        [-beta_max, +beta_max]. La degradation devient impossible au-dela d'un
        seuil choisi, et l'optimiseur doit trouver un filigrane DANS ce
        budget. tanh(0) = 0, donc l'identite exacte a l'initialisation est
        preservee.
        """
        super().__init__()
        if nbits <= 0:
            raise ValueError("nbits doit etre > 0")
        if not channels:
            raise ValueError("channels ne peut pas etre vide")
        if gamma_max < 0 or beta_max < 0:
            raise ValueError("gamma_max et beta_max doivent etre >= 0")

        self.nbits = nbits
        self.channels = list(channels)
        self.gamma_max = float(gamma_max)
        self.beta_max = float(beta_max)

        # reseau de mapping : Omega -> code latent partage
        couches: List[nn.Module] = []
        d_in = nbits
        for _ in range(max(1, n_couches_mapping - 1)):
            couches += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        couches += [nn.Linear(d_in, hidden)]
        self.mapping = nn.Sequential(*couches)

        # une tete par etage : code latent -> (gamma, beta) par canal
        self.tetes = nn.ModuleList([nn.Linear(hidden, 2 * c) for c in self.channels])
        for t in self.tetes:
            # identite au depart : gamma = beta = 0
            nn.init.zeros_(t.weight)
            nn.init.zeros_(t.bias)

        # rempli par omega(), lu par les hooks
        self._modulation: Optional[List[torch.Tensor]] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._index_par_module: Dict[nn.Module, int] = {}

    # ------------------------------------------------------------------ calcul

    def forward(self, omega: torch.Tensor) -> List[torch.Tensor]:
        """omega (B, nbits) -> liste de tenseurs (B, 2*C) par etage."""
        if omega.dim() != 2 or omega.shape[1] != self.nbits:
            raise ValueError(
                f"omega doit etre (B, {self.nbits}), recu {tuple(omega.shape)}")
        # {0,1} -> {-1,+1}, centre
        x = omega.to(dtype=self.mapping[0].weight.dtype) * 2.0 - 1.0
        w = self.mapping(x)
        return [t(w) for t in self.tetes]

    # ------------------------------------------------------------------ hooks

    def attach(self, cibles: Iterable[nn.Module]) -> None:
        """Pose un hook sur chaque module cible. L'ordre doit correspondre a
        celui de `channels`."""
        self.detach()
        cibles = list(cibles)
        if len(cibles) != len(self.channels):
            raise ValueError(
                f"{len(cibles)} cibles pour {len(self.channels)} entrees dans "
                "channels -- les deux listes doivent correspondre, dans l'ordre")
        for i, mod in enumerate(cibles):
            self._index_par_module[mod] = i
            self._handles.append(mod.register_forward_hook(self._hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._index_par_module.clear()

    def _hook(self, module, entree, sortie):
        if self._modulation is None:
            return sortie                      # pas de conditionnement actif
        i = self._index_par_module[module]
        gb = self._modulation[i]
        c = self.channels[i]
        gamma, beta = gb[:, :c], gb[:, c:]
        # bornes optionnelles -- tanh(0) = 0, l'identite a l'init est intacte
        if self.gamma_max > 0:
            gamma = self.gamma_max * torch.tanh(gamma)
        if self.beta_max > 0:
            beta = self.beta_max * torch.tanh(beta)
        if sortie.dim() != 4:
            raise RuntimeError(
                f"le hook attend une activation (B, C, H, W), recu "
                f"{tuple(sortie.shape)} sur l'etage {i}")
        if sortie.shape[1] != c:
            raise RuntimeError(
                f"etage {i} : {sortie.shape[1]} canaux observes, {c} attendus "
                "-- la liste channels ne correspond pas au decodeur")
        if gamma.shape[0] != sortie.shape[0]:
            raise RuntimeError(
                f"lot de {sortie.shape[0]} images mais Omega de taille "
                f"{gamma.shape[0]} -- un Omega par image est attendu")
        g = gamma.unsqueeze(-1).unsqueeze(-1).to(sortie.dtype)
        b = beta.unsqueeze(-1).unsqueeze(-1).to(sortie.dtype)
        return sortie * (1.0 + g) + b

    # ------------------------------------------------------------- contexte

    @contextlib.contextmanager
    def omega(self, omega_bits: torch.Tensor):
        """Active le conditionnement pour la duree du bloc.

        En dehors du bloc les hooks laissent passer l'activation inchangee, ce
        qui permet d'utiliser le meme decodeur avec et sans conditionnement --
        utile pour produire la reference non marquee.
        """
        precedent = self._modulation
        self._modulation = self.forward(omega_bits)
        try:
            yield self
        finally:
            self._modulation = precedent

    # ------------------------------------------------------------- utilitaire

    def n_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters())


def decouvre_etages(decodeur: nn.Module, latent_exemple: torch.Tensor):
    """Trouve les etages du decodeur et leur nombre de canaux, par observation.

    On ne devine pas les noms de modules : on lance une passe a blanc et on
    releve la forme reellement produite par chaque enfant direct du conteneur
    d'etages. C'est robuste aux variantes d'architecture (dc-ae f32/f64,
    maskgit-vqgan...), la ou une liste ecrite en dur casserait en silence.

    Retourne [(nom, module, canaux)], dans l'ordre d'execution.
    """
    conteneur = None
    for nom in ("stages", "blocks", "up_blocks", "layers"):
        if hasattr(decodeur, nom):
            c = getattr(decodeur, nom)
            if isinstance(c, (nn.ModuleList, nn.Sequential)) and len(c) > 0:
                conteneur = c
                break
    candidats = list(conteneur.named_children()) if conteneur is not None \
        else list(decodeur.named_children())

    vus: List[tuple] = []
    handles = []

    def faire_hook(nom, mod):
        def h(_m, _e, s):
            t = s[0] if isinstance(s, (tuple, list)) else s
            if torch.is_tensor(t) and t.dim() == 4:
                vus.append((nom, mod, int(t.shape[1])))
        return h

    for nom, mod in candidats:
        handles.append(mod.register_forward_hook(faire_hook(nom, mod)))
    try:
        with torch.no_grad():
            decodeur(latent_exemple)
    finally:
        for h in handles:
            h.remove()

    # un module appele plusieurs fois ne doit apparaitre qu'une fois
    uniques, deja = [], set()
    for nom, mod, c in vus:
        if id(mod) not in deja:
            deja.add(id(mod))
            uniques.append((nom, mod, c))
    return uniques

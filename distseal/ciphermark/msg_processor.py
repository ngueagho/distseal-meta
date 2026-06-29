"""
Drop-in replacement de distseal.modules.msg_processor.MsgProcessor.

L'API publique (forward(latents, msg), get_random_msg) reste identique
pour ne pas casser le reste de la pipeline. La grosse difference: au lieu
d'apprendre des embeddings, on injecte une perturbation gaussienne
construite cryptographiquement via le WitnessField, puis projetee sur le
sous-espace stable.

NOTE: pour rester compatible avec le code existant, `msg` est ici un
tenseur de bits (B, K) qui represente Omega_bits (champ temoin pre-calcule
par CipherMarkWam). On NE redirige PAS le tirage aleatoire vers OTP ici --
c'est le wrapper CipherMarkWam qui s'occupe de tout le crypto.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from .crypto import bits_to_gaussian
from .stable_subspace import LatentProjector


class CipherMarkMsgProcessor(nn.Module):
    """
    Args:
        nbits:        nombre de bits de Omega (typiquement 256)
        hidden_size:  dimension canal du latent (compat avec MsgProcessor)
        msg_mult:     facteur d'echelle (epsilon)
        proj:         LatentProjector optionnel (Pi_stable factorise)
        block_size:   bits par sample gaussien (16 par defaut)
    """

    def __init__(
        self,
        nbits: int,
        hidden_size: int,
        msg_mult: float = 1.0,
        proj: Optional[LatentProjector] = None,
        block_size: int = 16,
        msg_agg: str = "add",
    ):
        super().__init__()
        if nbits <= 0:
            raise ValueError("CipherMarkMsgProcessor requiert nbits > 0")
        if nbits % block_size != 0:
            raise ValueError(
                f"nbits ({nbits}) doit etre multiple de block_size ({block_size})"
            )

        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_mult = msg_mult
        self.block_size = block_size
        self.n_gauss = nbits // block_size

        # projecteur sur le canal (peut etre rebranche apres SSE-Hessian)
        if proj is None:
            proj = LatentProjector(
                n_channels=hidden_size,
                k=min(self.n_gauss, hidden_size),
                seed=0xC0FFEE,
            )
        self.proj = proj

        # agg additif uniquement (le concat n'a pas de sens crypto)
        if msg_agg != "add":
            raise ValueError("CipherMarkMsgProcessor: seul msg_agg='add' supporte")
        self.msg_agg = msg_agg

        # interface de compat: msg_type est tjs binary
        self.msg_type = "binary"
        self.msg_processor_type = "binary+add"

    # ---------------------------------------------------------- compat API --

    def get_random_msg(self, bsz: int = 1, nb_repetitions: int = 1) -> torch.Tensor:
        """
        Pour la compatibilite. En vrai, CipherMarkWam construira un msg avec
        OTP, mais l'entrainement standard fait des tirages aleatoires.
        """
        if nb_repetitions != 1:
            assert self.nbits % nb_repetitions == 0
            aux = torch.randint(0, 2, (bsz, self.nbits // nb_repetitions))
            return aux.unsqueeze(1).repeat(1, nb_repetitions, 1).view(bsz, self.nbits)
        return torch.randint(0, 2, (bsz, self.nbits))

    # ---------------------------------------------------------- forward -----

    def _bits_to_gauss_torch(self, msg: torch.Tensor) -> torch.Tensor:
        """
        msg: (B, K) en {0, 1}. Retourne (B, n_gauss) gaussiens.

        On passe par numpy (la conversion Phi_inverse est en CPU, pas un
        bottleneck en pratique vu que K est petit).
        """
        msg_np = msg.detach().cpu().numpy().astype(np.uint8)
        out = np.stack(
            [bits_to_gaussian(b, block_size=self.block_size) for b in msg_np],
            axis=0,
        )
        return torch.from_numpy(out).to(device=msg.device, dtype=torch.float32)

    def forward(
        self,
        latents: torch.Tensor,
        msg: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        latents: (B, C, H, W)
        msg    : (B, nbits)  -- attendu binaire {0, 1}
        """
        if self.nbits == 0:
            return latents

        B, C, H, W = latents.shape
        if C != self.hidden_size:
            # garde-fou utile pour ne pas embarquer un mauvais latent
            raise ValueError(
                f"hidden_size attendu={self.hidden_size}, C={C}"
            )

        # 1) bits -> gaussiens (B, n_gauss)
        coords = self._bits_to_gauss_torch(msg)

        # 2) on les "spread" sur la grille HxW: meme contenu repete + offset
        #    petit bruit pour eviter une perturbation purement constante.
        coords_grid = coords.view(B, self.n_gauss, 1, 1).expand(
            B, self.n_gauss, H, W
        ).contiguous()

        # 3) projection sur le sous-espace canal => (B, C, H, W)
        delta = self.proj.embed_from_coords(coords_grid, (B, C, H, W))

        if verbose:
            print(f"[ciphermark-msg] delta std={delta.std().item():.4f}, "
                  f"latents std={latents.std().item():.4f}")

        # 4) injection additive
        out = latents + self.msg_mult * delta
        return out

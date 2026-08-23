"""
Primitives crypto pour CipherMark.

On s'appuie autant que possible sur pycryptodome (ChaCha20, AES) et sur la
stdlib (hmac, hashlib). Si pycryptodome n'est pas dispo on bascule sur un
fallback AES-CTR pur stdlib -- moins rapide mais ca depanne.

Robert -- avril 2026
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import struct
from typing import Optional

import numpy as np
from scipy.stats import norm

try:
    from Crypto.Cipher import ChaCha20 as _ChaCha20  # pycryptodome
    _HAS_PYCRYPTO = True
except ImportError:  # pragma: no cover
    _HAS_PYCRYPTO = False


# ---------------------------------------------------------------------------
# HMAC / HKDF
# ---------------------------------------------------------------------------

def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    """HMAC-SHA256 classique. 32 octets en sortie."""
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key doit etre bytes")
    if not isinstance(msg, (bytes, bytearray)):
        raise TypeError("msg doit etre bytes")
    return _hmac.new(bytes(key), bytes(msg), hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Expand RFC5869. Pas besoin du Extract ici, on a deja un PRK."""
    if length > 255 * 32:
        raise ValueError("longueur demandee trop grande pour HKDF-SHA256")
    out = b""
    t = b""
    counter = 1
    while len(out) < length:
        t = hmac_sha256(prk, t + info + bytes([counter]))
        out += t
        counter += 1
    return out[:length]


# ---------------------------------------------------------------------------
# Pseudo-random generator (ChaCha20 ou fallback)
# ---------------------------------------------------------------------------

def _chacha20_stream(seed: bytes, nonce: bytes, nbytes: int) -> bytes:
    if len(seed) != 32:
        raise ValueError("seed ChaCha20 = 32 octets")
    if len(nonce) != 8:
        # pycryptodome accepte 8 ou 12, on standardise sur 8
        raise ValueError("nonce ChaCha20 = 8 octets")
    if _HAS_PYCRYPTO:
        cipher = _ChaCha20.new(key=seed, nonce=nonce)
        return cipher.encrypt(b"\x00" * nbytes)

    # --- repli HMAC-SHA256 en mode compteur --------------------------------
    # Ce n'est PAS un cryptosysteme faible : c'est la construction HMAC-DRBG
    # (NIST SP 800-90A), un generateur pseudo-aleatoire solide, simplement
    # plus lent que ChaCha20.
    #
    # Le danger n'est pas sa qualite, c'est sa DIVERGENCE. Il produit un
    # keystream different de celui de ChaCha20, donc un Omega different pour
    # la meme cle et le meme nonce. Une image marquee sur une machine equipee
    # de pycryptodome echoue alors a la verification sur une machine qui ne
    # l'a pas -- sans erreur, seulement un mauvais BER, qu'on attribuerait au
    # canal ou au modele. C'est pourquoi le repli est refuse par defaut et
    # doit etre demande explicitement.
    if not os.environ.get("CIPHERMARK_PRG_FALLBACK"):
        raise RuntimeError(
            "pycryptodome est absent : le PRG basculerait sur le repli "
            "HMAC-CTR, qui produit un keystream DIFFERENT de ChaCha20 et donc "
            "un Omega different pour la meme cle. Les marques posees ailleurs "
            "deviendraient inverifiables ici, en silence.\n"
            "  correctif  : pip install pycryptodome\n"
            "  contourner : CIPHERMARK_PRG_FALLBACK=1 (tests hors-ligne "
            "uniquement -- jamais pour produire ou verifier une marque "
            "destinee a une autre machine)")
    blocks = []
    counter = 0
    while sum(len(b) for b in blocks) < nbytes:
        block = hmac_sha256(seed, nonce + struct.pack(">Q", counter))
        blocks.append(block)
        counter += 1
    return b"".join(blocks)[:nbytes]


def prg(seed: bytes, nonce: bytes, nbytes: int) -> bytes:
    """
    Keystream pseudo-aleatoire de `nbytes` octets a partir de (seed, nonce).
    Reutilisable: meme (seed, nonce) -> meme sortie.

    Attention: NE JAMAIS reutiliser le meme nonce avec la meme seed sur deux
    messages differents (cf. cours OTP).
    """
    return _chacha20_stream(seed, nonce, nbytes)


def prg_backend() -> str:
    """Quel generateur sert reellement : "chacha20" ou "hmac-ctr".

    A journaliser dans toute mesure qui depend d'Omega. Deux machines qui ne
    repondent pas la meme chose ne produisent pas les memes marques.
    """
    if _HAS_PYCRYPTO:
        return "chacha20"
    return "hmac-ctr" if os.environ.get("CIPHERMARK_PRG_FALLBACK") else "indisponible"


def random_seed(n: int = 32) -> bytes:
    """Tirage cryptographique d'une seed (utilise os.urandom)."""
    return os.urandom(n)


def make_nonce(image_id: int) -> bytes:
    """Encode un compteur d'image en nonce 8 octets big-endian."""
    return struct.pack(">Q", image_id & ((1 << 64) - 1))


# ---------------------------------------------------------------------------
# Bits <-> octets
# ---------------------------------------------------------------------------

def bytes_to_bits(b: bytes) -> np.ndarray:
    """Decompose des octets en bits {0,1}, ordre MSB-first."""
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Inverse de bytes_to_bits. La longueur doit etre multiple de 8."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8 != 0:
        # pad a droite (rare en pratique mais bon)
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="big").tobytes()


def xor_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR bit a bit."""
    if a.shape != b.shape:
        raise ValueError(f"shapes incompatibles: {a.shape} vs {b.shape}")
    return np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("longueurs differentes pour xor_bytes")
    return bytes(x ^ y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Conversion bits <-> gaussienne (composant central pour l'embedding)
# ---------------------------------------------------------------------------

def bits_to_gaussian(bits: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Transforme une chaine de bits en samples gaussiens standards via la
    fonction inverse de la CDF normale.

    On groupe les bits par blocs de `block_size` (default = 16 -> 65536
    niveaux), on normalise dans (0, 1) en evitant 0 et 1 exacts, puis on
    applique Phi^{-1}.

    Le resultat a (len(bits) // block_size) floats N(0, 1).
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    if n % block_size != 0:
        # on tronque -- en pratique on appelle ca avec une taille bien choisie
        n = (n // block_size) * block_size
        bits = bits[:n]
    blocks = bits.reshape(-1, block_size)

    # bits -> entier non signe (MSB en premier)
    weights = (1 << np.arange(block_size - 1, -1, -1)).astype(np.uint64)
    vals = (blocks.astype(np.uint64) * weights).sum(axis=1)

    # normalise dans (0, 1) ouvert
    u = (vals.astype(np.float64) + 0.5) / float(1 << block_size)
    # securite numerique
    eps = 1e-9
    u = np.clip(u, eps, 1.0 - eps)

    return norm.ppf(u).astype(np.float32)


def gaussian_to_bits(values: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Inverse approximatif de bits_to_gaussian (pour extraction)."""
    values = np.asarray(values, dtype=np.float64)
    u = norm.cdf(values)
    u = np.clip(u, 0.0, 1.0 - 1e-12)
    vals = (u * float(1 << block_size)).astype(np.uint64)

    bits = np.zeros((vals.size, block_size), dtype=np.uint8)
    for i in range(block_size):
        shift = block_size - 1 - i
        bits[:, i] = (vals >> shift) & 1
    return bits.reshape(-1)


# ---------------------------------------------------------------------------
# Petit util de debug
# ---------------------------------------------------------------------------

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return int(np.sum(a != b))


if __name__ == "__main__":
    # petit smoke test
    k = random_seed()
    msg = b"hello ciphermark"
    tag = hmac_sha256(k, msg)
    print("HMAC:", tag.hex()[:32], "...")

    ks = prg(k, make_nonce(42), 64)
    print("keystream:", ks.hex()[:32], "...")

    bits = bytes_to_bits(tag)
    g = bits_to_gaussian(bits, block_size=8)
    print("gauss mean/std:", float(np.mean(g)), float(np.std(g)))

    # round-trip
    back = gaussian_to_bits(g, block_size=8)
    print("round-trip err (bits):", hamming(bits[:back.size], back))

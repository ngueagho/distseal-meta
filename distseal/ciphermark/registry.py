"""
Base de donnees de tracabilite CipherMark.

C'est le seul etat persistant du systeme. Pour chaque generation on stocke
l'entree (nonce, parite Reed-Solomon, metadonnees).

Ce qu'on NE stocke PAS, volontairement :
  * ni h        -- le verifieur le recalcule sur l'image observee, c'est ce
                   qui lie le verdict au contenu ;
  * ni Omega    -- il se reconstruit a partir des cles et du nonce.

La parite seule est necessaire au verifieur : elle lui permet de corriger le
hash observe vers le mot de code de generation, sans jamais lui reveler ce
mot de code.

Discipline OTP : un nonce ne doit JAMAIS servir deux fois avec la meme
s_master, sous peine de reutilisation de keystream. La table impose donc
l'unicite du nonce au niveau du schema, et `put` refuse un doublon par
defaut.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional


class NonceReuseError(RuntimeError):
    """Leve quand on tente de reutiliser un nonce deja enregistre."""


@dataclass(frozen=True)
class TraceEntry:
    nonce: int
    parity: bytes
    n_bits: int
    rs_nsym: int
    session: Optional[str]
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    nonce      INTEGER PRIMARY KEY,
    parity     BLOB    NOT NULL,
    n_bits     INTEGER NOT NULL,
    rs_nsym    INTEGER NOT NULL,
    session    TEXT,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session);
"""


class TraceRegistry:
    """
    Registre nonce -> parite, adosse a SQLite (stdlib, pas de dependance).

    Usage :

        with TraceRegistry("traces.db") as reg:
            reg.put(nonce=42, parity=pi, n_bits=256, rs_nsym=32)
            entry = reg.get(42)

    Passer ":memory:" donne un registre ephemere (tests).
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------- ecriture --

    def put(
        self,
        nonce: int,
        parity: bytes,
        n_bits: int,
        rs_nsym: int,
        session: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Enregistre la parite associee a un nonce.

        Leve NonceReuseError si le nonce existe deja et overwrite=False.
        C'est volontaire : un nonce reutilise casse la garantie OTP, et on
        prefere un echec bruyant a un keystream reutilise en silence.
        """
        if not isinstance(parity, (bytes, bytearray)):
            raise TypeError("parity doit etre bytes")
        if nonce < 0:
            raise ValueError("nonce doit etre positif")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            if overwrite:
                self._conn.execute(
                    "INSERT OR REPLACE INTO traces "
                    "(nonce, parity, n_bits, rs_nsym, session, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (nonce, bytes(parity), n_bits, rs_nsym, session, now),
                )
            else:
                self._conn.execute(
                    "INSERT INTO traces "
                    "(nonce, parity, n_bits, rs_nsym, session, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (nonce, bytes(parity), n_bits, rs_nsym, session, now),
                )
        except sqlite3.IntegrityError as exc:
            raise NonceReuseError(
                f"nonce {nonce} deja enregistre : reutiliser un nonce avec la "
                f"meme s_master reutilise le keystream (cf. discipline OTP)"
            ) from exc
        self._conn.commit()

    def put_many(self, entries: list, session: Optional[str] = None) -> None:
        """entries: liste de tuples (nonce, parity, n_bits, rs_nsym)."""
        for nonce, parity, n_bits, rs_nsym in entries:
            self.put(nonce, parity, n_bits, rs_nsym, session=session)

    # ------------------------------------------------------------- lecture ---

    def get(self, nonce: int) -> Optional[TraceEntry]:
        row = self._conn.execute(
            "SELECT * FROM traces WHERE nonce = ?", (nonce,)
        ).fetchone()
        if row is None:
            return None
        return TraceEntry(
            nonce=row["nonce"],
            parity=bytes(row["parity"]),
            n_bits=row["n_bits"],
            rs_nsym=row["rs_nsym"],
            session=row["session"],
            created_at=row["created_at"],
        )

    def parity_for(self, nonce: int) -> bytes:
        """Raccourci verifieur. Leve KeyError si le nonce est inconnu."""
        entry = self.get(nonce)
        if entry is None:
            raise KeyError(f"nonce {nonce} absent du registre")
        return entry.parity

    def next_free_nonce(self) -> int:
        """Plus grand nonce enregistre + 1 (0 si la table est vide)."""
        row = self._conn.execute("SELECT MAX(nonce) AS m FROM traces").fetchone()
        return 0 if row["m"] is None else int(row["m"]) + 1

    def sessions(self) -> list:
        rows = self._conn.execute(
            "SELECT DISTINCT session FROM traces WHERE session IS NOT NULL"
        ).fetchall()
        return [r["session"] for r in rows]

    # ------------------------------------------------------------- dunder ----

    def __contains__(self, nonce: int) -> bool:
        return self.get(nonce) is not None

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM traces").fetchone()
        return int(row["n"])

    def __iter__(self) -> Iterator[TraceEntry]:
        for row in self._conn.execute("SELECT nonce FROM traces ORDER BY nonce"):
            entry = self.get(row["nonce"])
            if entry is not None:
                yield entry

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TraceRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

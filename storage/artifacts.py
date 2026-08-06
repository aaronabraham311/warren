"""Content-addressed storage for immutable primary-filing artifacts.

SQLite stores only a manifest and relative artifact key. Large source documents and
derived text live under ``WARREN_FILINGS_DIR`` (``local/filings`` by default), keyed by
their SHA-256 digest so mirrors deduplicate and changed upstream content never overwrites
the evidence used by an earlier analysis.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FILINGS_DIR = Path("local/filings")

_MIME_EXTENSIONS: dict[str, str] = {
    "application/json": ".json",
    "application/pdf": ".pdf",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
}
_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactIntegrityError(RuntimeError):
    """Stored bytes do not match their content-addressed key."""


@dataclass(frozen=True)
class StoredArtifact:
    sha256: str
    relative_key: str
    byte_length: int | None
    mime_type: str


class ArtifactStore:
    """Persist and read immutable artifacts beneath a single configured root."""

    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("WARREN_FILINGS_DIR")
        self.root = Path(configured) if root is None and configured else root or DEFAULT_FILINGS_DIR

    @staticmethod
    def _extension(mime_type: str) -> str:
        normalized = mime_type.split(";", 1)[0].strip().lower()
        try:
            return _MIME_EXTENSIONS[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported filing artifact MIME type: {mime_type}") from exc

    def put(self, content: bytes, *, mime_type: str) -> StoredArtifact:
        """Atomically persist *content*, returning its stable relative manifest key."""
        checksum = hashlib.sha256(content).hexdigest()
        extension = self._extension(mime_type)
        relative = Path(checksum[:2]) / f"{checksum}{extension}"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            self._verify(target, checksum)
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{checksum}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as temporary:
                    temporary.write(content)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, target)
            finally:
                temporary_path = Path(temporary_name)
                if temporary_path.exists():
                    temporary_path.unlink()

        return StoredArtifact(
            sha256=checksum,
            relative_key=relative.as_posix(),
            byte_length=len(content),
            mime_type=mime_type.split(";", 1)[0].strip().lower(),
        )

    def read(self, artifact: StoredArtifact, *, max_bytes: int | None = None) -> bytes:
        """Read an artifact after checking its safe key, checksum, and recorded size."""
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        expected_key = self.relative_key(artifact.sha256, artifact.mime_type)
        if artifact.relative_key != expected_key:
            raise ArtifactIntegrityError("Artifact key does not match checksum and MIME type")
        path = self.root / expected_key
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise ArtifactIntegrityError("Artifact exceeds the configured read limit")
        content = self._verify(path, artifact.sha256)
        if artifact.byte_length is not None and len(content) != artifact.byte_length:
            raise ArtifactIntegrityError(
                "Artifact byte length mismatch: "
                f"expected {artifact.byte_length}, got {len(content)}"
            )
        return content

    @classmethod
    def relative_key(cls, checksum: str, mime_type: str) -> str:
        if not _CHECKSUM_RE.fullmatch(checksum):
            raise ValueError("Artifact checksum must be a lowercase SHA-256 hex digest")
        return (Path(checksum[:2]) / f"{checksum}{cls._extension(mime_type)}").as_posix()

    @staticmethod
    def _verify(path: Path, expected_checksum: str) -> bytes:
        if not path.is_file():
            raise ArtifactIntegrityError(f"Artifact is missing: {path}")
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected_checksum:
            raise ArtifactIntegrityError(
                f"Artifact checksum mismatch for {path}: expected {expected_checksum}, got {actual}"
            )
        return content

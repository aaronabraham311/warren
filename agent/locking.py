"""Process-safe, non-blocking lock shared by batch and interactive runs."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO


@dataclass(frozen=True, slots=True)
class RunLockMetadata:
    pid: int
    run_id: str
    mode: str
    started_at: str


class RunLockHeldError(Exception):
    """Raised when another process owns Warren's single-run lock."""

    def __init__(self, metadata: RunLockMetadata | None) -> None:
        self.metadata = metadata
        if metadata is None:
            detail = "owner metadata unavailable"
        else:
            detail = (
                f"pid {metadata.pid}, run {metadata.run_id}, mode {metadata.mode}, "
                f"started {metadata.started_at}"
            )
        super().__init__(f"Another Warren run is active ({detail})")


class _RunLockLease:
    def __init__(self, path: Path, fh: TextIO, metadata: RunLockMetadata) -> None:
        self.path = path
        self._fh = fh
        self.metadata = metadata
        self._released = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        # Clear stale display metadata while still holding the inode. Keep the lock file
        # itself: unlinking it around unlock can allow two processes to lock two inodes.
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except OSError:
            # Metadata is informational; releasing the kernel lock is authoritative.
            pass
        finally:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._released = True


class RunLock:
    """Repository-local advisory lock for write/run operations.

    ``acquire`` is intentionally non-blocking. A dead process automatically releases
    its kernel lock; the next caller safely overwrites any stale metadata.
    """

    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / "active-run.lock"

    def acquire(self, *, run_id: str, mode: str) -> _RunLockLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            metadata = self._read_metadata(fh)
            fh.close()
            raise RunLockHeldError(metadata) from exc

        metadata = RunLockMetadata(
            pid=os.getpid(),
            run_id=run_id,
            mode=mode,
            started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        try:
            fh.seek(0)
            fh.truncate()
            json.dump(asdict(metadata), fh, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        except BaseException:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
            raise
        return _RunLockLease(self.path, fh, metadata)

    @staticmethod
    def _read_metadata(fh: TextIO) -> RunLockMetadata | None:
        try:
            fh.seek(0)
            raw = json.load(fh)
            if not isinstance(raw, dict):
                return None
            pid = raw.get("pid")
            run_id = raw.get("run_id")
            mode = raw.get("mode")
            started_at = raw.get("started_at")
            if (
                not isinstance(pid, int)
                or isinstance(pid, bool)
                or not isinstance(run_id, str)
                or not isinstance(mode, str)
                or not isinstance(started_at, str)
            ):
                return None
            return RunLockMetadata(pid, run_id, mode, started_at)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

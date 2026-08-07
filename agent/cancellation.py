"""Cooperative cancellation primitives for Warren runs.

Cancellation is deliberately checked only at safe boundaries.  A first interrupt can
request cancellation without tearing down an in-flight filesystem write; callers then
raise :class:`RunCancelledError` before starting the next expensive operation.
"""

from __future__ import annotations

from threading import Event


class RunCancelledError(Exception):
    """Raised at a safe checkpoint after cancellation has been requested."""


class CancellationToken:
    """A small thread-safe cancellation token shared by one run."""

    def __init__(self) -> None:
        self._requested = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._requested.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation. Calling this repeatedly is harmless."""
        self._requested.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RunCancelledError("Run cancelled")


class NeverCancelToken(CancellationToken):
    """Compatibility token for batch callers that do not support cancellation."""

    @property
    def is_cancelled(self) -> bool:
        return False

    def cancel(self) -> None:
        """Ignore cancellation requests."""

    def raise_if_cancelled(self) -> None:
        return

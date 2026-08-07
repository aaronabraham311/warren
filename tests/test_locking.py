import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.locking import RunLock, RunLockHeldError


def test_lock_is_nonblocking_and_exposes_active_run_metadata(tmp_path: Path) -> None:
    first = RunLock(tmp_path)
    second = RunLock(tmp_path)
    with first.acquire(run_id="run-one", mode="gem_hunt") as lease:
        assert lease.metadata.run_id == "run-one"
        with pytest.raises(RunLockHeldError) as error:
            second.acquire(run_id="run-two", mode="tickers")
        assert error.value.metadata is not None
        assert error.value.metadata.run_id == "run-one"
        assert error.value.metadata.mode == "gem_hunt"
        assert "run-one" in str(error.value)


def test_released_lock_can_be_reacquired_and_clears_metadata(tmp_path: Path) -> None:
    lock = RunLock(tmp_path)
    lease = lock.acquire(run_id="old", mode="tickers")
    lease.release()
    lease.release()
    assert lock.path.read_text() == ""
    with lock.acquire(run_id="new", mode="portfolio") as current:
        saved = json.loads(lock.path.read_text())
        assert saved["run_id"] == "new"
        assert current.metadata.mode == "portfolio"


def test_stale_or_corrupt_metadata_is_overwritten_when_kernel_lock_is_free(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active-run.lock"
    path.write_text("not-json")
    with RunLock(tmp_path).acquire(run_id="recovered", mode="discovery"):
        saved = json.loads(path.read_text())
        assert saved["run_id"] == "recovered"


def test_release_failure_still_unlocks_and_closes(tmp_path: Path) -> None:
    lock = RunLock(tmp_path)
    lease = lock.acquire(run_id="first", mode="tickers")
    with patch("agent.locking.os.fsync", side_effect=OSError("disk error")):
        lease.release()
    with lock.acquire(run_id="second", mode="portfolio") as next_lease:
        assert next_lease.metadata.run_id == "second"

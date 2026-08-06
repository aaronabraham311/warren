#!/usr/bin/env python3
"""Deterministic PTY smoke test for Warren's interactive terminal."""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import tempfile
import termios
import time
from pathlib import Path

ANSI_RE = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
SGR_RE = re.compile(rb"\x1b\[([0-9;]*)m")


def _plain(output: bytes) -> str:
    return ANSI_RE.sub(b"", output).decode("utf-8", errors="replace").replace("\r", "")


def _has_color_sgr(output: bytes) -> bool:
    color_codes = {*range(30, 38), *range(40, 48), 38, 48, *range(90, 108)}
    for match in SGR_RE.finditer(output):
        parameters = {int(value) for value in match.group(1).split(b";") if value}
        if parameters & color_codes:
            return True
    return False


def _read_until(fd: int, needle: str, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if not readable:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output.extend(chunk)
        if needle in _plain(bytes(output)):
            return bytes(output)
    raise TimeoutError(f"timed out waiting for {needle!r}; transcript:\n{_plain(bytes(output))}")


def _set_size(fd: int, columns: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, columns, 0, 0))


def run_smoke(*, columns: int, no_color: bool, timeout: float) -> str:
    repo_root = Path(__file__).resolve().parents[4]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not available on PATH")

    with tempfile.TemporaryDirectory(prefix="warren-terminal-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        environment = os.environ.copy()
        environment.update(
            {
                "TERM": "xterm-256color",
                "PROMPT_TOOLKIT_NO_CPR": "1",
                "WARREN_DB": str(temp_root / "warren.db"),
                "WARREN_LOGS_DIR": str(temp_root / "logs" / "runs"),
                "WARREN_STATE_DIR": str(temp_root / "state"),
            }
        )
        if no_color:
            environment["NO_COLOR"] = "1"
        else:
            environment.pop("NO_COLOR", None)

        master_fd, slave_fd = pty.openpty()
        _set_size(slave_fd, columns)
        process = subprocess.Popen(
            [uv, "run", "warren"],
            cwd=repo_root,
            env=environment,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        transcript = bytearray()
        try:
            transcript.extend(_read_until(master_fd, "warren ›", timeout))
            os.write(master_fd, b"/help\r")
            transcript.extend(_read_until(master_fd, "Reference", timeout))
            os.write(master_fd, b"/quit\r")
            process.wait(timeout=timeout)
            while True:
                readable, _, _ = select.select([master_fd], [], [], 0)
                if not readable:
                    break
                try:
                    chunk = os.read(master_fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                transcript.extend(chunk)
        finally:
            os.close(master_fd)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    raw = bytes(transcript)
    plain = _plain(raw)
    required = ("Starting Warren…", "Warren", "Commands", "Analyze AAPL", "/history")
    missing = [value for value in required if value not in plain]
    if missing:
        raise AssertionError(f"missing terminal output {missing!r}; transcript:\n{plain}")
    forbidden = ("alembic.runtime.migration", "Running upgrade", "Traceback")
    present = [value for value in forbidden if value in plain]
    if present:
        raise AssertionError(f"unexpected terminal output {present!r}; transcript:\n{plain}")
    if process.returncode != 0:
        raise AssertionError(f"warren exited {process.returncode}; transcript:\n{plain}")
    if not no_color and b"\x1b[" not in raw:
        raise AssertionError("color smoke produced no ANSI styling")
    if no_color and _has_color_sgr(raw):
        raise AssertionError("NO_COLOR smoke produced ANSI color styling")
    return plain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.columns < 40:
        parser.error("--columns must be at least 40")
    print(run_smoke(columns=args.columns, no_color=args.no_color, timeout=args.timeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

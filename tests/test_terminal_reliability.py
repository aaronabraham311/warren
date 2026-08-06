from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent

import pexpect
import pyte
import pytest
from hypothesis import given
from hypothesis import strategies as st
from syrupy.assertion import SnapshotAssertion

from agent.activity import ActivityState, reduce_activity
from agent.events import (
    LlmCallStarted,
    RunCompleted,
    RunFailed,
    RunStarted,
    TickerStarted,
    ToolCallCompleted,
)
from agent.terminal.reliability import FakeClock, TerminalScenario


def test_reducer_tracks_external_wait_counts_and_terminal_outcome() -> None:
    state = reduce_activity(ActivityState(), RunStarted("run-1"), now=1.0)
    state = reduce_activity(state, TickerStarted("run-1", "AMD"), now=2.0)
    state = reduce_activity(
        state,
        LlmCallStarted("run-1", "AMD", "sonnet", "synthesis", 1, 0),
        now=3.0,
    )

    assert state.operation == "model"
    assert state.external_wait is True
    assert state.operation_age(18.0) == 15.0

    state = reduce_activity(
        state,
        ToolCallCompleted("run-1", "AMD", "get_quote", "ok", False, 12, 0),
        now=19.0,
    )
    state = reduce_activity(state, RunCompleted("run-1", "success", 0.01, 20.0), now=20.0)
    assert state.completed_tools == 1
    assert state.outcome == "completed"
    assert state.operation is None

    late = reduce_activity(state, RunFailed("run-1", "late failure"), now=21.0)
    assert late == state


def test_new_run_resets_terminal_state_but_unrelated_run_is_ignored() -> None:
    completed = reduce_activity(
        ActivityState(run_id="old"),
        RunCompleted("old", "success", 0.0, 1.0),
        now=2.0,
    )
    assert reduce_activity(completed, TickerStarted("other", "AAPL"), now=3.0) == completed

    fresh = reduce_activity(completed, RunStarted("new", tickers=("MSFT",)), now=4.0)
    assert fresh.run_id == "new"
    assert fresh.outcome is None
    assert fresh.completed_tools == 0


@given(
    started_at=st.floats(min_value=0, max_value=1_000_000, allow_nan=False),
    samples=st.lists(
        st.floats(min_value=0, max_value=10_000, allow_nan=False),
        min_size=1,
        max_size=30,
    ),
)
def test_elapsed_time_never_decreases(started_at: float, samples: list[float]) -> None:
    state = ActivityState(operation_started_at=started_at)
    observed: list[float] = []
    for elapsed in sorted(samples):
        age = state.operation_age(started_at + elapsed)
        assert age is not None
        observed.append(age)
    assert observed == sorted(observed)


def test_fake_clock_rejects_backwards_time() -> None:
    with pytest.raises(ValueError, match="cannot move backwards"):
        FakeClock().advance(-0.01)


def test_semantic_screen_checkpoint_captures_visible_wait_and_cursor(
    snapshot: SnapshotAssertion,
) -> None:
    scenario = TerminalScenario(width=48, height=8).start()
    try:
        checkpoint = (
            scenario.emit(RunStarted("run-1", "tickers", ("AMD",)))
            .emit(TickerStarted("run-1", "AMD"))
            .emit(ToolCallCompleted("run-1", "AMD", "get_quote", "ok", False, 178, 0))
            .emit(LlmCallStarted("run-1", "AMD", "sonnet", "synthesis", 4, 1))
            .advance(16)
            .checkpoint("model-wait-16s")
        )
    finally:
        scenario.close()

    semantic = {
        "name": checkpoint.name,
        "cells": checkpoint.cells,
        "cursor": checkpoint.cursor,
        "cursor_visible": checkpoint.cursor_visible,
        "scrollback": checkpoint.scrollback,
        "activity": asdict(checkpoint.activity),
    }
    assert "Waiting for model response · AMD" in "\n".join(checkpoint.cells)
    assert semantic == snapshot(name="model-wait-48x8")


@pytest.mark.parametrize("width", [32, 40, 47, 48, 60, 72, 80, 120])
def test_required_widths_keep_an_active_model_row_visible(width: int) -> None:
    scenario = TerminalScenario(width=width, height=8).start()
    try:
        checkpoint = (
            scenario.emit(LlmCallStarted("run-1", "AMD", "sonnet", "synthesis", 4, 7))
            .advance(46)
            .checkpoint(f"model-wait-{width}")
        )
    finally:
        scenario.close()

    visible = "\n".join(checkpoint.cells)
    assert "AMD" in visible
    assert any(token in visible for token in ("waiting", "Still waiting"))
    assert checkpoint.cursor_visible is False


def test_pipe_mode_is_plain_and_control_free() -> None:
    scenario = TerminalScenario(width=40, height=8, tty=False, color="never").start()
    try:
        checkpoint = (
            scenario.emit(LlmCallStarted("run-1", "AMD", "sonnet", "planning", 1, 0))
            .advance(90)
            .checkpoint("pipe")
        )
    finally:
        scenario.close()

    assert "\x1b" not in checkpoint.stdout + checkpoint.stderr
    assert "Planning research · AMD" in checkpoint.stderr


def test_resize_path_preserves_visible_activity() -> None:
    scenario = TerminalScenario(width=80, height=8).start()
    try:
        scenario.emit(LlmCallStarted("run-1", "AMD", "sonnet", "synthesis", 4, 3))
        narrow = scenario.advance(16).resize(32).checkpoint("narrow")
        wide = scenario.resize(120).advance(30).checkpoint("wide")
    finally:
        scenario.close()

    assert "AMD" in "\n".join(narrow.cells)
    assert "AMD" in "\n".join(wide.cells)


@pytest.mark.skipif(os.name == "nt", reason="PTY lifecycle is POSIX-specific")
def test_real_pty_asserts_intermediate_visible_screen_and_cursor_cleanup() -> None:
    child_program = dedent(
        """
        import sys
        from agent.events import LlmCallStarted
        from agent.terminal.renderer import TerminalRenderer

        renderer = TerminalRenderer(
            stdout=sys.stdout,
            stderr=sys.stderr,
            color="always",
            animation=True,
            width=80,
        )
        with renderer.activity("Preparing analysis…"):
            renderer.emit(
                LlmCallStarted("run-pty", "AMD", "sonnet", "synthesis", 4, 7)
            )
            sys.stdin.readline()
        """
    )
    environment = os.environ.copy()
    environment.pop("NO_COLOR", None)
    environment["TERM"] = "xterm-256color"
    child = pexpect.spawn(
        sys.executable,
        ["-c", child_program],
        cwd=str(Path(__file__).parents[1]),
        env=environment,
        dimensions=(8, 80),
        encoding=None,
        timeout=0.1,
    )
    screen = pyte.HistoryScreen(80, 8, history=100)
    stream = pyte.ByteStream(screen)
    deadline = time.monotonic() + 3.0
    try:
        while time.monotonic() < deadline:
            try:
                chunk = child.read_nonblocking(size=16_384, timeout=0.1)
            except pexpect.TIMEOUT:
                continue
            stream.feed(chunk)
            visible = "\n".join(line.rstrip() for line in screen.display)
            if "Synthesizing analysis · AMD" in visible:
                break
        else:
            pytest.fail("model activity did not become visible before the PTY deadline")

        assert screen.cursor.hidden is True
        child.sendline(b"")
        child.expect(pexpect.EOF, timeout=3)
        stream.feed(child.before)
        assert screen.cursor.hidden is False
    finally:
        child.close(force=True)

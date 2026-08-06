"""Full-screen, read-only viewer for a completed single-ticker analysis."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style

from agent.models import AnalysisOutput
from agent.service import RunResult
from agent.terminal.renderer import NAVY_BRAND, NAVY_MUTED, NAVY_STRONG, sanitize_terminal_text

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*[-*+]\s+")
_ORDERED_BULLET = re.compile(r"^\s*\d+[.)]\s+")
_LINK = re.compile(r"\[([^]]+)]\(([^)]+)\)")
_EMPHASIS = re.compile(r"(?<!\w)(?:\*\*|__|\*|_)(.+?)(?:\*\*|__|\*|_)(?!\w)")


@dataclass(frozen=True, slots=True)
class BinderBlock:
    """A titled, presentation-only section in a binder page."""

    title: str
    body: str


@dataclass(frozen=True, slots=True)
class BinderPage:
    """One keyboard-selectable page in the result binder."""

    label: str
    title: str
    blocks: tuple[BinderBlock, ...]


@dataclass(frozen=True, slots=True)
class BinderDocument:
    """Sanitized report data consumed by the interactive viewer."""

    ticker: str
    recommendation: str
    confidence: str
    pages: tuple[BinderPage, ...]


def _clean_markdown(value: object) -> str:
    """Remove presentation Markdown while preserving the model's wording."""

    safe = sanitize_terminal_text(value)
    lines: list[str] = []
    for raw_line in safe.splitlines():
        line = _HEADING.sub("", raw_line)
        if _BULLET.match(line) or _ORDERED_BULLET.match(line):
            line = _BULLET.sub("• ", line)
            line = _ORDERED_BULLET.sub("• ", line)
        line = _LINK.sub(r"\1 (\2)", line)
        line = line.replace("```", "").replace("`", "")
        previous = None
        while previous != line:
            previous = line
            line = _EMPHASIS.sub(r"\1", line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _items(values: list[str], *, empty: str = "None reported.") -> str:
    cleaned = [_clean_markdown(value) for value in values]
    return (
        "\n".join(value if value.startswith("• ") else f"• {value}" for value in cleaned if value)
        or empty
    )


def _lead(thesis: str) -> str:
    cleaned = _clean_markdown(thesis)
    paragraph = cleaned.split("\n\n", maxsplit=1)[0]
    return " ".join(line.strip() for line in paragraph.splitlines() if line.strip())


def _analysis_from(result: RunResult) -> AnalysisOutput:
    analyses = result.analyses
    if len(analyses) != 1:
        raise ValueError("the result binder requires exactly one completed analysis")
    return analyses[0]


def build_binder_document(
    result: RunResult,
    *,
    evidence_tools: tuple[str, ...] = (),
) -> BinderDocument:
    """Build the fixed five-page binder without model calls or mutable state."""

    analysis = _analysis_from(result)
    ticker = sanitize_terminal_text(analysis.ticker)
    recommendation = sanitize_terminal_text(analysis.recommendation).upper()
    confidence = f"{analysis.confidence:.0%}"
    risk = _clean_markdown(analysis.key_risks[0])
    termination = sanitize_terminal_text(analysis.termination_reason)

    summary_blocks = [
        BinderBlock("Overall recommendation", f"{recommendation} · {confidence} confidence"),
        BinderBlock("Why", _lead(analysis.thesis)),
        BinderBlock("Key risk", risk),
        BinderBlock(
            "Run",
            f"{sanitize_terminal_text(result.run_id)} · {result.total_tool_calls} tools · "
            f"{result.duration_seconds:.1f}s · ${result.total_cost_usd:.4f}",
        ),
    ]
    if termination != "success":
        summary_blocks.insert(3, BinderBlock("Termination", termination))

    signal_blocks: tuple[BinderBlock, ...] = (
        BinderBlock("Lynch · strengths", _items(analysis.lynch_signals.pros)),
        BinderBlock("Lynch · concerns", _items(analysis.lynch_signals.cons)),
        BinderBlock("Buffett · strengths", _items(analysis.buffett_signals.pros)),
        BinderBlock("Buffett · concerns", _items(analysis.buffett_signals.cons)),
    )
    if analysis.dirt_decision is not None:
        decision = analysis.dirt_decision
        signal_blocks += (
            BinderBlock(
                "DIRT decision",
                f"{sanitize_terminal_text(decision.outcome).upper()} · weighted IRR "
                f"{decision.probability_weighted_irr:.1%} · required entry "
                f"{sanitize_terminal_text(decision.currency)} "
                f"{decision.required_entry_price:,.2f}",
            ),
        )

    references: list[str] = []
    if analysis.dirt_signals is not None:
        references = list(analysis.dirt_signals.forensic_evidence_ids)
    evidence_blocks = (
        BinderBlock("Agent tools", _items(list(evidence_tools))),
        BinderBlock("References", _items(references)),
        BinderBlock("Data quality", _items(analysis.data_quality_notes)),
        BinderBlock(
            "Run metadata",
            f"Run {sanitize_terminal_text(result.run_id)}\n"
            f"Tokens {result.total_input_tokens} in / {result.total_output_tokens} out\n"
            f"Agent tools {result.total_tool_calls}\n"
            f"Duration {result.duration_seconds:.1f}s\n"
            f"Cost ${result.total_cost_usd:.4f}",
        ),
    )
    pages = (
        BinderPage("Summary", "Overall recommendation", tuple(summary_blocks)),
        BinderPage(
            "Thesis",
            "Investment thesis",
            (BinderBlock("Thesis", _clean_markdown(analysis.thesis)),),
        ),
        BinderPage("Signals", "Investment signals", signal_blocks),
        BinderPage(
            "Risks",
            "Key risks",
            (BinderBlock("Risks", _items(analysis.key_risks)),),
        ),
        BinderPage("Evidence", "Evidence and run details", evidence_blocks),
    )
    return BinderDocument(ticker, recommendation, confidence, pages)


def _page_text(page: BinderPage) -> str:
    parts = [page.title]
    for block in page.blocks:
        parts.extend(("", block.title, block.body))
    return "\n".join(parts)


class ResultBinder:
    """Temporary, full-screen, read-only result viewer."""

    def __init__(self, *, input: Input | None = None, output: Output | None = None) -> None:
        self._input = input
        self._output = output

    def run(self, document: BinderDocument) -> None:
        """Show the report until the user closes it, restoring terminal state."""

        active = 0
        help_visible = False
        bindings = KeyBindings()

        def header_text() -> AnyFormattedText:
            return FormattedText(
                [
                    ("class:binder.ticker", f" {document.ticker}  "),
                    ("class:binder.call", document.recommendation),
                    ("class:binder.muted", f"  {document.confidence} confidence"),
                    ("", " " * 4),
                    ("class:binder.muted", "q Close"),
                ]
            )

        def tabs_text() -> AnyFormattedText:
            fragments: list[tuple[str, str]] = []
            for index, page in enumerate(document.pages):
                style = "class:binder.tab.active" if index == active else "class:binder.tab"
                fragments.append((style, f" {index + 1} {page.label} "))
                fragments.append(("", " "))
            return FormattedText(fragments)

        def body_text() -> str:
            return _page_text(document.pages[active])

        def footer_text() -> str:
            if help_visible:
                return (
                    "←/→ or h/l tabs · j/k or ↑/↓ scroll · PgUp/PgDn · Home/End · "
                    "1–5 jump · q/Esc close · ? hide help"
                )
            return "←/→ tabs   j/k or PgUp/PgDn scroll   1–5 jump   ? help"

        body = Window(
            FormattedTextControl(body_text, focusable=True),
            wrap_lines=True,
            always_hide_cursor=True,
            style="class:binder.body",
        )

        def select(index: int) -> None:
            nonlocal active
            active = index % len(document.pages)
            body.vertical_scroll = 0

        def move(amount: int) -> None:
            body.vertical_scroll = max(0, body.vertical_scroll + amount)

        @bindings.add("q")
        @bindings.add("escape")
        @bindings.add("c-d")
        @bindings.add("c-c")
        def close(event: KeyPressEvent) -> None:
            event.app.exit()

        @bindings.add("right")
        @bindings.add("l")
        @bindings.add("tab")
        def next_page(event: KeyPressEvent) -> None:
            del event
            select(active + 1)

        @bindings.add("left")
        @bindings.add("h")
        @bindings.add("s-tab")
        def previous_page(event: KeyPressEvent) -> None:
            del event
            select(active - 1)

        for number in range(1, len(document.pages) + 1):
            bindings.add(str(number))(lambda event, index=number - 1: select(index))

        @bindings.add("down")
        @bindings.add("j")
        def scroll_down(event: KeyPressEvent) -> None:
            del event
            move(1)

        @bindings.add("up")
        @bindings.add("k")
        def scroll_up(event: KeyPressEvent) -> None:
            del event
            move(-1)

        @bindings.add("pagedown")
        def page_down(event: KeyPressEvent) -> None:
            height = body.render_info.window_height if body.render_info is not None else 10
            move(max(1, height - 2))

        @bindings.add("pageup")
        def page_up(event: KeyPressEvent) -> None:
            height = body.render_info.window_height if body.render_info is not None else 10
            move(-max(1, height - 2))

        @bindings.add("home")
        def scroll_home(event: KeyPressEvent) -> None:
            del event
            body.vertical_scroll = 0

        @bindings.add("end")
        def scroll_end(event: KeyPressEvent) -> None:
            del event
            body.vertical_scroll = len(body_text().splitlines())

        @bindings.add("?")
        def toggle_help(event: KeyPressEvent) -> None:
            del event
            nonlocal help_visible
            help_visible = not help_visible

        layout = Layout(
            HSplit(
                [
                    Window(FormattedTextControl(header_text), height=1),
                    Window(FormattedTextControl(tabs_text), height=1),
                    Window(height=1, char="─", style="class:binder.rule"),
                    body,
                    Window(FormattedTextControl(footer_text), height=1),
                ]
            ),
            focused_element=body,
        )
        no_color = bool(os.environ.get("NO_COLOR"))
        style = Style.from_dict(
            {
                "binder.ticker": "bold" if no_color else f"bold {NAVY_STRONG}",
                "binder.call": "bold" if no_color else f"bold {NAVY_BRAND}",
                "binder.muted": "" if no_color else NAVY_MUTED,
                "binder.tab": "" if no_color else NAVY_MUTED,
                "binder.tab.active": "bold reverse" if no_color else f"bold {NAVY_BRAND} reverse",
                "binder.rule": "" if no_color else NAVY_MUTED,
                "binder.body": "",
            }
        )
        Application[None](
            layout=layout,
            key_bindings=bindings,
            full_screen=True,
            style=style,
            input=self._input,
            output=self._output,
            mouse_support=False,
        ).run()

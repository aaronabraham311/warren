"""Deterministic parsing for Warren's deliberately small natural-language grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, TypeAlias

from data_sources.symbols import TICKER_PATTERN, canonical_symbol

if TYPE_CHECKING:
    from agent.service import RunRequest

PersonaName: TypeAlias = Literal["default", "dirt"]


class RequestIntent(StrEnum):
    ANALYZE = "analyze"
    COMPARE = "compare"
    PORTFOLIO = "portfolio"
    DISCOVERY = "discovery"
    GEM_HUNT = "gem_hunt"


class FollowUpKind(StrEnum):
    RISKS = "risks"
    DATA_QUALITY = "data_quality"
    LYNCH = "lynch"
    BUFFETT = "buffett"
    EVIDENCE = "evidence"
    WHY = "why"
    SELECT_TICKER = "select_ticker"


@dataclass(frozen=True, slots=True)
class RecentContext:
    """The minimal stored-result context the pure parser is allowed to inspect."""

    tickers: tuple[str, ...]
    selected_ticker: str | None = None


@dataclass(frozen=True, slots=True)
class RunnableRequest:
    intent: RequestIntent
    tickers: tuple[str, ...] = ()
    persona: PersonaName | None = None

    def to_run_request(
        self,
        *,
        max_cost_usd: float = 1.25,
        default_persona: str = "default",
    ) -> RunRequest:
        """Convert the parsed intent to the shared service's validated request."""
        from agent.service import RunMode, RunRequest

        modes = {
            RequestIntent.ANALYZE: RunMode.TICKERS,
            RequestIntent.COMPARE: RunMode.TICKERS,
            RequestIntent.PORTFOLIO: RunMode.PORTFOLIO,
            RequestIntent.DISCOVERY: RunMode.DISCOVERY,
            RequestIntent.GEM_HUNT: RunMode.GEM_HUNT,
        }
        fallback_persona: PersonaName = "dirt" if default_persona == "dirt" else "default"
        persona: PersonaName = self.persona or fallback_persona
        if self.intent is RequestIntent.GEM_HUNT:
            persona = "dirt"
        return RunRequest(
            mode=modes[self.intent],
            tickers=list(self.tickers),
            persona=persona,
            max_cost_usd=max_cost_usd,
        )


@dataclass(frozen=True, slots=True)
class Clarification:
    prompt: str


@dataclass(frozen=True, slots=True)
class Unsupported:
    explanation: str = (
        "I can analyze or compare tickers, review your portfolio, run discovery or gem hunt, "
        "and show structured fields from the latest result. Try “Analyze AAPL” or /help."
    )

    @property
    def message(self) -> str:
        return self.explanation


UnsupportedRequest: TypeAlias = Unsupported


@dataclass(frozen=True, slots=True)
class StoredResultFollowUp:
    kind: FollowUpKind
    ticker: str | None = None


RequestOutcome: TypeAlias = RunnableRequest | Clarification | Unsupported | StoredResultFollowUp

_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9.-])[A-Za-z0-9]{1,6}(?:[.-][A-Za-z]{1,2})?(?![A-Za-z0-9.-])")
_TICKER_STOP_WORDS = frozenset(
    {
        "A",
        "ABOUT",
        "AGAINST",
        "AN",
        "AND",
        "ANOTHER",
        "AT",
        "BUFFETT",
        "BUY",
        "CHEAP",
        "DATA",
        "DIRT",
        "EUROPE",
        "FOR",
        "FROM",
        "HOLD",
        "IN",
        "IT",
        "LATEST",
        "LOOK",
        "LYNCH",
        "ME",
        "MY",
        "NOTES",
        "OF",
        "ON",
        "PLEASE",
        "RECENT",
        "RESULT",
        "REVIEW",
        "RISKS",
        "RUN",
        "SELL",
        "SHOW",
        "STOCK",
        "STOCKS",
        "TAKE",
        "THE",
        "THESE",
        "THIS",
        "TICKER",
        "TO",
        "USING",
        "VERSUS",
        "VS",
        "WHY",
        "WITH",
    }
)


def _extract_tickers(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        ticker = canonical_symbol(match.group())
        if ticker in _TICKER_STOP_WORDS or re.fullmatch(TICKER_PATTERN, ticker) is None:
            continue
        if ticker not in found:
            found.append(ticker)
    return tuple(found)


def _persona(text: str) -> PersonaName | None:
    lowered = text.casefold()
    if re.search(r"\b(?:using|with)\s+dirt\b|\bdirt\s+persona\b|\bdeep[- ]value\b", lowered):
        return "dirt"
    if re.search(r"\b(?:using|with)\s+default\b|\bdefault\s+persona\b", lowered):
        return "default"
    return None


def _follow_up(text: str, recent: RecentContext | None) -> RequestOutcome | None:
    lowered = " ".join(text.casefold().split())
    kind: FollowUpKind | None = None
    if re.fullmatch(r"(?:show\s+)?(?:the\s+)?risks?", lowered):
        kind = FollowUpKind.RISKS
    elif re.fullmatch(r"(?:show\s+)?(?:the\s+)?data[- ]quality(?:\s+notes?)?", lowered):
        kind = FollowUpKind.DATA_QUALITY
    elif re.fullmatch(r"(?:show\s+)?(?:the\s+)?lynch(?:\s+signals?)?", lowered):
        kind = FollowUpKind.LYNCH
    elif re.fullmatch(r"(?:show\s+)?(?:the\s+)?buffett(?:\s+signals?)?", lowered):
        kind = FollowUpKind.BUFFETT
    elif re.fullmatch(r"(?:show\s+)?(?:the\s+)?evidence", lowered):
        kind = FollowUpKind.EVIDENCE
    elif re.fullmatch(r"why(?:\s+(?:buy|sell|hold))?\??", lowered):
        kind = FollowUpKind.WHY
    elif re.fullmatch(r"show\s+another\s+ticker(?:\s+from\s+this\s+run)?", lowered):
        kind = FollowUpKind.SELECT_TICKER
    elif re.fullmatch(
        r"show\s+[A-Za-z0-9]{1,6}(?:[.-][A-Za-z]{1,2})?\s+from\s+this\s+run",
        text.strip(),
        flags=re.IGNORECASE,
    ):
        kind = FollowUpKind.SELECT_TICKER
    if kind is None:
        return None
    if recent is None or not recent.tickers:
        return Clarification("Run an analysis first, then ask to show a field from its result.")
    ticker: str | None = None
    if kind is FollowUpKind.SELECT_TICKER:
        candidates = _extract_tickers(text)
        if candidates:
            ticker = candidates[0]
            if ticker not in recent.tickers:
                return Clarification(
                    f"{ticker} is not in the latest run. Choose: {', '.join(recent.tickers)}."
                )
        else:
            if len(recent.tickers) < 2:
                return Clarification("The latest run contains only one ticker.")
            try:
                selected_index = recent.tickers.index(recent.selected_ticker or "")
            except ValueError:
                ticker = recent.tickers[0]
            else:
                ticker = recent.tickers[(selected_index + 1) % len(recent.tickers)]
    return StoredResultFollowUp(kind, ticker=ticker)


def parse_request(text: str, *, recent: RecentContext | None = None) -> RequestOutcome:
    """Parse supported input without model, network, filesystem, or data-source calls."""
    stripped = text.strip()
    if not stripped:
        return Clarification("What would you like Warren to analyze?")
    if stripped.startswith("/"):
        return UnsupportedRequest(
            "Use the slash-command parser for commands; type /help for options."
        )

    follow_up = _follow_up(stripped, recent)
    if follow_up is not None:
        return follow_up

    lowered = " ".join(stripped.casefold().split())
    compare = bool(re.search(r"\bcompare\b|\bversus\b|\bvs\.?\b", lowered))
    analyze = bool(re.search(r"\banaly[sz]e\b|\btake\s+a\s+look\s+at\b|\bevaluate\b", lowered))
    portfolio = bool(re.search(r"\b(?:review|show|analy[sz]e)\s+my\s+portfolio\b", lowered))
    gem_hunt = bool(
        re.search(
            r"\bgem[- ]hunt\b|\brun\s+gem\s+hunt\b|\bdirt[- ]cheap\s+european\s+stocks?\b",
            lowered,
        )
    )
    discovery = bool(re.search(r"\brun\s+discovery\b|\bfind\s+candidates?\b", lowered))

    intents = sum((compare or analyze, portfolio, gem_hunt, discovery))
    if intents > 1:
        return Clarification(
            "Please choose one workflow: analysis, portfolio, discovery, or gem hunt."
        )

    tickers = _extract_tickers(stripped)
    persona = _persona(stripped)
    if compare:
        if not 2 <= len(tickers) <= 4:
            return Clarification("Name two to four explicit tickers to compare.")
        return RunnableRequest(RequestIntent.COMPARE, tickers, persona)
    if analyze:
        if len(tickers) != 1:
            return Clarification(
                "Name exactly one ticker to analyze, or say “compare” for several."
            )
        return RunnableRequest(RequestIntent.ANALYZE, tickers, persona)
    if portfolio:
        return RunnableRequest(RequestIntent.PORTFOLIO, persona=persona)
    if gem_hunt:
        return RunnableRequest(RequestIntent.GEM_HUNT, persona="dirt")
    if discovery:
        return RunnableRequest(RequestIntent.DISCOVERY, persona=persona)
    return UnsupportedRequest()

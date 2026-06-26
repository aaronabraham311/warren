"""get_adverse_media tool — GDELT DOC 2.0 cross-language adverse news.

Queries GDELT for negative-tone coverage of an entity, classifies hits into
adverse categories derived from GKG theme tags and title keywords, dedupes by
URL, and returns ranked flags (not full articles). Supports dual-query for
native-language spellings.
"""

import re
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import gdelt_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.gdelt_client import RawArticle

AdverseCategory = Literal[
    "fraud_accounting",
    "legal_regulatory",
    "sanctions",
    "management_misconduct",
    "environmental",
    "labor",
    "cybersecurity",
    "other_adverse",
]

# Lower index = higher priority in ranking.
_CATEGORY_PRIORITY: dict[str, int] = {
    "fraud_accounting": 0,
    "sanctions": 1,
    "legal_regulatory": 2,
    "management_misconduct": 3,
    "environmental": 4,
    "labor": 5,
    "cybersecurity": 6,
    "other_adverse": 7,
}

# (theme_prefix_or_exact, category) — checked before keywords.
_THEME_RULES: list[tuple[str, str]] = [
    ("CRIME_FRAUD", "fraud_accounting"),
    ("SANCTION", "sanctions"),
    ("CORRUPTION", "management_misconduct"),
    ("SCANDAL", "management_misconduct"),
    ("CRISISLEX", "legal_regulatory"),
    ("ENV_", "environmental"),
    ("LABOR_", "labor"),
    ("CYBERSECURITY", "cybersecurity"),
]

# (regex, category) — fallback when no theme prefix matched.
_KEYWORD_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"fraud|embezzl|restat|writedown|accounting irregularit", re.I),
        "fraud_accounting",
    ),
    (
        re.compile(r"sanction|ofac|blacklist|debarred|export control", re.I),
        "sanctions",
    ),
    (
        re.compile(r"lawsuit|litigat|penalty|fine|sec investigation|class action|settlement", re.I),
        "legal_regulatory",
    ),
    (
        re.compile(r"misconduct|bribery|corrupt|harass|discriminat", re.I),
        "management_misconduct",
    ),
    (re.compile(r"spill|epa|contamin|environmental damage", re.I), "environmental"),
    (re.compile(r"strike|labor dispute|child labor|sweatshop|exploitation", re.I), "labor"),
    (re.compile(r"data breach|hack|cyber|ransomware|security incident", re.I), "cybersecurity"),
]

_SEENDATE_FMT = "%Y%m%dT%H%M%SZ"
_MAX_HITS = 50


def _parse_date(seendate: str) -> date:
    try:
        return datetime.strptime(seendate, _SEENDATE_FMT).replace(tzinfo=timezone.utc).date()
    except ValueError:
        return date.today()


def classify_article(themes: str, title: str) -> str:
    """Return the adverse category for an article based on its GKG themes and title."""
    theme_tokens = set(re.split(r"[;, ]+", themes.upper()))
    for prefix, category in _THEME_RULES:
        if any(t == prefix or t.startswith(prefix) for t in theme_tokens):
            return category
    for pattern, category in _KEYWORD_RULES:
        if pattern.search(title):
            return category
    return "other_adverse"


class GetAdverseMediaInput(BaseModel):
    entity_name: str = Field(description="Romanised entity name to search, e.g. 'Apple Inc'")
    entity_type: Literal["company", "person"] = Field(
        description="Whether the entity is a company or a person"
    )
    lookback_days: int = Field(
        default=30, ge=1, le=90, description="Look-back window in days (1–90)"
    )
    native_name: str | None = Field(
        default=None,
        description="Optional local-script name for dual-query deduplication, e.g. '苹果公司'",
    )


class AdverseHit(BaseModel):
    url: str
    title: str
    date: date
    domain: str
    language: str
    # tone is not returned by GDELT DOC 2.0 ArtList; None means the filter
    # was applied server-side (tone<-2) but the value was not returned.
    tone: float | None
    category: str


class AdverseMediaResult(BaseModel):
    entity_name: str
    hits: list[AdverseHit]
    total_coverage_volume: int
    languages_with_coverage: list[str]
    categories: dict[str, int]


def _to_hit(article: RawArticle) -> AdverseHit:
    return AdverseHit(
        url=article.url,
        title=article.title,
        date=_parse_date(article.seendate),
        domain=article.domain,
        language=article.language,
        tone=article.tone,
        category=classify_article(article.themes, article.title),
    )


def _rank_key(hit: AdverseHit) -> tuple[int, float, int]:
    # category priority ASC, tone ASC (most negative first, 0.0 when absent), date DESC
    return (_CATEGORY_PRIORITY.get(hit.category, 7), hit.tone or 0.0, -hit.date.toordinal())


class GetAdverseMediaTool(Tool):
    name = "get_adverse_media"
    description = (
        "Search for adverse media coverage of an entity across 65 languages via GDELT. "
        "Returns ranked, categorised negative-tone news hits (fraud, sanctions, legal, "
        "misconduct, environmental, labor, cybersecurity) rather than full articles. "
        "Supports dual-query for native-script names to reach local-language coverage. "
        "Use for cross-border reputational due diligence."
    )
    input_schema = GetAdverseMediaInput
    output_schema = AdverseMediaResult

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetAdverseMediaInput)
        gdelt = gdelt_client()

        queries = [tool_input.entity_name]
        if tool_input.native_name and tool_input.native_name != tool_input.entity_name:
            queries.append(tool_input.native_name)

        seen_urls: dict[str, RawArticle] = {}
        for query in queries:
            result = gdelt.get_adverse_articles(query, tool_input.lookback_days)
            if isinstance(result, DataSourceError):
                return error_from_data_source(result)
            for article in result:
                if article.url not in seen_urls:
                    seen_urls[article.url] = article

        total_volume = len(seen_urls)
        hits = [_to_hit(a) for a in seen_urls.values()]
        hits.sort(key=_rank_key)
        hits = hits[:_MAX_HITS]

        languages = sorted({h.language for h in hits if h.language})
        categories: dict[str, int] = {}
        for h in hits:
            categories[h.category] = categories.get(h.category, 0) + 1

        return ToolResultOk(
            data=AdverseMediaResult(
                entity_name=tool_input.entity_name,
                hits=hits,
                total_coverage_volume=total_volume,
                languages_with_coverage=languages,
                categories=categories,
            )
        )

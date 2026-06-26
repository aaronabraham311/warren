"""Tests for the get_adverse_media tool and its classification/ranking logic.

Exercises:
- classify_article() — theme-prefix rules and keyword fallback rules
- _parse_date() — seendate format handling
- GetAdverseMediaTool.run() — happy path, deduplication, network error
- Tool registration in TOOL_REGISTRY
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.adverse_media import (
    AdverseHit,
    AdverseMediaResult,
    GetAdverseMediaInput,
    GetAdverseMediaTool,
    _parse_date,
    _rank_key,
    classify_article,
)
from agent.tools.base import ToolResultError, ToolResultOk
from data_sources.errors import DataSourceError
from data_sources.gdelt_client import RawArticle

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_article(
    url: str = "https://example.com/article",
    title: str = "Some news",
    seendate: str = "20260601T120000Z",
    language: str = "English",
    tone: float | None = None,
    themes: str = "",
) -> RawArticle:
    return RawArticle(
        url=url,
        title=title,
        seendate=seendate,
        domain="example.com",
        language=language,
        sourcecountry="US",
        tone=tone,
        themes=themes,
    )


def _make_ctx() -> MagicMock:
    return MagicMock()


# ── classify_article — theme rules ────────────────────────────────────────────


@pytest.mark.parametrize(
    "themes,expected_category",
    [
        ("CRIME_FRAUD", "fraud_accounting"),
        ("CRIME_FRAUD CORRUPTION", "fraud_accounting"),  # first match wins
        ("SANCTION", "sanctions"),
        ("CORRUPTION", "management_misconduct"),
        ("SCANDAL CORRUPTION", "management_misconduct"),
        ("CRISISLEX_CRISISLEXREC", "legal_regulatory"),
        ("ENV_DEFORESTATION", "environmental"),
        ("LABOR_UNIONS", "labor"),
        ("CYBERSECURITY", "cybersecurity"),
    ],
)
def test_classify_by_theme_prefix(themes: str, expected_category: str) -> None:
    assert classify_article(themes, "irrelevant title") == expected_category


def test_classify_theme_prefix_case_insensitive() -> None:
    assert classify_article("crime_fraud", "irrelevant") == "fraud_accounting"


# ── classify_article — keyword rules ─────────────────────────────────────────


@pytest.mark.parametrize(
    "title,expected_category",
    [
        ("Company faces fraud investigation", "fraud_accounting"),
        ("Executives charged over accounting irregularities", "fraud_accounting"),
        ("Apple settles class action lawsuit", "legal_regulatory"),
        ("Firm hit with $500M penalty from SEC investigation", "legal_regulatory"),
        ("Supplier sanctioned by OFAC over Iran ties", "sanctions"),
        ("Director accused of bribery and misconduct", "management_misconduct"),
        ("Oil spill causes environmental damage near plant", "environmental"),
        ("Workers launch strike over labor dispute", "labor"),
        ("Company suffers major data breach exposing millions", "cybersecurity"),
        ("Ransomware attack disrupts operations", "cybersecurity"),
        ("Vague negative headline without clear signal", "other_adverse"),
    ],
)
def test_classify_by_keyword(title: str, expected_category: str) -> None:
    # No themes — falls through to keyword rules
    assert classify_article("", title) == expected_category


def test_classify_theme_wins_over_keyword() -> None:
    # Title has "fraud" but theme says sanctions → theme wins (first-matched rule)
    result = classify_article("SANCTION", "Company fraud investigation")
    assert result == "sanctions"


def test_classify_other_adverse_when_no_match() -> None:
    assert classify_article("", "Quarterly results disappoint investors") == "other_adverse"


# ── _parse_date ────────────────────────────────────────────────────────────────


def test_parse_date_valid() -> None:
    assert _parse_date("20260601T120000Z") == date(2026, 6, 1)


def test_parse_date_invalid_returns_today() -> None:
    result = _parse_date("not_a_date")
    assert result == date.today()


# ── Ranking ───────────────────────────────────────────────────────────────────


def test_rank_key_category_priority() -> None:
    fraud_hit = AdverseHit(
        url="a",
        title="",
        date=date(2026, 1, 1),
        domain="",
        language="",
        tone=None,
        category="fraud_accounting",
    )
    sanctions_hit = AdverseHit(
        url="b",
        title="",
        date=date(2026, 1, 1),
        domain="",
        language="",
        tone=None,
        category="sanctions",
    )
    other_hit = AdverseHit(
        url="c",
        title="",
        date=date(2026, 1, 1),
        domain="",
        language="",
        tone=None,
        category="other_adverse",
    )
    hits = [other_hit, sanctions_hit, fraud_hit]
    hits.sort(key=_rank_key)
    assert [h.category for h in hits] == ["fraud_accounting", "sanctions", "other_adverse"]


def test_rank_key_recency_tiebreak() -> None:
    older = AdverseHit(
        url="a",
        title="",
        date=date(2026, 1, 1),
        domain="",
        language="",
        tone=None,
        category="other_adverse",
    )
    newer = AdverseHit(
        url="b",
        title="",
        date=date(2026, 6, 1),
        domain="",
        language="",
        tone=None,
        category="other_adverse",
    )
    hits = [older, newer]
    hits.sort(key=_rank_key)
    # Same category — most recent first (date DESC)
    assert hits[0].date == date(2026, 6, 1)


def test_rank_key_tone_tiebreak() -> None:
    less_negative = AdverseHit(
        url="a",
        title="",
        date=date(2026, 1, 1),
        domain="",
        language="",
        tone=-2.5,
        category="legal_regulatory",
    )
    more_negative = AdverseHit(
        url="b",
        title="",
        date=date(2026, 1, 1),
        domain="",
        language="",
        tone=-8.0,
        category="legal_regulatory",
    )
    hits = [less_negative, more_negative]
    hits.sort(key=_rank_key)
    # Same category and date — most negative tone first (tone ASC)
    assert hits[0].tone == pytest.approx(-8.0)


# ── GetAdverseMediaTool.run ───────────────────────────────────────────────────


def _tool_with_mock_client(
    articles: list[RawArticle] | DataSourceError,
) -> tuple[GetAdverseMediaTool, MagicMock]:
    tool = GetAdverseMediaTool()
    gdelt_mock = MagicMock()
    gdelt_mock.get_adverse_articles.return_value = articles
    return tool, gdelt_mock


def test_tool_happy_path(gdelt_fixture: dict[str, object]) -> None:
    from data_sources.gdelt_client import GDELTClient

    articles = GDELTClient._parse(gdelt_fixture)
    tool, gdelt_mock = _tool_with_mock_client(articles)

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        result = tool.run(
            GetAdverseMediaInput(entity_name="Apple", entity_type="company", lookback_days=30),
            _make_ctx(),
        )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, AdverseMediaResult)
    data = result.data
    assert data.entity_name == "Apple"
    assert data.total_coverage_volume == 5
    assert len(data.hits) == 5
    assert "English" in data.languages_with_coverage
    assert "French" in data.languages_with_coverage
    assert isinstance(data.categories, dict)


def test_tool_deduplicates_by_url() -> None:
    url = "https://example.com/same-article"
    articles = [
        _make_article(url=url, title="Fraud probe"),
        _make_article(url=url, title="Fraud probe"),  # same URL
        _make_article(url="https://example.com/other", title="Lawsuit filed"),
    ]
    tool, gdelt_mock = _tool_with_mock_client(articles)

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        result = tool.run(
            GetAdverseMediaInput(entity_name="Corp", entity_type="company"),
            _make_ctx(),
        )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, AdverseMediaResult)
    assert result.data.total_coverage_volume == 2
    assert len(result.data.hits) == 2


def test_tool_dual_query_deduplicates_across_queries() -> None:
    shared_url = "https://example.com/shared"
    native_only_url = "https://example.com/native-only"

    roman_articles = [_make_article(url=shared_url, title="Fraud probe")]
    native_articles = [
        _make_article(url=shared_url, title="Fraud probe"),  # duplicate
        _make_article(url=native_only_url, title="Lawsuit"),
    ]

    gdelt_mock = MagicMock()
    gdelt_mock.get_adverse_articles.side_effect = [roman_articles, native_articles]
    tool = GetAdverseMediaTool()

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        result = tool.run(
            GetAdverseMediaInput(
                entity_name="Apple Inc",
                entity_type="company",
                native_name="苹果公司",
            ),
            _make_ctx(),
        )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, AdverseMediaResult)
    assert result.data.total_coverage_volume == 2
    assert gdelt_mock.get_adverse_articles.call_count == 2


def test_tool_skips_second_query_when_native_name_same_as_entity_name() -> None:
    articles = [_make_article(title="Fraud")]
    tool, gdelt_mock = _tool_with_mock_client(articles)

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        tool.run(
            GetAdverseMediaInput(
                entity_name="Apple",
                entity_type="company",
                native_name="Apple",  # identical — should not trigger second query
            ),
            _make_ctx(),
        )

    assert gdelt_mock.get_adverse_articles.call_count == 1


def test_tool_truncates_to_50_hits() -> None:
    articles = [
        _make_article(url=f"https://example.com/{i}", title="Lawsuit filed") for i in range(60)
    ]
    tool, gdelt_mock = _tool_with_mock_client(articles)

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        result = tool.run(
            GetAdverseMediaInput(entity_name="Corp", entity_type="company"),
            _make_ctx(),
        )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, AdverseMediaResult)
    assert len(result.data.hits) == 50
    assert result.data.total_coverage_volume == 60


def test_tool_network_error_returns_tool_result_error() -> None:
    gdelt_mock = MagicMock()
    gdelt_mock.get_adverse_articles.return_value = DataSourceError(
        error_code="network", message="connection refused"
    )
    tool = GetAdverseMediaTool()

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        result = tool.run(
            GetAdverseMediaInput(entity_name="Corp", entity_type="company"),
            _make_ctx(),
        )

    assert isinstance(result, ToolResultError)
    assert result.error_code == "network"
    assert result.retryable is True


def test_tool_categories_counted_correctly() -> None:
    articles = [
        _make_article(url="https://a.com/1", title="Fraud investigation"),
        _make_article(url="https://a.com/2", title="Another fraud scheme"),
        _make_article(url="https://a.com/3", title="Lawsuit settlement"),
    ]
    tool, gdelt_mock = _tool_with_mock_client(articles)

    with patch("agent.tools.adverse_media.gdelt_client", return_value=gdelt_mock):
        result = tool.run(
            GetAdverseMediaInput(entity_name="Corp", entity_type="company"),
            _make_ctx(),
        )

    assert isinstance(result, ToolResultOk)
    assert isinstance(result.data, AdverseMediaResult)
    categories = result.data.categories
    assert categories.get("fraud_accounting") == 2
    assert categories.get("legal_regulatory") == 1


# ── Tool registration ─────────────────────────────────────────────────────────


def test_tool_registered_in_registry() -> None:
    from agent.tools import TOOL_REGISTRY

    assert "get_adverse_media" in TOOL_REGISTRY
    assert isinstance(TOOL_REGISTRY["get_adverse_media"], GetAdverseMediaTool)


def test_tool_has_valid_api_dict() -> None:
    tool = GetAdverseMediaTool()
    api_dict = tool.to_api_dict()
    assert api_dict["name"] == "get_adverse_media"
    assert "description" in api_dict
    assert "input_schema" in api_dict

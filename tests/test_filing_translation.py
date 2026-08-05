from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from data_sources.cache import CacheStore
from data_sources.errors import DataSourceError
from data_sources.filing_models import (
    DocumentPage,
    DocumentText,
    ExtractionMethod,
    TranslationStatus,
)
from data_sources.filing_translation import (
    AnthropicPageTranslator,
    AnthropicTranslationResult,
    CachedTranslationStore,
    TranslationCacheConflictError,
    TranslationCacheKey,
    TranslationLimits,
    translate_document,
    translate_document_with_provider,
)
from storage.artifacts import ArtifactStore


class _Translator:
    def __init__(
        self,
        *,
        version: str = "offline-v1",
        cost_per_page: Decimal = Decimal("0.10"),
        failing_pages: set[str] | None = None,
    ) -> None:
        self._version = version
        self.cost_per_page = cost_per_page
        self.failing_pages = failing_pages or set()
        self.calls: list[str] = []

    @property
    def version(self) -> str:
        return self._version

    def estimate_cost_usd(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> Decimal:
        del text, source_language, target_language
        return self.cost_per_page

    def translate_page(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> str:
        del source_language, target_language
        self.calls.append(text)
        if text in self.failing_pages:
            raise RuntimeError("offline translation failure")
        return f"EN:{text}"


class _AnthropicTransport:
    def __init__(self, *, stop_reason: str = "end_turn") -> None:
        self.calls: list[tuple[str, int, str]] = []
        self.stop_reason = stop_reason

    def translate(self, *, model: str, max_tokens: int, prompt: str) -> AnthropicTranslationResult:
        self.calls.append((model, max_tokens, prompt))
        return AnthropicTranslationResult(
            text="Translated annual report",
            stop_reason=self.stop_reason,
            input_tokens=10,
            output_tokens=5,
        )


class _UnderestimatingTranslator(_Translator):
    def __init__(self) -> None:
        super().__init__(cost_per_page=Decimal("0.01"))
        self._actual_cost = Decimal("0")

    @property
    def actual_cost_usd(self) -> Decimal:
        return self._actual_cost

    def translate_page(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> str:
        translated = super().translate_page(
            text, source_language=source_language, target_language=target_language
        )
        self._actual_cost += Decimal("0.75")
        return translated


def _document(*texts: str, source_language: str | None = "pl") -> DocumentText:
    return DocumentText(
        filing_id="filing_test",
        sha256="a" * 64,
        source_url="https://example.test/report.pdf",
        retrieved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        extraction_method=ExtractionMethod.EMBEDDED_TEXT,
        source_language=source_language,
        pages=[DocumentPage(page_number=index, text=text) for index, text in enumerate(texts, 1)],
        page_count=len(texts),
        original_char_count=sum(map(len, texts)),
        extracted_char_count=sum(map(len, texts)),
    )


@pytest.fixture
def cache(tmp_path: Path) -> CachedTranslationStore:
    return CachedTranslationStore(
        CacheStore(sqlite3.connect(":memory:")),
        ArtifactStore(tmp_path),
    )


def test_translation_preserves_pages_and_reuses_versioned_cache(
    cache: CachedTranslationStore,
) -> None:
    document = _document("Pierwsza", "Druga")
    first_translator = _Translator()
    first = translate_document(document, translator=first_translator, cache=cache)

    assert first.document.translation_status is TranslationStatus.TRANSLATED
    assert first.document.english_translation_pages == [
        DocumentPage(page_number=1, text="EN:Pierwsza"),
        DocumentPage(page_number=2, text="EN:Druga"),
    ]
    assert first.document.translation_missing_pages == []
    assert first.provider_pages_attempted == 2
    assert first.estimated_cost_usd == Decimal("0.20")

    cached_translator = _Translator()
    cached = translate_document(document, translator=cached_translator, cache=cache)
    assert cached.cache_hits == 2
    assert cached.provider_pages_attempted == 0
    assert cached_translator.calls == []

    changed_translator = _Translator(version="offline-v2")
    changed = translate_document(document, translator=changed_translator, cache=cache)
    assert changed.cache_hits == 0
    assert changed_translator.calls == ["Pierwsza", "Druga"]


def test_cache_key_isolated_by_every_contract_dimension(cache: CachedTranslationStore) -> None:
    text_hash = hashlib.sha256(b"page text").hexdigest()
    base = TranslationCacheKey("a" * 64, 1, text_hash, "pl", "en", "v1")
    cache.set(base, "cached")

    assert cache.get(base) == "cached"
    assert cache.get(TranslationCacheKey("b" * 64, 1, text_hash, "pl", "en", "v1")) is None
    assert cache.get(TranslationCacheKey("a" * 64, 2, text_hash, "pl", "en", "v1")) is None
    changed_hash = hashlib.sha256(b"changed text").hexdigest()
    assert cache.get(TranslationCacheKey("a" * 64, 1, changed_hash, "pl", "en", "v1")) is None
    assert cache.get(TranslationCacheKey("a" * 64, 1, text_hash, "it", "en", "v1")) is None
    assert cache.get(TranslationCacheKey("a" * 64, 1, text_hash, "pl", "en-gb", "v1")) is None
    assert cache.get(TranslationCacheKey("a" * 64, 1, text_hash, "pl", "en", "v2")) is None


def test_cache_is_first_value_wins_for_an_exact_versioned_key(
    cache: CachedTranslationStore,
) -> None:
    key = TranslationCacheKey("a" * 64, 1, hashlib.sha256(b"page").hexdigest(), "pl", "en", "v1")
    cache.set(key, "first")
    cache.set(key, "first")

    with pytest.raises(TranslationCacheConflictError, match="different text"):
        cache.set(key, "changed without a version bump")
    assert cache.get(key) == "first"


def test_translation_cache_keeps_page_body_out_of_sqlite(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    cache = CachedTranslationStore(CacheStore(connection), ArtifactStore(tmp_path))
    key = TranslationCacheKey("a" * 64, 1, hashlib.sha256(b"page").hexdigest(), "pl", "en", "v1")

    cache.set(key, "sensitive translated filing body")

    stored_metadata = str(connection.execute("SELECT data FROM cache").fetchone()[0])
    assert "sensitive translated filing body" not in stored_metadata
    assert "artifact_key" in stored_metadata
    assert cache.get(key) == "sensitive translated filing body"


def test_one_failed_page_reports_exact_partial_coverage(
    cache: CachedTranslationStore,
) -> None:
    outcome = translate_document(
        _document("pierwsza", "druga", "trzecia"),
        translator=_Translator(failing_pages={"druga"}),
        cache=cache,
    )

    assert outcome.document.translation_status is TranslationStatus.PARTIAL
    assert [page.page_number for page in outcome.document.english_translation_pages or []] == [1, 3]
    assert outcome.document.translation_missing_pages == [2]
    assert any("Translation coverage: 2/3" in note for note in outcome.document.coverage_notes)


def test_all_failed_pages_report_failed_with_every_page_missing(
    cache: CachedTranslationStore,
) -> None:
    outcome = translate_document(
        _document("pierwsza", "druga"),
        translator=_Translator(failing_pages={"pierwsza", "druga"}),
        cache=cache,
    )

    assert outcome.document.translation_status is TranslationStatus.FAILED
    assert outcome.document.english_translation_pages is None
    assert outcome.document.translation_missing_pages == [1, 2]
    assert any("Translation coverage: 0/2" in note for note in outcome.document.coverage_notes)


@pytest.mark.parametrize(
    ("limits", "expected_calls", "expected_missing", "note_fragment"),
    [
        (TranslationLimits(max_pages=1), ["one"], [2, 3], "page limit"),
        (
            TranslationLimits(max_characters=3),
            ["one"],
            [2, 3],
            "character limit",
        ),
        (
            TranslationLimits(max_cost_usd=Decimal("0.20")),
            ["one", "two"],
            [3],
            "cost limit",
        ),
    ],
)
def test_translation_limits_prevent_out_of_budget_provider_calls(
    cache: CachedTranslationStore,
    limits: TranslationLimits,
    expected_calls: list[str],
    expected_missing: list[int],
    note_fragment: str,
) -> None:
    translator = _Translator()
    outcome = translate_document(
        _document("one", "two", "three"),
        translator=translator,
        cache=cache,
        limits=limits,
    )

    assert translator.calls == expected_calls
    assert outcome.document.translation_status is TranslationStatus.PARTIAL
    assert outcome.document.translation_missing_pages == expected_missing
    assert any(note_fragment in note for note in outcome.document.coverage_notes)


def test_actual_provider_spend_stops_subsequent_calls_at_cost_limit(
    cache: CachedTranslationStore,
) -> None:
    translator = _UnderestimatingTranslator()

    outcome = translate_document(
        _document("one", "two", "three"),
        translator=translator,
        cache=cache,
        limits=TranslationLimits(max_cost_usd=Decimal("0.50")),
    )

    assert translator.calls == ["one"]
    assert outcome.document.translation_status is TranslationStatus.PARTIAL
    assert outcome.document.translation_missing_pages == [2, 3]
    assert outcome.actual_cost_usd == Decimal("0.75")
    assert any("cost limit" in note for note in outcome.document.coverage_notes)


def test_english_source_is_not_needlessly_translated(cache: CachedTranslationStore) -> None:
    translator = _Translator()
    outcome = translate_document(
        _document("Already English", source_language="en-US"),
        translator=translator,
        cache=cache,
        target_language="en-GB",
    )

    assert outcome.document.translation_status is TranslationStatus.NOT_NEEDED
    assert translator.calls == []


def test_unconfigured_provider_returns_typed_error(cache: CachedTranslationStore) -> None:
    result = translate_document_with_provider(
        _document("Tekst"),
        translator=None,
        cache=cache,
    )

    assert isinstance(result, DataSourceError)
    assert result.stage == "translate"
    assert result.source == "filing_translation"


def test_blank_extracted_page_is_missing_and_never_sent_or_cached(
    cache: CachedTranslationStore,
) -> None:
    translator = _Translator()

    outcome = translate_document(
        _document("pierwsza", "   "),
        translator=translator,
        cache=cache,
    )

    assert translator.calls == ["pierwsza"]
    assert outcome.document.translation_status is TranslationStatus.PARTIAL
    assert outcome.document.translation_missing_pages == [2]
    assert any("no extracted text" in note for note in outcome.document.coverage_notes)


def test_rejects_language_name_that_only_starts_with_en(
    cache: CachedTranslationStore,
) -> None:
    with pytest.raises(ValueError, match="English translation targets"):
        translate_document(
            _document("tekst"),
            translator=_Translator(),
            cache=cache,
            target_language="english",
        )


def test_opt_in_anthropic_provider_has_versioned_prompt_and_conservative_cost() -> None:
    transport = _AnthropicTransport()
    translator = AnthropicPageTranslator(
        transport,
        model="test-model",
        input_usd_per_million_tokens=Decimal("3"),
        output_usd_per_million_tokens=Decimal("15"),
        max_output_tokens=100,
    )

    estimate = translator.estimate_cost_usd("abcd", source_language="it", target_language="en")
    text = translator.translate_page("Bilancio", source_language="it", target_language="en")

    assert translator.version == "anthropic:test-model:filing-page-v1"
    assert estimate == Decimal("0.000036")
    assert text == "Translated annual report"
    assert translator.actual_cost_usd == Decimal("0.000105")
    assert transport.calls == [
        (
            "test-model",
            100,
            "Source language hint: it\nTarget language: en\n\nBilancio",
        )
    ]


def test_truncated_provider_response_is_missing_not_translated(
    tmp_path: Path,
) -> None:
    translator = AnthropicPageTranslator(
        _AnthropicTransport(stop_reason="max_tokens"),
        model="test-model",
        input_usd_per_million_tokens=Decimal("3"),
        output_usd_per_million_tokens=Decimal("15"),
        max_output_tokens=10,
    )
    cache = CachedTranslationStore(CacheStore(sqlite3.connect(":memory:")), ArtifactStore(tmp_path))

    outcome = translate_document(_document("Bilancio"), translator=translator, cache=cache)

    assert outcome.document.translation_status is TranslationStatus.FAILED
    assert outcome.document.translation_missing_pages == [1]
    assert any("translation failed" in note.lower() for note in outcome.document.coverage_notes)
    assert outcome.actual_cost_usd == Decimal("0.000105")

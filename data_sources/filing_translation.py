"""Bounded, page-preserving translation for source-neutral filing text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

import anthropic
from anthropic.types import TextBlock

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError
from data_sources.filing_models import DocumentPage, DocumentText, TranslationStatus
from storage.artifacts import ArtifactIntegrityError, ArtifactStore, StoredArtifact


@dataclass(frozen=True)
class TranslationCacheKey:
    """Every input that can change a translated page's meaning."""

    document_sha256: str
    page_number: int
    page_text_sha256: str
    source_language: str | None
    target_language: str
    translator_version: str

    def __post_init__(self) -> None:
        if len(self.document_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.document_sha256
        ):
            raise ValueError("document_sha256 must be a lowercase SHA-256 digest")
        if self.page_number < 1:
            raise ValueError("page_number must be positive")
        if len(self.page_text_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.page_text_sha256
        ):
            raise ValueError("page_text_sha256 must be a lowercase SHA-256 digest")
        if not self.target_language.strip():
            raise ValueError("target_language must be non-empty")
        if not self.translator_version.strip():
            raise ValueError("translator_version must be non-empty")


class TranslationCache(Protocol):
    def get(self, key: TranslationCacheKey) -> str | None: ...

    def set(self, key: TranslationCacheKey, text: str) -> None: ...


class TranslationCacheConflictError(RuntimeError):
    """A provider returned different text for an already-versioned cache key."""


class CachedTranslationStore:
    """Cache translated text as artifacts; SQLite stores only small manifest metadata."""

    def __init__(
        self,
        store: CacheStore,
        artifact_store: ArtifactStore,
        *,
        ttl_hours: float = 87_600,
    ) -> None:
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
        self._store = store
        self._artifacts = artifact_store
        self._ttl_hours = ttl_hours

    def get(self, key: TranslationCacheKey) -> str | None:
        metadata = self._store.get(_serialized_cache_key(key))
        if metadata is None:
            return None
        try:
            value = json.loads(metadata)
            artifact = StoredArtifact(
                sha256=str(value["sha256"]),
                relative_key=str(value["artifact_key"]),
                byte_length=int(value["byte_length"]),
                mime_type="text/plain",
            )
            return self._artifacts.read(artifact).decode("utf-8")
        except (
            ArtifactIntegrityError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            raise TranslationCacheConflictError(
                f"translation cache artifact failed integrity validation: {exc}"
            ) from exc

    def set(self, key: TranslationCacheKey, text: str) -> None:
        cache_key = _serialized_cache_key(key)
        existing = self.get(key)
        if existing is not None:
            if existing != text:
                raise TranslationCacheConflictError(
                    "translation cache key already contains different text"
                )
            return
        artifact = self._artifacts.put(text.encode("utf-8"), mime_type="text/plain")
        self._store.set(
            cache_key,
            json.dumps(
                {
                    "sha256": artifact.sha256,
                    "artifact_key": artifact.relative_key,
                    "byte_length": artifact.byte_length,
                },
                sort_keys=True,
            ),
            ttl_hours=self._ttl_hours,
        )


class PageTranslator(Protocol):
    """Replaceable translation provider; implementations own their network calls."""

    @property
    def version(self) -> str: ...

    def estimate_cost_usd(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> Decimal: ...

    def translate_page(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> str: ...


class AnthropicTranslationTransport(Protocol):
    def translate(
        self, *, model: str, max_tokens: int, prompt: str
    ) -> AnthropicTranslationResult: ...


@dataclass(frozen=True)
class AnthropicTranslationResult:
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


class AnthropicSdkTranslationTransport:
    """Thin, injectable Anthropic SDK transport kept inside data_sources."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def translate(self, *, model: str, max_tokens: int, prompt: str) -> AnthropicTranslationResult:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=(
                "Translate the filing page faithfully into English. Preserve headings, "
                "numbers, units, and uncertainty. Return only the translation."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return AnthropicTranslationResult(
            text="\n".join(
                block.text for block in response.content if isinstance(block, TextBlock)
            ).strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class AnthropicPageTranslator:
    """Opt-in bounded translator with explicit model, prompt version, and pricing."""

    PROMPT_VERSION = "filing-page-v1"

    def __init__(
        self,
        transport: AnthropicTranslationTransport,
        *,
        model: str,
        input_usd_per_million_tokens: Decimal,
        output_usd_per_million_tokens: Decimal,
        max_output_tokens: int = 8_192,
        max_input_characters: int = 100_000,
    ) -> None:
        if not model.strip():
            raise ValueError("translation model must be non-empty")
        if input_usd_per_million_tokens < 0 or output_usd_per_million_tokens < 0:
            raise ValueError("translation token prices cannot be negative")
        if max_output_tokens < 1 or max_input_characters < 1:
            raise ValueError("translation input/output bounds must be positive")
        self._transport = transport
        self._model = model.strip()
        self._input_rate = input_usd_per_million_tokens
        self._output_rate = output_usd_per_million_tokens
        self._max_output_tokens = max_output_tokens
        self._max_input_characters = max_input_characters
        self._actual_cost = Decimal("0")

    @property
    def version(self) -> str:
        return f"anthropic:{self._model}:{self.PROMPT_VERSION}"

    @property
    def actual_cost_usd(self) -> Decimal:
        return self._actual_cost

    def estimate_cost_usd(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> Decimal:
        del source_language, target_language
        # Two characters/token for both input and similarly sized translated output
        # is deliberately conservative for multilingual financial text.
        estimated_tokens = Decimal(max(1, (len(text) + 1) // 2))
        return (estimated_tokens * (self._input_rate + self._output_rate)) / Decimal(1_000_000)

    def translate_page(
        self, text: str, *, source_language: str | None, target_language: str
    ) -> str:
        if len(text) > self._max_input_characters:
            raise ValueError("filing page exceeds the provider input-character limit")
        prompt = (
            f"Source language hint: {source_language or 'unknown'}\n"
            f"Target language: {target_language}\n\n{text}"
        )
        result = self._transport.translate(
            model=self._model,
            max_tokens=self._max_output_tokens,
            prompt=prompt,
        )
        self._actual_cost += (
            Decimal(result.input_tokens) * self._input_rate
            + Decimal(result.output_tokens) * self._output_rate
        ) / Decimal(1_000_000)
        if result.stop_reason == "max_tokens":
            raise ValueError("translation response was truncated at max_tokens")
        if result.stop_reason not in {"end_turn", "stop_sequence"}:
            raise ValueError(f"translation stopped unexpectedly: {result.stop_reason}")
        return result.text


@runtime_checkable
class CostReportingTranslator(Protocol):
    @property
    def actual_cost_usd(self) -> Decimal: ...


@dataclass(frozen=True)
class TranslationLimits:
    """Hard limits on pages processed and uncached provider work."""

    max_pages: int = 100
    max_characters: int = 500_000
    max_cost_usd: Decimal = Decimal("5.00")

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be positive")
        if self.max_characters < 0:
            raise ValueError("max_characters cannot be negative")
        if self.max_cost_usd < 0:
            raise ValueError("max_cost_usd cannot be negative")


@dataclass(frozen=True)
class TranslationOutcome:
    document: DocumentText
    target_language: str
    translator_version: str
    cache_hits: int
    provider_pages_attempted: int
    provider_characters_attempted: int
    estimated_cost_usd: Decimal
    actual_cost_usd: Decimal | None


def translate_document(
    document: DocumentText,
    *,
    translator: PageTranslator,
    cache: TranslationCache,
    limits: TranslationLimits | None = None,
    target_language: str = "en",
) -> TranslationOutcome:
    """Translate independently by source page, never obscuring incomplete coverage.

    Character and cost limits apply to uncached provider submissions. ``max_pages``
    bounds the total returned translation work, including cache hits.
    """

    normalized_target = target_language.strip().lower()
    if not normalized_target:
        raise ValueError("target_language must be non-empty")
    if normalized_target != "en" and not normalized_target.startswith("en-"):
        raise ValueError("DocumentText currently supports English translation targets only")
    version = translator.version.strip()
    if not version:
        raise ValueError("translator version must be non-empty")
    effective_limits = limits or TranslationLimits()
    actual_cost_start = (
        translator.actual_cost_usd if isinstance(translator, CostReportingTranslator) else None
    )

    if _same_language(document.source_language, normalized_target):
        translated_document = document.model_copy(
            update={
                "english_translation_pages": None,
                "translation_missing_pages": [],
                "translation_status": TranslationStatus.NOT_NEEDED,
            }
        )
        return TranslationOutcome(
            document=translated_document,
            target_language=normalized_target,
            translator_version=version,
            cache_hits=0,
            provider_pages_attempted=0,
            provider_characters_attempted=0,
            estimated_cost_usd=Decimal("0"),
            actual_cost_usd=Decimal("0") if actual_cost_start is not None else None,
        )

    translated_pages: list[DocumentPage] = []
    notes: list[str] = []
    cache_hits = 0
    provider_pages = 0
    provider_characters = 0
    estimated_cost = Decimal("0")

    for index, page in enumerate(document.pages):
        if index >= effective_limits.max_pages:
            notes.append(f"Translation stopped at the {effective_limits.max_pages}-page limit.")
            break
        if not page.text.strip():
            notes.append(
                f"Page {page.page_number} has no extracted text and was not sent for translation."
            )
            continue
        key = TranslationCacheKey(
            document_sha256=document.sha256,
            page_number=page.page_number,
            page_text_sha256=hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
            source_language=(
                document.source_language.strip().lower() if document.source_language else None
            ),
            target_language=normalized_target,
            translator_version=version,
        )
        cached = cache.get(key)
        if cached is not None:
            translated_pages.append(DocumentPage(page_number=page.page_number, text=cached))
            cache_hits += 1
            continue

        character_count = len(page.text)
        if provider_characters + character_count > effective_limits.max_characters:
            notes.append(f"Page {page.page_number} skipped by the character limit.")
            continue
        try:
            page_cost = translator.estimate_cost_usd(
                page.text,
                source_language=document.source_language,
                target_language=normalized_target,
            )
            if page_cost < 0:
                raise ValueError("translator returned a negative cost estimate")
        except Exception as exc:
            notes.append(f"Page {page.page_number} cost estimation failed ({type(exc).__name__}).")
            continue
        actual_cost_so_far = (
            translator.actual_cost_usd - actual_cost_start
            if actual_cost_start is not None and isinstance(translator, CostReportingTranslator)
            else Decimal("0")
        )
        committed_cost = max(estimated_cost, actual_cost_so_far)
        if committed_cost + page_cost > effective_limits.max_cost_usd:
            notes.append(f"Page {page.page_number} skipped by the cost limit.")
            continue

        provider_pages += 1
        provider_characters += character_count
        estimated_cost += page_cost
        try:
            translated = translator.translate_page(
                page.text,
                source_language=document.source_language,
                target_language=normalized_target,
            )
            if page.text and not translated.strip():
                raise ValueError("translator returned empty text")
        except Exception as exc:
            notes.append(f"Page {page.page_number} translation failed ({type(exc).__name__}).")
            continue
        cache.set(key, translated)
        translated_pages.append(DocumentPage(page_number=page.page_number, text=translated))

    translated_numbers = {page.page_number for page in translated_pages}
    missing_pages = [
        page.page_number for page in document.pages if page.page_number not in translated_numbers
    ]
    if not missing_pages and translated_pages:
        status = TranslationStatus.TRANSLATED
    elif translated_pages:
        status = TranslationStatus.PARTIAL
    else:
        status = TranslationStatus.FAILED
    notes.append(
        f"Translation coverage: {len(translated_pages)}/{document.page_count} pages; "
        f"missing pages: {missing_pages or 'none'}."
    )
    actual_cost = (
        translator.actual_cost_usd - actual_cost_start
        if actual_cost_start is not None and isinstance(translator, CostReportingTranslator)
        else None
    )
    cost_note = f"Translation cost estimate for provider attempts: ${estimated_cost:.6f}."
    if actual_cost is not None:
        cost_note += f" Provider-reported actual cost: ${actual_cost:.6f}."
    notes.append(cost_note)
    translated_document = document.model_copy(
        update={
            "english_translation_pages": translated_pages or None,
            "translation_missing_pages": missing_pages,
            "translation_status": status,
            "coverage_notes": [*document.coverage_notes, *notes],
            "provenance": [
                *document.provenance,
                f"translation:{version}:target={normalized_target}",
                f"translation_cost:estimated={estimated_cost}:actual={actual_cost}",
            ],
        }
    )
    # model_copy(update=...) does not revalidate Pydantic models.
    translated_document = DocumentText.model_validate(translated_document.model_dump())
    return TranslationOutcome(
        document=translated_document,
        target_language=normalized_target,
        translator_version=version,
        cache_hits=cache_hits,
        provider_pages_attempted=provider_pages,
        provider_characters_attempted=provider_characters,
        estimated_cost_usd=estimated_cost,
        actual_cost_usd=actual_cost,
    )


def translate_document_with_provider(
    document: DocumentText,
    *,
    translator: PageTranslator | None,
    cache: TranslationCache,
    limits: TranslationLimits | None = None,
    target_language: str = "en",
) -> TranslationOutcome | DataSourceError:
    """Fail explicitly when no live translation provider has been configured."""

    if translator is None:
        return DataSourceError(
            error_code="not_found",
            message="No filing translation provider is configured",
            stage="translate",
            source="filing_translation",
        )
    return translate_document(
        document,
        translator=translator,
        cache=cache,
        limits=limits,
        target_language=target_language,
    )


def _serialized_cache_key(key: TranslationCacheKey) -> str:
    return make_key(
        "filing_translation",
        key.document_sha256,
        str(key.page_number),
        key.page_text_sha256,
        key.source_language or "unknown",
        key.target_language,
        key.translator_version,
    )


def _same_language(source_language: str | None, target_language: str) -> bool:
    if source_language is None:
        return False
    return (
        source_language.strip().lower().split("-", maxsplit=1)[0]
        == target_language.split("-", maxsplit=1)[0]
    )

"""Shared bounded HTTP policy for official regional filing archives."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode, urljoin, urlparse

import requests

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError, ErrorStage
from data_sources.filing_models import SourceSystem


@dataclass(frozen=True)
class HttpDocument:
    """One fetched archive page with response provenance."""

    url: str
    text: str
    fetched_at: datetime
    etag: str | None
    last_modified: str | None
    status_code: int
    from_cache: bool = False


@dataclass(frozen=True)
class HttpBytesDocument:
    """One fetched binary attachment with response provenance."""

    url: str
    content: bytes
    fetched_at: datetime
    etag: str | None
    last_modified: str | None
    content_type: str | None
    status_code: int
    from_cache: bool = False


class RegionalHttpClient:
    """GET-only client enforcing hosts, timeouts, throttling, retries, and cache."""

    def __init__(
        self,
        *,
        source: SourceSystem,
        allowed_hosts: frozenset[str],
        db_conn: sqlite3.Connection | None = None,
        session: requests.Session | None = None,
        timeout: tuple[float, float] = (5.0, 20.0),
        max_attempts: int = 3,
        max_redirects: int = 3,
        max_response_bytes: int = 5_000_000,
        min_interval_seconds: float = 0.25,
        _sleep: Callable[[float], None] = time.sleep,
        _monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not allowed_hosts:
            raise ValueError("at least one official archive host must be allowed")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self._source = source
        self._allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self._cache = CacheStore(db_conn) if db_conn is not None else None
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._max_redirects = max_redirects
        self._max_response_bytes = max_response_bytes
        self._min_interval = min_interval_seconds
        self._sleep = _sleep
        self._monotonic = _monotonic
        self._last_request_at: float | None = None

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        """Fetch text from an allowlisted HTTPS host without raising expected errors."""
        host_error = self._validate_url(url)
        if host_error is not None:
            return host_error
        normalized_params = tuple(sorted((params or {}).items()))
        request_url = url
        if normalized_params:
            request_url = f"{url}?{urlencode(normalized_params)}"
        cache_key = make_key("regional_http", self._source.value, request_url)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                try:
                    document = self._decode_cached(cached)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    return self._error("parse", f"invalid cached archive response: {exc}")
                cached_url_error = self._validate_url(document.url)
                if cached_url_error is not None:
                    return cached_url_error
                return document

        for attempt in range(self._max_attempts):
            try:
                response_or_error = self._get_with_redirects(
                    url, params=dict(normalized_params), stream=True
                )
            except requests.RequestException as exc:
                if attempt + 1 < self._max_attempts:
                    self._sleep(2.0**attempt)
                    continue
                return self._error("network", f"official archive request failed: {exc}")
            if isinstance(response_or_error, DataSourceError):
                return response_or_error
            response = response_or_error
            if response.status_code == 429:
                response.close()
                if attempt + 1 < self._max_attempts:
                    self._sleep(self._retry_after(response, attempt))
                    continue
                return self._error("rate_limit", "official archive rate limit exhausted")
            if response.status_code >= 500:
                response.close()
                if attempt + 1 < self._max_attempts:
                    self._sleep(2.0**attempt)
                    continue
                return self._error(
                    "network", f"official archive returned HTTP {response.status_code}"
                )
            if response.status_code >= 400:
                code = "not_found" if response.status_code == 404 else "network"
                response.close()
                return self._error(code, f"official archive returned HTTP {response.status_code}")

            try:
                response_text = self._read_bounded_text(response)
            finally:
                response.close()
            if isinstance(response_text, DataSourceError):
                return response_text
            document = HttpDocument(
                url=str(response.url or request_url),
                text=response_text,
                fetched_at=datetime.now(timezone.utc),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                status_code=response.status_code,
            )
            if self._cache is not None:
                self._cache.set(cache_key, self._encode_cached(document), cache_ttl_hours)
            return document

        raise AssertionError("positive max_attempts guarantees a return")

    def get_bytes(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        max_bytes: int = 25_000_000,
    ) -> HttpBytesDocument | DataSourceError:
        """Fetch a bounded binary attachment through the same safe HTTP boundary."""
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        host_error = self._validate_url(url, stage="download")
        if host_error is not None:
            return host_error
        normalized_params = tuple(sorted((params or {}).items()))
        request_url = url
        if normalized_params:
            request_url = f"{url}?{urlencode(normalized_params)}"
        for attempt in range(self._max_attempts):
            try:
                response_or_error = self._get_with_redirects(
                    url, params=dict(normalized_params), stage="download", stream=True
                )
            except requests.RequestException as exc:
                if attempt + 1 < self._max_attempts:
                    self._sleep(2.0**attempt)
                    continue
                return self._error(
                    "network", f"official attachment request failed: {exc}", stage="download"
                )
            if isinstance(response_or_error, DataSourceError):
                return response_or_error
            response = response_or_error
            try:
                if response.status_code == 429:
                    if attempt + 1 < self._max_attempts:
                        self._sleep(self._retry_after(response, attempt))
                        continue
                    return self._error(
                        "rate_limit", "official attachment rate limit exhausted", stage="download"
                    )
                if response.status_code >= 500:
                    if attempt + 1 < self._max_attempts:
                        self._sleep(2.0**attempt)
                        continue
                    return self._error(
                        "network",
                        f"official attachment returned HTTP {response.status_code}",
                        stage="download",
                    )
                if response.status_code >= 400:
                    code = "not_found" if response.status_code == 404 else "network"
                    return self._error(
                        code,
                        f"official attachment returned HTTP {response.status_code}",
                        stage="download",
                    )
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None:
                    try:
                        parsed_length = int(declared_length)
                        if parsed_length < 0:
                            return self._error(
                                "parse",
                                "attachment has invalid Content-Length",
                                stage="download",
                            )
                        if parsed_length > max_bytes:
                            return self._error(
                                "parse",
                                "attachment exceeds configured byte limit",
                                stage="download",
                            )
                    except ValueError:
                        return self._error(
                            "parse", "attachment has invalid Content-Length", stage="download"
                        )
                content = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    content.extend(chunk)
                    if len(content) > max_bytes:
                        return self._error(
                            "parse", "attachment exceeds configured byte limit", stage="download"
                        )
                return HttpBytesDocument(
                    url=str(response.url or request_url),
                    content=bytes(content),
                    fetched_at=datetime.now(timezone.utc),
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                    content_type=response.headers.get("Content-Type"),
                    status_code=response.status_code,
                )
            except requests.RequestException as exc:
                if attempt + 1 < self._max_attempts:
                    self._sleep(2.0**attempt)
                    continue
                return self._error(
                    "network", f"official attachment stream failed: {exc}", stage="download"
                )
            finally:
                response.close()

        raise AssertionError("positive max_attempts guarantees a return")

    def _get_with_redirects(
        self,
        url: str,
        *,
        params: dict[str, str],
        stage: ErrorStage = "discovery",
        stream: bool = False,
    ) -> requests.Response | DataSourceError:
        current_url = url
        current_params: dict[str, str] | None = params
        prepared_url = requests.Request("GET", current_url, params=current_params).prepare().url
        if prepared_url is None:
            return self._error("parse", "could not prepare official archive URL")
        visited = {prepared_url}
        for redirect_count in range(self._max_redirects + 1):
            self._throttle()
            response = self._session.get(
                current_url,
                params=current_params,
                timeout=self._timeout,
                allow_redirects=False,
                stream=stream,
            )
            response_url = str(response.url or current_url)
            response_url_error = self._validate_url(response_url, stage=stage)
            if response_url_error is not None:
                if stream:
                    response.close()
                return response_url_error
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("Location")
            if not location:
                if stream:
                    response.close()
                return self._error(
                    "parse", "official archive redirect omitted Location", stage=stage
                )
            next_url = urljoin(response_url, location)
            next_url_error = self._validate_url(next_url, stage=stage)
            if next_url_error is not None:
                if stream:
                    response.close()
                return next_url_error
            if next_url in visited:
                if stream:
                    response.close()
                return self._error(
                    "network", "official archive redirect loop detected", stage=stage
                )
            if redirect_count == self._max_redirects:
                if stream:
                    response.close()
                return self._error(
                    "network", "official archive redirect limit exceeded", stage=stage
                )
            visited.add(next_url)
            if stream:
                response.close()
            current_url = next_url
            # Location is a complete next request target. Reapplying the original
            # params can duplicate or alter its query string.
            current_params = None
        raise AssertionError("bounded redirect loop always returns")

    def _read_bounded_text(self, response: requests.Response) -> str | DataSourceError:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                if int(declared) > self._max_response_bytes:
                    return self._error("parse", "official archive response exceeded size limit")
            except ValueError:
                return self._error("parse", "official archive returned invalid Content-Length")
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > self._max_response_bytes:
                return self._error("parse", "official archive response exceeded size limit")
        encoding = response.encoding or "utf-8"
        return bytes(content).decode(encoding, errors="replace")

    def validate_url(self, url: str) -> DataSourceError | None:
        """Validate a discovered landing or attachment URL against the host policy."""
        return self._validate_url(url)

    def _validate_url(self, url: str, *, stage: ErrorStage = "discovery") -> DataSourceError | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or host not in self._allowed_hosts:
            return self._error(
                "parse", f"refusing non-official or non-HTTPS archive URL: {url}", stage=stage
            )
        return None

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self._min_interval - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    @staticmethod
    def _retry_after(response: requests.Response, attempt: int) -> float:
        value = response.headers.get("Retry-After")
        if value is not None:
            try:
                return min(max(float(value), 0.0), 30.0)
            except ValueError:
                pass
        return 2.0**attempt

    def _error(
        self, code: str, message: str, *, stage: ErrorStage = "discovery"
    ) -> DataSourceError:
        return DataSourceError(
            error_code=code,
            message=message,
            stage=stage,
            source=self._source,
        )

    @staticmethod
    def _encode_cached(document: HttpDocument) -> str:
        return json.dumps(
            {
                "url": document.url,
                "text": document.text,
                "fetched_at": document.fetched_at.isoformat(),
                "etag": document.etag,
                "last_modified": document.last_modified,
                "status_code": document.status_code,
            },
            sort_keys=True,
        )

    @staticmethod
    def _decode_cached(payload: str) -> HttpDocument:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("cached regional response must be an object")
        return HttpDocument(
            url=str(value["url"]),
            text=str(value["text"]),
            fetched_at=datetime.fromisoformat(str(value["fetched_at"])),
            etag=str(value["etag"]) if value.get("etag") is not None else None,
            last_modified=(
                str(value["last_modified"]) if value.get("last_modified") is not None else None
            ),
            status_code=int(str(value["status_code"])),
            from_cache=True,
        )

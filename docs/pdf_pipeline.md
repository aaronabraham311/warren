# Regional PDF extraction and translation

Warren treats filing bytes, extracted text, and translated text as immutable evidence.
SQLite stores discovery metadata, checksums, versions, statuses, and relative artifact
keys; bodies live only in the SHA-256-addressed `ArtifactStore`.

## Dependency bakeoff

The project Python 3.13 environment initially contained no `pypdf`, `pdfplumber`,
PyMuPDF, Tesseract wrapper, or language detector. The bundled desktop runtime exposed
`pypdf` 6.10, `pdfplumber` 0.11.9, ReportLab 4.4.9, and Poppler 26.05, but those Python
3.12 packages are not valid production dependencies for Warren's Python 3.13 runtime.
Poppler supplied `pdftoppm`/`pdfinfo`; no Tesseract executable was available.

`pypdf` is therefore pinned as the single runtime extraction dependency. It preserves
PDF pages, handles encryption detection, and exposes page content streams for preflight
bounds. Adding `pdfplumber` would duplicate its parser dependency without improving the
current page-text contract. OCR remains an injected backend: the default requires both
Poppler and Tesseract and reports missing/partial coverage when Tesseract is unavailable.

## Safety and provenance

- Binary downloads use `RegionalHttpClient` with HTTPS host allowlists, manually
  validated redirects, timeouts, retry/throttle policy, streamed 64 KiB chunks, declared
  and decoded byte limits, and no SQLite body cache.
- PDFs require a compatible content type, `%PDF-` signature, and trailing EOF marker
  before storage. `ArtifactStore` re-verifies SHA-256 on every read.
- Extraction temporarily lowers pypdf's decompression ceiling under a process lock, then
  enforces per-page and cumulative uncompressed-content limits plus a cumulative extracted-
  text limit. It is also bounded by document pages and selective OCR page count. Output
  always preserves every 1-based source page; unresolved sparse pages are empty and marked
  incomplete.
- Language detection is conservative and reports confidence. Discovery language metadata
  is never substituted for detector output.
- Translation is page-by-page and keyed by document SHA-256, page number, extracted-page
  text SHA-256, source/target languages, and translator/prompt version. Page bodies are
  artifacts; the shared SQLite cache stores only their artifact metadata. Blank pages are
  never sent to a provider.
- Default OCR subprocesses have 30-second timeouts and Poppler renders at a maximum
  2400-pixel page dimension before Tesseract runs, bounding decoded raster geometry without
  relying on platform-specific process resource controls.
- `TRANSLATED`, `PARTIAL`, and `FAILED` reflect exact page coverage. An unset translation
  model is an explicit unavailable-provider result, not a translation claim.

## Opt-in translation

Live translation is disabled unless all three variables are configured:
`WARREN_TRANSLATION_MODEL`,
`WARREN_TRANSLATION_INPUT_USD_PER_MILLION_TOKENS`, and
`WARREN_TRANSLATION_OUTPUT_USD_PER_MILLION_TOKENS`. Prices are explicit because model
pricing changes; they feed the hard estimated-cost limit. The provider version includes
the model and prompt version, so changing either invalidates cached pages.
Translation provider spend is currently bounded and reported by the PDF translation
limits, but it is not debited from `RunContext`'s agent-loop token budget. Treat the
translation cap as separate accounting until those budgets are unified.

## `read_filing`

For regional issuers, `read_filing` selects manifests by exact document kind and by
reporting-period year (falling back to publication year), never retrieval order alone.
SEC aliases remain compatible: `10-K` maps to `annual` and `10-Q` maps to `quarterly` for
stored PDFs, while unchanged SEC calls fall through to EDGAR. Regional section boundaries
are not standardized, so `full_document` is preferred; other section requests return
bounded full text with an explicit warning. Every returned source page has one citation.

# Eval fixtures

Two kinds of fixture live under this directory. They serve different callers and are
recorded by different commands — don't mix them up.

| Path | Contains | Used by | Recorded with |
|---|---|---|---|
| `{TICKER}/{client}/{method}/{hash}.json` | the **raw upstream payload** a data-source client parses | data-fetcher unit tests (they exercise the real parsing path) | `python -m eval.fixtures --record AAPL` |
| `{TICKER}/tools/{tool_name}/{hash}.json` | the **serialized `ToolResult`** a tool returns | the eval harness, via `eval.tool_fixtures.FixtureToolRunner` | `python -m eval.fixtures.recorder AAPL` |

The filename stem is `sha256(json.dumps(input, sort_keys=True))[:8]` over the validated
tool input, so `get_news(ticker="AAPL")` and `get_news(ticker="AAPL", days=7)` resolve to
one file. Error cases recorded by hand use descriptive stems (`error_not_found.json`).

## Replaying

`eval/tool_fixtures.py` owns the tool-level format — both ends of it. `FixtureToolRunner`
satisfies `agent.loop.ToolRunner` and serves results straight from disk, so replay never
constructs a data-source client and cannot reach the network *by construction*. A call with
no recorded fixture is recorded on `.misses` and comes back as a non-retryable
`ToolResultError`, which the eval runner grades as a `fixture_missing` failure rather than
letting the loop retry into backoff.

A tool that failed at record time is stored as a `ToolResultError` and replays as that
same error. Errors are data in this codebase (Tech Spec §5), so a data source that was
genuinely unavailable stays deterministic rather than leaving a hole.

## Recording

```bash
python -m eval.fixtures.recorder                # every golden-set ticker
python -m eval.fixtures.recorder AAPL BRK.B     # named tickers
```

Hits live APIs once per `RECORDED_CALLS` entry and **overwrites** existing fixtures in
place. Needs network; `get_news` and `get_insider_activity` additionally need
`FINNHUB_API_KEY` (they record as errors without it).

## Rotation policy

Fixtures rot. yfinance's schema drifts, filing dates advance, news windows slide, and a
`recorded_at` older than 90 days makes replay emit a staleness warning.

Refresh:

- **quarterly, at minimum** — the 90-day warning is the reminder;
- after any S&P 500 rebalance that adds or removes a golden-set ticker;
- whenever a golden-set YAML's `last_curated` changes, so expectations and data agree;
- after a data-source client changes what it parses (the recorded `ToolResult` shape
  moves with the model).

Re-record, run `pytest`, and eyeball the diff before committing: a fixture diff is a
change in what the eval measures.

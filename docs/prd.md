# PRD: Local Stock Analysis Agent Harness
 
**Version:** 1.0 (Draft)
**Owner:** [You]
**Last updated:** 2026-05-06
**Status:** Planning
 
---
 
## 1. Project Overview
 
### 1.1 What this is
A locally-running, autonomous research agent that analyzes a personal stock portfolio overnight and surfaces buy/sell/hold perspectives plus discovery candidates, viewable through a local dashboard each morning.
 
### 1.2 Why this exists (primary goal)
**This is an agent-harness learning project. Stock analysis is the problem domain, not the deliverable.**
 
The primary success metric is the user's understanding of how to design, instrument, and iterate on production-quality LLM agents. The investment outputs are a secondary, motivating output that keeps the project grounded in something real.
 
This framing has direct consequences throughout the doc:
- We optimize for **instructive architecture**, not maximum signal quality.
- We invest in **observability and evals from day one**, not as an afterthought.
- We pick **boring data sources** so cognitive budget goes to the agent, not the plumbing.
- We treat **cost-aware model routing** as a first-class harness skill to practice.
### 1.3 Why an agent at all (vs. a script)?
A deterministic script could produce ratios and screens. An agent earns its complexity when:
- The task requires **flexible reasoning across heterogeneous data** (filings, news, fundamentals, prices)
- **Tool selection** is non-obvious (when to read a 10-K vs. check news vs. compute a ratio)
- **Output quality benefits from iteration** (re-checking, reconsidering, synthesizing)
Stock research has all three properties — making it a legitimate (and instructive) agent use case.
 
---
 
## 2. Non-Goals
 
Explicitly **out of scope** for v1:
 
- ❌ **Automated trading.** The agent never places orders. Outputs are advisory only.
- ❌ **Real-time monitoring.** Nightly batch only. No intraday alerts.
- ❌ **Multi-user / cloud deployment.** Local single-user only.
- ❌ **Brokerage API integration.** Manual CSV portfolio input.
- ❌ **Paid data sources.** Free APIs only.
- ❌ **Backtesting framework.** Recommendation tracking is for *qualitative* eval, not quantitative backtests.
- ❌ **Tax-aware advice, options, derivatives, crypto.** US equities only.
- ❌ **Polished UX.** Streamlit's defaults are fine.
- ❌ **Multi-agent orchestration in v1.** Reserved for v2.
---
 
## 3. Users & Use Cases
 
### 3.1 User
A single user (you) running the system on a personal machine.
 
### 3.2 Primary use cases
1. **Morning portfolio review.** Open dashboard, see overnight analysis of current holdings with buy/sell/hold perspective and reasoning trace.
2. **Discovery review.** See 3-5 new candidates the agent flagged, with thesis fit explanation.
3. **Recommendation history review.** Look back at past recommendations to evaluate agent quality over time.
4. **Harness iteration.** Modify prompts/tools/models, replay against a golden eval set, compare results.
### 3.3 Anti-use cases
- Quick lookups during the trading day (use Fidelity for that)
- Definitive buy/sell decisions (the agent is a research assistant, not a decision-maker)
---
 
## 4. System Architecture
 
### 4.1 v1 — Single-agent loop with tools
 
```
┌─────────────────────────────────────────────────────────────┐
│                    Scheduler (cron / Python)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ nightly trigger
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       Agent Runner                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Single Claude Agent Loop                              │ │
│  │  - System prompt: blended Lynch/Buffett persona        │ │
│  │  - Model routing: Haiku → Sonnet → Opus by phase       │ │
│  │  - Tools: see §6.1                                     │ │
│  │  - Token budget enforcement                            │ │
│  │  - Max iteration cap                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Data sources │   │   SQLite     │   │  Logs / JSONL│
│ - yfinance   │   │ - holdings   │   │ - traces     │
│ - SEC EDGAR  │   │ - analyses   │   │ - tool calls │
│ - Finnhub    │   │ - recs       │   │ - tokens     │
└──────────────┘   └──────────────┘   └──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Streamlit Dashboard (localhost)               │
│  - Today's analysis     - History     - Eval / replay        │
└─────────────────────────────────────────────────────────────┘
```
 
### 4.2 v2 — Multi-agent evolution (future)
Same data layer, but the single agent splits into:
- **Orchestrator** — plans the run, dispatches work
- **Lynch analyst** — applies "invest in what you know" + GARP heuristics
- **Buffett analyst** — applies moat + owner earnings + margin-of-safety heuristics
- **Discovery agent** — screens universe for thesis fit
- **Synthesizer** — consolidates perspectives into final output
The v1 architecture should make this transition cheap. Concretely: the analysis function should be cleanly callable so v2 can call it N times with different personas.
 
---
 
## 5. Component Specifications
 
### 5.1 Portfolio reader
- **Input:** `portfolio.csv` with columns: `ticker, shares, cost_basis, purchase_date`
- **Source of truth:** manually maintained from Fidelity export
- **Validation:** ticker must be valid (verifiable via yfinance), no negative shares
- **Watchlist:** separate `watchlist.csv` with just `ticker, notes`
### 5.2 Data fetchers
Wrapped as agent tools (see §6.1). Each fetcher:
- Caches results to SQLite with TTL (24h for fundamentals, 1h for prices)
- Handles rate limits gracefully with exponential backoff
- Returns structured dicts, never raw HTML/JSON dumps
| Source | Use | Rate limit | Cost |
|---|---|---|---|
| **yfinance** | Prices, basic fundamentals, ratios | Soft (be polite) | Free |
| **SEC EDGAR** | 10-K, 10-Q, 8-K filings | 10 req/sec | Free |
| **Finnhub free** | News, basic estimates | 60/min | Free |
 
### 5.3 Agent runner
See §6 for full spec. Core responsibilities:
- Load portfolio + watchlist
- Run cheap screen across S&P 500 + watchlist (Haiku)
- Deep-analyze portfolio holdings + top discovery candidates (Sonnet)
- Synthesize daily summary (Opus, only if needed)
- Write structured outputs to SQLite
- Emit observability events
### 5.4 Storage layer (SQLite)
 
```sql
-- Current holdings snapshot (overwritten each run)
holdings(ticker, shares, cost_basis, purchase_date, current_price, updated_at)
 
-- Watchlist
watchlist(ticker, notes, added_at)
 
-- Analysis output (append-only)
analyses(
  id, run_id, ticker, analysis_type,  -- 'holding' | 'discovery'
  recommendation,                      -- 'buy' | 'sell' | 'hold'
  confidence,                          -- 0.0-1.0
  thesis,                              -- markdown blob
  lynch_signals,                       -- JSON
  buffett_signals,                     -- JSON
  key_risks,                           -- JSON array
  created_at
)
 
-- Run metadata for observability + cost tracking
runs(
  id, started_at, completed_at, status,
  total_input_tokens, total_output_tokens,
  total_cost_usd, num_tool_calls, error_msg
)
 
-- Tool call log (one row per call)
tool_calls(
  id, run_id, tool_name, input_json, output_json,
  latency_ms, error_msg, created_at
)
 
-- Eval / golden set
eval_examples(ticker, expected_recommendation, expected_thesis_keywords, notes)
eval_runs(id, run_id, example_ticker, passed, diff_notes)
```
 
### 5.5 Scheduler
**v1:** macOS `launchd` plist or Linux `cron` job. Runs at 2 AM local time.
**Why not a Python scheduler library?** OS-level scheduling is more robust; the agent process should be short-lived, not long-running.
 
### 5.6 Streamlit dashboard
Three pages:
1. **Today** — most recent run's analysis cards (holdings + discoveries), with expandable reasoning trace
2. **History** — searchable table of past recommendations, filterable by ticker/date/recommendation
3. **Eval** — golden set status, replay button, diff view of prompt-version-over-prompt-version
---
 
## 6. Agent Design Details
 
### 6.1 Tools available to the agent
 
| Tool | Description | Returns |
|---|---|---|
| `get_quote(ticker)` | Current price, day change, volume | dict |
| `get_fundamentals(ticker)` | P/E, P/B, ROE, debt/equity, FCF, margins | dict |
| `get_growth_metrics(ticker)` | Revenue/earnings CAGR (3y, 5y), PEG | dict |
| `read_filing(ticker, type, section?)` | Pull 10-K/10-Q section (MD&A, risk factors, etc.) | str |
| `get_news(ticker, days=7)` | Recent headlines + summaries | list[dict] |
| `screen_universe(criteria)` | Apply quantitative filters across S&P 500 | list[ticker] |
| `get_holding_context(ticker)` | User's cost basis, purchase date, current P/L | dict |
 
**Design principle:** tools return small, structured payloads. Filing-reading is the one exception — it returns text, but a *section*, not the whole filing. The agent must specify which section it wants.
 
### 6.2 System prompt structure (blended Lynch/Buffett persona)
 
```
You are a research analyst who blends two investing philosophies:
 
LYNCH HEURISTICS (apply when evaluating growth):
- Invest in what you understand
- Prefer companies with simple, explainable businesses
- Look for "stalwarts" and "fast growers" with reasonable PEG (<1.5)
- Be skeptical of stories without numbers
 
BUFFETT HEURISTICS (apply when evaluating quality):
- Look for durable competitive advantages (moats)
- Prefer high ROE sustained over years without excessive leverage
- Demand a margin of safety vs. intrinsic value estimate
- Owner earnings > reported earnings
 
OUTPUT REQUIREMENTS:
- Recommendation: buy / sell / hold
- Confidence: 0.0-1.0
- Thesis: 3-5 bullet points
- Lynch signals: which heuristics apply, pro/con
- Buffett signals: which heuristics apply, pro/con
- Key risks: 2-3 specific, concrete risks (no platitudes)
 
EXPLICIT GUARDRAILS:
- Never recommend without citing specific numbers from your tools
- If data is missing, say so — do not invent figures
- "Hold" is a valid answer; do not feel pressure to act
- Flag any data that looks stale or implausible
```
 
### 6.3 Cost control: model routing strategy
 
| Phase | Model | Why |
|---|---|---|
| **Screen S&P 500** | Claude Haiku 4.5 | Cheap pass over many tickers; just needs to flag candidates |
| **Deep analysis** (holdings + top discovery) | Claude Sonnet 4.6 | Reasoning-heavy, needs strong tool use |
| **Final synthesis** (only if results conflict) | Claude Opus 4.7 | Reserved for hard cases; most runs skip this |
 
**Prompt caching:** the system prompt + persona definitions + user holdings are static across the run. Cache them aggressively (Anthropic's prompt caching gives ~90% discount on cached tokens). Realistic savings: 40-60% on input cost.
 
**Hard caps per run:**
- Max iterations per ticker: 8 tool calls
- Max total tokens per run: 1.5M input / 200K output
- Hard cost ceiling: $1.50/run; abort if exceeded
### 6.4 Output schema (per ticker)
JSON, validated with Pydantic before write:
```python
{
  "ticker": str,
  "analysis_type": "holding" | "discovery",
  "recommendation": "buy" | "sell" | "hold",
  "confidence": float,  # 0.0-1.0
  "thesis": str,  # markdown
  "lynch_signals": {"pros": [str], "cons": [str]},
  "buffett_signals": {"pros": [str], "cons": [str]},
  "key_risks": [str],
  "data_quality_notes": [str],  # flag stale/missing data
  "tool_calls_made": int,
  "tokens_used": int
}
```
 
---
 
## 7. Observability (Medium tier)
 
### 7.1 What gets logged
- **Per run:** start/end time, status, total tokens, total cost, tool call count
- **Per tool call:** name, input, output (truncated), latency, error if any
- **Per agent decision point:** the model's stated reasoning when choosing a tool (extracted from response)
### 7.2 Where it goes
- Structured logs → JSONL files in `logs/runs/{run_id}.jsonl`
- Aggregates → SQLite (`runs`, `tool_calls` tables)
- Surfaced in dashboard's "Today" and "History" pages
### 7.3 Metrics tracked
- Cost per run (USD)
- Tokens in/out per run
- Tool calls per ticker (distribution)
- Wall-clock time per run
- Error rate (failed tool calls / total)
- Recommendation distribution (% buy/sell/hold)
### 7.4 What we explicitly skip in v1
- Distributed tracing (OpenTelemetry, etc.) — overkill for single-process
- Real-time dashboards — overnight batch doesn't need them
- LangSmith / external observability platforms — local-only requirement
---
 
## 8. Eval Infrastructure
 
### 8.1 Why this exists
The single highest-leverage harness skill is **knowing whether a change made things better or worse**. Without evals, prompt iteration is vibes.
 
### 8.2 v1 eval design
- **Golden set:** 10-15 hand-picked tickers across sectors, with notes on what good analysis should surface (e.g., "AAPL: should mention ecosystem moat, services growth; should NOT recommend sell at current valuation without specific catalyst")
- **Replay command:** `python -m agent.eval --golden-set` runs the agent against the golden set with a fixed timestamp seed
- **Diff view:** dashboard page showing current run vs. last run vs. expected, side-by-side
- **Pass/fail criteria:** keyword presence in thesis, recommendation directionality, no hallucinated metrics
### 8.3 What this enables
- Confident prompt iteration ("did changing the persona prompt break anything?")
- Regression detection on data source changes
- Honest comparison of model routing strategies (does swapping Sonnet → Haiku for screening hurt quality?)
---
 
## 9. Phased Roadmap (6 weeks @ ~5 hrs/week = ~30 hrs total)
 
### Week 1 — Skeleton (5 hrs)
**Goal: end-to-end "hello world" with one ticker**
- Repo setup, `.env` for API keys, dependency management (uv or poetry)
- SQLite schema + migrations
- Single hardcoded ticker (e.g., AAPL)
- One tool (`get_quote`) wired into a minimal Claude Sonnet agent loop
- JSONL logging
- **Done when:** running `python -m agent.run AAPL` produces a recommendation in the DB
### Week 2 — Data + Tools (5 hrs)
**Goal: full tool surface working**
- Implement all 7 tools from §6.1
- yfinance, SEC EDGAR, Finnhub clients with caching + rate limit handling
- Pydantic models for all tool I/O
- Unit tests for data fetchers (against recorded fixtures)
- **Done when:** agent can analyze any single ticker using all tools
### Week 3 — Persona + Routing + Caching (5 hrs)
**Goal: cost-aware harness**
- System prompt with blended Lynch/Buffett persona
- Model routing logic (Haiku/Sonnet/Opus)
- Prompt caching wired up
- Token budget enforcement + hard cost cap
- **Done when:** full portfolio run lands under $1/night with full quality
### Week 4 — Discovery + Scheduling (5 hrs)
**Goal: nightly autonomous run**
- S&P 500 screening pass
- Watchlist integration
- launchd/cron setup
- Output validation and DB persistence for full run
- **Done when:** agent runs unattended overnight and writes results to DB
### Week 5 — Dashboard (5 hrs)
**Goal: morning review experience**
- Streamlit app with Today / History pages
- Reasoning trace viewer (expandable per ticker)
- Cost + token usage charts
- **Done when:** you can review a morning's results without touching the DB directly
### Week 6 — Evals + Polish (5 hrs)
**Goal: ability to iterate confidently**
- Golden set creation (10-15 tickers)
- Eval replay command
- Eval page in dashboard
- README, runbook for common failures
- **Done when:** you can change a prompt, run evals, and see whether quality changed
### v2 backlog (post-week-6)
- Split into multi-agent (orchestrator + Lynch + Buffett + discovery + synthesizer)
- Separate Lynch and Buffett perspectives in output
- Better discovery (sector balance heuristics)
- Recommendation quality tracking (compare past recs to subsequent price action)
- Backtesting framework
---
 
## 10. Open Questions & Risks
 
### 10.1 Open questions
- **Q1: Filing reading depth.** Full 10-K is 100K+ tokens. Strategy: agent requests specific sections. But which sections matter most for Lynch/Buffett analysis? Worth iterating on in week 3.
- **Q2: Discovery candidate count.** 5 per night feels right for review fatigue, but unclear if S&P 500 screening surfaces 5 quality candidates daily. May need cooldown logic.
- **Q3: Stale data handling.** yfinance occasionally returns stale or missing data. How aggressive should the agent's "I don't have enough data" output be?
### 10.2 Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| yfinance breaks (it's scraping-based) | Medium | High | Cache aggressively; add fallback to Finnhub fundamentals |
| Cost blows past $20/mo | Medium | Medium | Hard cap per run; weekly cost review in dashboard |
| Agent hallucinates numbers | Medium | High | Schema validation + "data quality notes" field; eval golden set catches regressions |
| Rate limit lockouts | Low | Medium | Exponential backoff; cache TTLs |
| Recommendations feel like noise | High | Low | This is *expected* in v1; eval infrastructure is the answer, not signal-chasing |
 
### 10.3 Honest disclaimer (worth internalizing)
LLM-driven investment recommendations are not a solved problem. Published research (e.g., FinGPT, BloombergGPT) shows LLMs are useful as **research accelerators** — fast filing analysis, sentiment extraction, anomaly flagging — but autonomous buy/sell calls remain unreliable as standalone signals. Treat this agent's outputs as *prompts for your own thinking*, not as decisions.
 
---
 
## 11. Success Criteria
 
**v1 is done when all of the following are true:**
 
1. ✅ Agent runs unattended nightly and writes structured results to SQLite
2. ✅ Dashboard shows today's analysis + searchable history
3. ✅ Monthly cost stays under $20
4. ✅ Golden eval set runs reproducibly and produces a diff
5. ✅ I can change a prompt, replay evals, and see whether quality improved or regressed
6. ✅ All seven tools work, with caching and rate limit handling
7. ✅ Recommendation outputs validate against the Pydantic schema 100% of the time
8. ✅ I understand the harness well enough to write a 1-page summary of the design choices and tradeoffs
The eighth criterion is the real success metric. The first seven are how we get there.
 
---
 
## Appendix A — Repo structure (proposed)
 
```
stock-agent/
├── pyproject.toml
├── README.md
├── .env.example
├── data/
│   ├── portfolio.csv
│   └── watchlist.csv
├── agent/
│   ├── __init__.py
│   ├── run.py              # entrypoint
│   ├── loop.py             # core agent loop
│   ├── persona.py          # system prompts
│   ├── routing.py          # model selection
│   ├── budget.py           # cost/token enforcement
│   └── tools/
│       ├── quote.py
│       ├── fundamentals.py
│       ├── filings.py
│       ├── news.py
│       ├── screen.py
│       └── holdings.py
├── data_sources/
│   ├── yfinance_client.py
│   ├── edgar_client.py
│   └── finnhub_client.py
├── storage/
│   ├── db.py
│   ├── schema.sql
│   └── migrations/
├── eval/
│   ├── golden_set.py
│   ├── runner.py
│   └── examples/
├── dashboard/
│   ├── app.py              # Streamlit entrypoint
│   ├── pages/
│   │   ├── today.py
│   │   ├── history.py
│   │   └── eval.py
│   └── components/
└── logs/
    └── runs/               # JSONL traces
```
 
## Appendix B — Decisions log
 
| # | Decision | Rationale |
|---|---|---|
| 1 | Python | Best agent ecosystem, matches your skill goals |
| 2 | Single-agent v1 → multi-agent v2 | Learn loop mechanics first, then orchestration |
| 3 | Streamlit for dashboard | Removes frontend cognitive load |
| 4 | Free data sources only | Adds zero learning value vs. paid; skip the distraction |
| 5 | Manual CSV portfolio | Fidelity has no public retail API; CSV is reliable |
| 6 | Blended Lynch/Buffett persona | Simpler v1; natural split point for v2 multi-agent |
| 7 | S&P 500 + watchlist universe | Manageable size, real discovery surface |
| 8 | <$20/mo via Path C | Forces you to learn model routing + caching — high-leverage skill |
| 9 | Recommendation tracking from day 1 | Eval infrastructure is the most important harness skill |
| 10 | Medium observability | Enough to learn from; not so much it becomes the project |
 

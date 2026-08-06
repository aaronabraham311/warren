-- Reference DDL — not executed at runtime; storage/db.py drives schema creation via SQLAlchemy.

CREATE TABLE IF NOT EXISTS prompt_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  version_tag TEXT NOT NULL,
  persona_system_prompt TEXT,
  routing_policy_name TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,           -- UUID
  prompt_version_id INTEGER REFERENCES prompt_versions(id),
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  status TEXT,                   -- 'success'|'cost_aborted'|'partial'|'failed'
  total_input_tokens INTEGER,
  total_output_tokens INTEGER,
  total_cost_usd REAL,
  num_tool_calls INTEGER,
  error_msg TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
  ticker TEXT PRIMARY KEY,
  shares REAL,
  cost_basis REAL,
  purchase_date DATE,
  current_price REAL,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
  ticker TEXT PRIMARY KEY,
  notes TEXT,
  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(id),
  ticker TEXT,
  analysis_type TEXT,            -- 'holding'|'discovery'
  recommendation TEXT,           -- 'buy'|'sell'|'hold'
  confidence REAL,
  thesis TEXT,
  lynch_signals TEXT,            -- JSON
  buffett_signals TEXT,          -- JSON
  key_risks TEXT,                -- JSON array
  data_quality_notes TEXT,       -- JSON array
  tool_calls_made INTEGER,
  tokens_used INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(id),
  tool_name TEXT,
  input_json TEXT,
  output_json TEXT,              -- truncated to 8KB; see truncation rule in db.py
  output_file_path TEXT,         -- set when output_json exceeds 8KB
  latency_ms INTEGER,
  cached INTEGER DEFAULT 0,      -- 1 = cache hit
  error_msg TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eval_examples (
  ticker TEXT PRIMARY KEY,
  expected_recommendation TEXT,
  expected_thesis_keywords TEXT, -- JSON
  notes TEXT,
  last_curated DATE
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(id),
  example_ticker TEXT REFERENCES eval_examples(ticker),
  passed INTEGER,                -- 1 = pass
  check_results TEXT,            -- JSON
  diff_notes TEXT
);

CREATE TABLE IF NOT EXISTS discovery_cooldown (
  ticker TEXT PRIMARY KEY,
  flagged_at TIMESTAMP,
  expires_at TIMESTAMP,
  suppression_reason TEXT
);

CREATE TABLE IF NOT EXISTS security_identities (
  venue TEXT NOT NULL,
  isin TEXT NOT NULL,
  canonical_ticker TEXT NOT NULL,
  mic TEXT,
  exchange_symbol TEXT NOT NULL,
  legal_name TEXT NOT NULL,
  identity_source_url TEXT NOT NULL,
  resolved_at TIMESTAMP NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  superseded_by_isin TEXT,
  PRIMARY KEY (venue, isin)
);

CREATE TABLE IF NOT EXISTS filing_manifests (
  filing_id TEXT NOT NULL,
  checksum TEXT NOT NULL,
  issuer_isin TEXT,
  venue TEXT NOT NULL,
  source_system TEXT NOT NULL,
  upstream_id TEXT,
  document_kind TEXT,
  title TEXT,
  publication_date DATE,
  reporting_period_end DATE,
  landing_page_url TEXT,
  direct_document_url TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  byte_length INTEGER NOT NULL,
  retrieved_at TIMESTAMP NOT NULL,
  etag TEXT,
  last_modified TEXT,
  status TEXT NOT NULL,
  source_language TEXT,
  parser_version TEXT,
  extraction_version TEXT,
  translation_version TEXT,
  extracted_text_checksum TEXT,
  extracted_text_artifact_key TEXT,
  translated_text_checksum TEXT,
  translated_text_artifact_key TEXT,
  artifact_key TEXT NOT NULL,
  supersedes_checksum TEXT,
  created_at TIMESTAMP NOT NULL,
  PRIMARY KEY (filing_id, checksum),
  CONSTRAINT ck_filing_manifests_checksum_length CHECK (length(checksum) = 64),
  CONSTRAINT ck_filing_manifests_byte_length CHECK (byte_length >= 0),
  CONSTRAINT fk_filing_manifests_supersedes
    FOREIGN KEY (filing_id, supersedes_checksum)
    REFERENCES filing_manifests(filing_id, checksum)
);

CREATE TABLE IF NOT EXISTS forensic_snapshots (
  ticker TEXT NOT NULL,
  issuer_isin TEXT NOT NULL,
  as_of DATE NOT NULL,
  lookback_start DATE NOT NULL,
  extractor_version TEXT NOT NULL,
  corpus_hash TEXT NOT NULL,
  venue TEXT NOT NULL,
  generated_at TIMESTAMP NOT NULL,
  evidence_json JSON NOT NULL,
  coverage_json JSON NOT NULL,
  warnings_json JSON NOT NULL,
  PRIMARY KEY (ticker, issuer_isin, venue, as_of, lookback_start, extractor_version, corpus_hash)
);

-- Performance indexes (Tech Spec §7.5)
CREATE INDEX IF NOT EXISTS idx_analyses_ticker_created ON analyses(ticker, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_run             ON analyses(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run           ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_started             ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_run            ON eval_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_security_identities_isin ON security_identities(isin);
CREATE INDEX IF NOT EXISTS idx_security_identities_current_ticker
  ON security_identities(canonical_ticker, is_active);
CREATE INDEX IF NOT EXISTS idx_filing_manifests_checksum
  ON filing_manifests(checksum);
CREATE INDEX IF NOT EXISTS idx_filing_manifests_issuer_date
  ON filing_manifests(issuer_isin, retrieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_filing_manifests_document_versions
  ON filing_manifests(filing_id, retrieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_filing_manifests_selection
  ON filing_manifests(issuer_isin, document_kind,
                      reporting_period_end DESC, publication_date DESC);
CREATE INDEX IF NOT EXISTS idx_forensic_snapshots_ticker_as_of
  ON forensic_snapshots(ticker, as_of DESC, generated_at DESC);

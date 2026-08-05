from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import edgar_client, yfinance_client
from agent.tools.base import TICKER_PATTERN, Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.edgar_client import SC13Holder
from data_sources.yfinance_client import KeyPersonsRaw

_CONTROLLING_THRESHOLD = 0.20  # 20% ownership → controlling interest
_MAJOR_HOLDER_THRESHOLD = 0.05  # 5% minimum to appear in persons list


class GetKeyPersonsInput(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN, description="Stock ticker, e.g. AAPL")


class KeyPerson(BaseModel):
    name: str
    role: str
    ownership_pct: float | None
    birth_year: int | None = None
    age: int | None = None
    age_as_of: date | None = None
    source: Literal["yfinance_officers", "yfinance_holders", "edgar_13g", "edgar_13d"]


class KeyPersonsData(BaseModel):
    ticker: str
    as_of: date
    persons: list[KeyPerson]
    controlling_holder_identified: bool | None
    source_notes: list[str]
    data_age_hours: int
    source: Literal["combined"] = "combined"


def _persons_from_yf(raw: KeyPersonsRaw) -> list[KeyPerson]:
    persons: list[KeyPerson] = []
    for off in raw.officers:
        age = (
            off.reported_age
            if off.reported_age is not None and 0 <= off.reported_age <= 130
            else raw.as_of.year - off.year_born
            if off.year_born is not None
            else None
        )
        persons.append(
            KeyPerson(
                name=off.name,
                role=off.title,
                ownership_pct=None,
                birth_year=off.year_born,
                age=age,
                age_as_of=raw.as_of if age is not None else None,
                source="yfinance_officers",
            )
        )
    for ih in raw.institutional_holders:
        if ih.pct_held is not None:
            company_pct = ih.pct_held if 0.0 <= ih.pct_held <= 1.0 else None
        else:
            company_pct = (
                ih.shares / raw.shares_outstanding
                if ih.shares is not None
                and raw.shares_outstanding is not None
                and raw.shares_outstanding > 0
                else None
            )
        if company_pct is None or company_pct < _MAJOR_HOLDER_THRESHOLD:
            continue
        pct = round(company_pct * 100, 4)
        persons.append(
            KeyPerson(
                name=ih.name,
                role="Institutional Holder",
                ownership_pct=pct,
                source="yfinance_holders",
            )
        )
    return persons


def _persons_from_edgar(
    holders: list[SC13Holder],
    existing_names: set[str],
) -> list[KeyPerson]:
    persons: list[KeyPerson] = []
    for h in holders:
        if h.name.upper() in existing_names:
            continue
        src: Literal["edgar_13g", "edgar_13d"] = (
            "edgar_13d" if "13D" in h.form_type.upper() else "edgar_13g"
        )
        persons.append(
            KeyPerson(
                name=h.name,
                role="5%+ Beneficial Owner",
                ownership_pct=None,
                source=src,
            )
        )
    return persons


def _is_controlling(
    ticker: str,
    persons: list[KeyPerson],
    edgar_holders: list[SC13Holder],
    *,
    edgar_coverage: bool,
) -> bool | None:
    for p in persons:
        if p.ownership_pct is not None and p.ownership_pct >= _CONTROLLING_THRESHOLD * 100:
            return True
    if ticker.upper().endswith((".MI", ".MC", ".WA")):
        return None
    # SC 13D is only a coarse US intent proxy; amendments retain the same filing family.
    if any("13D" in holder.form_type.upper() for holder in edgar_holders):
        return True
    if not edgar_coverage and not any(person.ownership_pct is not None for person in persons):
        return None
    return False


class GetKeyPersonsTool(Tool):
    name = "get_key_persons"
    description = (
        "Resolve the key persons for a ticker: C-suite officers, directors, and 5%+ "
        "beneficial owners. Returns a list of persons with name, role, and ownership % "
        "where available, plus a tri-state controlling_holder_identified field. None means "
        "coverage is insufficient and must never be read as no controller. "
        "Use this tool before running adverse-news searches so you can search people, "
        "not just the ticker symbol. Data sourced from yfinance companyOfficers, "
        "institutional holders, and EDGAR SC 13G/D filings."
    )
    input_schema = GetKeyPersonsInput
    output_schema = KeyPersonsData

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetKeyPersonsInput)
        ticker = tool_input.ticker

        try:
            yf = yfinance_client()
            raw = yf.get_key_persons(ticker)
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_key_persons failed for {ticker}: {exc}",
                retryable=False,
            )

        if not isinstance(raw, KeyPersonsRaw):
            return ToolResultError(
                error_code="not_found",
                message=f"No key persons data available for {ticker}",
                retryable=False,
            )

        persons = _persons_from_yf(raw)
        source_notes: list[str] = []
        edgar_holders: list[SC13Holder] = []
        edgar_coverage = False

        try:
            ed = edgar_client()
            sc13_result = ed.get_sc13_holders(ticker)
            if isinstance(sc13_result, list):
                edgar_coverage = True
                edgar_holders = sc13_result
                existing_names = {p.name.upper() for p in persons}
                persons.extend(_persons_from_edgar(edgar_holders, existing_names))
            else:
                source_notes.append(
                    f"EDGAR SC 13G/D unavailable ({sc13_result.error_code}): {sc13_result.message}"
                )
        except Exception as exc:
            source_notes.append(f"EDGAR SC 13G/D lookup failed: {exc}")

        return ToolResultOk(
            data=KeyPersonsData(
                ticker=ticker,
                as_of=date.today(),
                persons=persons,
                controlling_holder_identified=_is_controlling(
                    ticker,
                    persons,
                    edgar_holders,
                    edgar_coverage=edgar_coverage,
                ),
                source_notes=source_notes,
                data_age_hours=raw.data_age_hours,
            )
        )

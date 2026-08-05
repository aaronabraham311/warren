import csv
import json
from pathlib import Path
from typing import cast

import requests

from data_sources.bme_client import BMEGrowthSource
from data_sources.errors import DataSourceError
from data_sources.euronext_client import EuronextProductDirectorySource
from data_sources.security_identity import SecurityIdentity
from data_sources.tradingview_client import TradingViewScannerSource

FIXTURES = Path(__file__).parents[1] / "eval" / "fixtures"


def _fixture(path: str) -> object:
    return json.loads((FIXTURES / path).read_text())


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(
        self,
        responses: list[object],
        *,
        error: requests.RequestException | None = None,
    ) -> None:
        self.responses = list(responses)
        self.error = error
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.headers: dict[str, str] = {}

    def _next(self, method: str, url: str, kwargs: dict[str, object]) -> _Response:
        self.calls.append((method, url, kwargs))
        if self.error is not None:
            raise self.error
        return _Response(self.responses.pop(0))

    def get(self, url: str, **kwargs: object) -> _Response:
        return self._next("GET", url, kwargs)

    def post(self, url: str, **kwargs: object) -> _Response:
        return self._next("POST", url, kwargs)


def test_euronext_parses_identities_and_filters_warrants() -> None:
    session = _Session([_fixture("EXGM/euronext/get_stocks/c352555c.json")])
    source = EuronextProductDirectorySource(
        mics="EXGM", suffix=".MI", venue="euronext_growth_milan"
    )

    result = source.fetch(session)  # type: ignore[arg-type]

    assert not isinstance(result, DataSourceError)
    assert [item.canonical_ticker for item in result] == ["AAT.MI", "ABC.MI"]
    assert result[0].isin == "IT0005548521"
    assert result[0].mic == "EXGM"
    _, _, kwargs = session.calls[0]
    assert kwargs["params"] == {
        "mics": "EXGM",
        "display_datapoints": "dp_stocks",
        "display_filters": "df_stocks",
        "display_type": "all",
    }
    assert kwargs["data"] == {
        "iDisplayStart": "0",
        "iDisplayLength": "300",
        "args[initialLetter]": "",
    }


def test_euronext_paginates_full_pages() -> None:
    row_a = ["", "Alpha", "IT0000000001", "AAA", ""]
    row_b = ["", "Beta", "IT0000000002", "BBB", ""]
    row_c = ["", "Gamma", "IT0000000003", "CCC", ""]
    session = _Session([{"aaData": [row_a, row_b]}, {"aaData": [row_c]}])
    source = EuronextProductDirectorySource(
        mics="EXGM",
        suffix=".MI",
        venue="euronext_growth_milan",
        page_size=2,
    )

    result = source.fetch(session)  # type: ignore[arg-type]

    assert not isinstance(result, DataSourceError)
    assert [item.canonical_ticker for item in result] == ["AAA.MI", "BBB.MI", "CCC.MI"]
    assert [call[2]["data"]["iDisplayStart"] for call in session.calls] == ["0", "2"]  # type: ignore[index]


def test_bme_resolves_isin_to_ticker() -> None:
    session = _Session(
        [
            _fixture("BMEGROWTH/bme/listed_companies/0f720129.json"),
            _fixture("BMEGROWTH/bme/share_details/81d30eb1.json"),
        ]
    )
    source = BMEGrowthSource(mtf_segment="BMEGrowth", suffix=".MC", venue="bme_growth")

    result = source.fetch(session)  # type: ignore[arg-type]

    assert not isinstance(result, DataSourceError)
    assert result[0].canonical_ticker == "480S.MC"
    assert result[0].isin == "ES0105509006"
    assert len(session.calls) == 2
    assert session.calls[1][2]["params"] == {
        "tradingSystem": "MTF",
        "mtfSegment": "BMEGrowth",
        "ISIN": "ES0105509006",
    }


def test_bme_rejects_missing_legal_name() -> None:
    session = _Session([{"data": [{"isin": "ES0105509006", "name": ""}]}])
    source = BMEGrowthSource(mtf_segment="BMEGrowth", suffix=".MC", venue="bme_growth")

    result = source.fetch(session)  # type: ignore[arg-type]

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def test_bme_tolerates_one_detail_failure_above_completeness_threshold() -> None:
    companies = [
        {"isin": f"ES00000000{index:02d}", "name": f"Issuer {index}"} for index in range(5)
    ]
    details: list[object] = [{"ticker": f"T{index}"} for index in range(4)]
    details.append(ValueError("invalid json"))
    session = _Session([{"data": companies}, *details])
    source = BMEGrowthSource(mtf_segment="BMEGrowth", suffix=".MC", venue="bme_growth")

    result = source.fetch(session)  # type: ignore[arg-type]

    assert not isinstance(result, DataSourceError)
    assert len(result) == 4


def test_tradingview_returns_verified_isin_backed_identities() -> None:
    session = _Session([_fixture("NEWCONNECT/tradingview/scan/7e9a6044.json")])
    source = TradingViewScannerSource(
        country="poland", exchange="NEWCONNECT", suffix=".WA", venue="newconnect"
    )

    result = source.fetch(session)  # type: ignore[arg-type]

    assert not isinstance(result, DataSourceError)
    assert [item.canonical_ticker for item in result] == ["CFG.WA", "NOV.WA"]
    assert [item.isin for item in result] == ["PLCRFRG00016", "PLBABY000016"]
    _, _, kwargs = session.calls[0]
    request_json = cast(dict[str, object], kwargs["json"])
    assert request_json["columns"] == [
        "name",
        "description",
        "isin",
        "market_cap_basic",
    ]


def test_source_transport_errors_never_raise() -> None:
    session = _Session([], error=requests.ConnectionError("down"))
    source = TradingViewScannerSource(
        country="poland", exchange="NEWCONNECT", suffix=".WA", venue="newconnect"
    )

    result = source.fetch(session)  # type: ignore[arg-type]

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_bme_malformed_json_is_a_parse_error() -> None:
    session = _Session([ValueError("not json")])
    source = BMEGrowthSource(mtf_segment="BMEGrowth", suffix=".MC", venue="bme_growth")

    result = source.fetch(session)  # type: ignore[arg-type]

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def test_identity_is_dataclass_boundary() -> None:
    session = _Session([_fixture("EXGM/euronext/get_stocks/c352555c.json")])
    result = EuronextProductDirectorySource(
        mics="EXGM", suffix=".MI", venue="euronext_growth_milan"
    ).fetch(session)  # type: ignore[arg-type]
    assert not isinstance(result, DataSourceError)
    assert isinstance(result[0], SecurityIdentity)


def test_regenerated_fallbacks_are_junior_scale_and_milan_is_not_ftse_mib() -> None:
    data_dir = Path(__file__).parents[1] / "data"

    def tickers(name: str) -> set[str]:
        with (data_dir / name).open(newline="") as handle:
            return {row["ticker"] for row in csv.DictReader(handle)}

    milan = tickers("milan.csv")
    madrid = tickers("madrid.csv")
    warsaw = tickers("warsaw.csv")
    reference = _fixture("EXGM/ftse_mib_reference.json")
    assert isinstance(reference, dict)
    ftse_mib = set(reference["tickers"])

    assert len(ftse_mib) == 40
    assert len(milan | madrid | warsaw) >= 500
    assert milan.isdisjoint(ftse_mib)

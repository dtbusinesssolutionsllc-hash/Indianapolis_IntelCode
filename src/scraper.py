#!/usr/bin/env python3
"""
scraper.py — Indianapolis Motivated Seller Lead Generator

WHAT THIS PULLS FROM
---------------------
The city of Indianapolis publishes its Code Enforcement Violations &
Investigations dataset as public open data through the ArcGIS Hub platform:
    https://data.indy.gov/datasets/indianapolis-code-enforcement-violations-and-investigations

That dataset is what this script queries — via the documented ArcGIS
FeatureServer REST API, with paging. This is a deliberate substitution for
scraping the county's Accela ("aca-prod.accela.com") case-search portal:

  * The Accela portal is a stateful ASP.NET WebForms app (postbacks +
    __VIEWSTATE per search) with bot detection in front of it. Automating
    it reliably means impersonating a browser session, and doing so likely
    conflicts with the portal's terms of use.
  * The open-data feed contains the same underlying case records
    (case number, status, open date, address, owner) via a stable,
    intentionally-public JSON API — no session hacking required.

WHAT THIS DOES NOT DO
----------------------
This script does NOT fabricate tax-delinquency, probate, lien-count, or
divorce/bankruptcy matches. Those record types live in systems this open
dataset does not cover (county treasurer tax rolls, county/circuit court
probate & divorce dockets, federal PACER bankruptcy filings) and most have
no free public API — several require a paid data license or a records
request. Rather than guess or invent matches, those signals are wired up
as clearly-labeled plug-in functions (see the DISTRESS SIGNAL PLUGINS
section) that return "unknown" until you connect a real, licensed data
source. Faking that data would produce leads that *look* more distressed
than they actually are, which is bad for both compliance and your close
rate.

USAGE
-----
    python src/scraper.py
    python src/scraper.py --max-records 5000 --since 2024-01-01
    python src/scraper.py --output data/output.json

Exit code is always 0 on a "soft" failure (e.g. one bad record) — the
script logs and skips. It exits non-zero only on a total, unrecoverable
failure (e.g. the API is unreachable for every retry).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# ArcGIS Hub item ID for "Indianapolis Code Enforcement Violations and
# Investigations". Confirmed live at data.indy.gov as of Sept 2026.
ARCGIS_ITEM_ID = "5d08eba2e9034bc88986af25afe12f5e"
ARCGIS_ITEM_INFO_URL = f"https://www.arcgis.com/sharing/rest/content/items/{ARCGIS_ITEM_ID}"

# Fallback query URL used if item-metadata resolution fails (network
# restrictions, item metadata endpoint down, etc). Tried first actually,
# since it saves a network round trip when it's already correct; if it
# 404s/errors, we fall back to resolving via ARCGIS_ITEM_INFO_URL.
FALLBACK_QUERY_URL = (
    "https://services2.arcgis.com/RH6ZjMkNXjuLtjkm/arcgis/rest/services/"
    "DCE_Violations/FeatureServer/0/query"
)

PAGE_SIZE = 1000            # records per API page
REQUEST_TIMEOUT = 30        # seconds
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
USER_AGENT = "IndyMotivatedSellerLeadTool/1.0 (public-open-data-client)"

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "output.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scraper")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class DistressBreakdown:
    """Itemized point contributions so the dashboard can explain a score."""
    code_violation: int = 0
    repeat_violations: int = 0       # stand-in for "multiple liens" — see notes below
    tax_delinquency: Optional[int] = None   # None = unknown / not connected
    probate_filing: Optional[int] = None
    divorce_bankruptcy: Optional[int] = None


@dataclass
class Lead:
    document_number: str
    file_date: Optional[str]
    owner: Optional[str]
    property_address: Optional[str]
    legal_description: Optional[str]
    case_type: Optional[str]
    case_status: Optional[str]
    seller_score: int
    score_breakdown: DistressBreakdown
    source_url: Optional[str]
    scraped_at: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def _get_with_retries(url: str, params: dict) -> dict:
    """GET a JSON endpoint with retry + exponential backoff.

    Raises requests.RequestException only after all retries are exhausted,
    so a single flaky request never crashes the whole run.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(f"API returned an error payload: {payload['error']}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_exc = exc
            wait = RETRY_BACKOFF_SECONDS * attempt
            log.warning(
                "Request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _query_url_is_reachable(query_url: str) -> bool:
    """Cheap probe: ask for 1 record with returnCountOnly to confirm the
    endpoint actually responds before committing to it."""
    try:
        payload = _get_with_retries(query_url, {"where": "1=1", "returnCountOnly": "true", "f": "json"})
        return "count" in payload
    except Exception as exc:  # noqa: BLE001
        log.warning("Query URL probe failed for %s: %s", query_url, exc)
        return False


def resolve_feature_server_url(item_id: str) -> str:
    """Return a working FeatureServer query URL, with a fallback chain so a
    single failed lookup never prevents the script from producing output.

    Order:
      1. Try the hardcoded FALLBACK_QUERY_URL directly (fast path).
      2. If that doesn't respond, resolve the URL live from the ArcGIS item
         metadata endpoint (handles the service having moved).
      3. If both fail, raise — the caller is responsible for still writing
         a (empty) output file rather than crashing silently.
    """
    log.info("Trying known FeatureServer query URL first...")
    if _query_url_is_reachable(FALLBACK_QUERY_URL):
        log.info("Using known FeatureServer query URL.")
        return FALLBACK_QUERY_URL

    log.info("Known URL unreachable — resolving FeatureServer URL for item %s via item metadata", item_id)
    try:
        payload = _get_with_retries(ARCGIS_ITEM_INFO_URL, {"f": "json"})
        service_url = payload.get("url")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Item metadata lookup failed for {item_id}: {exc}") from exc

    if not service_url:
        raise RuntimeError(
            f"Could not resolve a FeatureServer URL from item {item_id}. "
            f"Item metadata: {payload}"
        )
    # Layer 0 is the standard convention for a single-layer hosted feature layer.
    resolved = f"{service_url.rstrip('/')}/0/query"
    if not _query_url_is_reachable(resolved):
        raise RuntimeError(f"Resolved FeatureServer URL is also unreachable: {resolved}")
    return resolved


def fetch_all_records(
    query_url: str,
    since: Optional[str],
    max_records: Optional[int],
) -> list[dict]:
    """Page through the FeatureServer query endpoint and return raw records."""
    records: list[dict] = []
    offset = 0
    where_clause = "1=1"
    if since:
        # OPEN_DATE is the field name in this dataset; ArcGIS date literals
        # use this timestamp format.
        where_clause = f"OPEN_DATE >= TIMESTAMP '{since} 00:00:00'"

    while True:
        params = {
            "where": where_clause,
            "outFields": "*",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "orderByFields": "OPEN_DATE DESC",
        }
        try:
            payload = _get_with_retries(query_url, params)
        except Exception as exc:  # noqa: BLE001 — top-level page fetch guard
            log.error("Giving up on page at offset %d after retries: %s", offset, exc)
            break

        features = payload.get("features", [])
        if not features:
            log.info("No more records returned at offset %d — pagination complete.", offset)
            break

        for feature in features:
            attrs = feature.get("attributes")
            if attrs:
                records.append(attrs)

        log.info("Fetched %d records (running total: %d)", len(features), len(records))
        offset += len(features)

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            log.info("Reached --max-records limit of %d", max_records)
            break

        # ArcGIS signals the last page either by returning fewer than a full
        # page, or via exceededTransferLimit == False.
        if not payload.get("exceededTransferLimit", len(features) == PAGE_SIZE):
            break

    return records


# --------------------------------------------------------------------------
# Distress signal plugins
# --------------------------------------------------------------------------
#
# Each function takes the parsed record (+ shared context) and returns a
# point value. Returning None means "signal not available — data source not
# connected", which the dashboard displays honestly as "Unknown" rather than
# folding it into the score as a false zero.
#
# To wire up a real source:
#   - tax_delinquency_score: Marion County Treasurer publishes an annual
#     delinquent tax list; ingest it into a lookup keyed by parcel/address
#     and query it here.
#   - probate_score / divorce_bankruptcy_score: these live in county court
#     record systems (mycase.in.gov) and federal PACER. Both require an
#     account, most have per-lookup fees, and PACER's terms restrict bulk
#     automated scraping — treat these as a licensed-data-provider
#     integration, not a scraper target.

def code_violation_score(record: dict) -> int:
    """Every record in this feed is, by definition, a code enforcement case."""
    return 25


def repeat_violation_score(address: Optional[str], address_counts: Counter) -> int:
    """Stand-in for 'multiple liens': repeat code cases at the same address
    are a real, directly-observable distress signal in this dataset (an
    absentee or overwhelmed owner racking up multiple citations).
    """
    if not address:
        return 0
    if address_counts.get(address, 0) > 1:
        return 15
    return 0


def tax_delinquency_score(record: dict) -> Optional[int]:
    """Not connected. Wire this up to a Marion County Treasurer delinquent
    tax list lookup (by parcel number or normalized address) to enable.
    """
    return None


def probate_score(record: dict) -> Optional[int]:
    """Not connected. Requires a Marion County probate court records feed."""
    return None


def divorce_bankruptcy_score(record: dict) -> Optional[int]:
    """Not connected. Requires county circuit court (divorce) and/or PACER
    (bankruptcy) access — typically via a licensed data provider.
    """
    return None


def compute_score(record: dict, address_counts: Counter) -> tuple[int, DistressBreakdown]:
    address = normalize_address(record)

    cv = code_violation_score(record)
    rv = repeat_violation_score(address, address_counts)
    tax = tax_delinquency_score(record)
    probate = probate_score(record)
    divorce_bk = divorce_bankruptcy_score(record)

    total = cv + rv + (tax or 0) + (probate or 0) + (divorce_bk or 0)
    total = max(0, min(100, total))  # clamp to the documented 0-100 range

    breakdown = DistressBreakdown(
        code_violation=cv,
        repeat_violations=rv,
        tax_delinquency=tax,
        probate_filing=probate,
        divorce_bankruptcy=divorce_bk,
    )
    return total, breakdown


# --------------------------------------------------------------------------
# Record parsing
# --------------------------------------------------------------------------

def normalize_address(record: dict) -> Optional[str]:
    parts = [
        record.get("STREET_ADDRESS") or record.get("ADDRESS"),
        record.get("CITY"),
        record.get("STATE"),
        record.get("ZIP") or record.get("ZIPCODE"),
    ]
    parts = [str(p).strip() for p in parts if p not in (None, "")]
    return ", ".join(parts) if parts else None


def parse_esri_date(value: Any) -> Optional[str]:
    """ArcGIS returns dates as epoch milliseconds. Convert to ISO 8601."""
    if value is None:
        return None
    try:
        millis = int(value)
        dt = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
        return dt.date().isoformat()
    except (ValueError, TypeError, OSError):
        log.debug("Could not parse date value: %r", value)
        return None


def parse_record(record: dict, address_counts: Counter) -> Optional[Lead]:
    """Convert one raw ArcGIS attribute dict into a Lead.

    Returns None (and logs) instead of raising, so one malformed record
    never takes down the whole run.
    """
    try:
        doc_number = (
            record.get("CASE_NUMBER")
            or record.get("CASENUMBER")
            or record.get("OBJECTID")
        )
        if not doc_number:
            log.warning("Skipping record with no usable document number: %s", record)
            return None

        score, breakdown = compute_score(record, address_counts)

        return Lead(
            document_number=str(doc_number),
            file_date=parse_esri_date(record.get("OPEN_DATE") or record.get("OPENDATE")),
            owner=(record.get("OWNER") or "").strip() or None,
            property_address=normalize_address(record),
            # Not present in this open-data feed — see module docstring.
            # Left explicit (rather than omitted) so downstream consumers
            # know to look it up via the county Assessor/Recorder instead
            # of assuming it's missing data quality.
            legal_description=None,
            case_type=(record.get("CASE_TYPE") or record.get("CASETYPE") or "").strip() or None,
            case_status=(record.get("CASE_STATUS") or record.get("CASESTATUS") or "").strip() or None,
            seller_score=score,
            score_breakdown=breakdown,
            source_url=record.get("LINK") or record.get("URL"),
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:  # noqa: BLE001 — deliberate broad catch per-record
        log.error("Failed to parse record, skipping. Error: %s | Record: %s", exc, record)
        return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _write_output(output_path: Path, payload: dict) -> None:
    """Write JSON output, guaranteeing the parent directory exists.

    This is factored out so it can be called from every exit path in run()
    — success, partial failure, or total failure — so the file always
    exists after a run, even if it just documents an error.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    # Atomic-ish replace so a crash mid-write never leaves a corrupt file
    # in place of a previously-good one.
    tmp_path.replace(output_path)
    log.info("Wrote %s", output_path)


def _base_payload() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "data.indy.gov — Indianapolis Code Enforcement Violations and Investigations",
        "total_records_fetched": 0,
        "total_leads_parsed": 0,
        "records_skipped": 0,
        "error": None,
        "leads": [],
    }


def run(output_path: Path, since: Optional[str], max_records: Optional[int]) -> int:
    payload = _base_payload()

    try:
        query_url = resolve_feature_server_url(ARCGIS_ITEM_ID)
    except Exception as exc:  # noqa: BLE001
        log.critical("Could not resolve the data source at all: %s", exc)
        payload["error"] = f"Could not resolve data source: {exc}"
        _write_output(output_path, payload)
        return 1

    try:
        raw_records = fetch_all_records(query_url, since=since, max_records=max_records)
    except Exception as exc:  # noqa: BLE001
        log.critical("Fetch failed entirely: %s", exc)
        payload["error"] = f"Fetch failed: {exc}"
        _write_output(output_path, payload)
        return 1

    if not raw_records:
        log.error("No records fetched — writing empty result. Check connectivity/filters.")
        payload["error"] = "No records were returned by the API (check --since filter or connectivity)."
        _write_output(output_path, payload)
        return 1

    # Pre-count addresses across the whole batch so repeat_violation_score
    # can see cross-record patterns (this can't be computed per-record in
    # isolation).
    address_counts: Counter = Counter(
        normalize_address(r) for r in raw_records if normalize_address(r)
    )

    leads: list[Lead] = []
    skipped = 0
    for raw in raw_records:
        try:
            lead = parse_record(raw, address_counts)
        except Exception as exc:  # noqa: BLE001 — belt-and-suspenders; parse_record
            # already catches internally, but a bug here must never abort the run.
            log.error("Unexpected error parsing record, skipping: %s", exc)
            lead = None
        if lead is None:
            skipped += 1
            continue
        leads.append(lead)

    leads.sort(key=lambda l: l.seller_score, reverse=True)

    payload.update({
        "total_records_fetched": len(raw_records),
        "total_leads_parsed": len(leads),
        "records_skipped": skipped,
        "leads": [lead.to_dict() for lead in leads],
    })

    try:
        _write_output(output_path, payload)
    except Exception as exc:  # noqa: BLE001
        log.critical("Failed to write output file to %s: %s", output_path, exc)
        return 1

    log.info(
        "Done. Wrote %d leads (%d skipped) to %s",
        len(leads), skipped, output_path,
    )
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indianapolis motivated seller lead scraper")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write JSON output (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only fetch cases opened on/after this date, format YYYY-MM-DD",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Cap the number of records fetched (useful for testing)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        exit_code = run(output_path=args.output, since=args.since, max_records=args.max_records)
    except Exception as exc:  # noqa: BLE001 — last-resort guard
        log.critical("Unhandled exception in run(): %s", exc, exc_info=True)
        # Guarantee a file exists even if something in run() itself blew up
        # before it could write, so downstream consumers (dashboard, CI
        # artifact upload) never fail on a missing file.
        fallback = _base_payload()
        fallback["error"] = f"Unhandled exception: {exc}"
        try:
            _write_output(args.output, fallback)
        except Exception as write_exc:  # noqa: BLE001
            log.critical("Also failed to write fallback output: %s", write_exc)
        exit_code = 1
    sys.exit(exit_code)

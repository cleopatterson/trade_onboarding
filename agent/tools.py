"""Tools for the Trade Onboarding wizard"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import re
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

from agent.config import (
    ABR_GUID, NSW_TRADES_API_KEY, NSW_TRADES_AUTH_HEADER, BRAVE_SEARCH_API_KEY,
    GOOGLE_PLACES_API_KEY, ANTHROPIC_API_KEY, MODEL_FAST,
)

RESOURCES_DIR = Path(__file__).parent.parent / "resources"

# Persistent HTTP client — reuses connections across API calls (saves TLS handshake time)
_http_client = httpx.AsyncClient(timeout=15.0)

# Shared LLM client for vision tasks (AI photo filter) — avoids creating per-call instances
from langchain_anthropic import ChatAnthropic as _ChatAnthropic
_llm_vision = _ChatAnthropic(
    model=MODEL_FAST,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=256,
    temperature=0,
)


# ────────── ABR LOOKUP ──────────

async def abr_lookup(search_term: str, search_type: str = "name") -> dict:
    """Search the Australian Business Register for business details."""
    if not ABR_GUID:
        return _mock_abr_lookup(search_term, search_type)

    try:
        if search_type == "abn":
            url = "https://abr.business.gov.au/json/AbnDetails.aspx"
            params = {
                "abn": search_term.replace(" ", ""),
                "callback": "c",
                "guid": ABR_GUID,
            }
        else:
            url = "https://abr.business.gov.au/json/MatchingNames.aspx"
            params = {
                "name": search_term,
                "maxResults": "15",
                "callback": "c",
                "guid": ABR_GUID,
            }

        resp = await _http_client.get(url, params=params)
        if resp.status_code != 200:
            return {"results": [], "count": 0, "error": f"ABR API returned {resp.status_code}"}

        return _parse_jsonp_response(resp.text, search_type)

    except Exception as e:
        logger.error(f"ABR lookup error ({type(e).__name__}): {e}")
        if not ABR_GUID:
            return _mock_abr_lookup(search_term, search_type)
        return {"results": [], "count": 0, "error": f"ABR lookup failed: {e}"}


def _title_case(name: str) -> str:
    """Convert ALL CAPS ABR name to readable title case.

    Only converts if the name is fully uppercase. Fixes Python's title()
    apostrophe bug ("SMITH'S" → "Smith's" not "Smith'S").
    """
    if not name or not name.isupper():
        return name  # Already mixed case, leave it
    result = name.title()
    # Fix apostrophe word boundary: "Smith'S" → "Smith's"
    result = re.sub(r"'S\b", "'s", result)
    result = re.sub(r"\u2019S\b", "\u2019s", result)
    return result


def _parse_jsonp_response(jsonp_text: str, search_type: str) -> dict:
    """Parse ABR JSONP response into structured dict."""
    try:
        # Strip JSONP callback wrapper: c({...})
        match = re.search(r'c\((.*)\)', jsonp_text, re.DOTALL)
        if not match:
            return {"results": [], "count": 0, "error": "Failed to parse JSONP"}

        data = json.loads(match.group(1))

        if search_type == "abn":
            abn = data.get("Abn", "")
            raw_entity_name = data.get("EntityName", "Unknown")
            business_names = data.get("BusinessName") or []
            trading_name = business_names[0] if business_names else ""
            # Display name: prefer trading/business name over entity name
            display_name = trading_name if trading_name else raw_entity_name
            entity_type = data.get("EntityTypeName", "Unknown")
            gst = data.get("Gst", "")
            state = data.get("AddressState", "")
            postcode = data.get("AddressPostcode", "")

            if not abn:
                return {"results": [], "count": 0, "error": data.get("Message", "No result")}

            return {
                "results": [{
                    "abn": abn,
                    "entity_name": _title_case(display_name),
                    "legal_name": raw_entity_name,
                    "_has_registered_trading_name": bool(trading_name),
                    "entity_type": entity_type,
                    "gst_registered": bool(gst),
                    "state": state,
                    "postcode": postcode,
                    "status": "Active" if data.get("AbnStatus", "") == "Active" else data.get("AbnStatus", "Unknown"),
                    "entity_start_date": data.get("EntityStartDate", ""),
                }],
                "count": 1,
            }
        else:
            # Name search returns a Names array — same ABN can appear
            # multiple times (once as Entity Name, once as Business/Trading Name).
            # Deduplicate by ABN, preferring Business/Trading Name for display.
            # Preserve Entity Name as legal_name for licence lookups.
            names = data.get("Names", [])
            by_abn: dict[str, dict] = {}
            entity_names: dict[str, str] = {}  # abn → Entity Name (raw)
            has_trading: set[str] = set()  # ABNs with Business/Trading Name
            for entry in names:
                abn = entry.get("Abn", "")
                if not abn:
                    continue
                name = entry.get("Name", "Unknown")
                name_type = entry.get("NameType", "")
                state = entry.get("State", "")
                postcode = entry.get("Postcode", "")
                score = entry.get("Score", 0)

                # Track Entity Name separately for legal_name
                if name_type == "Entity Name":
                    entity_names[abn] = name
                elif name_type in ("Business Name", "Trading Name"):
                    has_trading.add(abn)

                record = {
                    "abn": abn,
                    "entity_name": _title_case(name),
                    "entity_type": name_type,
                    "gst_registered": False,
                    "state": state,
                    "postcode": postcode,
                    "status": "Active",
                    "score": score,
                }

                if abn not in by_abn:
                    by_abn[abn] = record
                elif name_type in ("Business Name", "Trading Name"):
                    # Trading/Business Name wins for display
                    by_abn[abn] = record

            # Populate legal_name and trading name flag from tracked data
            for abn, record in by_abn.items():
                record["legal_name"] = entity_names.get(abn, record["entity_name"])
                record["_has_registered_trading_name"] = abn in has_trading

            results = list(by_abn.values())[:8]
            return {"results": results, "count": len(results)}

    except (json.JSONDecodeError, KeyError) as e:
        return {"results": [], "count": 0, "error": f"Parse error: {e}"}


async def enrich_abr_with_entity_names(results: list[dict]) -> list[dict]:
    """Enrich ABR name search results with entity names via parallel ABN lookups.

    For results where legal_name is missing or same as entity_name (meaning ABR
    name search only returned one entry for that ABN), do a parallel ABN detail
    lookup to fetch the EntityName.
    """
    if not ABR_GUID:
        return results  # Mock mode — nothing to enrich

    async def _lookup_entity_name(result: dict) -> dict:
        """Fetch entity name for a single ABR result via ABN detail lookup."""
        abn = result.get("abn", "").replace(" ", "")
        if not abn:
            return result
        try:
            resp = await _http_client.get(
                "https://abr.business.gov.au/json/AbnDetails.aspx",
                params={"abn": abn, "callback": "c", "guid": ABR_GUID},
            )
            if resp.status_code != 200:
                return result
            parsed = _parse_jsonp_response(resp.text, "abn")
            detail = (parsed.get("results") or [{}])[0]
            if detail.get("legal_name"):
                result["legal_name"] = detail["legal_name"]
            if detail.get("_has_registered_trading_name"):
                result["_has_registered_trading_name"] = True
        except Exception as e:
            logger.warning(f"ABN detail lookup failed for {abn}: {e}")
        return result

    # Only enrich results where legal_name is missing or same as display name
    # Case-insensitive: _title_case() changes entity_name casing but legal_name stays raw
    needs_enrichment = [
        r for r in results
        if not r.get("legal_name") or r["legal_name"].lower() == r.get("entity_name", "").lower()
    ]
    if not needs_enrichment:
        return results

    await asyncio.gather(*[_lookup_entity_name(r) for r in needs_enrichment])
    return results


def _mock_abr_lookup(search_term: str, search_type: str) -> dict:
    """Mock ABR response for development."""
    term = search_term.lower()

    # Simulate realistic results
    if search_type == "abn":
        clean_abn = search_term.replace(" ", "")
        entity_name = f"Business with ABN {clean_abn}"
        return {
            "results": [{
                "abn": clean_abn,
                "entity_name": entity_name,
                "legal_name": entity_name.upper() + " PTY LTD",
                "_has_registered_trading_name": True,
                "entity_type": "Australian Private Company",
                "gst_registered": True,
                "state": "NSW",
                "postcode": "2000",
                "status": "Active",
            }],
            "count": 1
        }

    # Name search - generate plausible result
    name_title = search_term.strip().title()
    display_name = name_title if "pty" in term or "ltd" in term else f"{name_title} Pty Ltd"
    return {
        "results": [{
            "abn": "51 824 753 556",
            "entity_name": display_name,
            "legal_name": display_name.upper(),
            "_has_registered_trading_name": False,
            "entity_type": "Australian Private Company",
            "gst_registered": True,
            "state": "NSW",
            "postcode": "2093",
            "status": "Active",
        }],
        "count": 1
    }


# ────────── SERVICE CATEGORY MAPPER ──────────

_categories_cache = None

def _load_categories() -> dict:
    """Load SS category taxonomy (dict keyed by category name)."""
    global _categories_cache
    if _categories_cache is not None:
        return _categories_cache

    cat_file = RESOURCES_DIR / "subcategories.json"
    if cat_file.exists():
        with open(cat_file) as f:
            _categories_cache = json.load(f)
        return _categories_cache
    return {}


def get_category_taxonomy_text() -> str:
    """Get a text representation of the category taxonomy for the LLM."""
    categories = _load_categories()
    if not categories:
        return "Category taxonomy not available."

    lines = []
    for cat_key, cat_data in categories.items():
        cat_name = cat_data.get("category_name", cat_key)
        cat_id = cat_data.get("category_id", 0)
        subcats = cat_data.get("subcategories", [])
        if subcats:
            subcat_strs = []
            for sc in subcats:
                sc_name = sc.get("subcategory_name", "Unknown")
                sc_id = sc.get("subcategory_id", 0)
                subcat_strs.append(f"  - {sc_name} (id: {sc_id})")
            lines.append(f"{cat_name} (id: {cat_id}):")
            lines.extend(subcat_strs)
        else:
            lines.append(f"{cat_name} (id: {cat_id})")
    return "\n".join(lines)


# ────────── SERVICE GAP COMPUTATION ──────────

# Map trade keywords (from business name) to category keys in subcategories.json
_TRADE_CATEGORY_MAP = {
    "electri": "Electrician",
    "plumb": "Plumber",
    "paint": "Painter",
    "clean": "Cleaner",
    "garden": "Gardener",
    "landscap": "Landscaper",
    "carpent": "Carpenter",
    "build": "Builder",
    "roof": "Roofer",
    "tile": "Tiler",
    "concret": "Concreter",
    "fenc": "Fencing & Gate Company",
    "glass": "Glass Repair Company",
    "locksmith": "Locksmith",
    "handyman": "Handyman",
    "plaster": "Plasterer",
    "brick": "Bricklayer",
    "render": "Rendering Company",
    "pool": "Pool & Spa Company",
    "solar": "Solar Company",
    "air con": "Air Conditioning & Heating Technician",
    "hvac": "Air Conditioning & Heating Technician",
    "pest": "Exterminator",
    "waterproof": "Waterproofing Company",
    "insul": "Insulation Company",
    "floor": "Flooring Company",
    "kitchen": "Kitchen Renovation Company",
    "bathroom": "Bathroom Renovation Company",
    "secur": "Security Company",
    "gas fit": "Gas Fitter",
}


# ────────── STATE LICENCE CONFIG ──────────
# Keyed by state → trade. Each entry has: regulator, label, patterns (regex),
# context_keywords, optional, default_classes. WA DMIRS trades get extra dmirs_search_code.

_STATE_LICENCE_CONFIG = {
    "VIC": {
        "Electrician": {
            "regulator": "Electrical Safety Office (ESV)",
            "label": "REC or ESV licence number",
            "patterns": [r"REC\s?\d{4,6}", r"ESV\s?\d{4,6}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Electrical Work"],
        },
        "Plumber": {
            "regulator": "Victorian Building Authority (VBA)",
            "label": "VBA registration number",
            "patterns": [r"LIC\s*(?:No\.?|#)\s*\d{5,8}", r"VBA\s*\d{5,8}", r"\d{5,8}"],
            "context_keywords": ["vba", "plumb", "registered plumber", "plumbing registration", "lic"],
            "optional": False,
            "default_classes": ["Plumbing and Drainage"],
        },
        "Builder": {
            "regulator": "Building Practitioners Board (BPC)",
            "label": "BPC registration number",
            "patterns": [r"DB-U\s?\d{3,6}", r"CDB\s?\d{3,6}", r"CB\s?\d{3,6}", r"DB-L\s?\d{3,6}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Building Work"],
        },
        "Painter": {
            "regulator": "Building Practitioners Board (BPC)",
            "label": "BPC registration number",
            "patterns": [r"DB-L\s?\d{3,6}"],
            "context_keywords": None,
            "optional": True,
            "default_classes": ["Painting"],
        },
        "Carpenter": {
            "regulator": "Building Practitioners Board (BPC)",
            "label": "BPC registration number",
            "patterns": [r"DB-L\s?\d{3,6}"],
            "context_keywords": None,
            "optional": True,
            "default_classes": ["Carpentry"],
        },
    },
    "WA": {
        "Electrician": {
            "regulator": "WA Department of Mines, Industry Regulation and Safety (DMIRS)",
            "label": "EC licence number",
            "patterns": [r"EC\s?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Electrical Work"],
            "dmirs_search_code": "EC",
        },
        "Plumber": {
            "regulator": "WA Department of Mines, Industry Regulation and Safety (DMIRS)",
            "label": "PL or TL licence number",
            "patterns": [r"(?:PL|TL)\s?\d{4,8}"],
            "context_keywords": ["plumb", "registered plumber", "plumbing"],
            "optional": False,
            "default_classes": ["Plumbing and Drainage"],
            "dmirs_search_code": "PL",
        },
        "Gas Fitter": {
            "regulator": "WA Department of Mines, Industry Regulation and Safety (DMIRS)",
            "label": "GF licence number",
            "patterns": [r"GF\s?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Gas Fitting"],
            "dmirs_search_code": "GF",
        },
    },
    "SA": {
        "Electrician": {
            "regulator": "SA Office of the Technical Regulator (OTR)",
            "label": "PGE licence number",
            "patterns": [r"PGE\s?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Electrical Work"],
        },
        "Plumber": {
            "regulator": "SA Office of the Technical Regulator (OTR)",
            "label": "plumber licence number",
            "patterns": [r"PGP\s?\d{4,8}"],
            "context_keywords": ["plumb", "registered plumber"],
            "optional": False,
            "default_classes": ["Plumbing and Drainage"],
        },
        "Gas Fitter": {
            "regulator": "SA Office of the Technical Regulator (OTR)",
            "label": "gas fitter licence number",
            "patterns": [r"PGG\s?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Gas Fitting"],
        },
        "Builder": {
            "regulator": "SA Consumer and Business Services (CBS)",
            "label": "BLD licence number",
            "patterns": [r"BLD\s?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Building Work"],
        },
    },
    "TAS": {
        "Electrician": {
            "regulator": "TAS Consumer, Building and Occupational Services (CBOS)",
            "label": "electrical licence number",
            "patterns": [r"EL\s?\d{4,6}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Electrical Work"],
        },
        "Plumber": {
            "regulator": "TAS Consumer, Building and Occupational Services (CBOS)",
            "label": "plumber licence number",
            "patterns": [r"PL\s?\d{4,6}"],
            "context_keywords": ["plumb"],
            "optional": False,
            "default_classes": ["Plumbing and Drainage"],
        },
        "Builder": {
            "regulator": "TAS Consumer, Building and Occupational Services (CBOS)",
            "label": "builder accreditation number",
            "patterns": [r"(?:CC|CB)\s?\d{4,6}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Building Work"],
        },
    },
    "ACT": {
        "Electrician": {
            "regulator": "ACT Access Canberra",
            "label": "electrician licence number",
            "patterns": [r"EL\s?\d{4,8}", r"E\d{5,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Electrical Work"],
        },
        "Plumber": {
            "regulator": "ACT Access Canberra",
            "label": "plumber licence number",
            "patterns": [r"PL\s?\d{4,8}"],
            "context_keywords": ["plumb"],
            "optional": False,
            "default_classes": ["Plumbing and Drainage"],
        },
        "Builder": {
            "regulator": "ACT Access Canberra",
            "label": "builder licence number",
            "patterns": [r"BL\s?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Building Work"],
        },
    },
    "NT": {
        "Electrician": {
            "regulator": "NT Licensing Commission",
            "label": "electrician licence number",
            "patterns": [r"C\d{4,8}"],
            "context_keywords": ["electric", "electrical contractor"],
            "optional": False,
            "default_classes": ["Electrical Work"],
        },
        "Plumber": {
            "regulator": "NT Licensing Commission",
            "label": "plumber licence number",
            "patterns": [r"C\d{4,8}"],
            "context_keywords": ["plumb"],
            "optional": False,
            "default_classes": ["Plumbing and Drainage"],
        },
        "Builder": {
            "regulator": "NT Building Practitioners Board",
            "label": "builder registration number",
            "patterns": [r"BR?\d{4,8}"],
            "context_keywords": None,
            "optional": False,
            "default_classes": ["Building Work"],
        },
    },
}

# Backward-compatible alias
_VIC_LICENCE_CONFIG = _STATE_LICENCE_CONFIG["VIC"]


def get_licence_config(state: str, trade: str) -> dict | None:
    """Look up licence config for a state + trade combination."""
    state_cfg = _STATE_LICENCE_CONFIG.get(state)
    if not state_cfg:
        return None
    return state_cfg.get(trade)


def extract_licence_from_text(text: str, trade: str, state: str = "VIC") -> dict | None:
    """Extract a licence number from combined website/Brave text for any state.

    Scans text for patterns matching the trade's regulator format.
    Plumber patterns require context keywords to avoid false positives on random numbers.
    Returns licence_info dict with licence_source='web_extracted', or None.
    """
    if not text or not trade:
        return None

    config = get_licence_config(state, trade)
    if not config:
        return None

    text_lower = text.lower()

    # Plumber has generic digit patterns — require context keywords
    if config.get("context_keywords"):
        has_context = any(kw in text_lower for kw in config["context_keywords"])
        if not has_context:
            return None

    # Scan for patterns
    for pattern in config["patterns"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(0).strip()
            # Extract just the number portion (strip LIC No., VBA, etc. prefixes)
            num_match = re.search(r'\d[\d\s-]*\d|\d+', raw)
            licence_number = num_match.group(0).strip() if num_match else raw
            return {
                "licence_number": licence_number,
                "licence_type": trade,
                "status": "Web-extracted",
                "licence_source": "web_extracted",
                "classes": [{"name": c, "active": True} for c in config["default_classes"]],
                "compliance_clean": True,
                "associated_parties": [],
                "business_address": "",
            }

    return None


async def scan_website_for_licence(url: str, trade: str, state: str = "VIC") -> dict | None:
    """Fetch a website and scan the FULL text for licence patterns (any state).

    Unlike scrape_website_text (capped at 5000 chars for evidence keywords),
    this scans ALL extracted text — licence numbers often appear deep in footers,
    buried under reviews and other content.
    Returns licence_info dict or None.
    """
    if not url or not trade or not get_licence_config(state, trade):
        return None
    try:
        resp = await _http_client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
            follow_redirects=True,
            timeout=8.0,
        )
        if resp.status_code != 200:
            return None
        html = resp.text
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return extract_licence_from_text(text, trade, state)
    except Exception as e:
        logger.error(f"[LICENCE-SCAN] {url}: {type(e).__name__}: {e}")
        return None


# ────────── WA DMIRS LICENCE LOOKUP ──────────

_WA_DMIRS_URL = "https://occupationallicensing.dmirs.wa.gov.au/onlinelicencesearch/licenceSearch.jspx"
_WA_DMIRS_PARAMS = {"BranchGroupCode": "WS"}


def _wa_dmirs_extract_viewstate(html: str) -> str | None:
    """Extract javax.faces.ViewState value from DMIRS search page HTML."""
    if not html:
        return None
    m = re.search(
        r'<input[^>]*name=["\']javax\.faces\.ViewState["\'][^>]*value=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if m:
        return m.group(1)
    # Try value-before-name order
    m = re.search(
        r'<input[^>]*value=["\']([^"\']+)["\'][^>]*name=["\']javax\.faces\.ViewState["\']',
        html, re.IGNORECASE,
    )
    return m.group(1) if m else None


def _wa_dmirs_parse_results(html: str) -> list[dict]:
    """Parse DMIRS search results HTML into licence dicts.

    Results table uses licenceElementTitle links (alternating name, licence_no)
    and licenceStatus spans for status.
    """
    if not html:
        return []

    results = []

    # Find all result blocks — each has a title link + status span
    # Pattern: licenceElementTitle links come in pairs [name, licence_no]
    title_pattern = re.compile(
        r'class="licenceElementTitle"[^>]*>([^<]+)<', re.IGNORECASE,
    )
    status_pattern = re.compile(
        r'class="licenceStatus"[^>]*>([^<]+)<', re.IGNORECASE,
    )

    titles = title_pattern.findall(html)
    statuses = status_pattern.findall(html)

    # Titles alternate: [name, licence_no, name, licence_no, ...]
    for i in range(0, len(titles) - 1, 2):
        name = titles[i].strip()
        licence_no = titles[i + 1].strip()
        status_idx = i // 2
        status = statuses[status_idx].strip() if status_idx < len(statuses) else "Unknown"
        results.append({
            "licensee": name,
            "licence_number": licence_no,
            "status": status,
        })

    return results


async def wa_dmirs_lookup(search_name: str, trade: str) -> dict | None:
    """Look up a WA trade licence via the DMIRS online search.

    1. GET search page → extract ViewState + cookies
    2. POST PrimeFaces AJAX with search name
    3. Parse results, best-match by name
    4. Returns standard licence_info dict or None.
    """
    config = get_licence_config("WA", trade)
    if not config or "dmirs_search_code" not in config:
        return None

    try:
        import httpx
        # Per-call client for cookie persistence across GET→POST
        async with httpx.AsyncClient(
            timeout=12.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
            follow_redirects=True,
        ) as client:
            # Step 1: GET search page
            resp = await client.get(_WA_DMIRS_URL, params=_WA_DMIRS_PARAMS)
            if resp.status_code != 200:
                logger.error(f"[WA-DMIRS] GET failed: {resp.status_code}")
                return None

            viewstate = _wa_dmirs_extract_viewstate(resp.text)
            if not viewstate:
                logger.error("[WA-DMIRS] Could not extract ViewState")
                return None

            # Step 2: POST PrimeFaces AJAX search
            # Find the form ID and search input from the page
            form_id_match = re.search(r'<form[^>]*id="([^"]+)"', resp.text)
            form_id = form_id_match.group(1) if form_id_match else "mainForm"

            post_data = {
                "javax.faces.partial.ajax": "true",
                "javax.faces.source": f"{form_id}:searchButton",
                "javax.faces.partial.execute": f"{form_id}",
                "javax.faces.partial.render": f"{form_id}:resultsPanel",
                f"{form_id}:searchButton": f"{form_id}:searchButton",
                f"{form_id}:nameInput": search_name,
                "javax.faces.ViewState": viewstate,
            }

            resp2 = await client.post(
                _WA_DMIRS_URL,
                data=post_data,
                params=_WA_DMIRS_PARAMS,
                headers={"Faces-Request": "partial/ajax"},
            )
            if resp2.status_code != 200:
                logger.error(f"[WA-DMIRS] POST failed: {resp2.status_code}")
                return None

            # Step 3: Parse results
            results = _wa_dmirs_parse_results(resp2.text)
            if not results:
                logger.info(f"[WA-DMIRS] No results for '{search_name}' ({trade})")
                return None

            # Step 4: Best match — prefer current, substring match on name
            search_lower = search_name.lower()
            # Strip common suffixes for matching
            clean_search = re.sub(
                r'\s*(PTY\.?\s*LTD\.?|LTD\.?|INC\.?)\s*$', '',
                search_lower, flags=re.IGNORECASE,
            ).strip()

            best = None
            for r in results:
                if r.get("status", "").lower() != "current":
                    continue
                lic_name = r.get("licensee", "").lower()
                if clean_search in lic_name or lic_name in clean_search:
                    best = r
                    break

            # Fallback: word overlap >= 50%
            if not best:
                current = [r for r in results if r.get("status", "").lower() == "current"]
                if len(current) == 1:
                    best = current[0]
                elif current:
                    search_words = set(clean_search.split())
                    for r in current:
                        lic_words = set(r.get("licensee", "").lower().split())
                        if search_words and lic_words:
                            overlap = len(search_words & lic_words) / min(len(search_words), len(lic_words))
                            if overlap >= 0.5:
                                best = r
                                break

            if not best:
                logger.info(f"[WA-DMIRS] No matching current licence for '{search_name}'")
                return None

            logger.info(f"[WA-DMIRS] Found: {best['licensee']} — {best['licence_number']} ({best['status']})")
            return {
                "licence_number": best["licence_number"],
                "licensee": best["licensee"],
                "licence_type": trade,
                "status": best["status"],
                "licence_source": "wa_dmirs",
                "classes": [{"name": c, "active": True} for c in config["default_classes"]],
                "compliance_clean": True,
                "associated_parties": [],
                "business_address": "",
            }

    except Exception as e:
        logger.error(f"[WA-DMIRS] {type(e).__name__}: {e}")
        return None


def _gaps_for_category(cat_key: str, cat_data: dict, mapped_ids: set[int],
                       mapped_names: set[str], seen_names: set[str]) -> list[dict]:
    """Compute gaps for a single category. Mutates seen_names for cross-category dedup."""
    cat_id = cat_data.get("category_id", 0)
    cat_name = cat_data.get("category_name", cat_key)
    gaps = []
    for sc in cat_data.get("subcategories", []):
        sc_id = sc.get("subcategory_id")
        sc_name = sc.get("subcategory_name", "")
        if sc_id and int(sc_id) not in mapped_ids and sc_name not in mapped_names and sc_name not in seen_names:
            seen_names.add(sc_name)
            gaps.append({
                "subcategory_id": sc_id,
                "subcategory_name": sc_name,
                "category_id": cat_id,
                "category_name": cat_name,
            })
    return gaps


def compute_service_gaps(services: list[dict], business_name: str,
                         licence_classes: list[str] | None = None,
                         google_business_name: str = "",
                         google_primary_type: str = "") -> list[dict]:
    """Compute which subcategories are NOT yet mapped for this trade.

    Supports multi-category: collects ALL distinct category_name values from
    existing services + _detect_categories(), computes gaps for each.
    Returns merged flat list of {"subcategory_id": ..., "subcategory_name": ...} dicts.
    """
    categories = _load_categories()
    if not categories:
        return []

    # Collect all category keys from existing services
    matched_cat_keys: list[str] = []
    seen_keys: set[str] = set()
    if services:
        for s in services:
            cn = s.get("category_name", "")
            if cn and cn in categories and cn not in seen_keys:
                seen_keys.add(cn)
                matched_cat_keys.append(cn)

    # Also detect categories from business signals (catches categories not yet serviced)
    detected = _detect_categories(
        business_name, licence_classes or [],
        google_business_name, google_primary_type,
    )
    for dk in detected:
        if dk in categories and dk not in seen_keys:
            seen_keys.add(dk)
            matched_cat_keys.append(dk)

    # Fallback: single detect (backward compat for edge cases)
    if not matched_cat_keys:
        single = _detect_category(
            business_name, licence_classes,
            google_business_name, google_primary_type,
        )
        if single and single in categories:
            matched_cat_keys = [single]

    if not matched_cat_keys:
        return []

    # Get IDs and names already mapped (coerce to int for safe comparison)
    mapped_ids: set[int] = set()
    mapped_names: set[str] = set()
    for s in services:
        sid = s.get("subcategory_id")
        if sid is not None:
            try:
                mapped_ids.add(int(sid))
            except (ValueError, TypeError):
                pass
        sn = s.get("subcategory_name", "")
        if sn:
            mapped_names.add(sn)

    # Compute gaps for each category (deduplicate by name across categories)
    all_gaps: list[dict] = []
    seen_gap_names: set[str] = set()
    for cat_key in matched_cat_keys:
        cat_data = categories[cat_key]
        gaps = _gaps_for_category(cat_key, cat_data, mapped_ids, mapped_names, seen_gap_names)
        all_gaps.extend(gaps)

    return all_gaps


# ────────── TIERED SERVICE MAPPING ──────────

_tiers_cache = None

def _load_service_tiers() -> dict:
    """Load tier definitions for guided trades. Cached on first read."""
    global _tiers_cache
    if _tiers_cache is not None:
        return _tiers_cache

    tiers_file = RESOURCES_DIR / "service_tiers.json"
    if tiers_file.exists():
        with open(tiers_file) as f:
            _tiers_cache = json.load(f)
        return _tiers_cache
    _tiers_cache = {}
    return _tiers_cache


# ────────── RELATED CATEGORIES (CO-OCCURRENCE DATA) ──────────

_related_categories_cache = None

def _load_related_categories() -> dict:
    """Load related_categories.json. Cached on first read."""
    global _related_categories_cache
    if _related_categories_cache is not None:
        return _related_categories_cache

    rc_file = RESOURCES_DIR / "related_categories.json"
    if rc_file.exists():
        with open(rc_file) as f:
            _related_categories_cache = json.load(f)
        return _related_categories_cache
    _related_categories_cache = {}
    return _related_categories_cache


def suggest_related_categories(
    detected_categories: list[str],
    min_pct: int = 15,
    max_suggestions: int = 4,
) -> list[dict]:
    """Return related categories not already detected.

    Uses co-occurrence data from related_categories.json.
    Returns [{"category": "Handyman", "pct": 42}, ...] sorted by pct desc.
    """
    related = _load_related_categories()
    if not related or not detected_categories:
        return []

    detected_set = set(detected_categories)
    # Gather suggestions from all detected categories, deduplicate
    seen: set[str] = set()
    candidates: list[dict] = []

    for cat in detected_categories:
        for entry in related.get(cat, []):
            name = entry["category"]
            pct = entry["pct"]
            if name in detected_set or name in seen or pct < min_pct:
                continue
            seen.add(name)
            candidates.append({"category": name, "pct": pct})

    candidates.sort(key=lambda x: x["pct"], reverse=True)
    return candidates[:max_suggestions]


def map_extra_categories(
    category_names: list[str],
    existing_services: list[dict],
    existing_mapped_names: set[str],
    evidence_lower: str,
    licence_classes: list[str],
) -> tuple[list[dict], list[dict], set[str]]:
    """Map services + gaps for additional accepted categories.

    For tiered categories: runs _map_single_tier() to get core/evidence/licence services.
    For non-tiered categories: loads all subcategories as gaps for LLM.
    Returns (new_services, new_gaps, updated_mapped_names).
    """
    tiers = _load_service_tiers()
    categories = _load_categories()
    if not categories:
        return [], [], set(existing_mapped_names)

    all_new_services: list[dict] = []
    all_new_gaps: list[dict] = []
    mapped_names = set(existing_mapped_names)

    # Also exclude names already in existing_services
    for svc in existing_services:
        mapped_names.add(svc.get("subcategory_name", ""))

    for cat_name in category_names:
        cat_data = categories.get(cat_name)
        if not cat_data:
            continue

        tier_def = tiers.get(cat_name)
        if tier_def and cat_data:
            # Tiered: map core/evidence/licence services
            new_svcs, mapped_names = _map_single_tier(
                tier_def, cat_data, evidence_lower, licence_classes, mapped_names,
            )
            all_new_services.extend(new_svcs)

            # Specialist gaps for this category (unmapped subcategories)
            cat_id = cat_data.get("category_id", 0)
            cat_display = cat_data.get("category_name", cat_name)
            for sc in cat_data.get("subcategories", []):
                sc_name = sc["subcategory_name"]
                if sc_name not in mapped_names:
                    all_new_gaps.append({
                        "subcategory_id": sc["subcategory_id"],
                        "subcategory_name": sc_name,
                        "category_id": cat_id,
                        "category_name": cat_display,
                    })
                    mapped_names.add(sc_name)
        else:
            # Non-tiered: all subcategories become gaps for LLM
            cat_id = cat_data.get("category_id", 0)
            cat_display = cat_data.get("category_name", cat_name)
            for sc in cat_data.get("subcategories", []):
                sc_name = sc["subcategory_name"]
                if sc_name not in mapped_names:
                    all_new_gaps.append({
                        "subcategory_id": sc["subcategory_id"],
                        "subcategory_name": sc_name,
                        "category_id": cat_id,
                        "category_name": cat_display,
                    })
                    mapped_names.add(sc_name)

    logger.info(f"[EXTRA-CAT] Mapped {len(all_new_services)} services + {len(all_new_gaps)} gaps "
                f"from extra categories: {category_names}")
    return all_new_services, all_new_gaps, mapped_names


def _detect_category(business_name: str, licence_classes: list[str] | None,
                     google_business_name: str, google_primary_type: str) -> str | None:
    """Detect the trade category using the standard priority chain.

    Same logic as compute_service_gaps but without requiring existing services.
    Returns category key (e.g. "Electrician") or None.
    """
    # Priority 1: business name
    name_lower = business_name.lower()
    for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
        if keyword in name_lower:
            return cat_key

    # Priority 2: licence classes
    if licence_classes:
        for lc in licence_classes:
            lc_lower = lc.lower()
            for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
                if keyword in lc_lower:
                    return cat_key

    # Priority 3: Google Places business name
    if google_business_name:
        gname_lower = google_business_name.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in gname_lower:
                return cat_key

    # Priority 4: Google Places primary type
    if google_primary_type:
        gtype_lower = google_primary_type.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in gtype_lower:
                return cat_key

    return None


def _detect_categories(business_name: str, licence_classes: list[str] | None,
                       google_business_name: str, google_primary_type: str,
                       max_categories: int = 2) -> list[str]:
    """Detect ALL matching trade categories (up to max_categories) using the priority chain.

    Unlike _detect_category() which returns on first hit, this scans ALL keywords
    across all 4 priority levels and collects unique matches. Capped at max_categories.
    Returns [] if no match.
    """
    seen: set[str] = set()
    results: list[str] = []

    def _add(cat_key: str):
        if cat_key not in seen and len(results) < max_categories:
            seen.add(cat_key)
            results.append(cat_key)

    # Priority 1: business name
    name_lower = business_name.lower()
    for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
        if keyword in name_lower:
            _add(cat_key)

    # Priority 2: licence classes
    if licence_classes:
        for lc in licence_classes:
            lc_lower = lc.lower()
            for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
                if keyword in lc_lower:
                    _add(cat_key)

    # Priority 3: Google Places business name
    if google_business_name:
        gname_lower = google_business_name.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in gname_lower:
                _add(cat_key)

    # Priority 4: Google Places primary type
    if google_primary_type:
        gtype_lower = google_primary_type.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in gtype_lower:
                _add(cat_key)

    return results


def _map_single_tier(
    tier_def: dict, cat_data: dict, evidence_lower: str,
    licence_classes: list[str], already_mapped_names: set[str],
) -> tuple[list[dict], set[str]]:
    """Map services for one category using core/evidence/licence tiers.

    Returns (new_services, new_mapped_names) for this category.
    Skips any service name already in already_mapped_names.
    """
    cat_id = cat_data.get("category_id", 0)
    cat_name = cat_data.get("category_name", "")
    all_subcats = cat_data.get("subcategories", [])

    # Build name→subcat lookup
    subcat_by_name: dict[str, dict] = {}
    for sc in all_subcats:
        subcat_by_name[sc["subcategory_name"]] = sc

    def _resolve(name: str, confidence: str = "high") -> dict | None:
        sc = subcat_by_name.get(name)
        if not sc:
            return None
        return {
            "input": name,
            "category_name": cat_name,
            "category_id": cat_id,
            "subcategory_name": sc["subcategory_name"],
            "subcategory_id": sc["subcategory_id"],
            "confidence": confidence,
        }

    services: list[dict] = []
    mapped_names: set[str] = set(already_mapped_names)

    # 1. Core services — always mapped
    for name in tier_def.get("core", []):
        if name in mapped_names:
            continue
        svc = _resolve(name)
        if svc:
            services.append(svc)
            mapped_names.add(name)

    # 2. Evidence-based services
    for svc_name, keywords in tier_def.get("evidence_keywords", {}).items():
        if svc_name in mapped_names:
            continue
        for kw in keywords:
            if kw in evidence_lower:
                svc = _resolve(svc_name, "evidence")
                if svc:
                    services.append(svc)
                    mapped_names.add(svc_name)
                break

    # 3. Licence signal services
    for svc_name, signals in tier_def.get("licence_signals", {}).items():
        if svc_name in mapped_names:
            continue
        # First check evidence text
        for sig in signals:
            if sig in evidence_lower:
                svc = _resolve(svc_name, "evidence")
                if svc:
                    services.append(svc)
                    mapped_names.add(svc_name)
                break
        if svc_name in mapped_names:
            continue
        # Then check licence classes
        for lc in licence_classes:
            lc_lower = lc.lower()
            matched = False
            for sig in signals:
                if sig in lc_lower:
                    svc = _resolve(svc_name, "licence")
                    if svc:
                        services.append(svc)
                        mapped_names.add(svc_name)
                    matched = True
                    break
            if matched:
                break

    return services, mapped_names


def compute_initial_services(
    business_name: str,
    licence_classes: list[str],
    google_business_name: str,
    google_primary_type: str,
    google_reviews: list[dict],
    web_results: list[dict],
    website_text: str = "",
) -> dict:
    """Build initial service list using tiered mapping from guides.

    Supports multi-category businesses (e.g. "Smith Building & Carpentry").
    Returns:
      {"services": [...], "specialist_gaps": [...], "category_name": str,
       "category_names": [str, ...], "tiered": True}
      or {"tiered": False} if no tier definition exists for any detected trade.
    """
    tiers = _load_service_tiers()
    categories = _load_categories()
    if not tiers or not categories:
        return {"tiered": False}

    # Detect categories (multi-category)
    cat_keys = _detect_categories(business_name, licence_classes,
                                  google_business_name, google_primary_type)
    # Filter to categories that have both a tier definition and category data
    tiered_keys = [k for k in cat_keys if k in tiers and k in categories]
    if not tiered_keys:
        return {"tiered": False}

    # Build evidence text once (shared across all categories)
    evidence_text = ""
    for rev in google_reviews:
        evidence_text += " " + rev.get("text", "")
    for wr in web_results:
        evidence_text += " " + wr.get("title", "") + " " + wr.get("description", "")
    if website_text:
        evidence_text += " " + website_text
    evidence_lower = evidence_text.lower()

    # Map services for each tiered category
    all_services: list[dict] = []
    all_mapped_names: set[str] = set()
    cat_names: list[str] = []

    for cat_key in tiered_keys:
        tier_def = tiers[cat_key]
        cat_data = categories[cat_key]
        cat_names.append(cat_data.get("category_name", cat_key))

        new_services, all_mapped_names = _map_single_tier(
            tier_def, cat_data, evidence_lower, licence_classes, all_mapped_names,
        )
        all_services.extend(new_services)

        logger.info(f"[TIERS] {cat_key}: {len(new_services)} pre-mapped "
                    f"(core={sum(1 for s in new_services if s.get('confidence') == 'high')}, "
                    f"evidence={sum(1 for s in new_services if s.get('confidence') == 'evidence')}, "
                    f"licence={sum(1 for s in new_services if s.get('confidence') == 'licence')})")

    # Specialist gaps — all subcategories NOT yet mapped across ALL tiered categories
    specialist_gaps = []
    seen_gap_names: set[str] = set()
    for cat_key in tiered_keys:
        cat_data = categories[cat_key]
        cat_id = cat_data.get("category_id", 0)
        cat_name = cat_data.get("category_name", cat_key)
        for sc in cat_data.get("subcategories", []):
            sc_name = sc["subcategory_name"]
            if sc_name not in all_mapped_names and sc_name not in seen_gap_names:
                seen_gap_names.add(sc_name)
                specialist_gaps.append({
                    "subcategory_id": sc["subcategory_id"],
                    "subcategory_name": sc_name,
                    "category_id": cat_id,
                    "category_name": cat_name,
                })

    logger.info(f"[TIERS] Total: {len(all_services)} pre-mapped, "
                f"{len(specialist_gaps)} specialist gaps across {cat_names}")

    return {
        "services": all_services,
        "specialist_gaps": specialist_gaps,
        "category_name": cat_names[0],  # backward-compat: primary category
        "category_names": cat_names,
        "tiered": True,
    }


def get_filtered_cluster_groups(
    gaps: list[dict],
    business_name: str,
    licence_classes: list[str] | None = None,
    google_business_name: str = "",
    google_primary_type: str = "",
) -> list[dict]:
    """Return pre-defined cluster groups filtered to remaining gaps.

    Supports multi-category: loads cluster_groups for each detected tiered
    category, concatenates (primary first, secondary appended).
    Returns list of {"label": str, "services": [{"name": str, "id": int}]} dicts.
    """
    tiers = _load_service_tiers()
    if not tiers:
        return []

    cat_keys = _detect_categories(business_name, licence_classes or [],
                                  google_business_name, google_primary_type)
    tiered_keys = [k for k in cat_keys if k in tiers]
    if not tiered_keys:
        return []

    # Build gap name→id lookup from current gaps
    gap_lookup = {g["subcategory_name"]: g["subcategory_id"] for g in gaps}

    filtered = []
    for cat_key in tiered_keys:
        tier_def = tiers[cat_key]
        cluster_defs = tier_def.get("cluster_groups", [])
        for cluster in cluster_defs:
            label = cluster["label"]
            remaining = []
            for svc_name in cluster["services"]:
                if svc_name in gap_lookup:
                    remaining.append({"name": svc_name, "id": gap_lookup[svc_name]})
            if remaining:
                filtered.append({"label": label, "services": remaining})

    return filtered


# ────────── LOCATION PARSER ──────────

_suburbs_cache = None

def _load_suburbs() -> list[dict]:
    """Load suburbs database."""
    global _suburbs_cache
    if _suburbs_cache is not None:
        return _suburbs_cache

    suburbs_file = RESOURCES_DIR / "suburbs.csv"
    if not suburbs_file.exists():
        return []

    _suburbs_cache = []
    with open(suburbs_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            _suburbs_cache.append(row)

    return _suburbs_cache


def search_suburbs_by_postcode(postcode: str) -> list[dict]:
    """Find suburbs matching a postcode."""
    suburbs = _load_suburbs()
    return [s for s in suburbs if s.get("postcode") == postcode]


def get_suburbs_within_radius(lat: float, lng: float, radius_km: float) -> list[dict]:
    """Get suburbs within radius of a point using haversine."""
    suburbs = _load_suburbs()
    results = []

    for s in suburbs:
        try:
            s_lat = float(s.get("lat", s.get("latitude", 0)))
            s_lng = float(s.get("lng", s.get("longitude", 0)))
            if s_lat == 0 or s_lng == 0:
                continue
            dist = _haversine(lat, lng, s_lat, s_lng)
            if dist <= radius_km:
                results.append({**s, "distance_km": round(dist, 1)})
        except (ValueError, TypeError):
            continue

    return sorted(results, key=lambda x: x.get("distance_km", 999))


def _haversine(lat1, lng1, lat2, lng2):
    """Calculate distance in km between two lat/lng points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def get_suburbs_in_radius_grouped(base_postcode: str, radius_km: float = 20.0) -> dict:
    """Get suburbs within radius, grouped by area. Returns base info + grouped results."""
    base_suburbs = search_suburbs_by_postcode(base_postcode)
    if not base_suburbs:
        return {"base_suburb": None, "by_area": {}, "total": 0}

    base = base_suburbs[0]
    base_name = base.get("name", "Unknown")
    base_state = base.get("state", "")
    try:
        lat = float(base.get("lat", 0))
        lng = float(base.get("lng", 0))
    except (ValueError, TypeError):
        return {"base_suburb": base_name, "by_area": {}, "total": 0}

    if not lat or not lng:
        return {"base_suburb": base_name, "by_area": {}, "total": 0}

    nearby = get_suburbs_within_radius(lat, lng, radius_km)

    # Filter to same state (CSV has bad data — some NT suburbs have Sydney coords)
    if base_state:
        nearby = [s for s in nearby if s.get("state", "") == base_state]

    # Group by area
    by_area = {}
    for s in nearby:
        area = s.get("area", s.get("region", "Other"))
        if not area:
            area = "Other"
        if area not in by_area:
            by_area[area] = []
        by_area[area].append({
            "name": s.get("name", ""),
            "postcode": s.get("postcode", ""),
            "distance_km": s.get("distance_km", 0),
        })

    return {
        "base_suburb": base_name,
        "base_postcode": base_postcode,
        "base_state": base.get("state", ""),
        "base_lat": lat,
        "base_lng": lng,
        "radius_km": radius_km,
        "by_area": by_area,
        "total": len(nearby),
    }


_regional_cache: dict[str, str] = {}


def get_regional_guide(state_code: str) -> str:
    """Get the regional guide for a state (sydney, melbourne, etc). Cached on first read."""
    key = state_code.upper()
    if key in _regional_cache:
        return _regional_cache[key]
    state_map = {"NSW": "sydney", "VIC": "melbourne", "QLD": "brisbane", "WA": "perth"}
    city = state_map.get(key, "")
    if not city:
        _regional_cache[key] = ""
        return ""
    guide_path = RESOURCES_DIR / f"{city}_regions.md"
    content = guide_path.read_text() if guide_path.exists() else ""
    _regional_cache[key] = content
    return content


_guide_cache: dict[str, str] = {}


def find_subcategory_guide(business_name: str) -> str:
    """Find relevant subcategory guides based on business name trade type.

    Multi-category: scans ALL matching keywords and concatenates guides.
    Deduplicates by filename (a keyword may map to the same file).
    Cached on first read.
    """
    name_lower = business_name.lower()

    # Map trade keywords to guide files
    trade_guides = {
        "plumb": ["plumber-subcategory-guide.md", "plumbing_subcategories.md"],
        "paint": ["painter_subcategories.md"],
        "electri": ["electrician-subcategory-guide.md", "electrical_subcategories.md"],
        "clean": ["cleaner-subcategory-guide.md"],
        "garden": ["gardener-subcategory-guide.md"],
        "carpent": ["carpentry_subcategories.md"],
        "build": ["builder-subcategory-guide.md"],
    }

    parts: list[str] = []
    loaded_files: set[str] = set()

    for keyword, files in trade_guides.items():
        if keyword in name_lower:
            for fname in files:
                if fname in loaded_files:
                    continue
                loaded_files.add(fname)
                if fname in _guide_cache:
                    parts.append(_guide_cache[fname])
                else:
                    fpath = RESOURCES_DIR / fname
                    if fpath.exists():
                        content = fpath.read_text()
                        _guide_cache[fname] = content
                        parts.append(content)

    return "\n\n---\n\n".join(parts) if parts else ""


# ────────── NSW FAIR TRADING LICENCE LOOKUP ──────────

_nsw_trades_token: dict = {"access_token": "", "expires_at": 0}


async def _get_nsw_trades_token() -> str:
    """Get or refresh OAuth token for NSW Trades API.

    Swagger: https://apinsw.onegov.nsw.gov.au/api/swagger/spec/25
    Token endpoint: GET /oauth/client_credential/accesstoken?grant_type=client_credentials
    Returns BearerToken valid for ~12 hours.
    """
    import time
    global _nsw_trades_token

    if not NSW_TRADES_API_KEY:
        return ""

    # Return cached token if still valid (with 60s buffer)
    if _nsw_trades_token["access_token"] and time.time() < _nsw_trades_token["expires_at"] - 60:
        return _nsw_trades_token["access_token"]

    try:
        # Per Swagger spec: GET with grant_type as query param, Basic auth in header
        resp = await _http_client.get(
            "https://api.onegov.nsw.gov.au/oauth/client_credential/accesstoken",
            params={"grant_type": "client_credentials"},
            headers={"Authorization": NSW_TRADES_AUTH_HEADER},
        )
        if resp.status_code == 200:
            try:
                data = resp.json()
                token = data.get("access_token", "")
                expires_in = int(data.get("expires_in", 43200))  # Default 12h
                if token:
                    _nsw_trades_token["access_token"] = token
                    _nsw_trades_token["expires_at"] = time.time() + expires_in
                    logger.info(f"[NSW TRADES] Got OAuth token, expires in {expires_in}s")
                    return token
                logger.warning(f"[NSW TRADES] Token response missing access_token: {resp.text[:200]}")
                return ""
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"[NSW TRADES] Token response not JSON: {resp.text[:200]}")
                return ""
        else:
            logger.error(f"[NSW TRADES] Token request failed: {resp.status_code}")
            return ""
    except Exception as e:
        logger.error(f"[NSW TRADES] Token error: {e}")
        return ""


async def nsw_licence_browse(search_term: str) -> dict:
    """Browse the NSW Fair Trading Trades Register by name.

    Uses GET /tradesregister/v1/browse?searchText=...
    Returns list of matching licences with: licenceID, licensee, licenceNumber,
    licenceType, status, suburb, postcode, expiryDate, categories, classes.
    """
    if not NSW_TRADES_API_KEY:
        logger.warning("[NSW TRADES] No API key configured, skipping licence lookup")
        return {"results": [], "error": "NSW Trades API not configured"}

    token = await _get_nsw_trades_token()
    if not token:
        logger.warning("[NSW TRADES] Could not get OAuth token, skipping licence lookup")
        return {"results": [], "error": "Could not authenticate with NSW Trades API"}

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": NSW_TRADES_API_KEY,
            "Accept": "application/json",
        }

        resp = await _http_client.get(
            "https://api.onegov.nsw.gov.au/tradesregister/v1/browse",
            headers=headers,
            params={"searchText": search_term},
        )

        if resp.status_code != 200:
            logger.error(f"[NSW TRADES] Browse failed: {resp.status_code} - {resp.text[:200]}")
            return {"results": [], "error": f"API returned {resp.status_code}"}

        data = resp.json()
        entries = data if isinstance(data, list) else []

        results = []
        for entry in entries[:10]:
            results.append({
                "licence_id": entry.get("licenceID", ""),
                "licensee": entry.get("licensee", ""),
                "licence_number": entry.get("licenceNumber", ""),
                "licence_type": entry.get("licenceType", ""),
                "status": entry.get("status", ""),
                "suburb": entry.get("suburb", ""),
                "postcode": entry.get("postcode", ""),
                "expiry_date": entry.get("expiryDate", ""),
                "categories": entry.get("categories"),
                "classes": entry.get("classes"),
                "business_names": entry.get("businessNames"),
            })

        logger.info(f"[NSW TRADES] Found {len(results)} licence results for '{search_term}'")
        return {"results": results, "count": len(results)}

    except httpx.TimeoutException as e:
        logger.error(f"[NSW TRADES] Lookup timeout: {e}")
        return {"results": [], "error": "timeout"}
    except httpx.HTTPError as e:
        logger.error(f"[NSW TRADES] Lookup HTTP error: {e}")
        return {"results": [], "error": str(e)}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"[NSW TRADES] Lookup parse error: {e}")
        return {"results": [], "error": str(e)}
    except Exception as e:
        logger.error(f"[NSW TRADES] Lookup unexpected error ({type(e).__name__}): {e}")
        return {"results": [], "error": str(e)}


async def nsw_licence_details(licence_id: str) -> dict:
    """Get detailed info for a specific licence.

    Uses GET /tradesregister/v1/details?licenceid=...
    Returns: licenceDetail (ABN, ACN, status, dates), licenceClasses (trade classes),
    conditions, complianceActions, associatedParties, etc.
    """
    if not NSW_TRADES_API_KEY:
        return {"error": "NSW Trades API not configured"}

    token = await _get_nsw_trades_token()
    if not token:
        return {"error": "Could not authenticate"}

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": NSW_TRADES_API_KEY,
            "Accept": "application/json",
        }

        resp = await _http_client.get(
            "https://api.onegov.nsw.gov.au/tradesregister/v1/details",
            headers=headers,
            params={"licenceid": licence_id},
        )

        if resp.status_code != 200:
            logger.error(f"[NSW TRADES] Details failed: {resp.status_code}")
            return {"error": f"API returned {resp.status_code}"}

        data = resp.json()

        # Extract the key fields
        detail = data.get("licenceDetail", {})
        classes = data.get("licenceClasses", [])
        conditions = data.get("conditions", [])
        compliance = data.get("complianceActions", {})

        return {
            "licensee": detail.get("licensee", ""),
            "licence_number": detail.get("licenceNumber", ""),
            "licence_type": detail.get("licenceType", ""),
            "status": detail.get("status", ""),
            "start_date": detail.get("startDate", ""),
            "expiry_date": detail.get("expiryDate", ""),
            "abn": detail.get("abn", ""),
            "acn": detail.get("acn", ""),
            "classes": [
                {"name": c.get("className", ""), "active": c.get("isActive") == "True"}
                for c in classes
            ],
            "conditions": [c for c in conditions if c],
            "compliance_clean": (
                compliance.get("publicWarningsCount", 0) == 0
                and compliance.get("cautionReprimandCount", 0) == 0
                and not compliance.get("suspensions")
                and not compliance.get("prosecutions")
            ) if compliance else True,
            "associated_parties": [
                {
                    "name": p.get("name", ""),
                    "role": p.get("role", ""),
                    "party_type": p.get("partyType", ""),
                }
                for p in data.get("associatedParties", [])
                if p.get("isActive") == "True"
            ],
            "business_address": detail.get("address", ""),
            "raw": data,
        }

    except httpx.TimeoutException as e:
        logger.error(f"[NSW TRADES] Details timeout: {e}")
        return {"error": "timeout"}
    except httpx.HTTPError as e:
        logger.error(f"[NSW TRADES] Details HTTP error: {e}")
        return {"error": str(e)}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"[NSW TRADES] Details parse error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"[NSW TRADES] Details unexpected error ({type(e).__name__}): {e}")
        return {"error": str(e)}


# ────────── QBCC LICENCE CSV (QLD) ──────────

_qbcc_licences: dict = {"abn_index": {}, "name_index": {}, "loaded": False}


def qbcc_load_csv() -> None:
    """Load QBCC licensed contractors CSV into memory indexes.

    Reads resources/qbcc_licences.csv (UTF-8 with BOM), builds ABN and name indexes.
    All rows in the published CSV are active licences ("Licence In Force").
    Called once at server startup.
    """
    csv_path = RESOURCES_DIR / "qbcc_licences.csv"
    if not csv_path.exists():
        logger.warning("[QBCC] CSV not found at %s — QLD licence lookup disabled", csv_path)
        return

    import time as _time
    t0 = _time.time()
    abn_index: dict[str, list[dict]] = {}
    name_index: dict[str, list[dict]] = {}
    row_count = 0

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            abn_raw = row.get("ABN", "").strip().replace(" ", "")
            name_raw = row.get("Licensee Name", "").strip()
            if abn_raw:
                abn_index.setdefault(abn_raw, []).append(row)
            if name_raw:
                name_key = name_raw.upper()
                name_index.setdefault(name_key, []).append(row)

    _qbcc_licences["abn_index"] = abn_index
    _qbcc_licences["name_index"] = name_index
    _qbcc_licences["loaded"] = True

    t1 = _time.time()
    logger.info(f"[QBCC] Loaded {row_count} active licences for {len(abn_index)} unique ABNs ({t1 - t0:.1f}s)")


def qbcc_licence_lookup(abn: str, legal_name: str) -> dict | None:
    """Look up a QLD licence from the pre-loaded QBCC CSV.

    Primary: match by ABN. Fallback: match by normalised legal name.
    Returns a licence_info dict matching the NSW format, or None if no match.
    """
    if not _qbcc_licences["loaded"]:
        return None

    abn_clean = abn.strip().replace(" ", "") if abn else ""
    matched_rows = _qbcc_licences["abn_index"].get(abn_clean, [])

    if not matched_rows and legal_name:
        # Exact match first
        name_key = legal_name.strip().upper()
        matched_rows = _qbcc_licences["name_index"].get(name_key, [])

        # Normalised fallback: strip punctuation and common suffixes
        if not matched_rows:
            normalised = re.sub(r'[.\',]', '', name_key)
            normalised = re.sub(r'\s+', ' ', normalised).strip()
            if normalised != name_key:
                matched_rows = _qbcc_licences["name_index"].get(normalised, [])
            # Try word-overlap against index keys if still no match
            if not matched_rows:
                search_words = set(normalised.split()) - {"PTY", "LTD", "LIMITED", "THE", "AND", "&"}
                if len(search_words) >= 2:
                    for idx_key, rows in _qbcc_licences["name_index"].items():
                        idx_words = set(idx_key.split()) - {"PTY", "LTD", "LIMITED", "THE", "AND", "&"}
                        if idx_words and search_words:
                            overlap = len(search_words & idx_words) / min(len(search_words), len(idx_words))
                            if overlap >= 0.8:
                                matched_rows = rows
                                break

    if not matched_rows:
        return None

    # Aggregate all licence class types across matched rows, dedup by name
    seen_classes = set()
    classes = []
    for row in matched_rows:
        cls_name = row.get("Licence Class Type", "").strip()
        if cls_name and cls_name not in seen_classes:
            seen_classes.add(cls_name)
            classes.append({"name": cls_name, "active": True})

    first = matched_rows[0]
    address = first.get("Licensee Business Address", "").strip()

    return {
        "licensee": first.get("Licensee Name", "").strip(),
        "licence_number": first.get("Licence Number", "").strip(),
        "licence_type": first.get("Licence Grade", "").strip(),
        "status": "Current",
        "expiry_date": "",
        "classes": classes,
        "compliance_clean": True,
        "associated_parties": [],
        "business_address": address,
        "licence_source": "qbcc_csv",
    }


# ────────── BRAVE WEB SEARCH ──────────

async def brave_web_search(query: str, count: int = 5) -> list[dict]:
    """Search the web using Brave Search API.

    Returns top results with title, url, and description.
    Retries once on 429 (rate limit) after a short delay.
    """
    import asyncio as _aio

    if not BRAVE_SEARCH_API_KEY:
        logger.warning("[BRAVE] No API key configured, skipping web search")
        return []

    for attempt in range(2):
        try:
            resp = await _http_client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
                },
                params={
                    "q": query,
                    "count": str(count),
                    "country": "AU",
                },
            )

            if resp.status_code == 429 and attempt == 0:
                logger.warning(f"[BRAVE] Rate limited, retrying in 1s...")
                await _aio.sleep(1.0)
                continue

            if resp.status_code != 200:
                logger.error(f"[BRAVE] Search failed: {resp.status_code}")
                return []

            data = resp.json()
            results = []
            for item in data.get("web", {}).get("results", [])[:count]:
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                }
                thumb = item.get("thumbnail", {})
                if thumb and thumb.get("src"):
                    result["thumbnail"] = thumb["src"]
                results.append(result)

            logger.info(f"[BRAVE] Found {len(results)} results for '{query}'")
            return results

        except httpx.TimeoutException as e:
            logger.error(f"[BRAVE] Search timeout: {e}")
            return []
        except httpx.HTTPError as e:
            logger.error(f"[BRAVE] Search HTTP error: {e}")
            return []
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"[BRAVE] Search parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"[BRAVE] Search unexpected error ({type(e).__name__}): {e}")
            return []

    return []


# ────────── GOOGLE PLACES API ──────────

async def google_places_search(business_name: str, state_code: str = "") -> dict:
    """Search Google Places for a business and return rating, reviews, website.

    Uses the Places API (New) Text Search endpoint.
    Returns dict with: rating, review_count, website, maps_url, address, name.
    Returns empty dict on failure.
    """
    if not GOOGLE_PLACES_API_KEY:
        logger.warning("[GOOGLE] No API key configured, skipping Places search")
        return {}

    try:
        query = business_name
        if state_code:
            query += f" {state_code} Australia"

        resp = await _http_client.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount,places.websiteUri,places.googleMapsUri,places.formattedAddress,places.reviews,places.primaryType,places.types,places.photos",
            },
            json={"textQuery": query},
        )

        if resp.status_code != 200:
            logger.error(f"[GOOGLE] Places search failed: {resp.status_code} - {resp.text[:200]}")
            return {}

        data = resp.json()
        places = data.get("places", [])
        if not places:
            logger.info(f"[GOOGLE] No places found for '{query}'")
            return {}

        place = places[0]
        rating = place.get("rating", 0.0)
        review_count = place.get("userRatingCount", 0)
        website = place.get("websiteUri", "")
        maps_url = place.get("googleMapsUri", "")
        address = place.get("formattedAddress", "")
        name = place.get("displayName", {}).get("text", "")

        # Extract review text snippets
        reviews = []
        for rev in place.get("reviews", []):
            text = rev.get("text", {}).get("text", "")
            rev_rating = rev.get("rating", 0)
            if text:
                reviews.append({"text": text, "rating": rev_rating})

        primary_type = place.get("primaryType", "")
        types = place.get("types", [])

        # Extract Google Business photos (real work photos uploaded by owner/customers)
        # Resolve to actual googleusercontent.com URLs to avoid leaking API key
        photo_names = [p.get("name", "") for p in place.get("photos", [])[:10] if p.get("name")]
        photos = []
        async def _resolve_photo(photo_name: str) -> str:
            try:
                r = await _http_client.get(
                    f"https://places.googleapis.com/v1/{photo_name}/media",
                    params={"maxWidthPx": 800, "key": GOOGLE_PLACES_API_KEY, "skipHttpRedirect": "true"},
                )
                if r.status_code == 200:
                    return r.json().get("photoUri", "")
            except Exception:
                pass
            return ""
        if photo_names:
            resolved = await asyncio.gather(*[_resolve_photo(n) for n in photo_names])
            photos = [u for u in resolved if u]

        result = {
            "name": name,
            "rating": rating,
            "review_count": review_count,
            "website": website,
            "maps_url": maps_url,
            "address": address,
            "reviews": reviews[:5],
            "primary_type": primary_type,
            "types": types,
            "photos": photos,
        }

        logger.info(f"[GOOGLE] Found: {name} — {rating}★ ({review_count} reviews), website={bool(website)}, type={primary_type}, {len(photos)} photos")
        return result

    except httpx.TimeoutException as e:
        logger.error(f"[GOOGLE] Places search timeout: {e}")
        return {}
    except httpx.HTTPError as e:
        logger.error(f"[GOOGLE] Places search HTTP error: {e}")
        return {}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"[GOOGLE] Places search parse error: {e}")
        return {}
    except Exception as e:
        logger.error(f"[GOOGLE] Places search unexpected error ({type(e).__name__}): {e}")
        return {}


# ────────── WEBSITE TEXT SCRAPER (lightweight, for evidence keywords) ──────────

async def scrape_website_text(url: str, max_chars: int = 5000) -> str:
    """Fetch a website and extract visible text content for keyword matching.

    Returns plain text (stripped of HTML tags), capped at max_chars.
    Used during business confirmation to feed evidence into tiered mapping.
    """
    if not url:
        return ""
    try:
        resp = await _http_client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
            follow_redirects=True,
            timeout=8.0,
        )
        if resp.status_code != 200:
            return ""
        html = resp.text
        # Strip script/style blocks, then all tags
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        logger.info(f"[SCRAPE-TEXT] {url}: {len(text)} chars extracted")
        return text[:max_chars]
    except Exception as e:
        logger.error(f"[SCRAPE-TEXT] {url}: {type(e).__name__}: {e}")
        return ""


# ────────── BUSINESS WEBSITE DISCOVERY ──────────

# Suffixes to strip from business names before building domain guesses
_BIZ_SUFFIXES = re.compile(
    r'\b(pty\.?\s*ltd\.?|proprietary\s+limited|limited|inc\.?|llc|group)\b',
    re.IGNORECASE,
)

async def discover_business_website(business_name: str) -> str:
    """Try to find a business website by inferring common AU domain patterns.

    Strips 'PTY LTD' etc, builds candidate domains, does HEAD requests.
    Returns the first URL that resolves, or empty string.
    """
    # Clean: "AT YOUR SERVICE PLUMBING PTY LTD" → "at your service plumbing"
    clean = _BIZ_SUFFIXES.sub("", business_name).strip()
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean).strip()
    slug = clean.lower().replace(" ", "")

    if not slug or len(slug) < 4:
        return ""

    # Try common AU TLDs
    candidates = [
        f"https://www.{slug}.com.au",
        f"https://{slug}.com.au",
        f"https://www.{slug}.au",
        f"https://{slug}.au",
        f"https://www.{slug}.net.au",
    ]

    import asyncio

    async def _check(url: str) -> str:
        try:
            resp = await _http_client.head(url, follow_redirects=True, timeout=8.0)
            if resp.status_code < 400:
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type or "application" in content_type:
                    logger.info(f"[DISCOVER] Found: {url} → {resp.status_code}")
                    return str(resp.url)  # Return final URL after redirects
        except Exception as e:
            logger.error(f"[DISCOVER] Failed: {url} → {type(e).__name__}")
        return ""

    results = await asyncio.gather(*[_check(u) for u in candidates])

    # Prefer .com.au over .au (more likely to be the real content site)
    for r in results:
        if r:
            return r

    logger.info(f"[DISCOVER] No website found for '{business_name}' (tried {slug}.*)")
    return ""


# ────────── WEBSITE IMAGE SCRAPER ──────────

_scrape_cache: dict[str, dict] = {}   # domain → {"logo": ..., "photos": [...]}
_SCRAPE_CACHE_MAX = 100

# Patterns that indicate an image is a logo
_LOGO_PATTERNS = re.compile(r'logo|brand|header-img|site-icon', re.IGNORECASE)
# Patterns that indicate an image is junk (tracking pixels, social icons, etc.)
_JUNK_PATTERNS = re.compile(
    r'pixel|tracker|badge|icon-\d|sprite|spacer|gravatar|avatar'
    r'|facebook|twitter|linkedin|instagram|youtube|google|yelp'
    r'|\.svg$|1x1|blank\.|widget|button|arrow|caret|loading|spinner',
    re.IGNORECASE,
)
# Extensions we care about
_IMG_EXTENSIONS = re.compile(r'\.(jpe?g|png|webp)(\?|$)', re.IGNORECASE)


async def scrape_website_images(url: str) -> dict:
    """Fetch a website and extract logo + photo URLs from HTML.

    Returns {"logo": "url_or_empty", "photos": ["url1", ...]}
    Results are cached by domain (cleared on process restart).
    """
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    if domain in _scrape_cache:
        return _scrape_cache[domain]

    result = {"logo": "", "photos": []}

    try:
        resp = await _http_client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
            follow_redirects=True,
        )
        if resp.status_code != 200:
            logger.info(f"[SCRAPE] {url} returned {resp.status_code}")
            return result

        html = resp.text
        base_url = str(resp.url)  # After redirects

        # ── Extract logo ──
        # Priority: og:image > twitter:image > apple-touch-icon > link[rel=icon]
        logo = ""
        for pattern in [
            r'<meta\s+(?:property|name)=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
            r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\']og:image["\']',
            r'<meta\s+(?:property|name)=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']',
            r'<link\s+[^>]*rel=["\']apple-touch-icon["\'][^>]*href=["\']([^"\']+)["\']',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                logo = _resolve_url(m.group(1), base_url)
                break

        # If no meta logo, look for <img> with logo-like attributes
        if not logo:
            for m in re.finditer(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*/?\s*>', html, re.IGNORECASE):
                tag = m.group(0)
                src = m.group(1)
                if _LOGO_PATTERNS.search(tag):
                    logo = _resolve_url(src, base_url)
                    break

        result["logo"] = logo

        # ── Extract photos ──
        # Collect candidates from <img> tags, then rank to prioritize real photos
        seen = set()
        candidates = []  # (score, url) — higher score = more likely a real photo

        def _score_and_add(src: str, tag: str):
            """Score an image URL and add to candidates if it passes filters."""
            if not src or src.startswith('data:'):
                return
            # Only check junk patterns on the URL, not the full tag
            # (tag contains attrs like loading="lazy" which false-match "loading")
            if _JUNK_PATTERNS.search(src):
                return
            if not _IMG_EXTENSIONS.search(src):
                return
            if logo and src in logo:
                return
            width_m = re.search(r'width=["\']?(\d+)', tag)
            height_m = re.search(r'height=["\']?(\d+)', tag)
            if width_m and int(width_m.group(1)) < 100:
                return
            if height_m and int(height_m.group(1)) < 100:
                return

            full_url = _resolve_url(src, base_url)
            if full_url in seen:
                return
            seen.add(full_url)

            # Score: prefer JPEGs (real photos) over PNGs (often icons/graphics)
            score = 0
            src_lower = src.lower()
            if '.jpg' in src_lower or '.jpeg' in src_lower:
                score += 10  # JPEGs are almost always real photos
            if re.search(r'(\d{3,4})x(\d{3,4})', src):
                score += 5  # URL contains large dimensions (e.g. -1024x768)
            if 'scaled' in src_lower:
                score += 5  # WordPress scaled images are typically gallery photos
            if width_m and int(width_m.group(1)) >= 400:
                score += 5  # Large explicit width
            if re.search(r'gallery|portfolio|project|work|photo|slider|slide', src_lower):
                score += 5  # Gallery-like path
            if re.search(r'gallery|portfolio|project|work', tag.lower()):
                score += 3  # Gallery-like alt/class
            candidates.append((score, full_url))

        for m in re.finditer(r'<img\s+[^>]*?>', html, re.IGNORECASE):
            tag = m.group(0)
            # Try src first, then lazy-load attributes (WordPress, WP Rocket, etc.)
            for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                attr_m = re.search(rf'{attr}=["\']([^"\']+)["\']', tag, re.IGNORECASE)
                if attr_m:
                    _score_and_add(attr_m.group(1), tag)

            # Also check srcset for high-res versions
            srcset_m = re.search(r'srcset=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if srcset_m:
                # srcset format: "url1 300w, url2 600w, url3 1024w"
                # Pick the largest one
                parts = srcset_m.group(1).split(',')
                best_url, best_w = "", 0
                for part in parts:
                    tokens = part.strip().split()
                    if len(tokens) >= 2 and tokens[1].rstrip('w').isdigit():
                        w = int(tokens[1].rstrip('w'))
                        if w > best_w:
                            best_w = w
                            best_url = tokens[0]
                    elif len(tokens) == 1:
                        best_url = tokens[0]
                if best_url and best_w >= 400:
                    _score_and_add(best_url, tag)

            if len(candidates) >= 30:
                break

        # Also extract gallery images from non-<img> elements:
        # Elementor galleries use <a href="...jpg"> and <div data-thumbnail="...jpg">
        for pattern in [
            r'<a\s+[^>]*href=["\']([^"\']+\.(?:jpe?g|png|webp))["\']',
            r'data-thumbnail=["\']([^"\']+\.(?:jpe?g|png|webp))["\']',
        ]:
            for m in re.finditer(pattern, html, re.IGNORECASE):
                url = m.group(1)
                _score_and_add(url, m.group(0))

        # Extract CSS background images (Divi, Elementor, many WordPress themes)
        for m in re.finditer(r'background-image:\s*url\(["\']?([^)"\']+\.(?:jpe?g|png|webp))["\']?\)', html, re.IGNORECASE):
            bg_url = m.group(1)
            if not _JUNK_PATTERNS.search(bg_url):
                full_url = _resolve_url(bg_url, base_url)
                if full_url not in seen and (not logo or full_url != logo):
                    seen.add(full_url)
                    candidates.append((8, full_url))  # Score 8 — likely hero/section photos

        # Sort by score (highest first), take top 8 for AI filter
        candidates.sort(key=lambda x: -x[0])
        photos = [url for _, url in candidates[:8]]

        result["photos"] = photos
        logger.info(f"[SCRAPE] {url}: logo={'yes' if logo else 'no'}, {len(photos)} photos")

    except httpx.TimeoutException as e:
        logger.error(f"[SCRAPE] Timeout fetching {url}: {e}")
    except httpx.HTTPError as e:
        logger.error(f"[SCRAPE] HTTP error fetching {url}: {e}")
    except Exception as e:
        logger.error(f"[SCRAPE] Unexpected error fetching {url} ({type(e).__name__}): {e}")

    if len(_scrape_cache) >= _SCRAPE_CACHE_MAX:
        _scrape_cache.pop(next(iter(_scrape_cache)))
    _scrape_cache[domain] = result
    return result


# ────────── SOCIAL MEDIA IMAGE SCRAPER ──────────

async def scrape_social_images(urls: list[str]) -> dict:
    """Fetch og:image from Facebook/Instagram pages.

    Returns {"logo": "url", "photos": ["url1", ...]}.
    Facebook profile/cover photos → logo candidate.
    Instagram profile pic → logo candidate, post images → photos.
    """
    result = {"logo": "", "photos": []}

    import asyncio
    if not urls:
        return result

    async def _fetch_og_image(url: str) -> tuple[str, str]:
        """Fetch a URL using shared client, return (og_image_url, source_type)."""
        try:
            resp = await _http_client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return "", ""

            html = resp.text
            # Extract og:image
            for pattern in [
                r'<meta\s+(?:property|name)=["\']og:image["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+content=["\']([^"\']+)["\']\s+(?:property|name)=["\']og:image["\']',
            ]:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    img_url = m.group(1)
                    if img_url and not img_url.endswith('.svg'):
                        source = "facebook" if "facebook.com" in url else "instagram"
                        return img_url, source
        except Exception as e:
            logger.error(f"[SOCIAL] Error fetching {url}: {e}")
        return "", ""

    tasks = [_fetch_og_image(u) for u in urls[:4]]  # Cap at 4 fetches
    fetched = await asyncio.gather(*tasks)

    for img_url, source in fetched:
        if not img_url:
            continue

        # Facebook profile/cover → use as logo if we don't have one
        if source == "facebook" and not result["logo"]:
            result["logo"] = img_url
            logger.info(f"[SOCIAL] Facebook logo: {img_url[:80]}")
        # Instagram profile → logo fallback; post images → photos
        elif source == "instagram":
            if not result["logo"]:
                result["logo"] = img_url
                logger.info(f"[SOCIAL] Instagram logo: {img_url[:80]}")
            else:
                result["photos"].append(img_url)
                logger.info(f"[SOCIAL] Instagram photo: {img_url[:80]}")

    return result


# ────────── AI IMAGE FILTER ──────────

async def ai_filter_photos(photo_urls: list[str], business_type: str = "tradesperson") -> list[str]:
    """Use Haiku vision to filter photos, keeping only real work/gallery images.

    Downloads each image, sends batch to Haiku for classification.
    Returns filtered list of URLs that look like genuine work photos.
    """
    import asyncio
    import base64
    from langchain_core.messages import HumanMessage

    if not photo_urls:
        return []

    # Download images in parallel using shared client
    async def _download(url: str) -> tuple[str, str, str, int]:
        """Download image, return (url, base64_data, media_type, size_bytes) or empty on failure."""
        try:
            resp = await _http_client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return url, "", "", 0
            content_type = resp.headers.get("content-type", "")
            if "jpeg" in content_type or "jpg" in content_type:
                media_type = "image/jpeg"
            elif "png" in content_type:
                media_type = "image/png"
            elif "webp" in content_type:
                media_type = "image/webp"
            elif "gif" in content_type:
                media_type = "image/gif"
            else:
                # Guess from URL
                if ".png" in url.lower():
                    media_type = "image/png"
                elif ".webp" in url.lower():
                    media_type = "image/webp"
                else:
                    media_type = "image/jpeg"
            size_bytes = len(resp.content)
            size_kb = size_bytes / 1024
            # Skip if too small (<5KB likely a logo/icon) or too large (>2MB)
            if size_bytes < 5000 or size_bytes > 2_000_000:
                logger.info(f"[AI-FILTER] Skip {url[:60]} — {size_kb:.0f}KB (too {'small' if size_kb < 5 else 'large'})")
                return url, "", "", 0
            logger.info(f"[AI-FILTER] Downloaded {url[:60]} — {size_kb:.0f}KB {media_type}")
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return url, b64, media_type, size_bytes
        except Exception as e:
            logger.error(f"[AI-FILTER] Download error {url[:60]}: {e}")
            return url, "", "", 0

    downloads = await asyncio.gather(*[_download(u) for u in photo_urls[:8]])
    valid = [(url, b64, mt, sz) for url, b64, mt, sz in downloads if b64]

    if not valid:
        logger.info("[AI-FILTER] No images downloaded successfully")
        return []

    # Build multimodal message with all images
    content_parts = [{
        "type": "text",
        "text": f"""You are reviewing images scraped from a {business_type}'s website.
For each image, classify it as one of:
- WORK: A photo showing completed work, a job in progress, before/after, or a work portfolio image
- SKIP: A logo, icon, banner, stock photo, headshot, decoration, UI element, map, or anything NOT a work photo

Respond with one line per image in format: IMAGE_N: WORK or IMAGE_N: SKIP
where N is the image number (1-indexed)."""
    }]

    for i, (url, b64, mt, sz) in enumerate(valid):
        content_parts.append({
            "type": "text",
            "text": f"IMAGE_{i+1}:"
        })
        content_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mt,
                "data": b64,
            }
        })

    try:
        response = await _llm_vision.ainvoke([HumanMessage(content=content_parts)])
        response_text = response.content.strip()
        logger.info(f"[AI-FILTER] Response: {response_text}")

        # Parse response — keep only WORK images
        kept = []
        for i, (url, b64, mt, sz) in enumerate(valid):
            marker = f"IMAGE_{i+1}"
            if f"{marker}: WORK" in response_text or f"{marker}:WORK" in response_text:
                kept.append(url)

        logger.info(f"[AI-FILTER] Kept {len(kept)}/{len(valid)} images as work photos")

        # Trust the AI verdict — if it says all SKIP, return empty (no work photos)
        return kept

    except Exception as e:
        logger.error(f"[AI-FILTER] LLM error: {e}")
        # On LLM failure only, fall back to large JPEGs as best guess
        jpeg_fallback = [url for url, b64, mt, sz in valid
                        if mt == "image/jpeg" and sz >= 20_000]
        if jpeg_fallback:
            logger.info(f"[AI-FILTER] LLM failed — falling back to {len(jpeg_fallback)} large JPEG(s)")
            return jpeg_fallback[:4]
        return []


# ────────── SEARCH RESULT EXTRACTORS ──────────

def extract_facebook_url(results: list[dict]) -> str:
    """Pick the best Facebook business page URL from Brave search results.

    Skips marketplace, events, groups, profile.php, and other non-page URLs.
    Returns the first clean facebook.com page URL, or empty string.
    """
    _skip = ("/marketplace", "/events", "/groups", "/profile.php", "/watch", "/reel", "/stories", "/login")
    for r in results:
        url = r.get("url", "")
        if "facebook.com" not in url:
            continue
        if any(seg in url for seg in _skip):
            continue
        return url
    return ""


def _resolve_url(src: str, base_url: str) -> str:
    """Resolve a potentially relative URL against a base URL."""
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http"):
        return src
    if src.startswith("/"):
        # Absolute path — extract origin from base
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        return f"{parsed.scheme}://{parsed.netloc}{src}"
    # Relative path
    if "/" in base_url:
        return base_url.rsplit("/", 1)[0] + "/" + src
    return src

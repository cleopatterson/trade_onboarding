"""Tools for the Trade Onboarding wizard"""
from __future__ import annotations

import csv
import json
import os
import math
import re
from pathlib import Path
from typing import Optional
import httpx

from agent.config import (
    ABR_GUID, NSW_TRADES_API_KEY, NSW_TRADES_AUTH_HEADER, BRAVE_SEARCH_API_KEY,
    GOOGLE_PLACES_API_KEY,
)

RESOURCES_DIR = Path(__file__).parent.parent / "resources"

# Persistent HTTP client — reuses connections across API calls (saves TLS handshake time)
_http_client = httpx.AsyncClient(timeout=15.0)


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
                "maxResults": "5",
                "callback": "c",
                "guid": ABR_GUID,
            }

        resp = await _http_client.get(url, params=params)
        if resp.status_code != 200:
            return {"results": [], "count": 0, "error": f"ABR API returned {resp.status_code}"}

        return _parse_jsonp_response(resp.text, search_type)

    except Exception as e:
        print(f"ABR lookup error: {e}")
        return _mock_abr_lookup(search_term, search_type)


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
            name = data.get("EntityName", "") or data.get("BusinessName", [""])[0] if data.get("BusinessName") else ""
            if not name:
                name = data.get("EntityName", "Unknown")
            entity_type = data.get("EntityTypeName", "Unknown")
            gst = data.get("Gst", "")
            state = data.get("AddressState", "")
            postcode = data.get("AddressPostcode", "")

            if not abn:
                return {"results": [], "count": 0, "error": data.get("Message", "No result")}

            return {
                "results": [{
                    "abn": abn,
                    "entity_name": name,
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
            # Deduplicate by ABN, preferring Business/Trading Name over Entity Name.
            names = data.get("Names", [])
            by_abn: dict[str, dict] = {}
            for entry in names:
                abn = entry.get("Abn", "")
                if not abn:
                    continue
                name = entry.get("Name", "Unknown")
                name_type = entry.get("NameType", "")
                state = entry.get("State", "")
                postcode = entry.get("Postcode", "")
                score = entry.get("Score", 0)

                record = {
                    "abn": abn,
                    "entity_name": name,
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
                    by_abn[abn] = record

            results = list(by_abn.values())[:5]
            return {"results": results, "count": len(results)}

    except (json.JSONDecodeError, KeyError) as e:
        return {"results": [], "count": 0, "error": f"Parse error: {e}"}


def _mock_abr_lookup(search_term: str, search_type: str) -> dict:
    """Mock ABR response for development."""
    term = search_term.lower()

    # Simulate realistic results
    if search_type == "abn":
        clean_abn = search_term.replace(" ", "")
        return {
            "results": [{
                "abn": clean_abn,
                "entity_name": f"Business with ABN {clean_abn}",
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
    return {
        "results": [{
            "abn": "51 824 753 556",
            "entity_name": name_title if "pty" in term or "ltd" in term else f"{name_title} Pty Ltd",
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


def compute_service_gaps(services: list[dict], business_name: str,
                         licence_classes: list[str] | None = None,
                         google_business_name: str = "",
                         google_primary_type: str = "") -> list[dict]:
    """Compute which subcategories are NOT yet mapped for this trade.

    Loads the full subcategory list for the matched category, diffs against
    subcategory_ids already in the services list.
    Returns list of {"subcategory_id": ..., "subcategory_name": ...} dicts.
    """
    categories = _load_categories()
    if not categories:
        return []

    # Priority 1: if services are already mapped, detect category from them
    # (avoids mismatch for multi-trade businesses, e.g. licence says Electrician
    #  but mapped services are all Plumber)
    matched_cat_key = None
    if services:
        cat_counts: dict[str, int] = {}
        for s in services:
            cn = s.get("category_name", "")
            if cn:
                cat_counts[cn] = cat_counts.get(cn, 0) + 1
        if cat_counts:
            most_common = max(cat_counts, key=cat_counts.get)
            if most_common in categories:
                matched_cat_key = most_common

    # Priority 2: match business name to category
    if not matched_cat_key:
        name_lower = business_name.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in name_lower:
                matched_cat_key = cat_key
                break

    # Priority 3: match licence classes (for sole traders with personal names)
    if not matched_cat_key and licence_classes:
        for lc in licence_classes:
            lc_lower = lc.lower()
            for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
                if keyword in lc_lower:
                    matched_cat_key = cat_key
                    break
            if matched_cat_key:
                break

    # Priority 4: match Google Places business name (e.g. "Stacey Electrical"
    # when ABR name is "STACEY, MATTHEW GREGORY")
    if not matched_cat_key and google_business_name:
        gname_lower = google_business_name.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in gname_lower:
                matched_cat_key = cat_key
                break

    # Priority 5: match Google Places primary type (e.g. "electrician", "plumber")
    if not matched_cat_key and google_primary_type:
        gtype_lower = google_primary_type.lower()
        for keyword, cat_key in _TRADE_CATEGORY_MAP.items():
            if keyword in gtype_lower:
                matched_cat_key = cat_key
                break

    if not matched_cat_key or matched_cat_key not in categories:
        return []

    cat_data = categories[matched_cat_key]
    all_subcats = cat_data.get("subcategories", [])

    # Get IDs already mapped (coerce to int for safe comparison)
    mapped_ids = set()
    for s in services:
        sid = s.get("subcategory_id")
        if sid is not None:
            try:
                mapped_ids.add(int(sid))
            except (ValueError, TypeError):
                pass

    cat_id = cat_data.get("category_id", 0)

    # Return unmatched
    gaps = []
    for sc in all_subcats:
        sc_id = sc.get("subcategory_id")
        if sc_id and int(sc_id) not in mapped_ids:
            gaps.append({
                "subcategory_id": sc_id,
                "subcategory_name": sc.get("subcategory_name", ""),
                "category_id": cat_id,
                "category_name": cat_data.get("category_name", matched_cat_key),
            })

    return gaps


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


def search_suburbs_by_name(name: str) -> list[dict]:
    """Find suburbs matching a name (case insensitive)."""
    suburbs = _load_suburbs()
    name_lower = name.lower()
    return [s for s in suburbs if name_lower in s.get("name", "").lower()]


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


def get_region_suburbs(region_name: str) -> list[str]:
    """Look up suburbs for a named region from region guide files."""
    for fname in os.listdir(RESOURCES_DIR):
        if fname.endswith("_regions.md"):
            fpath = RESOURCES_DIR / fname
            content = fpath.read_text()
            if region_name.lower() in content.lower():
                return content
    return ""


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


def get_regional_guide(state_code: str) -> str:
    """Get the regional guide for a state (sydney, melbourne, etc)."""
    state_map = {"NSW": "sydney", "VIC": "melbourne", "QLD": "brisbane", "WA": "perth"}
    city = state_map.get(state_code.upper(), "")
    if not city:
        return ""
    guide_path = RESOURCES_DIR / f"{city}_regions.md"
    if guide_path.exists():
        return guide_path.read_text()
    return ""


def find_subcategory_guide(business_name: str) -> str:
    """Find the relevant subcategory guide based on business name trade type."""
    name_lower = business_name.lower()

    # Map trade keywords to guide files
    trade_guides = {
        "plumb": ["plumber-subcategory-guide.md", "plumbing_subcategories.md"],
        "paint": ["painter_subcategories.md"],
        "electri": ["electrician-subcategory-guide.md", "electrical_subcategories.md"],
        "clean": ["cleaner-subcategory-guide.md"],
        "garden": ["gardener-subcategory-guide.md"],
        "carpent": ["carpentry_subcategories.md"],
        "build": ["carpentry_subcategories.md"],
    }

    for keyword, files in trade_guides.items():
        if keyword in name_lower:
            for fname in files:
                fpath = RESOURCES_DIR / fname
                if fpath.exists():
                    return fpath.read_text()

    return ""


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
                    print(f"[NSW TRADES] Got OAuth token, expires in {expires_in}s")
                    return token
                print(f"[NSW TRADES] Token response missing access_token: {resp.text[:200]}")
                return ""
            except (json.JSONDecodeError, ValueError):
                print(f"[NSW TRADES] Token response not JSON: {resp.text[:200]}")
                return ""
        else:
            print(f"[NSW TRADES] Token request failed: {resp.status_code}")
            return ""
    except Exception as e:
        print(f"[NSW TRADES] Token error: {e}")
        return ""


async def nsw_licence_browse(search_term: str) -> dict:
    """Browse the NSW Fair Trading Trades Register by name.

    Uses GET /tradesregister/v1/browse?searchText=...
    Returns list of matching licences with: licenceID, licensee, licenceNumber,
    licenceType, status, suburb, postcode, expiryDate, categories, classes.
    """
    if not NSW_TRADES_API_KEY:
        print("[NSW TRADES] No API key configured, skipping licence lookup")
        return {"results": [], "error": "NSW Trades API not configured"}

    token = await _get_nsw_trades_token()
    if not token:
        print("[NSW TRADES] Could not get OAuth token, skipping licence lookup")
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
            print(f"[NSW TRADES] Browse failed: {resp.status_code} - {resp.text[:200]}")
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

        print(f"[NSW TRADES] Found {len(results)} licence results for '{search_term}'")
        return {"results": results, "count": len(results)}

    except Exception as e:
        print(f"[NSW TRADES] Lookup error: {e}")
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
            print(f"[NSW TRADES] Details failed: {resp.status_code}")
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

    except Exception as e:
        print(f"[NSW TRADES] Details error: {e}")
        return {"error": str(e)}


# ────────── BRAVE WEB SEARCH ──────────

async def brave_web_search(query: str, count: int = 5) -> list[dict]:
    """Search the web using Brave Search API.

    Returns top results with title, url, and description.
    Retries once on 429 (rate limit) after a short delay.
    """
    import asyncio as _aio

    if not BRAVE_SEARCH_API_KEY:
        print("[BRAVE] No API key configured, skipping web search")
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
                print(f"[BRAVE] Rate limited, retrying in 1s...")
                await _aio.sleep(1.0)
                continue

            if resp.status_code != 200:
                print(f"[BRAVE] Search failed: {resp.status_code}")
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

            print(f"[BRAVE] Found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            print(f"[BRAVE] Search error: {e}")
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
        print("[GOOGLE] No API key configured, skipping Places search")
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
                "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount,places.websiteUri,places.googleMapsUri,places.formattedAddress,places.reviews,places.primaryType,places.types",
            },
            json={"textQuery": query},
        )

        if resp.status_code != 200:
            print(f"[GOOGLE] Places search failed: {resp.status_code} - {resp.text[:200]}")
            return {}

        data = resp.json()
        places = data.get("places", [])
        if not places:
            print(f"[GOOGLE] No places found for '{query}'")
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
        }

        print(f"[GOOGLE] Found: {name} — {rating}★ ({review_count} reviews), website={bool(website)}, type={primary_type}")
        return result

    except Exception as e:
        print(f"[GOOGLE] Places search error: {e}")
        return {}


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

    # Use a fresh client for discovery — shared client can have stale connections
    import asyncio
    async with httpx.AsyncClient(timeout=8.0) as client:
        async def _check(url: str) -> str:
            try:
                resp = await client.head(url, follow_redirects=True)
                if resp.status_code < 400:
                    content_type = resp.headers.get("content-type", "")
                    if "text/html" in content_type or "application" in content_type:
                        print(f"[DISCOVER] Found: {url} → {resp.status_code}")
                        return str(resp.url)  # Return final URL after redirects
            except Exception as e:
                print(f"[DISCOVER] Failed: {url} → {type(e).__name__}")
            return ""

        results = await asyncio.gather(*[_check(u) for u in candidates])

    # Prefer .com.au over .au (more likely to be the real content site)
    for r in results:
        if r:
            return r

    print(f"[DISCOVER] No website found for '{business_name}' (tried {slug}.*)")
    return ""


# ────────── WEBSITE IMAGE SCRAPER ──────────

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
    """
    result = {"logo": "", "photos": []}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ServiceSeeking/1.0)"},
                follow_redirects=True,
            )
        if resp.status_code != 200:
            print(f"[SCRAPE] {url} returned {resp.status_code}")
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

        # Sort by score (highest first), take top 8 for AI filter
        candidates.sort(key=lambda x: -x[0])
        photos = [url for _, url in candidates[:8]]

        result["photos"] = photos
        print(f"[SCRAPE] {url}: logo={'yes' if logo else 'no'}, {len(photos)} photos")

    except Exception as e:
        print(f"[SCRAPE] Error fetching {url}: {e}")

    return result


# ────────── SOCIAL MEDIA IMAGE SCRAPER ──────────

async def scrape_social_images(urls: list[str]) -> dict:
    """Fetch og:image from Facebook/Instagram pages.

    Returns {"logo": "url", "photos": ["url1", ...]}.
    Facebook profile/cover photos → logo candidate.
    Instagram profile pic → logo candidate, post images → photos.
    """
    result = {"logo": "", "photos": []}

    async def _fetch_og_image(url: str) -> tuple[str, str]:
        """Fetch a URL, return (og_image_url, source_type)."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
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
            print(f"[SOCIAL] Error fetching {url}: {e}")
        return "", ""

    import asyncio
    if not urls:
        return result

    tasks = [_fetch_og_image(u) for u in urls[:4]]  # Cap at 4 fetches
    fetched = await asyncio.gather(*tasks)

    for img_url, source in fetched:
        if not img_url:
            continue

        # Facebook profile/cover → use as logo if we don't have one
        if source == "facebook" and not result["logo"]:
            result["logo"] = img_url
            print(f"[SOCIAL] Facebook logo: {img_url[:80]}")
        # Instagram profile → logo fallback; post images → photos
        elif source == "instagram":
            if not result["logo"]:
                result["logo"] = img_url
                print(f"[SOCIAL] Instagram logo: {img_url[:80]}")
            else:
                result["photos"].append(img_url)
                print(f"[SOCIAL] Instagram photo: {img_url[:80]}")

    return result


# ────────── AI IMAGE FILTER ──────────

async def ai_filter_photos(photo_urls: list[str], business_type: str = "tradesperson") -> list[str]:
    """Use Haiku vision to filter photos, keeping only real work/gallery images.

    Downloads each image, sends batch to Haiku for classification.
    Returns filtered list of URLs that look like genuine work photos.
    """
    import asyncio
    import base64
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from agent.config import ANTHROPIC_API_KEY, MODEL_FAST

    if not photo_urls:
        return []

    # Download images in parallel
    async def _download(url: str) -> tuple[str, str, str, int]:
        """Download image, return (url, base64_data, media_type, size_bytes) or empty on failure."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
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
                print(f"[AI-FILTER] Skip {url[:60]} — {size_kb:.0f}KB (too {'small' if size_kb < 5 else 'large'})")
                return url, "", "", 0
            print(f"[AI-FILTER] Downloaded {url[:60]} — {size_kb:.0f}KB {media_type}")
            b64 = base64.b64encode(resp.content).decode("utf-8")
            return url, b64, media_type, size_bytes
        except Exception as e:
            print(f"[AI-FILTER] Download error {url[:60]}: {e}")
            return url, "", "", 0

    downloads = await asyncio.gather(*[_download(u) for u in photo_urls[:8]])
    valid = [(url, b64, mt, sz) for url, b64, mt, sz in downloads if b64]

    if not valid:
        print("[AI-FILTER] No images downloaded successfully")
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
        llm = ChatAnthropic(
            model=MODEL_FAST,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=256,
            temperature=0,
        )
        response = await llm.ainvoke([HumanMessage(content=content_parts)])
        response_text = response.content.strip()
        print(f"[AI-FILTER] Response: {response_text}")

        # Parse response — keep only WORK images
        kept = []
        for i, (url, b64, mt, sz) in enumerate(valid):
            marker = f"IMAGE_{i+1}"
            if f"{marker}: WORK" in response_text or f"{marker}:WORK" in response_text:
                kept.append(url)

        print(f"[AI-FILTER] Kept {len(kept)}/{len(valid)} images as work photos")

        # Fallback: if AI skipped everything, keep large JPEGs only (>=20KB)
        # Small JPEGs (<20KB) are almost always logos/icons, not work photos
        if not kept:
            jpeg_fallback = [url for url, b64, mt, sz in valid
                            if mt == "image/jpeg" and sz >= 20_000]
            if jpeg_fallback:
                print(f"[AI-FILTER] All SKIP — falling back to {len(jpeg_fallback)} large JPEG(s)")
                return jpeg_fallback[:4]

        return kept

    except Exception as e:
        print(f"[AI-FILTER] LLM error: {e}")
        # On failure, return all images unfiltered
        return [url for url, b64, mt, sz in valid]


# ────────── SEARCH RESULT EXTRACTORS ──────────

def extract_google_rating(results: list[dict]) -> tuple[float, int]:
    """Extract Google Business rating + review count from Brave search results.

    Looks for patterns like "4.8 · 47 reviews", "Rated 4.8/5 based on 47 reviews",
    "4.8 stars (47)", "Rating: 4.8 (47 reviews)" in result descriptions.
    Returns (rating, review_count) or (0.0, 0) if not found.
    """
    patterns = [
        # "4.8 · 47 reviews" or "4.8 · 47 Google reviews"
        r'(\d\.\d)\s*[·•]\s*(\d+)\s*(?:Google\s+)?reviews?',
        # "Rated 4.8/5 based on 47 reviews"
        r'[Rr]ated\s+(\d\.\d)/5\s+(?:based on\s+)?(\d+)\s*reviews?',
        # "4.8 stars (47 reviews)" or "4.8 stars · 47 reviews"
        r'(\d\.\d)\s*stars?\s*[·•(]\s*(\d+)\s*reviews?\)?',
        # "Rating: 4.8 (47 reviews)" or "Rating: 4.8 (47)"
        r'[Rr]ating:?\s*(\d\.\d)\s*\((\d+)\s*(?:reviews?)?\)',
        # "4.8(47)" — compact format
        r'(\d\.\d)\((\d+)\)',
    ]
    for r in results:
        desc = r.get("description", "") + " " + r.get("title", "")
        for pat in patterns:
            m = re.search(pat, desc)
            if m:
                rating = float(m.group(1))
                count = int(m.group(2))
                if 1.0 <= rating <= 5.0 and count > 0:
                    return rating, count
    return 0.0, 0


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

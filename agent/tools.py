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
                }],
                "count": 1,
            }
        else:
            # Name search returns a Names array
            names = data.get("Names", [])
            results = []
            for entry in names[:5]:
                abn = entry.get("Abn", "")
                name = entry.get("Name", "Unknown")
                name_type = entry.get("NameType", "")
                state = entry.get("State", "")
                postcode = entry.get("Postcode", "")
                score = entry.get("Score", 0)

                results.append({
                    "abn": abn,
                    "entity_name": name,
                    "entity_type": name_type,
                    "gst_registered": False,  # Not available in name search
                    "state": state,
                    "postcode": postcode,
                    "status": "Active",
                    "score": score,
                })

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
    Useful for finding a business's website, Facebook page, reviews, etc.
    """
    if not BRAVE_SEARCH_API_KEY:
        print("[BRAVE] No API key configured, skipping web search")
        return []

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

        if resp.status_code != 200:
            print(f"[BRAVE] Search failed: {resp.status_code}")
            return []

        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", [])[:count]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            })

        print(f"[BRAVE] Found {len(results)} results for '{query}'")
        return results

    except Exception as e:
        print(f"[BRAVE] Search error: {e}")
        return []

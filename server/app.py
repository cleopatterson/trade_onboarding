"""FastAPI server for the Trade Onboarding wizard"""
from __future__ import annotations
from typing import Optional

import asyncio
import uuid
import time
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import os
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import (
    welcome_node, business_verification_node, service_discovery_node,
    service_area_node, profile_node, pricing_node,
    complete_node, assessment_node, _enrich_business,
)
from agent.config import PORT, ALLOWED_ORIGINS, validate_env
from agent.tools import _get_nsw_trades_token, qbcc_load_csv, ss_get_business

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ────────── SESSION STORE ──────────

sessions: dict = {}

SESSION_TTL_SECONDS = 30 * 60   # 30 minutes inactive
MAX_SESSIONS = 500


# ────────── RATE LIMITING ──────────

_rate_log: dict[str, list[float]] = {}   # key → [timestamp, ...]
RATE_LIMIT = 15          # max chat requests per session
RATE_WINDOW = 60.0       # per 60 seconds

SESSION_CREATE_LIMIT = 5     # max new sessions per IP
SESSION_CREATE_WINDOW = 60.0  # per 60 seconds


def _check_rate_limit(key: str, limit: int = RATE_LIMIT, window: float = RATE_WINDOW) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    window_start = now - window
    timestamps = _rate_log.get(key, [])
    # Trim old entries
    timestamps = [t for t in timestamps if t > window_start]
    if len(timestamps) >= limit:
        _rate_log[key] = timestamps
        return False
    timestamps.append(now)
    _rate_log[key] = timestamps
    return True


# ────────── SESSION LOGGING ──────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _log_turn(session_id: str, turn: dict):
    """Append a turn entry to the session's JSONL log file.

    Redacts PII: abn → last 3 digits, contact_name/contact_phone omitted.
    """
    log_file = LOG_DIR / f"{session_id}.jsonl"
    turn["timestamp"] = datetime.now(timezone.utc).isoformat()
    # Redact PII
    turn.pop("contact_name", None)
    turn.pop("contact_phone", None)
    if "abn" in turn:
        abn = str(turn["abn"])
        turn["abn"] = f"***{abn[-3:]}" if len(abn) >= 3 else "***"
    with open(log_file, "a") as f:
        f.write(json.dumps(turn, default=str) + "\n")


# ────────── SESSION CLEANUP ──────────

async def _session_cleanup_loop():
    """Background task: expire stale sessions every 5 minutes."""
    while True:
        await asyncio.sleep(300)  # 5 min
        now = time.time()
        expired = [
            sid for sid, s in sessions.items()
            if now - s.get("_last_active", s.get("_created_at", now)) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del sessions[sid]
            _rate_log.pop(sid, None)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions, {len(sessions)} active")

        # Cap total sessions — evict oldest by _last_active
        if len(sessions) > MAX_SESSIONS:
            by_age = sorted(sessions.items(), key=lambda kv: kv[1].get("_last_active", 0))
            to_evict = len(sessions) - MAX_SESSIONS
            for sid, _ in by_age[:to_evict]:
                del sessions[sid]
                _rate_log.pop(sid, None)
            logger.info(f"Evicted {to_evict} oldest sessions (cap={MAX_SESSIONS})")


# ────────── LIFESPAN ──────────

@asynccontextmanager
async def lifespan(app):
    """Startup/shutdown lifecycle: validate env, pre-warm tokens, start cleanup."""
    validate_env()
    await asyncio.to_thread(qbcc_load_csv)

    token = await _get_nsw_trades_token()
    if token:
        logger.info("NSW Trades OAuth token pre-warmed")
    else:
        logger.warning("NSW Trades OAuth token failed — licence lookups will retry on first request")

    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    logger.info(f"Server started — CORS origins: {ALLOWED_ORIGINS}")
    yield
    cleanup_task.cancel()


# ────────── APP ──────────

app = FastAPI(title="Trade Onboarding Wizard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
web_dir = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# ────────── MODELS ──────────

class StartRequest(BaseModel):
    ss_business_id: Optional[str] = None

class MessageRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)


# ────────── NODE DISPATCH ──────────

NODE_FUNCTIONS = {
    "welcome": welcome_node,
    "business_verification": business_verification_node,
    "service_discovery": service_discovery_node,
    "service_area": service_area_node,
    "profile": profile_node,
    "pricing": pricing_node,
    "complete": complete_node,
    "assessment": assessment_node,
}


def determine_node(state: dict) -> str:
    """Determine which node to run based on state."""
    # Improve mode: assessment node first
    if state.get("_flow_mode") == "improve" and not state.get("_assessment_shown"):
        return "assessment"
    # Assessment Turn 2+: user is still interacting with assessment
    # (verification sub-flow, or hasn't picked a fix yet)
    if (state.get("current_node") == "assessment"
            and state.get("_assessment_shown")
            and not state.get("_improve_fixes")):
        return "assessment"
    if not state.get("business_verified"):
        return "business_verification"
    if not state.get("services_confirmed"):
        return "service_discovery"
    if not state.get("service_areas_confirmed"):
        return "service_area"
    if not state.get("profile_saved"):
        return "profile"
    if not state.get("subscription_plan"):
        # Skip pricing for improve mode
        if state.get("_flow_mode") == "improve":
            state["subscription_plan"] = "existing"
            return "complete"
        return "pricing"
    return "complete"


async def _auto_chain_remaining(state: dict, _merge):
    """Run remaining auto-chain transitions: profile -> pricing -> complete."""
    if (state.get("service_areas_confirmed")
            and not state.get("profile_saved")
            and not state.get("profile_description_draft")):
        state["confirmed"] = True
        _merge(await NODE_FUNCTIONS["profile"](state))
    if state.get("profile_saved") and not state.get("subscription_plan"):
        if state.get("_flow_mode") == "improve":
            state["subscription_plan"] = "existing"
        else:
            _merge(await NODE_FUNCTIONS["pricing"](state))
    if state.get("subscription_plan") and not state.get("output_json"):
        _merge(await NODE_FUNCTIONS["complete"](state))


async def run_node(state: dict) -> dict:
    """Run the appropriate node and merge results into state."""
    node_name = determine_node(state)
    node_fn = NODE_FUNCTIONS.get(node_name)

    if not node_fn:
        return state

    def _merge(result):
        """Merge node result into state, clearing stale buttons."""
        state.pop("buttons", None)
        for key, value in result.items():
            if key == "messages":
                state["messages"] = state.get("messages", []) + value
            else:
                state[key] = value

    def _raw_merge(result):
        """Merge without clearing buttons (for initial node result)."""
        for key, value in result.items():
            if key == "messages":
                state["messages"] = state.get("messages", []) + value
            else:
                state[key] = value

    # ── Parallel path: service_discovery turn 2+ alongside speculative service_area ──
    # Fire both LLM calls simultaneously — but only use the area result if services
    # were ALREADY confirmed going in (i.e. this is an area-focused turn).
    # If services get confirmed THIS turn, discard the speculative area result
    # entirely and fall through to the normal sequential auto-chain below, which
    # runs area with the AI response already in state (so area sees the right context).
    parallel_handled = False
    already_confirmed = state.get("services_confirmed", False)
    if (node_name == "service_discovery"
            and state.get("services")  # turn 2 — services already mapped
            and not state.get("service_areas", {}).get("base_suburb")  # base_suburb guard ([] is falsy, base_suburb isn't)
            and not state.get("service_areas_confirmed")):
        svc_result, area_result = await asyncio.gather(
            node_fn(state),
            NODE_FUNCTIONS["service_area"](state),
        )
        _raw_merge(svc_result)
        parallel_handled = True
        if state.get("services_confirmed") and already_confirmed:
            # Services were already confirmed — this turn's message was for area
            _merge(area_result)
            await _auto_chain_remaining(state, _merge)
            return state
        # Services just confirmed THIS turn (or not confirmed yet) — discard area result,
        # fall through to sequential auto-chain which handles area with clean context

    # ── Normal sequential path (skip if parallel path already ran the node) ──
    if not parallel_handled:
        result = await node_fn(state)
        _raw_merge(result)

    # Auto-chain: assessment fix route → immediately run the target fix node
    # Do NOT call _auto_chain_remaining — let the user interact with the fix node
    if (state.get("_flow_mode") == "improve"
            and state.get("_assessment_shown")
            and state.get("_improve_fixes")
            and node_name == "assessment"):
        next_node = determine_node(state)
        logger.info(f"[IMPROVE] Assessment fix auto-chain → {next_node} "
                     f"(fixes={state.get('_improve_fixes')}, "
                     f"svc_confirmed={state.get('services_confirmed')}, "
                     f"area_confirmed={state.get('service_areas_confirmed')}, "
                     f"profile_saved={state.get('profile_saved')})")
        if next_node != "assessment" and next_node in NODE_FUNCTIONS:
            state["_auto_chained"] = True
            _merge(await NODE_FUNCTIONS[next_node](state))
            state.pop("_auto_chained", None)
            return state

    # In improve mode, don't cascade through auto-chains — each fix node
    # completes and stops. When a fix finishes (e.g. services confirmed),
    # chain once to the NEXT fix node, then stop.
    if state.get("_flow_mode") == "improve" and state.get("_improve_fixes"):
        next_node = determine_node(state)
        current = state.get("current_node", "")
        # Only auto-chain if we've moved to a new node (fix just completed)
        if next_node != current and next_node in NODE_FUNCTIONS and next_node != "complete":
            state["_auto_chained"] = True
            state["_improve_fix_index"] = state.get("_improve_fix_index", 0) + 1
            _merge(await NODE_FUNCTIONS[next_node](state))
            state.pop("_auto_chained", None)
        return state

    # Auto-chain: business verified → immediately run service discovery
    if state.get("business_verified") and not state.get("services") and not state.get("services_confirmed"):
        _merge(await NODE_FUNCTIONS["service_discovery"](state))

    # Auto-chain: services confirmed → immediately run service area
    # Use base_suburb as the guard (set on first area run) instead of regions_included (can be empty [])
    if state.get("services_confirmed") and not state.get("service_areas", {}).get("base_suburb") and not state.get("service_areas_confirmed"):
        state["_auto_chained"] = True  # Tell area node to ignore stale user message
        _merge(await NODE_FUNCTIONS["service_area"](state))
        state.pop("_auto_chained", None)

    await _auto_chain_remaining(state, _merge)

    return state


# ────────── ENDPOINTS ──────────

@app.get("/")
async def root():
    """Serve the landing page."""
    landing = web_dir / "landing.html"
    if landing.exists():
        return HTMLResponse(content=landing.read_text())
    return {"status": "Trade Onboarding Wizard API"}


@app.get("/health")
async def health():
    return {"status": "healthy", "sessions": len(sessions)}


# ────────── CSV SEARCH FOR IMPROVE ──────────

_csv_businesses: list[dict] = []


def _load_csv_businesses():
    """Load the business CSV into memory for search."""
    global _csv_businesses
    if _csv_businesses:
        return
    csv_path = Path(__file__).parent.parent / "all-contacts_with ID.csv"
    if not csv_path.exists():
        logger.warning(f"[CSV] Business CSV not found at {csv_path}")
        return
    import csv as csv_mod
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            biz_id = (row.get("Business ID") or "").strip()
            if not biz_id:
                continue
            _csv_businesses.append({
                "id": biz_id,
                "name": (row.get("Business Name") or "").strip(),
                "industry": (row.get("Industry") or "").strip(),
                "city": (row.get("City") or "").strip(),
                "reviews": row.get("Number of Reviews", "0"),
                "rating": row.get("Star Rating", "0"),
                "status": (row.get("Business Status") or "").strip(),
                "membership": (row.get("membership_status1") or "").strip(),
            })
    logger.info(f"[CSV] Loaded {len(_csv_businesses)} businesses from CSV")


@app.get("/api/search-businesses")
async def search_businesses(q: str = "", limit: int = 20):
    """Search the CSV business list by name or ID."""
    _load_csv_businesses()
    if not q or len(q) < 2:
        return {"results": [], "total": len(_csv_businesses)}

    q_lower = q.lower().strip()
    results = []
    # Exact ID match first
    if q.isdigit():
        for b in _csv_businesses:
            if b["id"] == q:
                results.append(b)
                break

    # Name search
    for b in _csv_businesses:
        if q_lower in b["name"].lower() or q_lower in b["industry"].lower():
            if b not in results:
                results.append(b)
            if len(results) >= limit:
                break

    return {"results": results, "total": len(_csv_businesses)}


# ────────── TEST HARNESS: Mock SS profile for improve mode testing ──────────

_MOCK_SS_PROFILES = {
    "demo-electrician": {
        "id": 99901,
        "businessName": "Spark Right Electrical",
        "businessDescription": "Electrician",
        "businessNumber": "51 824 753 556",
        "phoneNumber": "0412 345 678",
        "websiteUrl": "",
        "logoUrl": "",
        "hasALogo": False,
        "hasPortfolio": False,
        "badges": {"licenceVerified": False, "abnVerified": True, "identityVerified": False},
        "reviewsCount": 12,
        "reviewsScore": 4.5,
        "jobFilter": {
            "subCategories": [
                {"id": 848, "name": "Lighting", "categoryID": 30, "categoryName": "Electrician"},
                {"id": 849, "name": "Powerpoints", "categoryID": 30, "categoryName": "Electrician"},
                {"id": 851, "name": "Switchboards", "categoryID": 30, "categoryName": "Electrician"},
            ],
            "suburb": {"suburb": "Parramatta", "state": "NSW", "postcode": "2150", "lat": -33.8151, "lng": 151.0011},
            "radius": 15,
        },
    },
    "demo-plumber": {
        "id": 99902,
        "businessName": "Watertight Plumbing Services",
        "businessDescription": "",
        "businessNumber": "65 432 109 876",
        "phoneNumber": "",
        "websiteUrl": "",
        "logoUrl": "",
        "hasALogo": False,
        "hasPortfolio": False,
        "badges": {"licenceVerified": False, "abnVerified": True, "identityVerified": False},
        "reviewsCount": 3,
        "reviewsScore": 5.0,
        "jobFilter": {
            "subCategories": [
                {"id": 774, "name": "Toilet Installation & Repair", "categoryID": 74, "categoryName": "Plumber"},
                {"id": 775, "name": "Tap Installation & Repair", "categoryID": 74, "categoryName": "Plumber"},
            ],
            "suburb": {"suburb": "Bondi", "state": "NSW", "postcode": "2026", "lat": -33.8914, "lng": 151.2743},
            "radius": 10,
        },
    },
}


@app.get("/api/test-profiles")
async def test_profiles():
    """List available mock profiles for improve mode testing."""
    return {
        "profiles": {
            k: {"name": v["businessName"], "services": len(v["jobFilter"]["subCategories"]),
                 "suburb": v["jobFilter"]["suburb"]["suburb"]}
            for k, v in _MOCK_SS_PROFILES.items()
        },
        "usage": "Open http://localhost:8001?business_id=demo-electrician",
    }


def _init_base_state(session_id: str) -> dict:
    """Create a base state dict with all fields initialized."""
    now = time.time()
    return {
        "session_id": session_id,
        "current_node": "welcome",
        "messages": [],
        "business_name_input": "",
        "abn_input": "",
        "abr_results": [],
        "business_name": "",
        "legal_name": "",
        "abn": "",
        "entity_type": "",
        "gst_registered": False,
        "business_verified": False,
        "business_postcode": "",
        "business_state": "",
        "services_raw": "",
        "services": [],
        "services_confirmed": False,
        "location_raw": "",
        "service_areas": {},
        "service_areas_confirmed": False,
        "confirmed": False,
        "output_json": {},
        "abn_registration_date": "",
        "years_in_business": 0,
        "profile_description": "",
        "profile_description_draft": "",
        "profile_logo": "",
        "profile_photos": [],
        "profile_saved": False,
        "profile_intro": "",
        "google_rating": 0.0,
        "google_review_count": 0,
        "google_reviews": [],
        "google_business_name": "",
        "google_primary_type": "",
        "google_types": [],
        "business_website": "",
        "business_suburb": "",
        "google_address": "",
        "licence_info": {},
        "licence_classes": [],
        "contact_name": "",
        "contact_phone": "",
        "pricing_shown": False,
        "subscription_plan": "",
        "subscription_billing": "",
        "subscription_price": "",
        "_svc_turn": 1,
        "_specialist_gap_ids": [],
        "_pending_cluster_ids": [],
        "_selected_plan": "",
        "_needs_trading_name": False,
        "_needs_licence_number": False,
        "_licence_self_report": {},
        "_flow_mode": "",
        "_ss_profile": {},
        "_ss_business_id": "",
        "_assessment": {},
        "_assessment_shown": False,
        "_improve_fixes": [],
        "_improve_fix_total": 0,
        "_improve_fix_index": 0,
        "_verification_mode": False,
        "_description_comparison": False,
        "_description_improved": "",
        "_needs_logo": False,
        "_needs_photos": False,
        "_created_at": now,
        "_last_active": now,
    }


def _init_improve_state(ss_profile: dict, session_id: str) -> dict:
    """Map SS API response to wizard state for improve mode."""
    state = _init_base_state(session_id)

    state["_flow_mode"] = "improve"
    state["_ss_profile"] = ss_profile
    state["_ss_business_id"] = str(ss_profile.get("id", ""))

    # Business identity
    state["business_name"] = ss_profile.get("businessName", "")
    state["profile_description"] = ss_profile.get("businessDescription", "")

    # ABN — v3 API doesn't expose businessNumber, check badges.abnVerified too
    abn = ss_profile.get("businessNumber", "") or ""
    abn_verified = (ss_profile.get("badges") or {}).get("abnVerified", False)
    state["abn"] = abn
    state["business_verified"] = bool(abn.strip()) or abn_verified

    # Location from jobFilter.suburb
    job_filter = ss_profile.get("jobFilter") or {}
    suburb_data = job_filter.get("suburb") or {}
    state["business_state"] = suburb_data.get("state", "")
    state["business_postcode"] = str(suburb_data.get("postcode", ""))
    state["business_suburb"] = suburb_data.get("suburb", "")

    # Service areas — compute actual regions from SS radius
    if suburb_data:
        postcode = str(suburb_data.get("postcode", ""))
        radius = job_filter.get("radius", 20)
        regions_included = []
        if postcode:
            from agent.tools import get_suburbs_in_radius_grouped
            grouped = get_suburbs_in_radius_grouped(postcode, float(radius))
            if grouped.get("by_area"):
                # Include all regions with 3+ suburbs (same threshold as service_area_node)
                regions_included = [area for area, suburbs in grouped["by_area"].items()
                                    if len(suburbs) >= 3]
        state["service_areas"] = {
            "base_suburb": suburb_data.get("suburb", ""),
            "base_postcode": postcode,
            "base_lat": suburb_data.get("lat", 0),
            "base_lng": suburb_data.get("lng", 0),
            "radius_km": radius,
            "regions_included": regions_included,
            "regions_excluded": [],
        }
        # Mark areas as confirmed so we don't re-run unless user chooses to fix
        state["service_areas_confirmed"] = True

    # Existing services from SS subcategories
    ss_subcats = job_filter.get("subCategories") or []
    services = []
    for sc in ss_subcats:
        services.append({
            "input": sc.get("name", ""),
            "category_name": sc.get("categoryName", ""),
            "category_id": sc.get("categoryID", 0),
            "subcategory_name": sc.get("name", ""),
            "subcategory_id": sc.get("id", 0),
            "confidence": "existing",
            "source": "existing",
        })
    state["services"] = services
    # Mark services as confirmed so we don't re-run unless user chooses to fix
    state["services_confirmed"] = True

    # Profile data
    state["profile_logo"] = ss_profile.get("logoUrl", "") or ""
    state["contact_phone"] = ss_profile.get("phoneNumber", "") or ""
    state["business_website"] = ss_profile.get("websiteUrl", "") or ""
    state["contact_name"] = ss_profile.get("ownerName", "") or ""

    # Mark profile as saved (don't re-run unless user wants to)
    state["profile_saved"] = True

    # Set current node for assessment
    state["current_node"] = "assessment"

    return state


@app.post("/api/session")
async def create_session(req: StartRequest, request: Request):
    """Create a new onboarding session and get the welcome message.

    If ss_business_id is provided, creates an improve-mode session by fetching the
    existing SS profile and running assessment.
    """
    # IP-based rate limit on session creation
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(f"ip:{client_ip}", SESSION_CREATE_LIMIT, SESSION_CREATE_WINDOW):
        raise HTTPException(
            status_code=429,
            detail="Too many sessions created — please wait a moment",
            headers={"Retry-After": str(int(SESSION_CREATE_WINDOW))},
        )

    session_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # ── Improve mode: existing SS user ──
    if req.ss_business_id:
        # Check mock profiles first (for testing), then real API
        ss_profile = _MOCK_SS_PROFILES.get(req.ss_business_id) or await ss_get_business(req.ss_business_id)
        if not ss_profile:
            raise HTTPException(status_code=404, detail=f"Business {req.ss_business_id} not found on Service Seeking")

        state = _init_improve_state(ss_profile, session_id)

        # Run assessment node (enrichment + gap analysis)
        result = await assessment_node(state)
        for key, value in result.items():
            if key == "messages":
                state["messages"] = state.get("messages", []) + value
            else:
                state[key] = value

        turn_time = round(time.time() - start_time, 2)
        sessions[session_id] = state

        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        response_text = ai_messages[-1].content if ai_messages else "Let me have a look at your profile..."
        buttons = state.pop("buttons", None) or []

        _log_turn(session_id, {
            "turn": 0,
            "node": "assessment",
            "turn_time": turn_time,
            "ai_response": response_text,
            "business_name": state.get("business_name", ""),
            "flow_mode": "improve",
        })

        resp = {
            "text": response_text,
            "buttons": buttons,
            "node": "assessment",
            "turn_time": turn_time,
        }

        # Attach assessment findings (strip internal _ fields)
        assessment = state.get("_assessment", {})
        if assessment.get("findings"):
            resp["_assessment_findings"] = [
                {k: v for k, v in f.items() if not k.startswith("_")}
                for f in assessment["findings"]
            ]
            if assessment.get("strengths"):
                resp["_assessment_strengths"] = assessment["strengths"]
            if assessment.get("profile_score") is not None:
                resp["_profile_score"] = assessment["profile_score"]

        api_trace = state.pop("_api_trace", [])

        return {
            "session_id": session_id,
            "response": resp,
            "state": _safe_state(state),
            "api_trace": api_trace,
        }

    # ── New user mode ──
    state = _init_base_state(session_id)

    result = await welcome_node(state)

    # Merge
    for key, value in result.items():
        if key == "messages":
            state["messages"] = state.get("messages", []) + value
        else:
            state[key] = value

    turn_time = round(time.time() - start_time, 2)
    sessions[session_id] = state

    # Extract AI response
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    response_text = ai_messages[-1].content if ai_messages else "G'day! What's your business name or ABN?"

    _log_turn(session_id, {
        "turn": 0,
        "node": "welcome",
        "turn_time": turn_time,
        "ai_response": response_text,
    })

    return {
        "session_id": session_id,
        "response": {
            "text": response_text,
            "buttons": [],
            "node": "welcome",
            "turn_time": turn_time,
        },
        "state": _safe_state(state),
    }


@app.post("/api/chat")
async def chat(req: MessageRequest):
    """Send a message and get a response."""
    state = sessions.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    # Rate limit
    if not _check_rate_limit(req.session_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment",
            headers={"Retry-After": str(int(RATE_WINDOW))},
        )

    state["_last_active"] = time.time()

    # Add user message
    state["messages"].append(HumanMessage(content=req.message))
    state["_api_trace"] = []  # Reset trace for this turn

    start_time = time.time()

    # Run the appropriate node
    state = await run_node(state)

    turn_time = round(time.time() - start_time, 2)
    sessions[req.session_id] = state

    # Extract response
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    response_text = ai_messages[-1].content if ai_messages else ""

    # Buttons come from: node-generated (LLM) or ABR results (data-driven)
    # None = no buttons set by node, fall back to data-driven; [] = explicitly no buttons
    buttons = state.pop("buttons", None)
    multiselect = state.pop("_multiselect", False)
    if buttons is None:
        buttons = _get_buttons_for_state(state)

    node = state.get("current_node", "")
    completed = node == "complete"

    _log_turn(req.session_id, {
        "turn": len([m for m in state["messages"] if isinstance(m, HumanMessage)]),
        "node": node,
        "turn_time": turn_time,
        "user_message": req.message,
        "ai_response": response_text,
        "buttons": [b.get("label", "") if isinstance(b, dict) else b for b in buttons],
        "business_name": state.get("business_name", ""),
        "business_verified": state.get("business_verified", False),
        "services_count": len(state.get("services", [])),
        "services_confirmed": state.get("services_confirmed", False),
        "service_areas_confirmed": state.get("service_areas_confirmed", False),
        "confirmed": state.get("confirmed", False),
        "completed": completed,
    })

    api_trace = state.pop("_api_trace", [])

    resp = {
        "text": response_text,
        "buttons": buttons,
        "node": node,
        "turn_time": turn_time,
    }
    if multiselect:
        resp["_multiselect"] = True
    fallback_form = state.pop("_fallback_form", False)
    if fallback_form:
        resp["_fallback_form"] = True
        logger.info(f"[RESP] Fallback form flag set, buttons have headers: {any(b.get('header') for b in buttons if isinstance(b, dict))}")
    profile_question = state.pop("_profile_question", False)
    if profile_question:
        resp["_profile_question"] = True

    # Attach assessment findings for improve mode
    if node == "assessment":
        assessment = state.get("_assessment", {})
        if assessment.get("findings"):
            resp["_assessment_findings"] = [
                {k: v for k, v in f.items() if not k.startswith("_")}
                for f in assessment["findings"]
            ]
            if assessment.get("strengths"):
                resp["_assessment_strengths"] = assessment["strengths"]
            if assessment.get("profile_score") is not None:
                resp["_profile_score"] = assessment["profile_score"]

    return {
        "session_id": req.session_id,
        "response": resp,
        "state": _safe_state(state),
        "completed": completed,
        "api_trace": api_trace,
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "state": _safe_state(state)}


@app.get("/api/session/{session_id}/result")
async def get_result(session_id: str):
    """Get final structured output for a completed session."""
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    if not state.get("confirmed"):
        return {"status": "in_progress", "result": None}

    return {"status": "complete", "result": state.get("output_json", {})}


@app.get("/api/logs")
async def list_logs():
    """List all session logs."""
    logs = sorted(LOG_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [
        {
            "session_id": f.stem,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in logs[:50]
    ]


@app.get("/api/logs/{session_id}")
async def get_log(session_id: str):
    """Get full session log."""
    log_file = LOG_DIR / f"{session_id}.jsonl"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    turns = [json.loads(line) for line in log_file.read_text().strip().split("\n") if line.strip()]
    total_time = sum(t.get("turn_time", 0) for t in turns)
    return {
        "session_id": session_id,
        "total_turns": len(turns),
        "total_time": round(total_time, 2),
        "completed": turns[-1].get("completed", False) if turns else False,
        "business_name": next((t["business_name"] for t in reversed(turns) if t.get("business_name")), ""),
        "turns": turns,
    }


def _is_debug_allowed(request: Request) -> bool:
    """Check if debug endpoints are accessible: localhost or DEBUG=1 env var."""
    if os.environ.get("DEBUG") == "1":
        return True
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


def _debug_state(state: dict) -> dict:
    """Return full state for debug panel, minus non-serialisable messages."""
    from langchain_core.messages import BaseMessage
    out = {}
    for k, v in state.items():
        if k == "messages":
            out["_message_count"] = len(v) if isinstance(v, list) else 0
            continue
        # Skip large binary data
        if k in ("profile_logo", "profile_photos") and v:
            if k == "profile_logo":
                out[k] = "(base64 data)" if v else ""
            else:
                out[k] = [f"(photo {i+1})" for i in range(len(v))]
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


@app.get("/debug.html")
async def debug_page(request: Request):
    """Serve the debug page (localhost/DEBUG only)."""
    if not _is_debug_allowed(request):
        raise HTTPException(status_code=403, detail="Debug page not available in production")
    debug_file = web_dir / "debug.html"
    if debug_file.exists():
        return HTMLResponse(content=debug_file.read_text())
    raise HTTPException(status_code=404, detail="debug.html not found")


@app.get("/api/debug-state/{session_id}")
async def get_debug_state(session_id: str, request: Request):
    """Return full unfiltered session state for the debug panel.

    Gated: only available on localhost or when DEBUG=1 env var is set.
    """
    if not _is_debug_allowed(request):
        raise HTTPException(status_code=403, detail="Debug endpoint not available in production")
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "debug_state": _debug_state(state)}


MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_PHOTOS = 6


@app.post("/api/upload")
async def upload_image(
    session_id: str = Form(...),
    upload_type: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a logo or work photo. Stores as base64 data URL in session state."""
    state = sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    if upload_type not in ("logo", "photo"):
        raise HTTPException(status_code=400, detail="upload_type must be 'logo' or 'photo'")

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid image type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 5MB limit")

    data_url = f"data:{file.content_type};base64,{base64.b64encode(data).decode()}"

    if upload_type == "logo":
        state["profile_logo"] = data_url
    else:
        photos = state.get("profile_photos", [])
        if len(photos) >= MAX_PHOTOS:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_PHOTOS} photos allowed")
        photos.append(data_url)
        state["profile_photos"] = photos

    return {"ok": True, "upload_type": upload_type, "count": len(state.get("profile_photos", []))}


# ────────── HELPERS ──────────

def _get_buttons_for_state(state: dict) -> list:
    """Fallback buttons for data-driven steps (ABR results, confirmation).

    Service discovery and service area buttons come from the LLM.
    """
    node = state.get("current_node", "")

    if node == "business_verification":
        abr_results = state.get("abr_results", [])
        if abr_results and not state.get("business_verified"):
            # Detect near-duplicate names (e.g. "Foo Electrical" + "Foo Electrical Pty Ltd")
            # so we can add entity type to disambiguate
            name_stems = {}
            for r in abr_results[:8]:
                stem = re.sub(r'\s*(pty|ltd|limited|inc)\.?\s*', '', r.get("display_name", "").lower()).strip()
                name_stems.setdefault(stem, []).append(r.get("abn", ""))
            has_similar = any(len(abns) > 1 for abns in name_stems.values())

            buttons = []
            for r in abr_results[:8]:
                # Skip inactive/cancelled ABNs
                if r.get("status", "Active") != "Active":
                    continue
                name = r.get("display_name", "Unknown")
                legal = r.get("legal_name", "")
                abn = r.get("abn", "")
                location = r.get("state", "")
                if r.get("postcode"):
                    location = f"{location} {r['postcode']}"
                entity_type = r.get("entity_type", "")
                # Determine entity label
                if has_similar:
                    is_pty = bool(re.search(r'\bpty\b', name.lower()))
                    entity_label = "Company" if is_pty else "Sole Trader"
                elif entity_type:
                    entity_label = entity_type
                else:
                    entity_label = ""
                legal_display = legal.title() if legal.isupper() else legal
                buttons.append({
                    "label": name,
                    "value": f"Yes, it's {name} (ABN: {abn})",
                    "type": "abr_result",
                    "location": location,
                    "abn": abn,
                    "entity_type": entity_label,
                    "legal_name": legal_display if legal and legal.lower() != name.lower() else "",
                })
            buttons.append({"label": "None of these", "value": "No, none of those are my business"})
            return buttons

    return []


def _safe_state(state: dict) -> dict:
    """Return a JSON-safe version of the state for the frontend.

    Omits: messages, internal fields (_*), and server-only fields
    (web_results, entity_type, gst_registered, confirmed, subscription_billing/price,
    contact_phone, google_reviews).
    ABN is redacted (last 3 visible). Licence info stripped to number + classes.
    Full state is preserved server-side for /result endpoint.
    """
    # Redact ABN: "51 824 753 556" → "XX XXX XXX 556"
    abn = state.get("abn", "")
    if abn:
        clean = abn.replace(" ", "")
        if len(clean) >= 3:
            abn = f"XX XXX XXX {clean[-3:]}"

    # Strip licence_info to just what the profile builder shows
    licence_info = state.get("licence_info", {})
    safe_licence = {}
    if licence_info:
        safe_licence = {
            "licence_number": licence_info.get("licence_number", ""),
            "classes": licence_info.get("classes", []),
        }

    return {
        "session_id": state.get("session_id"),
        "current_node": state.get("current_node"),
        "business_name": state.get("business_name"),
        "abn": abn,
        "business_verified": state.get("business_verified"),
        "business_postcode": state.get("business_postcode"),
        "business_state": state.get("business_state", ""),
        "licence_classes": state.get("licence_classes", []),
        "licence_info": safe_licence,
        "services": state.get("services", []),
        "services_confirmed": state.get("services_confirmed"),
        "service_areas": state.get("service_areas", {}),
        "service_areas_confirmed": state.get("service_areas_confirmed"),
        "output_json": state.get("output_json"),
        "contact_name": state.get("contact_name", ""),
        "years_in_business": state.get("years_in_business", 0),
        "profile_description": state.get("profile_description", ""),
        "profile_description_draft": state.get("profile_description_draft", ""),
        "profile_intro": state.get("profile_intro", ""),
        "profile_logo": state.get("profile_logo", ""),
        "profile_photos": state.get("profile_photos", []),
        "profile_saved": state.get("profile_saved", False),
        "google_rating": state.get("google_rating", 0.0),
        "google_review_count": state.get("google_review_count", 0),
        "pricing_shown": state.get("pricing_shown", False),
        "subscription_plan": state.get("subscription_plan", ""),
        "_flow_mode": state.get("_flow_mode", ""),
        "_assessment_shown": state.get("_assessment_shown", False),
        "_improve_fix_total": state.get("_improve_fix_total", 0),
        "_improve_fix_index": state.get("_improve_fix_index", 0),
        "_verification_mode": state.get("_verification_mode", False),
        "_description_comparison": state.get("_description_comparison", False),
        "_description_improved": state.get("_description_improved", ""),
        "_needs_logo": state.get("_needs_logo", False),
        "_needs_photos": state.get("_needs_photos", False),
        "_ss_description": (state.get("_ss_profile") or {}).get("businessDescription", ""),
    }


# ────────── MAIN ──────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

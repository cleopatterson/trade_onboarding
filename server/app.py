"""FastAPI server for the Trade Onboarding wizard"""
from __future__ import annotations

import asyncio
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

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
    complete_node,
)
from agent.config import PORT, ALLOWED_ORIGINS, validate_env
from agent.tools import _get_nsw_trades_token

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
    pass

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
}


def determine_node(state: dict) -> str:
    """Determine which node to run based on state."""
    if not state.get("business_verified"):
        return "business_verification"
    if not state.get("services_confirmed"):
        return "service_discovery"
    if not state.get("service_areas_confirmed"):
        return "service_area"
    if not state.get("profile_saved"):
        return "profile"
    if not state.get("subscription_plan"):
        return "pricing"
    return "complete"


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
        if state.get("services_confirmed") and already_confirmed:
            # Services were already confirmed — this turn's message was for area
            _merge(area_result)
            # Continue auto-chain: area confirmed → profile
            if state.get("service_areas_confirmed") and not state.get("profile_saved") and not state.get("profile_description_draft"):
                state["confirmed"] = True
                _merge(await NODE_FUNCTIONS["profile"](state))
            if state.get("profile_saved") and not state.get("subscription_plan"):
                _merge(await NODE_FUNCTIONS["pricing"](state))
            if state.get("subscription_plan") and not state.get("output_json"):
                _merge(await NODE_FUNCTIONS["complete"](state))
            return state
        # Services just confirmed THIS turn (or not confirmed yet) — discard area result,
        # fall through to sequential auto-chain which handles area with clean context

    # ── Normal sequential path ──
    result = await node_fn(state)
    _raw_merge(result)

    # Auto-chain: business verified → immediately run service discovery
    if state.get("business_verified") and not state.get("services") and not state.get("services_confirmed"):
        _merge(await NODE_FUNCTIONS["service_discovery"](state))

    # Auto-chain: services confirmed → immediately run service area
    # Use base_suburb as the guard (set on first area run) instead of regions_included (can be empty [])
    if state.get("services_confirmed") and not state.get("service_areas", {}).get("base_suburb") and not state.get("service_areas_confirmed"):
        state["_auto_chained"] = True  # Tell area node to ignore stale user message
        _merge(await NODE_FUNCTIONS["service_area"](state))
        state.pop("_auto_chained", None)

    # Auto-chain: service areas confirmed → profile (skip confirmation step)
    if state.get("service_areas_confirmed") and not state.get("profile_saved") and not state.get("profile_description_draft"):
        state["confirmed"] = True  # Skip confirmation step
        _merge(await NODE_FUNCTIONS["profile"](state))

    # Auto-chain: profile saved → pricing
    if state.get("profile_saved") and not state.get("subscription_plan"):
        _merge(await NODE_FUNCTIONS["pricing"](state))

    # Auto-chain: subscription done → complete
    if state.get("subscription_plan") and not state.get("output_json"):
        _merge(await NODE_FUNCTIONS["complete"](state))

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


@app.post("/api/session")
async def create_session(req: StartRequest, request: Request):
    """Create a new onboarding session and get the welcome message."""
    # IP-based rate limit on session creation
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(f"ip:{client_ip}", SESSION_CREATE_LIMIT, SESSION_CREATE_WINDOW):
        raise HTTPException(
            status_code=429,
            detail="Too many sessions created — please wait a moment",
            headers={"Retry-After": str(int(SESSION_CREATE_WINDOW))},
        )

    session_id = str(uuid.uuid4())[:8]

    now = time.time()
    state = {
        "session_id": session_id,
        "current_node": "welcome",
        "messages": [],
        "business_name_input": "",
        "abn_input": "",
        "abr_results": [],
        "business_name": "",
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
        "facebook_url": "",
        "business_website": "",
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
        "_created_at": now,
        "_last_active": now,
    }

    # Run welcome node
    start_time = time.time()
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
        "buttons": [b.get("label", "") for b in buttons],
        "business_name": state.get("business_name", ""),
        "business_verified": state.get("business_verified", False),
        "services_count": len(state.get("services", [])),
        "services_confirmed": state.get("services_confirmed", False),
        "service_areas_confirmed": state.get("service_areas_confirmed", False),
        "confirmed": state.get("confirmed", False),
        "completed": completed,
    })

    api_trace = state.pop("_api_trace", [])

    return {
        "session_id": req.session_id,
        "response": {
            "text": response_text,
            "buttons": buttons,
            "node": node,
            "turn_time": turn_time,
        },
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
            if len(abr_results) == 1:
                return [
                    {"label": "Yes, that's me", "value": "Yes, that's my business"},
                    {"label": "No, that's not right", "value": "No, that's not my business"},
                ]
            else:
                buttons = []
                for r in abr_results[:5]:
                    name = r.get("entity_name", "Unknown")
                    abn = r.get("abn", "")
                    location = r.get("state", "")
                    if r.get("postcode"):
                        location = f"{location} {r['postcode']}"
                    # Show ABN suffix so tradies can distinguish near-duplicates
                    abn_short = f" · ABN ...{abn[-5:]}" if len(abn) >= 5 else ""
                    label = f"{name} ({location}{abn_short})" if location else f"{name}{abn_short}"
                    if len(label) > 60:
                        label = f"{name[:28]}... ({location}{abn_short})"
                    buttons.append({"label": label, "value": f"Yes, it's {name} (ABN: {abn})"})
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
        "facebook_url": state.get("facebook_url", ""),
        "pricing_shown": state.get("pricing_shown", False),
        "subscription_plan": state.get("subscription_plan", ""),
    }


# ────────── MAIN ──────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

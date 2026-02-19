"""FastAPI server for the Trade Onboarding wizard"""
from __future__ import annotations

import asyncio
import uuid
import time
import json
from datetime import datetime
from pathlib import Path

import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import (
    welcome_node, business_verification_node, service_discovery_node,
    service_area_node, confirmation_node, profile_node, pricing_node,
    complete_node,
)
from agent.config import PORT
from agent.tools import _get_nsw_trades_token


# ────────── SESSION STORE ──────────

sessions: dict = {}


# ────────── SESSION LOGGING ──────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _log_turn(session_id: str, turn: dict):
    """Append a turn entry to the session's JSONL log file."""
    log_file = LOG_DIR / f"{session_id}.jsonl"
    turn["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(log_file, "a") as f:
        f.write(json.dumps(turn, default=str) + "\n")


# ────────── APP ──────────

app = FastAPI(title="Trade Onboarding Wizard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-warm NSW Trades OAuth token on startup
@app.on_event("startup")
async def warmup():
    token = await _get_nsw_trades_token()
    if token:
        print(f"[STARTUP] NSW Trades OAuth token pre-warmed")
    else:
        print(f"[STARTUP] NSW Trades OAuth token failed — licence lookups will retry on first request")

# Serve static files
web_dir = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# ────────── MODELS ──────────

class StartRequest(BaseModel):
    pass

class MessageRequest(BaseModel):
    session_id: str
    message: str


# ────────── NODE DISPATCH ──────────

NODE_FUNCTIONS = {
    "welcome": welcome_node,
    "business_verification": business_verification_node,
    "service_discovery": service_discovery_node,
    "service_area": service_area_node,
    "confirmation": confirmation_node,
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

    # ── Parallel path: service_discovery turn 2 + service_area turn 1 ──
    # service_area only needs postcode + region data, not the services list,
    # so we can fire both LLM calls simultaneously (~3s instead of ~11s)
    if (node_name == "service_discovery"
            and state.get("services")  # turn 2 — services already mapped
            and not state.get("service_areas", {}).get("regions_included")
            and not state.get("service_areas_confirmed")):
        svc_result, area_result = await asyncio.gather(
            node_fn(state),
            NODE_FUNCTIONS["service_area"](state),
        )
        _raw_merge(svc_result)
        if state.get("services_confirmed"):
            _merge(area_result)
            # Continue auto-chain: area confirmed → profile
            if state.get("service_areas_confirmed") and not state.get("profile_saved") and not state.get("profile_description_draft"):
                state["confirmed"] = True  # Skip confirmation step
                _merge(await NODE_FUNCTIONS["profile"](state))
            if state.get("profile_saved") and not state.get("subscription_plan"):
                _merge(await NODE_FUNCTIONS["pricing"](state))
            if state.get("subscription_plan") and not state.get("output_json"):
                _merge(await NODE_FUNCTIONS["complete"](state))
        return state

    # ── Normal sequential path ──
    result = await node_fn(state)
    _raw_merge(result)

    # Auto-chain: business verified → immediately run service discovery
    if state.get("business_verified") and not state.get("services") and not state.get("services_confirmed"):
        _merge(await NODE_FUNCTIONS["service_discovery"](state))

    # Auto-chain: services confirmed → immediately run service area
    # Use base_suburb as the guard (set on first area run) instead of regions_included (can be empty [])
    if state.get("services_confirmed") and not state.get("service_areas", {}).get("base_suburb") and not state.get("service_areas_confirmed"):
        _merge(await NODE_FUNCTIONS["service_area"](state))

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
async def create_session(req: StartRequest):
    """Create a new onboarding session and get the welcome message."""
    session_id = str(uuid.uuid4())[:8]

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
        "pricing_shown": False,
        "subscription_plan": "",
        "subscription_billing": "",
        "subscription_price": "",
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
                    label = f"{name} ({location})" if location else name
                    if len(label) > 45:
                        label = f"{name[:30]}... ({location})"
                    buttons.append({"label": label, "value": f"Yes, it's {name} (ABN: {abn})"})
                buttons.append({"label": "None of these", "value": "No, none of those are my business"})
                return buttons

    if node == "confirmation":
        if not state.get("confirmed"):
            return [
                {"label": "All good, let's go", "value": "Yes, confirm and complete"},
                {"label": "Edit Services", "value": "I want to change my services"},
                {"label": "Edit Service Areas", "value": "I want to change my service areas"},
            ]

    return []


def _safe_state(state: dict) -> dict:
    """Return a JSON-safe version of the state (strip messages)."""
    return {
        "session_id": state.get("session_id"),
        "current_node": state.get("current_node"),
        "business_name": state.get("business_name"),
        "abn": state.get("abn"),
        "entity_type": state.get("entity_type"),
        "gst_registered": state.get("gst_registered"),
        "business_verified": state.get("business_verified"),
        "business_postcode": state.get("business_postcode"),
        "licence_classes": state.get("licence_classes", []),
        "licence_info": {
            k: v for k, v in state.get("licence_info", {}).items()
            if k != "raw"
        } if state.get("licence_info") else {},
        "web_results": state.get("web_results", []),
        "services": state.get("services", []),
        "services_confirmed": state.get("services_confirmed"),
        "service_areas": state.get("service_areas", {}),
        "service_areas_confirmed": state.get("service_areas_confirmed"),
        "confirmed": state.get("confirmed"),
        "output_json": state.get("output_json"),
        "contact_name": state.get("contact_name", ""),
        "contact_phone": state.get("contact_phone", ""),
        "years_in_business": state.get("years_in_business", 0),
        "profile_description": state.get("profile_description", ""),
        "profile_description_draft": state.get("profile_description_draft", ""),
        "profile_intro": state.get("profile_intro", ""),
        "profile_logo": state.get("profile_logo", ""),
        "profile_photos": state.get("profile_photos", []),
        "profile_saved": state.get("profile_saved", False),
        "pricing_shown": state.get("pricing_shown", False),
        "subscription_plan": state.get("subscription_plan", ""),
        "subscription_billing": state.get("subscription_billing", ""),
        "subscription_price": state.get("subscription_price", ""),
    }


# ────────── MAIN ──────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

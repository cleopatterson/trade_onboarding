"""LangGraph state machine for the Trade Onboarding wizard

CORE PRINCIPLE: No scripting.
Give the LLM context, goals, and guides — let it figure out the conversation.
No keyword matching, no canned responses, no rigid flows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import OnboardingState

logger = logging.getLogger(__name__)
from agent.config import ANTHROPIC_API_KEY, MODEL_FAST
from agent.tools import (
    abr_lookup, enrich_abr_with_entity_names, get_category_taxonomy_text,
    search_suburbs_by_postcode,
    get_suburbs_in_radius_grouped, get_regional_guide,
    find_subcategory_guide,
    nsw_licence_browse, nsw_licence_details,
    brave_web_search, scrape_website_images,
    discover_business_website, scrape_social_images, ai_filter_photos,
    google_places_search, compute_service_gaps, compute_initial_services,
    scrape_website_text,
    qbcc_licence_lookup, _detect_category, _detect_categories,
    extract_licence_from_text, scan_website_for_licence, _VIC_LICENCE_CONFIG,
    get_licence_config, match_licence,
    suggest_related_categories, map_extra_categories,
)


# ────────── PROTOCOL CONSTANTS ──────────
# Button value prefixes and magic strings shared between frontend and backend.
# Change here → grep for the constant name to find all usages.

MSG_SAVE_PROFILE = "__SAVE_PROFILE__:"
MSG_PLAN = "__PLAN__:"
MSG_BILLING = "__BILLING__:"
MSG_RESTART_BIZ = "__RESTART_BIZ__"
MSG_YES_ALL = "Yes, all of these"


# ────────── MODELS ──────────

llm_fast = ChatAnthropic(
    model=MODEL_FAST,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=512,
    temperature=0.3,
)

# Haiku with higher token limit for structured JSON responses (service lists, area mappings)
llm_fast_json = ChatAnthropic(
    model=MODEL_FAST,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=2048,
    temperature=0.3,
)


# ────────── API TRACE ──────────

def _trace(state: dict, name: str, duration: float, result_summary: str, data: dict = None):
    """Append an API call trace entry to state for dev tools visibility."""
    if "_api_trace" not in state:
        state["_api_trace"] = []
    state["_api_trace"].append({
        "api": name,
        "time": round(duration, 2),
        "summary": result_summary,
        "data": data or {},
    })


# ────────── LLM BUSINESS CLASSIFIER ──────────

_SS_CATEGORIES = [
    "Accountant","Air Conditioning & Heating Technician","App Developer","Arborist",
    "Asbestos Removalist","Asphalting Company","Auto Electrician","Auto Tuner",
    "Automatic Door & Gate Company","Automotive Glass Repair Company",
    "Bathroom Renovation Company","Blinds & Shutter Installer","Bookkeeper",
    "Bricklayer","Builder","Building Designer","Building Inspector","Car detailer",
    "Carpenter","Carpet Cleaner","Carpet Installer","Celebrant","Cleaner",
    "Computer Repairer & IT Service Provider","Concreter","Crane Hire Company",
    "Demolition Company","Dishwasher Installation & Repair Company",
    "Door Installation Company","Draftsman","Ducting and Ventiliation Company",
    "Earthworks contractor","Electrician","Exterminator","Fencing & Gate Company",
    "Financial Planner","Flooring Company","Fridge & Freezer Repair & Installation Company",
    "Gardener","Gas Fitter","Glass Repair Company","Graphic Designer","Hair Stylist",
    "Handyman","High pressure cleaning company","Home theatre installation company",
    "Housekeeper","Insulation Company","Insurance","Interior Designer",
    "Kitchen Renovation Company","Labourer","Landscaper",
    "Lawyers and legal professionals","Lighting installation company","Locksmith",
    "Make Up Artist","Mechanic","Musician & Entertainer","Oven Installation & Repairs",
    "Painter","Paver","Photographer","Plasterer","Plumber","Pool & Spa Company",
    "Printer","Removalist","Rendering Company","Roofer","Rubbish Removalist",
    "Scaffolding Company","Security Company","Shades and Sails Company","Sign Company",
    "Skylight Installation Company","Smash Repair Company","Solar Company","Stonemason",
    "Structural Engineer","Surveyor","TV Antenna Technician","TV repair technician",
    "Test and tag company","Tiler","Upholsterer","Videographer",
    "Washing Machine and Dryer Technician","Waterproofing Company",
    "Website Developers and Designers","Wedding & Event Supplier",
    "Welders and Boilermakers","Window Cleaner","Window and Glass Installation Company",
]

async def classify_business_from_web(
    business_name: str,
    google_name: str = "",
    google_type: str = "",
    google_reviews: list[dict] = None,
    website_text: str = "",
) -> dict:
    """Use Haiku to classify a business from its web presence.

    Returns {"is_trade": bool, "categories": [...], "reason": "..."}.
    Replaces keyword matching for Google/website signals.
    """
    # Build context from available signals
    signals = []
    if google_name:
        signals.append(f"Google Places listing: {google_name}")
    if google_type:
        signals.append(f"Google business type: {google_type}")
    if google_reviews:
        review_text = " | ".join(r.get("text", "")[:200] for r in google_reviews[:3])
        if review_text:
            signals.append(f"Customer reviews: {review_text}")
    if website_text:
        signals.append(f"Website content (excerpt): {website_text[:1500]}")

    if not signals:
        return {"is_trade": True, "categories": [], "reason": "No web data to classify"}

    signals_str = "\n".join(signals)
    cat_list = ", ".join(_SS_CATEGORIES)

    try:
        response = await llm_fast.ainvoke([
            SystemMessage(content=f"""You are classifying an Australian business for Service Seeking, a marketplace for trade and service professionals.

BUSINESS NAME: {business_name}

WEB PRESENCE DATA:
{signals_str}

SERVICE SEEKING CATEGORIES:
{cat_list}

Analyse the web presence and determine:
1. Is this a trade or service business that belongs on Service Seeking? (plumbers, electricians, builders, cleaners, photographers, accountants, designers, etc.)
   - NOT suitable: retail shops, restaurants, auction houses, real estate agencies, medical practices, recruitment agencies, mining companies, equipment dealers
2. If it IS a trade/service business, which Service Seeking categories match? Use EXACT category names from the list above. Only include categories where the web evidence clearly shows they offer that service.

Respond with JSON only:
{{"is_trade": true/false, "categories": ["Category Name", ...], "reason": "one sentence explanation"}}"""),
            HumanMessage(content="Classify this business."),
        ])
        parsed = json.loads(_extract_json(response.content))
        logger.info(f"[CLASSIFY] {business_name}: is_trade={parsed.get('is_trade')}, categories={parsed.get('categories', [])}, reason={parsed.get('reason', '')}")
        return parsed
    except Exception as e:
        logger.warning(f"[CLASSIFY] Failed for {business_name}: {e}")
        return {"is_trade": True, "categories": [], "reason": f"Classification failed: {e}"}


# ────────── NODE FUNCTIONS ──────────

async def welcome_node(state: OnboardingState) -> dict:
    """Greet the user and ask for business name/ABN."""
    response = await llm_fast.ainvoke([
        SystemMessage(content="""You are the Service Seeking onboarding assistant. You help Australian trade and service professionals get set up on the platform.
Service Seeking covers tradies (plumbers, electricians, builders, etc.) AND professional services (photographers, accountants, designers, IT, etc.). Most users are tradies, but welcome everyone.
You are warm, friendly, and speak in natural Australian English.

Write a welcome message that:
- Greets them warmly
- Briefly explains what's about to happen: you'll look up their business, figure out what services they offer, and sort out where they work — all in about 2 minutes
- Mentions you'll do most of the heavy lifting by pulling in their details automatically (ABN, licences, etc)
- Asks for their **business name** or **ABN** to kick things off — use bold markdown on these two terms so they stand out as the clear call-to-action
- Feels like a real person, not a corporate form. Keep it concise — 3-4 short sentences.
- If they include a postcode with their business name (e.g. "dans plumbing 2155") you can match them faster
- Do NOT ask what type of business they are (tradie vs consultant etc.) — just ask for their business name or ABN and you'll figure the rest out"""),
        HumanMessage(content="Hi, I'd like to get set up on Service Seeking."),
    ])

    return {
        "current_node": "welcome",
        "messages": [response],
    }



def _is_sole_trader_personal_name(abr: dict) -> bool:
    """Check if ABR result is a sole trader with only a personal name (no trading name).

    Returns True for "GRIFFITH, THOMAS" (Individual/Sole Trader) where no BusinessName
    is registered — meaning display_name == legal_name and both are personal names.
    Returns False when a trading name exists (display_name != legal_name).
    """
    dn = abr.get("display_name", "")
    legal_name = abr.get("legal_name", "")
    entity_type = abr.get("entity_type", "")

    # Has a registered trading/business name (e.g. "Watertight Plumbing" vs "SMITH, JACK") — no need to ask
    if abr.get("_has_registered_trading_name"):
        return False

    # Entity type check (reliable for ABN direct lookup)
    if "individual" in entity_type.lower() or "sole trader" in entity_type.lower():
        return True

    # Name format check (for name search results where entity_type is "Entity Name"):
    # "SURNAME, FIRSTNAME..." — all caps with comma. Check legal_name (original ABR case).
    check_name = legal_name or dn
    if re.match(r'^[A-Z][A-Z\s\'\-]+,\s+[A-Z]', check_name):
        return True

    return False


def _format_sole_trader_prompt(abr: dict) -> str:
    """Format the trading name prompt for a sole trader with personal name."""
    name = abr.get("display_name", "")
    etype = abr.get("entity_type", "")
    location = f"{abr.get('state', '')} {abr.get('postcode', '')}".strip()
    return (
        f"I found {name} registered as {etype} in {location}. "
        f"What name does your business trade under?"
    )


async def business_verification_node(state: OnboardingState) -> dict:
    """Look up and verify the business via ABR.

    This node has tool calls (ABR API) so it keeps some structure,
    but delegates conversation to the LLM.
    """
    messages = state.get("messages", [])
    last_msg = _get_last_human_message(messages)

    if not last_msg:
        return {"current_node": "business_verification"}

    # Guard: reject very short input (likely typo or junk)
    if last_msg and len(last_msg.strip()) < 2:
        return {
            "current_node": "business_verification",
            "messages": [AIMessage(content="Could you give me your full business name or ABN? I need at least a couple of characters to search.")],
        }

    # If no ABR results yet, do the lookup
    if not state.get("abr_results") and not state.get("business_verified"):
        clean = last_msg.strip().replace(" ", "")
        is_abn = clean.isdigit() and len(clean) == 11
        search_type = "abn" if is_abn else "name"

        # Detect trailing postcode (e.g. "dans plumbing 2155")
        postcode_match = re.search(r'\b(\d{4})\s*$', last_msg.strip())
        search_term = last_msg
        user_postcode = None
        if postcode_match and not is_abn:
            user_postcode = postcode_match.group(1)
            search_term = last_msg[:postcode_match.start()].strip()
            logger.info(f"[BIZ] Detected postcode {user_postcode} in input, searching for '{search_term}'")

        t0 = time.time()
        results = await abr_lookup(search_term, search_type)
        abr_results = results.get("results", [])
        _trace(state, "ABR Lookup", time.time() - t0,
               f"{len(abr_results)} results for '{search_term}'",
               {"search_type": search_type, "results": [
                   {"name": r.get("display_name"), "abn": r.get("abn"), "postcode": r.get("postcode")}
                   for r in abr_results[:5]
               ]})

        # Enrich name search results with entity names (parallel ABN lookups)
        # Runs for all name searches — even single results need the entity name
        # (e.g. sole trader "Watertight Plumbing" → entity "SMITH, JACK" for licence)
        if not is_abn and abr_results:
            t_enrich = time.time()
            abr_results = await enrich_abr_with_entity_names(abr_results)
            results["results"] = abr_results
            _trace(state, "ABR Enrich Names", time.time() - t_enrich,
                   f"Enriched {len(abr_results)} results with entity names",
                   {"results": [{"entity": r.get("display_name"), "legal": r.get("legal_name")} for r in abr_results[:5]]})

        # Filter to active results only (enrichment pulls real status from ABN detail lookups)
        abr_results = [r for r in abr_results if r.get("status", "Active") == "Active"]
        results["results"] = abr_results
        results["count"] = len(abr_results)

        if not abr_results:
            return {
                "current_node": "business_verification",
                "business_name_input": search_term,
                "abn_input": "",
                "abr_results": [],
                "messages": [AIMessage(content=f"I couldn't find any active businesses matching '{search_term}' on the ABR. Could you try a different name, or enter your ABN directly?")],
            }

        # If user provided a postcode, filter results and auto-confirm if single match
        if user_postcode and abr_results:
            filtered = [r for r in abr_results if r.get("postcode") == user_postcode]
            if len(filtered) == 1:
                r = filtered[0]
                if _is_sole_trader_personal_name(r):
                    logger.info(f"[BIZ] Single match in postcode {user_postcode}, sole trader — asking for trading name")
                    st_buttons = [{"label": "That's not me", "value": "No, that's not my business"}]
                    if search_term:
                        st_buttons.insert(0, {"label": search_term, "value": search_term})
                    return {
                        "current_node": "business_verification",
                        "abr_results": filtered,
                        "_needs_trading_name": True,
                        "buttons": st_buttons,
                        "messages": [AIMessage(content=_format_sole_trader_prompt(r))],
                    }
                logger.info(f"[BIZ] Single match in postcode {user_postcode}, auto-confirming")
                return await _confirm_business(r, state)
            elif filtered:
                # Multiple matches in postcode — show only those
                abr_results = filtered
                results["results"] = filtered
                results["count"] = len(filtered)
            else:
                # No matches in that postcode — show all with feedback
                postcode_note = f"None of the results matched postcode {user_postcode}, so here are all the matches:\n\n"
                return {
                    "current_node": "business_verification",
                    "business_name_input": search_term,
                    "abn_input": "",
                    "abr_results": abr_results,
                    "messages": [AIMessage(content=postcode_note + _format_abr_results(results, search_term))],
                }

        # Sole trader with personal name only — ask for trading name directly (skip "Is this you?")
        if len(abr_results) == 1 and _is_sole_trader_personal_name(abr_results[0]):
            logger.info(f"[BIZ] Single result is sole trader '{abr_results[0].get('display_name')}' — asking for trading name")
            st_buttons = [{"label": "That's not me", "value": "No, that's not my business"}]
            if search_term and not is_abn:
                st_buttons.insert(0, {"label": search_term, "value": search_term})
            return {
                "current_node": "business_verification",
                "business_name_input": search_term,
                "abn_input": last_msg if is_abn else "",
                "abr_results": abr_results,
                "_needs_trading_name": True,
                "buttons": st_buttons,
                "messages": [AIMessage(content=_format_sole_trader_prompt(abr_results[0]))],
            }

        return {
            "current_node": "business_verification",
            "business_name_input": search_term,
            "abn_input": last_msg if is_abn else "",
            "abr_results": abr_results,
            "messages": [AIMessage(content=_format_abr_results(results, search_term))],
        }

    # ── Sole trader trading name collection ──
    # If we asked for a trading name last turn, the user's response IS the trading name
    if state.get("_needs_trading_name"):
        abr_results = state.get("abr_results", [])
        # Check for rejection
        reject_words = {"no", "not me", "wrong", "not my", "that's not"}
        if any(w in last_msg.lower() for w in reject_words):
            return {
                "current_node": "business_verification",
                "abr_results": [],
                "_needs_trading_name": False,
                "business_verified": False,
                "messages": [AIMessage(content="No worries! Could you try a different business name, or enter your ABN directly?")],
            }
        # User provided their trading name — use it for display/web, keep personal name as legal
        trading_name = last_msg.strip()
        abr = abr_results[0].copy() if abr_results else {}
        abr["legal_name"] = abr.get("display_name", "")  # personal name → legal
        abr["display_name"] = trading_name  # trading name → display
        logger.info(f"[BIZ] Sole trader trading name: '{trading_name}', legal: '{abr['legal_name']}'")
        state["_needs_trading_name"] = False
        return await _confirm_business(abr, state)

    # We have ABR results — let the LLM interpret the user's response
    abr_results = state.get("abr_results", [])

    # Quick-match: if the message contains "Yes, it's" + an ABN from results, it's a button click confirmation
    selected_abr = None
    parsed = {}
    for r in abr_results:
        abn = r.get("abn", "")
        if abn and abn in last_msg and ("yes" in last_msg.lower() or "it's" in last_msg.lower()):
            logger.info(f"[BIZ] Quick-match: ABN {abn} found in message, confirming directly")
            selected_abr = r
            break

    if not selected_abr:
        # Smart LLM interpreter — understands natural language and extracts structured data
        abr_summary = json.dumps([
            {"abn": r.get("abn"), "name": r.get("display_name"), "state": r.get("state"), "postcode": r.get("postcode")}
            for r in abr_results[:8]
        ])
        response = await llm_fast.ainvoke([
            SystemMessage(content=f"""You are interpreting a user's response during business verification for Service Seeking onboarding.

CURRENT ABR SEARCH RESULTS SHOWN TO USER:
{abr_summary}

The user is looking at these results and has typed a message. Understand what they mean and respond with JSON:

{{
  "intent": "confirm|reject|new_search|question",
  "abn": "the ABN they're selecting (if confirming — match from the results above)",
  "search_term": "extracted business name or ABN to search (if new_search — pull just the name/ABN from their message, not the whole sentence)",
  "preferred_name": "if they mention a different trading name they want to use",
  "reply": "a helpful Australian-English response (only for question intent — when they're asking something, confused, or saying something unrelated to selecting a business)"
}}

INTENT RULES:
- "confirm": they're selecting one of the results (yes, that's me, the first one, clicking a name, etc). Set "abn" to the matching result's ABN.
- "reject": they're saying none of these are right (no, wrong, not me, none of these)
- "new_search": they're giving you a different business name or ABN to search. Extract JUST the business name or ABN into "search_term" — strip out filler words like "try", "search for", "I'm actually", "my business is", etc.
- "question": they're asking something, confused, or chatting. Write a natural, helpful reply in "reply" that guides them back to providing their business name or ABN. Be friendly and Australian.

Only include fields relevant to the intent. Respond with JSON only."""),
            HumanMessage(content=last_msg),
        ])

        try:
            parsed = json.loads(_extract_json(response.content))
        except (json.JSONDecodeError, ValueError):
            parsed = {"intent": "question", "reply": "Sorry, I didn't quite catch that. Could you tell me your business name or ABN?"}

        intent = parsed.get("intent", "question")
        logger.info(f"[BIZ] LLM intent: {intent} for message: {last_msg[:80]}")

        if intent == "confirm":
            # Find the selected result by ABN
            selected_abn = parsed.get("abn", "")
            if selected_abn:
                for r in abr_results:
                    if r.get("abn") == selected_abn:
                        selected_abr = r
                        break
            # Fallback: first result if LLM confirmed but didn't specify ABN
            if not selected_abr and abr_results:
                selected_abr = abr_results[0]

    if selected_abr:
        # Apply preferred name override if provided
        preferred = parsed.get("preferred_name", "")
        if preferred:
            selected_abr = selected_abr.copy()
            selected_abr["display_name"] = preferred
            logger.info(f"[BIZ] User preferred name override: '{preferred}'")

        if _is_sole_trader_personal_name(selected_abr):
            original_input = state.get("business_name_input", "")
            logger.info(f"[BIZ] Selected sole trader '{selected_abr.get('display_name')}' — asking for trading name (original input: '{original_input}')")
            buttons = [{"label": "That's not me", "value": "No, that's not my business"}]
            if original_input:
                # User already told us their business name — offer it as default
                buttons.insert(0, {"label": original_input, "value": original_input})
            return {
                "current_node": "business_verification",
                "abr_results": [selected_abr],
                "_needs_trading_name": True,
                "buttons": buttons,
                "messages": [AIMessage(content=_format_sole_trader_prompt(selected_abr))],
            }
        return await _confirm_business(selected_abr, state)

    intent = parsed.get("intent", "question")

    if intent == "reject":
        return {
            "current_node": "business_verification",
            "abr_results": [],
            "business_verified": False,
            "messages": [AIMessage(content="No worries! Could you try a different business name, or enter your ABN directly?")],
        }

    if intent == "new_search":
        search_term = parsed.get("search_term", last_msg).strip()
        if not search_term:
            search_term = last_msg.strip()
        clean = search_term.replace(" ", "")
        is_abn = clean.isdigit() and len(clean) == 11

        # Detect trailing postcode
        postcode_match = re.search(r'\b(\d{4})\s*$', search_term)
        user_postcode = None
        if postcode_match and not is_abn:
            user_postcode = postcode_match.group(1)
            search_term = search_term[:postcode_match.start()].strip()

        results = await abr_lookup(search_term, "abn" if is_abn else "name")
        new_abr = results.get("results", [])

        # Enrich with entity names
        if not is_abn and new_abr:
            new_abr = await enrich_abr_with_entity_names(new_abr)
            results["results"] = new_abr

        # Postcode filter
        if user_postcode and new_abr:
            filtered = [r for r in new_abr if r.get("postcode") == user_postcode]
            if len(filtered) == 1:
                if _is_sole_trader_personal_name(filtered[0]):
                    st_buttons = [{"label": "That's not me", "value": "No, that's not my business"}]
                    if search_term:
                        st_buttons.insert(0, {"label": search_term, "value": search_term})
                    return {
                        "current_node": "business_verification",
                        "abr_results": filtered,
                        "_needs_trading_name": True,
                        "buttons": st_buttons,
                        "messages": [AIMessage(content=_format_sole_trader_prompt(filtered[0]))],
                    }
                return await _confirm_business(filtered[0], state)
            elif filtered:
                new_abr = filtered
                results["results"] = filtered
                results["count"] = len(filtered)

        return {
            "current_node": "business_verification",
            "business_name_input": search_term,
            "abr_results": new_abr,
            "messages": [AIMessage(content=_format_abr_results(results, search_term))],
        }

    # question intent (or any fallback) — LLM generated a natural response
    reply = parsed.get("reply", "Sorry, I didn't quite catch that. Could you tell me your business name or ABN?")
    return {
        "current_node": "business_verification",
        "messages": [AIMessage(content=reply)],
    }


_NON_TRADE_TYPES = {
    "restaurant", "cafe", "bar", "bakery", "meal_delivery", "meal_takeaway",
    "store", "supermarket", "pharmacy", "clothing_store", "shoe_store",
    "jewelry_store", "book_store", "convenience_store", "department_store",
    "shopping_mall", "furniture_store", "hardware_store", "home_goods_store",
    "pet_store", "electronics_store",
    "doctor", "dentist", "hospital", "physiotherapist", "veterinary_care",
    "gym", "spa", "beauty_salon", "hair_care",
    "bank", "insurance_agency", "real_estate_agency",
    "school", "university", "library",
    "church", "mosque", "synagogue",
    "gas_station", "car_dealer", "car_rental", "car_wash",
    "travel_agency", "lodging", "hotel",
}


def _process_cluster_response(
    pending_ids: list, gaps: list[dict], services: list[dict], user_msg: str,
) -> tuple[list[dict], list[str], list[dict]]:
    """Process user's response to a cluster question (deterministic, no LLM).

    Returns (updated_services, added_names, remaining_gaps).
    """
    pending_set = set(pending_ids)
    cluster_gaps = [g for g in gaps if g["subcategory_id"] in pending_set]
    added_names: list[str] = []

    if user_msg == MSG_YES_ALL:
        # Legacy: "Yes, all of these" button (kept for backward compat)
        for g in cluster_gaps:
            services.append({
                "input": g["subcategory_name"],
                "category_name": g["category_name"],
                "category_id": g["category_id"],
                "subcategory_name": g["subcategory_name"],
                "subcategory_id": g["subcategory_id"],
                "confidence": "confirmed",
            })
            added_names.append(g["subcategory_name"])
        logger.info(f"[SVC] Pre-added {len(added_names)} cluster services: {added_names}")
    elif user_msg == "__CLUSTER_SKIP__" or (user_msg and user_msg.lower() in ("none of these", "not for us", "nah, move on", "not our thing")):
        logger.info(f"[SVC] Declined cluster: '{user_msg}'")
    else:
        # Multi-select: split comma-separated input, match each part by word overlap
        parts = [p.strip().removeprefix("just ").strip() for p in user_msg.split(",") if p.strip()]
        matched_ids: set[int] = set()
        for part in parts:
            part_words = set(part.lower().split())
            best_match = None
            best_score = 0
            for g in cluster_gaps:
                if g["subcategory_id"] in matched_ids:
                    continue
                name_words = set(g["subcategory_name"].lower().split())
                overlap = len(part_words & name_words)
                if overlap > best_score:
                    best_score = overlap
                    best_match = g
            if best_match and best_score >= 1:
                services.append({
                    "input": best_match["subcategory_name"],
                    "category_name": best_match["category_name"],
                    "category_id": best_match["category_id"],
                    "subcategory_name": best_match["subcategory_name"],
                    "subcategory_id": best_match["subcategory_id"],
                    "confidence": "confirmed",
                })
                added_names.append(best_match["subcategory_name"])
                matched_ids.add(best_match["subcategory_id"])
        if added_names:
            logger.info(f"[SVC] Multi-selection '{user_msg}' matched: {added_names}")
        else:
            logger.info(f"[SVC] Declined cluster: '{user_msg}'")

    # Remove processed cluster from gaps regardless of response
    remaining_gaps = [g for g in gaps if g["subcategory_id"] not in pending_set]
    return services, added_names, remaining_gaps


def _merge_llm_services(state_services: list[dict], llm_services: list[dict]) -> list[dict]:
    """Ensure pre-added services aren't dropped by LLM output. Dedup by subcategory_id."""
    existing_ids = {s.get("subcategory_id") for s in llm_services}
    for s in state_services:
        if s.get("subcategory_id") not in existing_ids:
            llm_services.append(s)
            existing_ids.add(s.get("subcategory_id"))
    return llm_services


def _format_services_context(services: list[dict], general_headings: list[str]) -> str:
    """Format services for the LLM context.

    Shows categories active (not individual services) to prevent the LLM
    from listing counts or individual services to the user.
    Evidence-matched specialist services ARE shown so the LLM can acknowledge
    what makes this business specific (e.g. solar, data cabling).
    """
    if not services:
        return "SERVICES MAPPED SO FAR: None yet"
    # Group by category name
    cat_names = list(dict.fromkeys(
        s.get("category_name", "") for s in services if s.get("category_name")
    ))
    # Highlight evidence-matched specialists — these tell the LLM what's unique about this business
    evidence_names = [s.get("subcategory_name", "") for s in services if s.get("source") == "evidence"]
    evidence_line = ""
    if evidence_names:
        evidence_line = (
            f"\nSPECIALIST SERVICES DETECTED (from reviews/website/licence): {', '.join(evidence_names)}. "
            f"Mention these naturally to show you understand the business — e.g. "
            f"\"I can see you're into {evidence_names[0].lower()}\" or similar."
        )
    return (
        f"SERVICES MAPPED SO FAR: All standard services auto-mapped under "
        f"{', '.join(cat_names) if cat_names else ', '.join(general_headings)}. "
        f"Do NOT mention service counts or list individual services to the user."
        f"{evidence_line}"
    )


def _build_service_prompt(
    state: dict, services: list[dict], gaps: list[dict],
    cluster_added: list[str],
    tiered_mode: bool, pending_cluster_ids: list,
    related_suggestions: list[dict] | None = None,
) -> tuple[str, str]:
    """Build static + dynamic context strings for the service discovery LLM call.

    Returns (static_context, dynamic_context).
    """
    business_name = state.get("business_name", "")
    licence_classes = state.get("licence_classes", [])
    licence_info = state.get("licence_info", {})
    web_results = state.get("web_results", [])
    messages = state.get("messages", [])

    # ── Gaps text ──
    gaps_text = ""
    if tiered_mode:
        gap_entries = [f"{g['subcategory_name']} (id: {g['subcategory_id']})" for g in gaps]
        if gap_entries:
            gaps_text = f"\nSPECIALIST SUBCATEGORIES TO ASK ABOUT ({len(gaps)} remaining):\n{', '.join(gap_entries)}"
        else:
            gaps_text = "\nSPECIALIST SUBCATEGORIES: None remaining — all specialists covered!"
    elif gaps:
        gap_entries = [f"{g['subcategory_name']} (id: {g['subcategory_id']})" for g in gaps]
        gaps_text = f"\nREMAINING UNCOVERED SUBCATEGORIES ({len(gaps)} remaining):\n{', '.join(gap_entries)}"
    elif services:
        gaps_text = "\nREMAINING UNCOVERED SUBCATEGORIES: None — full coverage achieved!"

    # ── Enrichment context ──
    licence_context = ""
    if licence_classes:
        source = licence_info.get("licence_source", "nsw")
        label = {
            "qbcc_csv": "QBCC LICENCE",
            "web_extracted": "LICENCE NUMBER (FROM WEBSITE — UNVERIFIED)",
            "self_reported": "LICENCE NUMBER (SELF-REPORTED — UNVERIFIED)",
        }.get(source, "NSW FAIR TRADING LICENCE")
        licence_context = f"\n{label} CLASSES: {', '.join(licence_classes)}"
        if licence_info.get("licence_number"):
            expiry = licence_info.get("expiry_date", "")
            expiry_text = f", Expiry: {expiry}" if expiry else ""
            licence_context += f"\nLicence #{licence_info['licence_number']} — Status: {licence_info.get('status', 'Unknown')}{expiry_text}"
        if licence_info.get("compliance_clean") is False:
            licence_context += "\n⚠️ Compliance issues on record"
    elif licence_info.get("_expired"):
        # Expired/cancelled licence — still useful signal for the AI
        expiry = licence_info.get("expiry_date", "")
        licence_context = (
            f"\n⚠️ EXPIRED LICENCE: {licence_info.get('licence_type', 'Trade')} licence #{licence_info.get('licence_number', '?')} "
            f"— Status: {licence_info.get('status', 'Unknown')}"
            f"{f', Expired: {expiry}' if expiry else ''}. "
            f"This tells us what trade they're in, but they may need to renew before taking jobs. "
            f"Mention this conversationally — don't block them."
        )
    elif gaps:
        licence_context = f"\nNO LICENCE ON FILE — category detected from business profile. Map all subcategories as a starting point and let the user confirm."

    web_context = ""
    if web_results:
        web_lines = [f"- {r.get('title', '')}: {r.get('url', '')}" for r in web_results[:3]]
        web_context = f"\nWEB PRESENCE:\n" + "\n".join(web_lines)

    # Data confidence summary — tells the LLM what we found AND what we didn't
    has_licence = bool(licence_classes)
    has_google = bool(state.get("google_rating") or state.get("google_reviews"))
    has_category = bool(services or gaps)
    confidence_parts = []
    if has_licence:
        confidence_parts.append("licence found")
    else:
        confidence_parts.append("no licence found")
    if has_google:
        confidence_parts.append(f"Google Places verified ({state.get('google_rating', 0)}★)")
    else:
        confidence_parts.append("no Google Places match")
    if has_category:
        confidence_parts.append("trade category detected")
    else:
        confidence_parts.append("no trade category detected")
    if web_results:
        confidence_parts.append(f"{len(web_results)} web results (may or may not be this business)")
    else:
        confidence_parts.append("no web results")
    data_confidence = f"\nDATA CONFIDENCE: {' · '.join(confidence_parts)}"
    if not has_licence and not has_google and not has_category:
        data_confidence += "\n⚠️ LOW CONFIDENCE — we know very little about this business. Do NOT make assumptions from web results. Simply ask what services they offer."

    google_reviews = state.get("google_reviews", [])
    reviews_context = ""
    if google_reviews:
        review_lines = [f"- [{r.get('rating', '?')}★] \"{r['text'][:200]}\"" for r in google_reviews[:5] if r.get("text")]
        if review_lines:
            reviews_context = f"\nGOOGLE REVIEWS ({state.get('google_rating', 0)}★, {state.get('google_review_count', 0)} reviews):\n" + "\n".join(review_lines)

    contact = state.get("contact_name", "")
    conv_history = _format_conversation(messages, max_turns=6)
    guide, guide_files = find_subcategory_guide(business_name, return_files=True)
    taxonomy = get_category_taxonomy_text()

    _trace(state, "Guides Loaded", 0,
           f"Subcategory: {', '.join(guide_files) if guide_files else 'none (full taxonomy)'} | Taxonomy: {len(taxonomy)} chars",
           {"subcategory_guides": guide_files,
            "guide_chars": len(guide),
            "taxonomy_chars": len(taxonomy),
            "has_guide": bool(guide)})

    # ── Licence acknowledgement hint for turn 1 ──
    licence_ack = ""
    if licence_info and licence_info.get("licence_number"):
        source = licence_info.get("licence_source", "nsw")
        if source == "qbcc_csv":
            licence_ack = "Mention briefly that you found their QBCC licence on file (e.g. \"I found your QBCC licence, so...\"). "
        elif source == "web_extracted":
            licence_ack = "Mention you spotted a licence number on their website and have noted it down — do NOT say it has been verified. "
        elif source == "self_reported":
            licence_ack = "Acknowledge you've noted their licence number — do NOT say it has been verified or confirmed. "
        else:
            licence_ack = "Mention briefly that you found their NSW trade licence on file. "

    # ── Related category suggestions hint ──
    suggestions_hint = ""
    if related_suggestions:
        sug_names = [s["category"] for s in related_suggestions]
        suggestions_hint = (
            f"RELATED CATEGORIES: Data shows most businesses in this field also list under "
            f"{', '.join(sug_names)} to get more leads. After confirming the category setup "
            f"and acknowledging the licence (if any), mention these extra categories naturally and "
            f"ask if they'd like to add any. Keep it casual — e.g. \"Most {sug_names[0].lower()} "
            f"businesses also list under {' and '.join(sug_names[:2])} for extra leads — want me to add any of those?\" "
            f"IMPORTANT: This turn is ONLY about related categories ({', '.join(sug_names)}). "
            f"Do NOT ask about any specialist services, gaps, or specific work types (e.g. gas fitting, solar, etc). "
            f"The category buttons will be provided separately so do NOT list them or number them in your response."
        )

    # ── Multi-category note ──
    multi_cat_note = ""
    if services:
        cat_names_in_services = list(dict.fromkeys(
            s.get("category_name", "") for s in services if s.get("category_name")
        ))
        if len(cat_names_in_services) > 1:
            multi_cat_note = (
                f"Services span {len(cat_names_in_services)} categories: "
                f"{' and '.join(cat_names_in_services)}. Mention both trades naturally. "
            )

    # ── General headings for prompt ──
    general_headings = state.get("_general_headings", [])
    if not general_headings and services:
        # Fallback: derive from category names
        general_headings = list(dict.fromkeys(
            s.get("category_name", "") for s in services if s.get("source") == "general"
        ))
    general_heading_text = " and ".join(general_headings) if general_headings else "their trade"
    general_count = sum(1 for s in services if s.get("source") == "general")

    # ── Turn 1 rule ──
    if tiered_mode:
        # Evidence-matched services to acknowledge in turn 1
        evidence_names = [s.get("subcategory_name", "") for s in services if s.get("source") == "evidence"]
        evidence_ack = ""
        if evidence_names:
            evidence_ack = (
                f"If SPECIALIST SERVICES DETECTED are listed, acknowledge them to show you understand "
                f"the business — e.g. \"I can see you specialise in {evidence_names[0].lower()}\" or "
                f"\"your reviews mention {evidence_names[0].lower()}\". This makes the user feel understood. "
            )

        if related_suggestions and suggestions_hint:
            # Turn 1 with related category suggestions — LLM handles naturally
            turn1_rule = (
                f"- TURN 1 RULE: You're setting this business up as {general_heading_text}. "
                f"Your response MUST be 2-3 SHORT sentences max. Confirm the category heading "
                f"and briefly mention the licence if found. "
                f"{evidence_ack}"
                f"FORBIDDEN: Do NOT mention service counts, list individual services, or say how many are mapped. "
                f"Include the pre-mapped services array exactly as-is in your JSON output. "
                f"{licence_ack}{multi_cat_note}"
                f"{suggestions_hint} "
                f"Set step_complete=false. cluster_ids should be empty. "
                f"Do NOT ask about specialist gaps this turn."
            )
        elif gaps:
            turn1_rule = (
                f"- TURN 1 RULE: You're setting this business up as {general_heading_text}. "
                f"Your response MUST be 2-3 SHORT sentences max. Confirm the category heading "
                f"and briefly mention the licence if found. "
                f"{evidence_ack}"
                f"FORBIDDEN: Do NOT mention service counts, list individual services, or say how many are mapped. "
                f"Include the pre-mapped services array exactly as-is in your JSON output. "
                f"{licence_ack}{multi_cat_note}"
                f"Then look at the REMAINING GAPS and pick the most relevant group of 3-6 services to ask about. "
                f"Group them by theme (e.g. 'cabling work' or 'solar and energy'). "
                f"Keep it to ONE simple question per turn."
            )
        else:
            turn1_rule = (
                f"- TURN 1 RULE: You're setting this business up as {general_heading_text} "
                f"and all specialist areas are covered. "
                f"Confirm the category heading and mention the licence if found. "
                f"{evidence_ack}"
                f"FORBIDDEN: Do NOT mention service counts or list individual services. "
                f"{licence_ack}{multi_cat_note}"
                f"Set step_complete=true."
            )
    else:
        turn1_rule = (
            "- TURN 1 RULE (FULL TAXONOMY): No tier guide exists for this trade. Use the full "
            "CATEGORY TAXONOMY to map services based on the business name, licence, and Google data. "
            "Map aggressively — most professionals do most things in their field. Add ALL subcategories "
            "from the REMAINING GAPS list using their exact IDs. Then ask a short confirmation: "
            "\"I have added all [N] services — anything you do not do that I should remove?\""
        )

    # ── Static context ──
    static_context = f"""You are the Service Seeking onboarding assistant helping a business set up their services.

GOAL: Map this business's services as completely as possible. Every missed subcategory is leads they'll never see. It's better to include a service they occasionally do than to miss one they do regularly.

SUBCATEGORY GUIDE:
{guide[:4000] if guide else "No specific guide available for this trade."}

CATEGORY TAXONOMY:
{taxonomy[:6000]}

GUIDELINES:
- This flows directly from business confirmation — the conversation is already going. Don't re-introduce yourself.
- Be conversational and Australian. Keep it short — people are busy. Say "tradie" for trade businesses, but adapt your language for professional services (photographers, designers, accountants, etc.).
- Licence classes are your strongest signal for trades — they tell you exactly what they're licensed for. Professional services may not have trade licences, and that's fine.
{turn1_rule}
- Google reviews are a strong signal — if customers mention specific work, that confirms those services. Use reviews to validate mapping.
- READ THE DATA CONFIDENCE LINE. When confidence is low (no licence, no Google, no category), web results are unreliable — they may be about a completely different person or business with the same name. In low-confidence situations, ignore web results and simply ask what services they offer.
- FOLLOW-UP RULE: Services from the user's previous answer have already been added to SERVICES MAPPED SO FAR (check JUST ADDED note). Your job: acknowledge what was added in one sentence, then look at the REMAINING GAPS and pick the most relevant group of 3-6 services to ask about next. Group them by theme — e.g. "cabling work" (Data Cabling, Cabling, TV Antenna), "solar and energy" (Solar panel installation, Energy efficiency checks). Pick high-value, commonly-offered services first. One question per turn. If no gaps remain, set step_complete=true. Your output "services" array should contain ONLY newly added services from this turn — existing services are preserved automatically.
- BUTTON RULE: Buttons are multi-select toggles — the user can tap individual services to select/deselect them, then send their selection. ALWAYS include "Yes, all of these" as the FIRST button and a clear decline as the LAST button — use "None of these" or "Not for us". In between, include one button per service in the cluster using the exact subcategory name (no "Just" prefix). NEVER use ambiguous options like "Occasionally", "Not regularly", "Sometimes", "Rarely" — we need a clear yes or no, not frequency. NEVER use "Skip these". Aim for 4-7 buttons total (all + services + decline).
- COMPLETION RULE: Set step_complete=true when there are no REMAINING GAPS left, OR the user says they are done / wants to move on. Do NOT drag it out — ask the most important gaps efficiently, then wrap up.
- FALLBACK RULE: Look at the REMAINING GAPS list. If the gaps are too numerous or too scattered across unrelated categories to cover naturally in 1-2 more questions, set "fallback_to_list": true. The system will show a multi-select checklist instead — much faster than answering questions one by one. Use this when: there are 10+ diverse gaps, OR the gaps span categories that don't relate to each other, OR you've already asked 2+ cluster questions and gaps remain. Do NOT try to force-fit unrelated services into one question — fallback is the better UX.
- The response text is a conversational summary. Keep it to 1-2 sentences: mention the total count and groups, not every service. No headers, no bullet points, no line breaks.
- Don't announce what you're doing, just do it.

CRITICAL — IDs MUST be exact:
- subcategory_id and category_id MUST come from the CATEGORY TAXONOMY or REMAINING GAPS list above. Copy the exact integer IDs shown in parentheses (e.g. "Switchboards (id: 854)" → subcategory_id: 854).
- NEVER invent or guess IDs. If you can't find an exact match in the taxonomy, omit the service.
- In the "services" array, only include NEWLY ADDED services from this turn. Do NOT repeat services already in SERVICES MAPPED SO FAR — they are preserved automatically.

Return a JSON object:
{{"response": "your conversational message", "services": [array of NEWLY ADDED services only, with input, category_name, category_id, subcategory_name, subcategory_id, confidence], "buttons": ["2-4 button options"], "cluster_ids": [subcategory_ids being asked about in this turn's question], "step_complete": true/false, "fallback_to_list": false}}

cluster_ids MUST contain the exact subcategory_id integers for every service mentioned in your question, matching the IDs from the SPECIALIST/REMAINING list. This is used to process the user's next response. If step_complete=true, cluster_ids should be empty.

Return ONLY the JSON object."""

    # ── Cluster processing context ──
    cluster_context = ""
    if cluster_added:
        cluster_context = f"\nJUST ADDED (from user's response): {', '.join(cluster_added)}. Acknowledge these briefly."
    elif pending_cluster_ids and not cluster_added:
        cluster_context = "\nUser declined the last cluster. Move on to the next one."

    # ── Dynamic context ──
    dynamic_context = f"""BUSINESS: {business_name}
{f'CONTACT: {contact}' if contact else ''}
{_format_services_context(services, general_headings)}
{licence_context}{data_confidence}
{web_context}{reviews_context}{gaps_text}{cluster_context}

CONVERSATION SO FAR:
{conv_history}"""

    return static_context, dynamic_context


async def service_discovery_node(state: OnboardingState) -> dict:
    """Discover and map services with deterministic gap tracking.

    Orchestrator: extract state → preprocess clusters → check fast-exit →
    build prompt → call LLM → parse → merge → return.
    """
    messages = state.get("messages", [])
    services = state.get("services", [])
    business_name = state.get("business_name", "")

    if state.get("services_confirmed"):
        return {
            "current_node": "service_discovery",
            "messages": [AIMessage(content="Services locked in!")],
        }

    last_msg = _get_last_human_message(messages)
    licence_classes = state.get("licence_classes", [])
    web_results = state.get("web_results", [])

    is_follow_up = bool(services)
    svc_turn = state.get("_svc_turn", 1)
    tiered_mode_check = bool(state.get("_specialist_gap_ids"))  # True if we entered tiered flow

    # ── Handle restart request: send back to business verification ──
    if last_msg == MSG_RESTART_BIZ:
        return {
            "current_node": "business_verification",
            "business_verified": False,
            "abr_results": [],
            "business_name": "",
            "abn": "",
            "services": [],
            "licence_classes": [],
            "licence_info": {},
            "google_primary_type": "",
            "messages": [AIMessage(content="No worries! What's the name of your trade business?")],
        }

    # ── Handle multi-select fallback response ──
    fallback_gaps = state.get("_fallback_gaps", [])
    if fallback_gaps and last_msg:
        if last_msg == "__FALLBACK_ALL__":
            # Add all fallback gap services
            for g in fallback_gaps:
                services.append({
                    "input": g["subcategory_name"],
                    "category_name": g.get("category_name", ""),
                    "category_id": g.get("category_id", 0),
                    "subcategory_name": g["subcategory_name"],
                    "subcategory_id": g["subcategory_id"],
                    "confidence": "user_selected",
                    "source": "fallback_select",
                })
            logger.info(f"[SVC] Fallback: user selected ALL {len(fallback_gaps)} remaining services")
        elif last_msg.startswith("__FALLBACK__:"):
            # Single button tap: parse ID
            selected_ids_raw = last_msg.replace("__FALLBACK__:", "").split(",")
            selected_ids = set()
            for raw in selected_ids_raw:
                try:
                    selected_ids.add(int(raw.strip()))
                except ValueError:
                    continue
            for g in fallback_gaps:
                if g["subcategory_id"] in selected_ids:
                    services.append({
                        "input": g["subcategory_name"],
                        "category_name": g.get("category_name", ""),
                        "category_id": g.get("category_id", 0),
                        "subcategory_name": g["subcategory_name"],
                        "subcategory_id": g["subcategory_id"],
                        "confidence": "user_selected",
                        "source": "fallback_select",
                    })
            logger.info(f"[SVC] Fallback: user selected {len(selected_ids)} of {len(fallback_gaps)} services")
        elif last_msg != "__FALLBACK_SKIP__":
            # Multi-select: comma-separated labels from input field
            selected_labels = {l.strip().lower() for l in last_msg.split(",") if l.strip()}
            matched = 0
            for g in fallback_gaps:
                if g["subcategory_name"].lower() in selected_labels:
                    services.append({
                        "input": g["subcategory_name"],
                        "category_name": g.get("category_name", ""),
                        "category_id": g.get("category_id", 0),
                        "subcategory_name": g["subcategory_name"],
                        "subcategory_id": g["subcategory_id"],
                        "confidence": "user_selected",
                        "source": "fallback_select",
                    })
                    matched += 1
            logger.info(f"[SVC] Fallback: matched {matched} services from labels: {last_msg[:200]}")
        else:
            logger.info(f"[SVC] Fallback: user skipped remaining services")

        count = len(services)
        return {
            "current_node": "service_discovery",
            "services": services,
            "services_confirmed": True,
            "_svc_turn": svc_turn,
            "_fallback_gaps": [],
            "messages": [AIMessage(content=f"All sorted — {count} services locked in! Let's move on to your service area.")],
        }

    # ── Multi-select fallback helper ──
    def _build_fallback(reason: str):
        """Show remaining gaps as multi-select UI, or confirm if none left."""
        remaining_gap_ids = state.get("_specialist_gap_ids", [])
        if not remaining_gap_ids:
            logger.info(f"[SVC] {reason} but no gaps remain — confirming")
            return {
                "current_node": "service_discovery",
                "services": services,
                "services_confirmed": True,
                "_svc_turn": svc_turn,
                "messages": [AIMessage(content="All sorted — let's move on to your service area!")],
            }
        all_gaps = compute_service_gaps(services, business_name, licence_classes,
                                       state.get("google_business_name", ""),
                                       state.get("google_primary_type", ""))
        remaining = [g for g in all_gaps if g["subcategory_id"] in remaining_gap_ids]
        if not remaining:
            logger.warning(f"[SVC] {reason}: {len(remaining_gap_ids)} gap IDs in state but none matched taxonomy — confirming")
            return {
                "current_node": "service_discovery",
                "services": services,
                "services_confirmed": True,
                "_svc_turn": svc_turn,
                "_specialist_gap_ids": [],
                "messages": [AIMessage(content="All sorted — let's move on to your service area!")],
            }
        logger.info(f"[SVC] {reason}: showing {len(remaining)} remaining gaps as multi-select (turn {svc_turn})")

        # Group gaps by category for cleaner UI
        from collections import OrderedDict
        grouped: OrderedDict[str, list] = OrderedDict()
        for g in remaining:
            cat = g.get("category_name", "Other")
            grouped.setdefault(cat, []).append(g)

        btn_list = [{"label": "Select all", "value": "__FALLBACK_ALL__"}]
        for cat_name, cat_gaps in grouped.items():
            # Category header (non-clickable, rendered as section divider)
            btn_list.append({"label": cat_name, "value": "__GROUP_HEADER__", "header": True})
            for g in cat_gaps[:8]:  # Cap per category
                btn_list.append({"label": g["subcategory_name"], "value": f"__FALLBACK__:{g['subcategory_id']}"})
        btn_list.append({"label": "None of these — move on", "value": "__FALLBACK_SKIP__"})

        return {
            "current_node": "service_discovery",
            "services": services,
            "_svc_turn": svc_turn,
            "_specialist_gap_ids": remaining_gap_ids,
            "_pending_cluster_ids": [],
            "_multiselect": True,
            "_fallback_form": True,
            "_fallback_gaps": remaining,
            "buttons": btn_list,
            "messages": [AIMessage(content=(
                "Nearly there! Here are a few specialist services grouped by category — "
                "just tap the ones you offer."
            ))],
        }

    # ── Deterministic fallback + safety cap (tiered mode) ──
    # Primary: after turn 1, if too many gaps to cover in 2 questions → immediate multi-select
    # Secondary: LLM can also trigger via `fallback_to_list` in its JSON output
    # Safety net: hard cap at turn 4 (turn 1 = confirm, turns 2-3 = gap questions, turn 4 = fallback)
    MAX_GAPS_FOR_QUESTIONS = 12  # More than this can't be covered in 2 clusters of 3-6
    SAFETY_CAP_TURN = 4
    if tiered_mode_check:
        remaining_gap_ids = state.get("_specialist_gap_ids", [])
        remaining_gap_count = len(remaining_gap_ids)
        if svc_turn > SAFETY_CAP_TURN:
            result = _build_fallback(f"Safety cap: turn {svc_turn} > {SAFETY_CAP_TURN}")
            if result:
                return result
        elif svc_turn > 1 and remaining_gap_count > MAX_GAPS_FOR_QUESTIONS:
            result = _build_fallback(f"Too many gaps ({remaining_gap_count}) for conversational questions")
            if result:
                return result

    # ── QLD ESO re-entry flag (set below if processing ESO licence response) ──
    _is_eso_reentry = False
    needs_licence = state.get("_needs_licence_number", False)

    # ── Non-trade business gate (turn 1 only, skip on ESO re-entry) ──
    # Uses LLM classification from _confirm_business (replaces brittle _NON_TRADE_TYPES keyword set)
    if svc_turn == 1 and not services and not licence_classes and not _is_eso_reentry:
        is_trade = state.get("_is_trade_business", True)
        if not is_trade:
            return {
                "current_node": "service_discovery",
                "services": [],
                "buttons": [
                    {"label": "Try a different business", "value": MSG_RESTART_BIZ},
                ],
                "messages": [AIMessage(content=(
                    f"It looks like {business_name} isn't a trade or service business based on what I found online. "
                    f"Service Seeking is for trade and service professionals — things like plumbing, electrical, building, cleaning, photography, design, and similar. "
                    f"If you have a different business that offers trade or professional services, I can help you get that one set up."
                ))],
            }

    # ── Licence holder name lookup (trusts/companies where legal name didn't match) ──
    _eso_updates: dict = {}
    if state.get("_needs_licence_holder_name") and svc_turn == 1:
        return {
            "current_node": "service_discovery",
            "_svc_turn": 2,
            "buttons": [
                {"label": "Skip for now", "value": "Skip licence"},
            ],
            "messages": [AIMessage(content=(
                f"I couldn't find a trade licence under {state.get('legal_name', 'the registered entity')} — "
                f"licences are usually registered under a person's name. "
                f"What name is the licence under? Or skip if you'd rather sort it out later."
            ))],
        }
    if state.get("_needs_licence_holder_name") and svc_turn == 2:
        _is_eso_reentry = True
        if last_msg and last_msg.lower() not in ("skip licence", "skip", "skip for now"):
            # User gave us a name — search for their licence
            holder_name = last_msg.strip()
            logger.info(f"[BIZ] Licence holder name search: '{holder_name}'")
            if business_state == "NSW":
                holder_results = await nsw_licence_browse(holder_name)
                matches = holder_results.get("results", [])
                if matches:
                    pre_detected = _detect_categories(business_name, [],
                                                      state.get("google_biz_name", ""),
                                                      state.get("google_primary_type", ""))
                    best_match, _ = match_licence(matches, holder_name,
                                                  detected_categories=pre_detected,
                                                  return_details=True)
                    if best_match:
                        lid = best_match.get("licence_id", "")
                        if lid:
                            details = await nsw_licence_details(lid)
                            if not details.get("error"):
                                classes = [c["name"] for c in details.get("classes", []) if c.get("active")]
                                if classes:
                                    state["licence_classes"] = classes
                                    state["licence_info"] = details
                                    state["contact_name"] = details.get("contact_name", holder_name)
                                    logger.info(f"[BIZ] Licence found under '{holder_name}': {classes}")
            elif business_state == "QLD":
                result = qbcc_licence_lookup("", holder_name)
                if result:
                    classes = [c["name"] for c in result.get("classes", []) if c.get("active")]
                    if classes:
                        state["licence_classes"] = classes
                        state["licence_info"] = result
                        logger.info(f"[BIZ] QBCC licence found under '{holder_name}': {classes}")
        state["_needs_licence_holder_name"] = False

    # ── Licence self-report (QLD ESO, VIC regulated trades) ──
    self_report = state.get("_licence_self_report", {})
    if needs_licence and svc_turn == 1:
        # First entry into service discovery — ask for licence number
        sr_trade = self_report.get("trade", "tradesperson")
        sr_state = self_report.get("state", "")
        sr_regulator = self_report.get("regulator", "the relevant authority")
        sr_label = self_report.get("label", "licence number")
        sr_optional = self_report.get("optional", False)

        if sr_optional:
            prompt_msg = (
                f"In {sr_state}, {sr_trade.lower()}s can optionally register through {sr_regulator}. "
                f"If you have a {sr_label}, type it in — otherwise just skip."
            )
        else:
            prompt_msg = (
                f"I couldn't find a trade licence for {business_name} — as a {sr_trade.lower()} in {sr_state} "
                f"you'd be registered through {sr_regulator}. "
                f"Do you have your {sr_label} handy? You can type it in, or skip for now."
            )
        return {
            "current_node": "service_discovery",
            "_svc_turn": 2,
            "buttons": [
                {"label": "Skip for now", "value": "Skip licence"},
            ],
            "messages": [AIMessage(content=prompt_msg)],
        }
    if needs_licence and svc_turn == 2:
        # Process the licence response, then fall through to normal turn-1 flow
        _eso_updates = {"_needs_licence_number": False, "_licence_self_report": {}}
        _is_eso_reentry = True
        sr_trade = self_report.get("trade", "Unknown")
        sr_default_classes = self_report.get("default_classes", [])
        if last_msg and not last_msg.lower().startswith(("skip", "i'll add")):
            lic_num = re.sub(r'[^A-Za-z0-9\-/ ]', '', last_msg.strip())[:30]
            _eso_updates["licence_info"] = {
                "licence_number": lic_num,
                "licence_type": sr_trade,
                "status": "Self-reported",
                "licence_source": "self_reported",
                "classes": [{"name": c, "active": True} for c in sr_default_classes],
                "compliance_clean": True,
                "associated_parties": [],
                "business_address": "",
            }
            _eso_updates["licence_classes"] = sr_default_classes
            licence_classes = sr_default_classes
            logger.info(f"[SVC] {sr_trade} licence self-reported: {lic_num}")
        else:
            logger.info(f"[SVC] {sr_trade} licence skipped")
        svc_turn = 1
        is_follow_up = False

    # ── Tiered mapping on turn 1 ──
    google_biz_name = state.get("google_business_name", "")
    google_type = state.get("google_primary_type", "")
    google_reviews = state.get("google_reviews", [])
    tiered_mode = False
    specialist_gap_ids = state.get("_specialist_gap_ids", [])
    pending_cluster_ids = state.get("_pending_cluster_ids", [])

    if svc_turn == 1 and not services:
        website_text = state.get("website_text", "")
        initial = compute_initial_services(
            business_name, licence_classes,
            google_biz_name, google_type,
            google_reviews, web_results,
            website_text=website_text,
            google_types=state.get("google_types", []),
        )
        if initial.get("tiered"):
            services = initial["services"]
            tiered_mode = True
            specialist_gap_ids = [g["subcategory_id"] for g in initial["specialist_gaps"]]
            _eso_updates["_general_headings"] = initial.get("general_headings", [])
            logger.info(f"[SVC] Tiered mapping: {len(services)} pre-mapped, {len(initial['specialist_gaps'])} specialist gaps")
            # Trace: category detection + tiered mapping breakdown
            general_headings = initial.get("general_headings", initial.get("category_names", []))
            _trace(state, "Tiered Mapping", 0,
                   f"Detected {', '.join(initial.get('category_names', []))} → {len(services)} auto-mapped, "
                   f"{len(specialist_gap_ids)} specialist gaps to ask about",
                   {"categories_detected": initial.get("category_names", []),
                    "general_headings": general_headings,
                    "auto_mapped": len(services),
                    "specialist_gaps": [g.get("subcategory_name", "") for g in initial["specialist_gaps"]]})

    # ── Category suggestion (turn 1, after initial mapping) ──
    related_suggestions = None
    if svc_turn == 1 and not state.get("_category_suggestions_shown") and services and tiered_mode:
        cat_names = initial.get("category_names", []) if tiered_mode else []
        suggestions = suggest_related_categories(cat_names)
        if suggestions:
            related_suggestions = suggestions
            sug_names = [s["category"] for s in suggestions]
            logger.info(f"[SVC] Related category suggestions available: {sug_names}")
            _trace(state, "Related Category Suggestions", 0,
                   f"Suggesting {len(suggestions)} related categories: {', '.join(sug_names)}",
                   {"primary_categories": cat_names,
                    "suggestions": suggestions,
                    "source": "co-occurrence data from related_categories.json"})

    # ── Process category suggestion response ──
    if state.get("_category_suggestions_shown") and state.get("_category_suggestions") and is_follow_up:
        suggestions = state["_category_suggestions"]
        sug_names = [s["category"] for s in suggestions]
        accepted: list[str] = []

        if last_msg == "__CAT_SKIP__" or (last_msg and last_msg.lower().startswith("skip")):
            accepted = []
        elif last_msg == MSG_YES_ALL or (last_msg and "yes" in last_msg.lower() and "all" in last_msg.lower()):
            accepted = sug_names
        elif last_msg:
            # Split comma-separated selections and match against suggestion names
            parts = [p.strip() for p in last_msg.split(",") if p.strip()]
            for part in parts:
                for s in suggestions:
                    if s["category"].lower() == part.lower() or s["category"].lower() in part.lower():
                        if s["category"] not in accepted:
                            accepted.append(s["category"])

        if accepted:
            # Build evidence text for mapping
            website_text = state.get("website_text", "")
            evidence_text = ""
            for rev in state.get("google_reviews", []):
                evidence_text += " " + rev.get("text", "")
            for wr in web_results:
                evidence_text += " " + wr.get("title", "") + " " + wr.get("description", "")
            if website_text:
                evidence_text += " " + website_text
            evidence_lower = evidence_text.lower()

            existing_mapped = {s.get("subcategory_name", "") for s in services}
            new_svcs, new_gaps, _ = map_extra_categories(
                accepted, services, existing_mapped, evidence_lower, licence_classes,
            )
            services.extend(new_svcs)
            specialist_gap_ids.extend(g["subcategory_id"] for g in new_gaps)
            logger.info(f"[SVC] Accepted extra categories {accepted}: +{len(new_svcs)} services, +{len(new_gaps)} gaps")
            _trace(state, "Extra Category Mapping", 0,
                   f"Accepted {', '.join(accepted)}: +{len(new_svcs)} services, +{len(new_gaps)} gaps",
                   {"accepted": accepted,
                    "new_services": [s.get("subcategory_name", "") for s in new_svcs],
                    "new_gaps": [g.get("subcategory_name", "") for g in new_gaps]})
        else:
            logger.info(f"[SVC] Category suggestions skipped")

        # Clear suggestion state, reset to turn-1 flow
        _eso_updates["_category_suggestions_shown"] = False
        _eso_updates["_category_suggestions"] = []
        svc_turn = 1
        is_follow_up = False

        # Immediate fallback if extra categories created too many gaps to ask about
        if len(specialist_gap_ids) > MAX_GAPS_FOR_QUESTIONS:
            # Update state with new services before falling back
            state["_specialist_gap_ids"] = specialist_gap_ids
            state["services"] = services
            result = _build_fallback(f"Post-category acceptance: {len(specialist_gap_ids)} gaps too many for questions")
            if result:
                result["services"] = services
                result["_specialist_gap_ids"] = specialist_gap_ids
                result.update(_eso_updates)
                return result

    # ── Compute remaining gaps deterministically ──
    if tiered_mode:
        gaps = initial["specialist_gaps"]
    elif specialist_gap_ids:
        tiered_mode = True
        all_gaps = compute_service_gaps(services, business_name, licence_classes,
                                        google_biz_name, google_type)
        gaps = [g for g in all_gaps if g["subcategory_id"] in specialist_gap_ids]
    else:
        gaps = compute_service_gaps(services, business_name, licence_classes,
                                    google_biz_name, google_type)

    if is_follow_up:
        _trace(state, "Service Gaps", 0,
               f"{len(gaps)} gaps remaining ({len(services)} services mapped)",
               {"gap_count": len(gaps), "service_count": len(services),
                "gaps": [g.get("subcategory_name", "") for g in gaps[:15]],
                "tiered_mode": tiered_mode})

    # ── Pre-process cluster response (deterministic) ──
    cluster_added: list[str] = []
    if pending_cluster_ids and is_follow_up and gaps:
        pre_svc_count = len(services)
        services, cluster_added, gaps = _process_cluster_response(
            pending_cluster_ids, gaps, services, last_msg,
        )
        action = "added" if cluster_added else "skipped"
        _trace(state, "Cluster Processing", 0,
               f"Cluster {action}: +{len(services) - pre_svc_count} services, {len(gaps)} gaps left",
               {"action": action,
                "pending_cluster_ids": pending_cluster_ids,
                "user_message": (last_msg or "")[:200],
                "services_added": cluster_added,
                "gaps_remaining": len(gaps)})

    # ── Fast-exit: no specialist gaps left in tiered mode — skip LLM ──
    if tiered_mode and not gaps and is_follow_up:
        count = len(services)
        added_text = f" Added {', '.join(cluster_added)}." if cluster_added else ""
        logger.info(f"[SVC] All specialist gaps covered ({count} services) — confirming without LLM")
        return {
            "current_node": "service_discovery",
            "services": services,
            "services_raw": last_msg or f"Inferred from: {business_name}",
            "services_confirmed": True,
            "_svc_turn": svc_turn + 1,
            "_specialist_gap_ids": specialist_gap_ids,
            "_pending_cluster_ids": [],
            "messages": [AIMessage(content=f"All sorted — {count} services locked in!{added_text} Let's move on to your service area.")],
            **_eso_updates,
        }

    # If gaps don't match what the user is describing, clear them
    if gaps and not services and svc_turn >= 3:
        logger.info(f"[SVC] Clearing unhelpful gaps (0 services after {svc_turn} turns)")
        gaps = []

    # ── Build prompt ──
    static_context, dynamic_context = _build_service_prompt(
        state, services, gaps, cluster_added,
        tiered_mode, pending_cluster_ids,
        related_suggestions=related_suggestions,
    )

    # ── Single LLM call ──
    t_llm = time.time()
    response = await llm_fast_json.ainvoke([
        SystemMessage(content=[
            {"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": dynamic_context},
        ]),
        HumanMessage(content=last_msg or "Let's set up my services"),
    ])
    llm_time = time.time() - t_llm

    # ── Parse response ──
    try:
        raw_json = _extract_json(response.content)
        if not raw_json or raw_json[0] != "{":
            logger.info(f"[SVC] LLM returned plain text, wrapping: {response.content[:200]}")
            data = {
                "response": response.content,
                "services": services,
                "buttons": [],
                "step_complete": False,
            }
        else:
            data = json.loads(raw_json)
        llm_services = data.get("services", [])
        message = data.get("response", "What services do you offer?")
        buttons = data.get("buttons", [])
        step_complete = data.get("step_complete", False)
        cluster_ids = data.get("cluster_ids", [])
        fallback_to_list = data.get("fallback_to_list", False)

        # LLM requested multi-select fallback — show remaining gaps as checklist
        if fallback_to_list and tiered_mode:
            logger.info(f"[SVC] LLM requested fallback_to_list (turn {svc_turn}, {len(specialist_gap_ids)} gaps)")
            # Merge any services the LLM added in this same response before falling back
            merged = _merge_llm_services(services, data.get("services", []))
            result = _build_fallback(f"LLM fallback_to_list (turn {svc_turn})")
            if result:
                result["services"] = merged
                return result

        # Merge LLM output (new services only) with existing services
        new_services = _merge_llm_services(services, llm_services)

        # Compute post-turn gaps for logging
        new_gaps = compute_service_gaps(new_services, business_name, licence_classes,
                                        google_biz_name, google_type)

        logger.info(f"[SVC] user='{last_msg}' | mapped={len(new_services)} | gaps={len(new_gaps)} | complete={step_complete} | tiered={tiered_mode} | cluster_ids={cluster_ids}")
        _trace(state, "LLM: Service Discovery", llm_time,
               f"Mapped {len(new_services)} services, {len(new_gaps)} gaps remaining, complete={step_complete}, tiered={tiered_mode}",
               {"services_count": len(new_services), "gaps_remaining": len(new_gaps),
                "step_complete": step_complete, "turn": svc_turn, "tiered": tiered_mode,
                "prompt_context": dynamic_context[:800],
                "user_message": (last_msg or "")[:200],
                "llm_response": (response.content or "")[:1000],
                "cluster_ids": cluster_ids,
                "buttons": buttons[:6]})

        # Build button list — mark as multi-select when there are cluster toggles
        _DECLINE_WORDS = {"none", "not", "nah", "skip", "move on"}
        _ALL_WORDS = {"yes", "all of these", "all of"}
        btn_list = [{"label": b, "value": b} for b in buttons]
        has_toggles = len(btn_list) >= 3 and cluster_ids and not step_complete
        multiselect = False
        if has_toggles:
            # Tag action buttons so frontend renders them as normal single-tap sends
            for btn in btn_list:
                lower = btn["label"].lower()
                if any(w in lower for w in _DECLINE_WORDS):
                    btn["value"] = "__CLUSTER_SKIP__"
                elif any(w in lower for w in _ALL_WORDS):
                    btn["value"] = MSG_YES_ALL
            multiselect = True

        # ── Override buttons for related category suggestions ──
        if related_suggestions:
            btn_list = [{"label": "Yes, all of these", "value": "Yes, all of these"}]
            for s in related_suggestions:
                btn_list.append({"label": s["category"], "value": s["category"]})
            btn_list.append({"label": "None of these", "value": "__CAT_SKIP__"})
            multiselect = True
            step_complete = False
            cluster_ids = []
            _eso_updates["_category_suggestions_shown"] = True
            _eso_updates["_category_suggestions"] = related_suggestions

        result = {
            "current_node": "service_discovery",
            "services": new_services,
            "services_raw": last_msg or f"Inferred from: {business_name}",
            "services_confirmed": step_complete,
            "_svc_turn": svc_turn if related_suggestions else svc_turn + 1,
            "_specialist_gap_ids": specialist_gap_ids,
            "_pending_cluster_ids": cluster_ids,
            "buttons": btn_list,
            "messages": [AIMessage(content=message)],
            **_eso_updates,
        }
        if multiselect:
            result["_multiselect"] = True
        return result
    except Exception as e:
        logger.error(f"[SVC] {e} | raw response: {response.content[:300]}")
        # Preserve existing services, try to extract text from the failed response
        fallback_text = ""
        if response and response.content:
            # Try to extract the "response" field even if JSON is broken
            import re as _re
            m = _re.search(r'"response"\s*:\s*"([^"]+)"', response.content)
            if m:
                fallback_text = m.group(1)
        if not fallback_text:
            count = len(services)
            fallback_text = f"I've got {count} services mapped so far. What else does {business_name} offer?" if services else f"What services does {business_name} offer? Just tell me in your own words."
        return {
            "current_node": "service_discovery",
            "services": services,
            "_svc_turn": svc_turn + 1,
            "_specialist_gap_ids": specialist_gap_ids,
            "_pending_cluster_ids": [],
            "messages": [AIMessage(content=fallback_text)],
            **_eso_updates,
        }


async def service_area_node(state: OnboardingState) -> dict:
    """Map service areas through natural, geography-aware conversation.

    NO SCRIPTING. The LLM gets goals, regional guides, and suburb data.
    It figures out the conversation — asks about barriers, corridors,
    congestion zones relevant to THIS business's location.

    Model selection:
    - Turn 1 → Haiku (regional guide provides intelligence, prompt is prescriptive)
    - Turn 2+ → Haiku with trimmed prompt (just set include/exclude from selection)
    """
    messages = state.get("messages", [])
    service_areas = state.get("service_areas", {})
    business_name = state.get("business_name", "")
    postcode = state.get("business_postcode", "")
    business_state = state.get("business_state", "NSW")

    if state.get("service_areas_confirmed"):
        return {
            "current_node": "service_area",
            "messages": [AIMessage(content="Service areas confirmed!")],
        }

    # When auto-chained from service confirmation, the last human message belongs
    # to service discovery (e.g. a decline response) — ignore it so we present a fresh area question.
    if state.get("_auto_chained"):
        last_msg = None
    else:
        last_msg = _get_last_human_message(messages)

    # Gather geographic context
    grouped = get_suburbs_in_radius_grouped(postcode, 20.0) if postcode else {}

    # Build region summary with suburb counts
    # Filter out regions with <3 suburbs — those are misclassified stray entries in the CSV
    region_list = ""
    valid_regions = 0
    total_suburbs = 0
    if grouped.get("by_area"):
        region_lines = []
        for area, suburbs in sorted(grouped["by_area"].items(), key=lambda x: len(x[1]), reverse=True):
            if len(suburbs) < 3:
                continue
            valid_regions += 1
            total_suburbs += len(suburbs)
            sample = ", ".join([s["name"] for s in suburbs[:3]])
            region_lines.append(f"  - {area} ({len(suburbs)} suburbs, e.g. {sample})")
        region_list = "\n".join(region_lines)
    if grouped:
        _trace(state, "Suburb Grouping", 0,
               f"{valid_regions} regions, {total_suburbs} suburbs within 20km of {postcode}",
               {"postcode": postcode, "regions": valid_regions, "total_suburbs": total_suburbs})

    # ── Extract location evidence from website text + Google reviews ──
    location_evidence = ""
    if not service_areas.get("base_suburb"):  # Only on turn 1
        valid_regions = set()
        if grouped.get("by_area"):
            valid_regions = {area for area, suburbs in grouped["by_area"].items() if len(suburbs) >= 3}

        if valid_regions:
            evidence_sources = []
            website_text = state.get("website_text", "")
            google_reviews = state.get("google_reviews", [])

            # Scan website text for region/suburb mentions
            if website_text:
                wt_lower = website_text.lower()
                matched_regions = [r for r in valid_regions if r.lower() in wt_lower]
                # Also check for suburb names within each region
                for area, suburbs in grouped["by_area"].items():
                    if area in matched_regions or len(suburbs) < 3:
                        continue
                    for s in suburbs:
                        if s["name"].lower() in wt_lower:
                            matched_regions.append(area)
                            break
                if matched_regions:
                    evidence_sources.append(f"Website mentions: {', '.join(sorted(set(matched_regions)))}")

            # Scan Google reviews for region/suburb mentions
            if google_reviews:
                review_text = " ".join(r.get("text", "") for r in google_reviews[:5]).lower()
                review_regions = set()
                for area, suburbs in grouped["by_area"].items():
                    if len(suburbs) < 3:
                        continue
                    if area.lower() in review_text:
                        review_regions.add(area)
                        continue
                    for s in suburbs:
                        if s["name"].lower() in review_text:
                            review_regions.add(area)
                            break
                if review_regions:
                    evidence_sources.append(f"Google reviews mention: {', '.join(sorted(review_regions))}")

            if evidence_sources:
                location_evidence = "\nLOCATION EVIDENCE (from business website/reviews):\n" + "\n".join(f"  - {e}" for e in evidence_sources)
                logger.info(f"[AREA] Location evidence: {evidence_sources}")

    # Turn 1 always populates base_suburb, so its presence means we're on turn 2+
    is_follow_up = bool(service_areas.get("base_suburb"))
    model = llm_fast_json
    model_name = MODEL_FAST

    # ── TURN 2+: Trimmed prompt — just update include/exclude ──
    if is_follow_up:
        # Collect all region names from grouped data for exclude list
        all_regions = [area for area, suburbs in (grouped.get("by_area", {})).items() if len(suburbs) >= 3]

        prompt = f"""You are finalizing a business's service area on Service Seeking.

BASE: {grouped.get("base_suburb", "Unknown")} ({postcode})
ALL REGIONS WITHIN 20KM: {json.dumps(all_regions)}
CURRENT SELECTION: included={json.dumps(service_areas.get("regions_included", []))}, excluded={json.dumps(service_areas.get("regions_excluded", []))}

USER SAID: "{last_msg}"

Update regions_included based on the user's response. IMPORTANT:
- If they say "add", "also", "as well", "plus", or mention extra regions → ADD those to the CURRENT included list (keep existing regions)
- If they select a button or give a complete list → replace the included list
- If they confirm or say "looks good" → keep current selection as-is
- The base region ({grouped.get("base_suburb", "")}'s region) should always be included unless they explicitly exclude it
Set regions_excluded to ALL regions from the list above that are NOT in regions_included.
Set step_complete = true.

Return a JSON object:
{{"response": "Locked in! (or brief confirmation)", "service_areas": {{"base_suburb": "{grouped.get('base_suburb', '')}", "base_postcode": "{postcode}", "base_lat": {grouped.get('base_lat', 0)}, "base_lng": {grouped.get('base_lng', 0)}, "radius_km": 20, "regions_included": ["all included regions"], "regions_excluded": ["all other regions"], "barriers": [], "travel_notes": ""}}, "buttons": [], "step_complete": true}}

Return ONLY the JSON object."""

        t_llm = time.time()
        response = await model.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_msg or "Looks good"),
        ])
        llm_time = time.time() - t_llm

    # ── TURN 1: Full prompt with regional guide ──
    else:
        regional_guide = get_regional_guide(business_state)
        _state_city_map = {"NSW": "sydney", "VIC": "melbourne", "QLD": "brisbane", "WA": "perth"}
        _trace(state, "Regional Guide Loaded", 0,
               f"{'Loaded ' + _state_city_map.get(business_state, '?') + '_regions.md' if regional_guide else 'No regional guide for ' + business_state} ({len(regional_guide)} chars)",
               {"state": business_state, "guide_file": _state_city_map.get(business_state, "none") + "_regions.md" if regional_guide else "none",
                "chars": len(regional_guide)})

        static_context = f"""You are the Service Seeking onboarding assistant helping a business define their service area.

GOAL: Figure out which REGIONS this business covers. Real coverage isn't a perfect circle — it's a blob shaped by traffic, barriers, and preferences. Your job is to identify which regions they include and which they exclude.

REGIONS WITHIN 20KM OF BASE:
{region_list or "No region data available"}
Total: {grouped.get("total", 0)} suburbs across these regions
{location_evidence}
REGIONAL GUIDE (barriers, congestion, corridors):
{regional_guide[:4000] if regional_guide else "No regional guide available."}

GUIDELINES:
- This step flows directly from the service confirmation — the conversation is already going. Don't re-introduce yourself or rehash what you already know. The user's last message was about services, not areas.
- Be conversational and Australian. Keep it short — people are busy.
- Talk in terms of regions/areas, not individual suburbs
- Use the regional guide to understand natural boundaries, but don't explain the geography — they already know it. Just ask which areas they cover.
- If LOCATION EVIDENCE is provided, use it to inform your button suggestions — the business already advertises in those areas, so they're likely inclusions. You can reference what you've seen (e.g. "Your website mentions Northern Beaches — do you cover there too?") but keep it brief.
- Present the nearby regions and ask which ones they cover. One simple question, no preamble. Offer buttons for likely groupings.
- When they tell you their areas, lock it in and move on. Two turns max — be decisive, don't ask for confirmation of what they just told you.
- Keep your total response under 2 sentences.

BUTTON RULE: Each button must represent a COMPLETE coverage selection — listing ALL regions included.
- Good: "Eastern Suburbs + North Shore only", "Eastern Suburbs + North Shore + Northern Beaches", "All of Sydney"
- Bad: "Add Inner West" or "Add Northern Beaches too" (unclear what the total coverage is)
- Pattern: tight option (just evidence-based regions), medium (those + 1-2 nearby), wide (whole metro)
- 3-4 buttons max. The user should know EXACTLY which regions they're selecting from the label alone.

Return a JSON object:
{{"response": "your conversational message", "service_areas": {{"base_suburb": "", "base_postcode": "", "base_lat": 0, "base_lng": 0, "radius_km": 20, "regions_included": ["region names they cover"], "regions_excluded": ["region names within radius they don't cover"], "barriers": ["relevant barriers from regional guide"], "travel_notes": "brief note on coverage shape"}}, "buttons": ["3-4 complete coverage options"], "step_complete": true/false}}

Use REAL region names from the grouped data above for regions_included and regions_excluded.
step_complete = true when the user has indicated which regions they cover.

Return ONLY the JSON object."""

        contact = state.get("contact_name", "")
        dynamic_context = f"""BUSINESS: {business_name}
{f'CONTACT: {contact}' if contact else ''}
BASE SUBURB: {grouped.get("base_suburb", "Unknown")} ({postcode})
STATE: {business_state}
CURRENT SERVICE AREA: Not set yet"""

        t_llm = time.time()
        response = await model.ainvoke([
            SystemMessage(content=[
                {"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic_context},
            ]),
            HumanMessage(content=last_msg or (
                "Include all the areas from our website and reviews as a starting point — I'll adjust if needed"
                if state.get("_auto_chained") and location_evidence
                else "Let's set up my service area"
            )),
        ])
        llm_time = time.time() - t_llm

    # ── Parse response (shared by both paths) ──
    try:
        data = json.loads(_extract_json(response.content))
        new_areas = data.get("service_areas", service_areas)
        message = data.get("response", "Where do you typically work?")
        buttons = data.get("buttons", [])
        step_complete = data.get("step_complete", False)

        # Fill in base coordinates from grouped data if LLM didn't
        if not new_areas.get("base_lat") and grouped:
            new_areas["base_suburb"] = grouped.get("base_suburb", "")
            new_areas["base_postcode"] = grouped.get("base_postcode", postcode)
            new_areas["base_lat"] = grouped.get("base_lat", 0)
            new_areas["base_lng"] = grouped.get("base_lng", 0)

        included = new_areas.get("regions_included", [])
        excluded = new_areas.get("regions_excluded", [])
        logger.info(f"[AREA] user='{last_msg}' | model={model_name} | included={included} | excluded={excluded} | complete={step_complete}")
        _area_prompt = dynamic_context[:800] if not is_follow_up else prompt[:800]
        _trace(state, "LLM: Service Area", llm_time,
               f"{len(included)} regions included, {len(excluded)} excluded, complete={step_complete}",
               {"model": model_name, "regions_included": included,
                "regions_excluded": excluded, "step_complete": step_complete, "follow_up": is_follow_up,
                "prompt_context": _area_prompt,
                "user_message": (last_msg or "")[:200],
                "llm_response": (response.content or "")[:1000]})

        return {
            "current_node": "service_area",
            "location_raw": last_msg or f"Based on {postcode}",
            "service_areas": new_areas,
            "service_areas_confirmed": step_complete,
            "buttons": [{"label": b, "value": b} for b in buttons],
            "messages": [AIMessage(content=message)],
        }
    except Exception as e:
        logger.error(f"[AREA] {e}")
        base = grouped.get("base_suburb", "your area")
        return {
            "current_node": "service_area",
            "messages": [AIMessage(content=f"Where do you typically work? You're based in {base} — do you mainly work locally or travel further afield?")],
        }


def _compute_years_in_business(state: dict) -> int:
    """Compute years in business from licence start_date or ABR registration date."""
    from datetime import datetime

    # Try licence start date first (more reliable for tradies)
    licence_info = state.get("licence_info", {})
    date_str = licence_info.get("start_date", "")

    # Fall back to ABR registration date
    if not date_str:
        date_str = state.get("abn_registration_date", "")

    if not date_str:
        return 0

    try:
        # Handle common date formats: "2015-01-15", "15/01/2015", "2015"
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y"):
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                years = datetime.now().year - dt.year
                return max(0, years)
            except ValueError:
                continue
        return 0
    except Exception:
        return 0


async def profile_node(state: OnboardingState) -> dict:
    """Generate a profile preview with LLM description and years in business.

    First call (auto-chained from confirmation): generate description draft.
    Save call (__SAVE_PROFILE__: prefix): store edited description, mark saved.
    """
    messages = state.get("messages", [])
    last_msg = _get_last_human_message(messages)

    # ── Save call: user clicked Save Profile ──
    if last_msg and last_msg.startswith(MSG_SAVE_PROFILE):
        payload = last_msg[len(MSG_SAVE_PROFILE):].strip()
        result = {
            "current_node": "profile",
            "profile_saved": True,
        }

        # Parse JSON payload with description + optional edits
        try:
            data = json.loads(payload)
            result["profile_description"] = data.get("description", "")

            # Handle service removals
            removed_svcs = data.get("removed_services", [])
            if removed_svcs:
                services = state.get("services", [])
                result["services"] = [s for s in services if s.get("subcategory_name", s.get("input", "")) not in removed_svcs]

            # Handle area removals
            removed_areas = data.get("removed_areas", [])
            if removed_areas:
                service_areas = dict(state.get("service_areas", {}))
                service_areas["regions_included"] = [r for r in service_areas.get("regions_included", []) if r not in removed_areas]
                service_areas["regions_excluded"] = service_areas.get("regions_excluded", []) + removed_areas
                result["service_areas"] = service_areas
        except (json.JSONDecodeError, AttributeError):
            # Backwards compat: plain text description
            result["profile_description"] = payload

        return result

    # ── Question handler: if draft exists and user typed something (not a save), answer it ──
    if state.get("profile_description_draft") and last_msg:
        business_name = state.get("business_name", "")
        contact_name = state.get("contact_name", "")
        response = await llm_fast.ainvoke([
            SystemMessage(content=f"""You are the Service Seeking onboarding assistant. The user is viewing their profile preview and has typed a question or comment instead of publishing.

BUSINESS: {business_name}
{f'CONTACT: {contact_name}' if contact_name else ''}

Answer their question helpfully and briefly (1-3 sentences). Be conversational and Australian.
- If they ask about pricing/plans: mention there are a few tiers depending on reach, and you'll walk them through it after publishing.
- If they ask about editing: mention they can edit the description, remove services or areas right on the preview.
- If they ask about what happens next: after publishing, you'll sort out their plan and they'll be live.
- If they want to change something about the description: suggest they edit it directly on the profile card.
- Keep it short, don't over-explain.
- End with a natural nudge back to the profile — something like "Want to jump back to your profile?" or "Ready to check out your profile again?" Keep it casual."""),
            HumanMessage(content=last_msg),
        ])
        return {
            "current_node": "profile",
            "_profile_question": True,
            "messages": [AIMessage(content=response.content)],
            "buttons": ["Back to my profile"],
        }

    # ── Idempotency: if draft already exists and no message, return early ──
    if state.get("profile_description_draft"):
        return {
            "current_node": "profile",
        }

    # ── No-website gate: search for social profiles, ask user to verify ──
    google_website = state.get("business_website", "")
    if not google_website and not state.get("_website_asked"):
        business_name = state.get("business_name", "")
        business_state = state.get("business_state", "")

        # Search Brave for social profiles
        social_results = await brave_web_search(
            f"{business_name} site:instagram.com OR site:facebook.com", count=5)

        # Filter: require distinctive name words + at least one other name word in URL path or title
        _GENERIC_TRADE_WORDS = {
            "painter", "painters", "painting", "plumber", "plumbing", "electrician", "electrical",
            "carpenter", "carpentry", "builder", "builders", "building", "cleaner", "cleaners",
            "cleaning", "gardener", "gardening", "landscaping", "handyman", "roofer", "roofing",
            "services", "service", "solutions",
        }
        name_words = {w.lower() for w in re.split(r'[\s\-&]+', business_name) if len(w) >= 2}
        name_words -= {"pty", "ltd", "limited", "the", "and", "group", "co"}
        distinctive = name_words - _GENERIC_TRADE_WORDS
        other_words = name_words - distinctive

        candidates = []
        seen_domains = set()
        for sr in social_results:
            url = sr.get("url", "")
            title = sr.get("title", "")
            is_fb = "facebook.com" in url
            is_ig = "instagram.com" in url
            if not is_fb and not is_ig:
                continue
            # Deduplicate by platform
            platform = "facebook" if is_fb else "instagram"
            if platform in seen_domains:
                continue
            # Check name match in URL path and title
            url_path = url.split(".com/")[-1].lower() if ".com/" in url else ""
            check_text = url_path + " " + title.lower()
            has_distinctive = any(w in check_text for w in distinctive) if distinctive else False
            has_other = any(w in check_text for w in other_words) if other_words else True
            if has_distinctive and has_other:
                seen_domains.add(platform)
                label = f"Instagram: @{url_path.strip('/')}" if is_ig and "/" not in url_path.strip("/") else f"{'Facebook' if is_fb else 'Instagram'}: {title}"
                candidates.append({"url": url, "label": label, "platform": platform})

        if candidates:
            logger.info(f"[PROFILE] Social search found {len(candidates)} candidates: {[c['url'] for c in candidates]}")
        else:
            logger.info(f"[PROFILE] Social search: no matching profiles for '{business_name}'")

        buttons = []
        if candidates:
            # Offer candidates as buttons for user confirmation
            for c in candidates:
                buttons.append({"label": f"Yes — {c['label']}", "value": c["url"]})
            buttons.append({"label": "No, I'll paste my link", "value": "__PASTE_LINK__"})
            buttons.append({"label": "Skip", "value": "__SKIP_WEBSITE__"})
            candidate_list = "\n".join(f"- **{c['label']}**" for c in candidates)
            msg = (
                f"I found these social profiles — are any of them yours?\n\n"
                f"{candidate_list}\n\n"
                f"If not, you can paste your own link below."
            )
        else:
            # Nothing found — ask for their link
            buttons.append({"label": "Skip", "value": "__SKIP_WEBSITE__"})
            msg = (
                f"I couldn't find a website or social page for {business_name}. "
                f"Got a website, Facebook, or Instagram I can grab your logo and photos from?\n\n"
                f"Just paste the link below, or skip and you can upload photos yourself."
            )

        return {
            "current_node": "profile",
            "_website_asked": True,
            "buttons": buttons,
            "messages": [AIMessage(content=msg)],
        }

    # ── Handle website/social URL response ──
    if state.get("_website_asked") and not state.get("_website_url_processed") and last_msg:
        if last_msg == "__PASTE_LINK__":
            # User said "no that's not mine" — ask them to paste their own link
            return {
                "current_node": "profile",
                "buttons": [{"label": "Skip", "value": "__SKIP_WEBSITE__"}],
                "messages": [AIMessage(content="No worries! Paste your website, Facebook, or Instagram link below — or skip to upload photos yourself.")],
            }
        if last_msg != "__SKIP_WEBSITE__":
            # Extract URL from message
            url_match = re.search(r'https?://[^\s<>"]+', last_msg)
            if url_match:
                user_url = url_match.group(0)
                logger.info(f"[PROFILE] User provided URL: {user_url}")
                is_social = "facebook.com" in user_url or "instagram.com" in user_url
                if is_social:
                    social_result = await scrape_social_images([user_url])
                    if social_result.get("logo"):
                        state["business_website"] = user_url
                        state["_user_social_logo"] = social_result["logo"]
                        state["_user_social_photos"] = social_result.get("photos", [])
                        logger.info(f"[PROFILE] Social scrape: logo={'yes' if social_result.get('logo') else 'no'}, {len(social_result.get('photos', []))} photos")
                else:
                    scraped_user = await scrape_website_images(user_url)
                    if scraped_user.get("logo") or scraped_user.get("photos"):
                        state["business_website"] = user_url
                        state["_user_website_scraped"] = scraped_user
                        logger.info(f"[PROFILE] Website scrape: logo={'yes' if scraped_user.get('logo') else 'no'}, {len(scraped_user.get('photos', []))} photos")
        state["_website_url_processed"] = True

    # ── First call: compute years, generate description, scrape website images ──
    years = _compute_years_in_business(state)

    business_name = state.get("business_name", "")
    licence_classes = state.get("licence_classes", [])
    services = state.get("services", [])
    service_areas = state.get("service_areas", {})
    web_results = state.get("web_results", [])
    contact_name = state.get("contact_name", "")

    services_text = ", ".join([s.get("subcategory_name", s.get("input", "")) for s in services])
    regions = service_areas.get("regions_included", [])
    regions_text = ", ".join(regions) if regions else "local area"

    web_context = ""
    if web_results:
        web_lines = [f"- {r.get('title', '')}: {r.get('description', '')}" for r in web_results[:3]]
        web_context = "\nWEB RESULTS:\n" + "\n".join(web_lines)

    # Add scraped website text for richer descriptions
    website_text = state.get("website_text", "")
    if website_text:
        web_context += f"\nBUSINESS WEBSITE TEXT:\n{website_text[:2000]}"

    google_rating = state.get("google_rating", 0.0)
    google_review_count = state.get("google_review_count", 0)
    rating_context = ""
    if google_rating:
        rating_context = f"\nGOOGLE RATING: {google_rating}/5 ({google_review_count} reviews)"

    # ── Run LLM description + intro in parallel with scrape ──
    llm_task = llm_fast_json.ainvoke([
        SystemMessage(content=f"""You're helping a business set up their Service Seeking profile. Do two things:

1. Write a short "intro" message (1 sentence) presenting their profile preview and inviting them to make changes. Speak as one person (not "we"), conversational and Australian, use their first name if known. Like "Here's your profile [name] — let me know if you want to change anything." Vary the wording naturally. Don't give specific UI instructions.
2. Write a "description" (2-3 sentences) for their profile listing.

BUSINESS: {business_name}
{f'OWNER: {contact_name}' if contact_name else ''}
{f'LICENCE CLASSES: {", ".join(licence_classes)}' if licence_classes else ''}
SERVICES: {services_text}
AREAS: {regions_text}
{f'YEARS IN BUSINESS: {years}' if years else ''}
{web_context}{rating_context}

DESCRIPTION GUIDELINES:
- Third person, professional but warm Australian tone
- Mention key services, areas covered, and years of experience if available
- If Google rating is available and strong (4+), mention it briefly (e.g. "highly rated")
- Keep under 500 characters — punchy and specific, not generic
- Don't mention ABN, licence numbers, or compliance details
- If BUSINESS WEBSITE TEXT is provided, use it to pick out specific details (specialties, taglines, unique selling points) — don't just repeat it, distil the best bits
- Focus on what makes this business worth hiring

Return JSON: {{"intro": "...", "description": "..."}}"""),
        HumanMessage(content="Generate the intro and profile description."),
    ])

    # Separate web results into: business site (only if verified), junk
    scrape_url = ""
    _directory_domains = [
        "yelp.com", "yellowpages", "truelocal", "hipages.com", "oneflare.com",
        "airtasker.com", "serviceseeking.com", "productreview.com", "localsearch.com",
        "hotfrog.com", "startlocal.com", "word-of-mouth.com", "mylocaltrades",
        ".gov.au", "wikipedia.org", "linkedin.com", "gumtree.com", "sgpgrid.com",
        "housetohomepros.com", "finditnowdirectory.com", "businessified.com",
        "companydirectory.com", "facebook.com", "instagram.com",
    ]
    # Build distinctive name words (exclude generic trade terms that match unrelated businesses)
    _GENERIC_TRADE_WORDS = {
        "painter", "painters", "painting", "plumber", "plumbing", "electrician", "electrical",
        "carpenter", "carpentry", "builder", "builders", "building", "cleaner", "cleaners",
        "cleaning", "gardener", "gardening", "landscaping", "landscaper", "handyman",
        "roofer", "roofing", "concreter", "concreting", "tiler", "tiling", "fencing",
        "services", "service", "solutions", "group", "company", "pro", "pros",
    }
    _bname_words = {w.lower() for w in re.split(r'[\s\-&]+', business_name) if len(w) >= 2}
    _bname_words -= {"pty", "ltd", "limited", "the", "and", "group", "co"}
    _distinctive_words = _bname_words - _GENERIC_TRADE_WORDS
    for wr in web_results:
        url = wr.get("url", "")
        if not scrape_url and not any(d in url for d in _directory_domains):
            # Only use Brave result if title or URL contains distinctive business name words
            title_lower = wr.get("title", "").lower()
            url_lower = url.lower()
            if _distinctive_words and any(w in title_lower or w in url_lower for w in _distinctive_words):
                scrape_url = url
            else:
                logger.info(f"[PROFILE] Skipping Brave URL (no distinctive name match): {url}")

    # Google Places website is the most reliable source — use it as primary scrape URL
    google_website = state.get("business_website", "")
    google_is_social = google_website and ("facebook.com" in google_website or "instagram.com" in google_website)
    if google_website and not google_is_social:
        scrape_url = google_website
        logger.info(f"[PROFILE] Using Google Places website: {google_website}")
    elif google_is_social:
        logger.info(f"[PROFILE] Google Places website is social: {google_website}")

    # ── Run everything in parallel: LLM + domain discovery (fallback) + scrape + social ──
    t0 = time.time()
    async def _noop(): return ""
    async def _noop_dict(): return {"logo": "", "photos": []}
    tasks = {
        "llm": llm_task,
        "discover": discover_business_website(business_name) if not google_website else _noop(),
        "scrape": scrape_website_images(scrape_url) if scrape_url else _noop_dict(),
        "social": scrape_social_images([google_website]) if google_is_social else _noop_dict(),
    }
    results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values())))

    response = results["llm"]
    discovered_url = results["discover"] if not google_website else ""
    brave_scraped = results["scrape"]
    google_social_result = results.get("social", {"logo": "", "photos": []})

    if discovered_url:
        _trace(state, "Website Discovery", time.time() - t0,
               f"Discovered: {discovered_url}",
               {"url": discovered_url, "business_name": business_name})
    elif not google_website:
        _trace(state, "Website Discovery", time.time() - t0,
               "No website found (tried .com.au/.au/.net.au)", {})

    # If we have a Google website, we already scraped it directly above
    # If not, discovered domain takes priority over Brave result scrape
    if google_website:
        scraped = brave_scraped  # Actually the Google website scrape (scrape_url was set to google_website)
    elif discovered_url:
        scraped = await scrape_website_images(discovered_url)
        scrape_url = discovered_url
        if not scraped.get("logo") and not scraped.get("photos"):
            scraped = brave_scraped
    else:
        scraped = brave_scraped

    llm_time = time.time() - t0

    # Parse LLM response — JSON with intro + description
    raw = response.content.strip()
    # Strip markdown code fences if present (```json ... ```)
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    logger.info(f"[PROFILE] Raw LLM response: {raw[:200]}")
    try:
        parsed = json.loads(raw)
        intro = parsed.get("intro", "")
        description = parsed.get("description", "")
        logger.info(f"[PROFILE] Parsed intro: {intro[:80]}")
        logger.info(f"[PROFILE] Parsed desc: {description[:80]}")
    except json.JSONDecodeError:
        # Fallback: treat entire response as description
        intro = ""
        description = raw.strip('"')
        logger.info(f"[PROFILE] JSON parse failed, using raw as description")

    # ── Merge images: Google Places > verified website > user-provided URL only ──
    logo = scraped.get("logo", "")

    # Google Business photos are highest quality — real work photos from owner/customers
    google_photos = state.get("google_photos", [])
    photos = list(google_photos[:8])
    if google_photos:
        logger.info(f"[PROFILE] {len(google_photos)} Google Business photos available")

    # Layer in website scraped images (only from verified sources: Google website or domain discovery)
    for p in scraped.get("photos", []):
        if p not in photos and p != logo and len(photos) < 8:
            photos.append(p)

    # Layer in Google Places social result (when website is a Facebook/Instagram page)
    if google_is_social and google_social_result:
        if not logo and google_social_result.get("logo"):
            logo = google_social_result["logo"]
            logger.info(f"[PROFILE] Using Google-verified social logo from {google_website}")
        for p in google_social_result.get("photos", []):
            if p not in photos and p != logo and len(photos) < 8:
                photos.append(p)

    # Layer in user-provided URL results (social or website they gave us)
    user_social_logo = state.get("_user_social_logo", "")
    user_social_photos = state.get("_user_social_photos", [])
    user_website_scraped = state.get("_user_website_scraped", {})
    if user_social_logo and not logo:
        logo = user_social_logo
    for p in user_social_photos:
        if p not in photos and p != logo and len(photos) < 8:
            photos.append(p)
    if user_website_scraped:
        if not logo and user_website_scraped.get("logo"):
            logo = user_website_scraped["logo"]
        for p in user_website_scraped.get("photos", []):
            if p not in photos and p != logo and len(photos) < 8:
                photos.append(p)

    if not logo and not photos:
        logger.info(f"[PROFILE] No verified logo or photos found — monogram will be shown")

    _trace(state, "LLM: Profile Description", llm_time,
           f"Generated {len(description)} char description",
           {"years_in_business": years,
            "prompt_inputs": {"business": business_name, "contact": contact_name,
                              "services": services_text[:300], "areas": regions_text[:200],
                              "years": years, "google_rating": google_rating},
            "llm_response": {"intro": intro[:300], "description": description[:500]}})
    if scrape_url:
        _trace(state, "Website Scrape", llm_time,
               f"logo={'yes' if logo else 'no'}, {len(scraped.get('photos', []))} photos from site, {len(photos)} total",
               {"url": scrape_url, "logo": bool(logo), "site_photos": len(scraped.get("photos", [])),
                "brave_thumbs": len(photos) - len(scraped.get("photos", []))})

    # ── AI filter: use vision to keep only real work photos ──
    pre_filter_count = len(photos)
    logger.info(f"[PROFILE] {pre_filter_count} candidate photos before AI filter: {[u[:60] for u in photos]}")
    if photos:
        trade_type = ""
        if services:
            cats = list(dict.fromkeys(s.get("category_name", "") for s in services if s.get("category_name")))
            trade_type = cats[0].lower() if cats else "tradesperson"
        t_filter = time.time()
        photos = await ai_filter_photos(photos[:8], trade_type or "tradesperson")
        _trace(state, "AI Photo Filter", time.time() - t_filter,
               f"{pre_filter_count} candidates → {len(photos)} kept (Haiku vision WORK/SKIP)",
               {"before": pre_filter_count, "after": len(photos), "trade_type": trade_type})

    return {
        "current_node": "profile",
        "years_in_business": years,
        "profile_description": description,
        "profile_description_draft": description,
        "profile_intro": intro,
        "profile_logo": logo,
        "profile_photos": photos[:6],
        "messages": [AIMessage(content=intro or description)],
    }


PRICING_PLANS = {
    "standard": {"coverage": "10km", "monthly": 49, "quarterly": 118, "annual": 349,
                  "monthly_equiv_q": 39, "monthly_equiv_a": 29},
    "plus":     {"coverage": "20km", "monthly": 79, "quarterly": 190, "annual": 569,
                  "monthly_equiv_q": 63, "monthly_equiv_a": 47},
    "pro":      {"coverage": "50km", "monthly": 119, "quarterly": 286, "annual": 859,
                  "monthly_equiv_q": 95, "monthly_equiv_a": 72},
}


async def pricing_node(state: OnboardingState) -> dict:
    """Present subscription options after profile publish.

    Turn 1 (auto-chained): recommend plan based on region count, show buttons.
    Turn 2: user selected plan → ask billing frequency (or skip).
    Turn 3: user selected billing → set subscription, done.
    """
    messages = state.get("messages", [])
    last_msg = _get_last_human_message(messages)

    service_areas = state.get("service_areas", {})
    regions = service_areas.get("regions_included", [])
    region_count = len(regions)
    regions_text = ", ".join(regions) if regions else "your area"

    # ── Turn 1: Recommend a plan (no LLM call) ──
    if not state.get("pricing_shown"):
        # Pick recommendation based on region count
        if region_count <= 2:
            rec = "standard"
        elif region_count <= 5:
            rec = "plus"
        else:
            rec = "pro"

        plan = PRICING_PLANS[rec]
        rec_label = rec.capitalize()

        message = (
            f"Your profile is live! Based on your coverage in {regions_text}, "
            f"I'd recommend the **{rec_label}** plan at ${plan['monthly']}/mo — "
            f"covers up to {plan['coverage']} and gets you all leads in your area. "
            f"No commissions, no per-lead fees."
        )

        # Build buttons: recommended first, then others, then skip
        button_order = [rec] + [p for p in ["standard", "plus", "pro"] if p != rec]
        buttons = []
        for p in button_order:
            info = PRICING_PLANS[p]
            label = f"{p.capitalize()} — ${info['monthly']}/mo"
            if p == rec:
                label += " (Recommended)"
            buttons.append({"label": label, "value": f"{MSG_PLAN}{p}"})
        buttons.append({"label": "Not ready yet — skip", "value": f"{MSG_PLAN}skip"})

        return {
            "current_node": "pricing",
            "pricing_shown": True,
            "buttons": buttons,
            "messages": [AIMessage(content=message)],
        }

    # ── Turn 3: User selected billing (plan already chosen on prev turn) ──
    selected_plan = state.get("_selected_plan", "")
    if selected_plan:
        billing = ""
        if last_msg and MSG_BILLING in last_msg:
            billing = last_msg.split(MSG_BILLING)[1].strip().lower()
        elif last_msg:
            lower = last_msg.lower()
            for b in ["annual", "quarterly", "monthly"]:
                if b in lower:
                    billing = b
                    break

        if billing and selected_plan in PRICING_PLANS:
            plan = PRICING_PLANS[selected_plan]
            if billing == "monthly":
                price = f"${plan['monthly']}/mo"
            elif billing == "quarterly":
                price = f"${plan['quarterly']}/qtr"
            else:
                price = f"${plan['annual']}/yr"

            return {
                "current_node": "pricing",
                "subscription_plan": selected_plan,
                "subscription_billing": billing,
                "subscription_price": price,
                "messages": [AIMessage(content=f"Locked in — {selected_plan.capitalize()} plan, {billing}. Let's get you set up!")],
            }

        # Couldn't parse billing — re-show billing buttons
        plan = PRICING_PLANS[selected_plan]
        buttons = [
            {"label": f"Monthly — ${plan['monthly']}/mo", "value": f"{MSG_BILLING}monthly"},
            {"label": f"Quarterly — ${plan['quarterly']} (save 20%)", "value": f"{MSG_BILLING}quarterly"},
            {"label": f"Annual — ${plan['annual']} (save 40%)", "value": f"{MSG_BILLING}annual"},
        ]
        return {
            "current_node": "pricing",
            "buttons": buttons,
            "messages": [AIMessage(content="How would you like to pay — monthly, quarterly, or annual?")],
        }

    # ── Turn 2: User selected a plan ──
    selected = ""
    if last_msg and MSG_PLAN in last_msg:
        selected = last_msg.split(MSG_PLAN)[1].strip().lower()
    elif last_msg:
        lower = last_msg.lower()
        if "skip" in lower or "not ready" in lower:
            selected = "skip"
        else:
            for p in ["standard", "plus", "pro"]:
                if p in lower:
                    selected = p
                    break

    if selected == "skip":
        return {
            "current_node": "pricing",
            "subscription_plan": "skip",
            "messages": [AIMessage(content="No worries — you can always upgrade later from your dashboard.")],
        }

    if selected in PRICING_PLANS:
        plan = PRICING_PLANS[selected]
        label = selected.capitalize()
        buttons = [
            {"label": f"Monthly — ${plan['monthly']}/mo", "value": f"{MSG_BILLING}monthly"},
            {"label": f"Quarterly — ${plan['quarterly']} (save 20%)", "value": f"{MSG_BILLING}quarterly"},
            {"label": f"Annual — ${plan['annual']} (save 40%)", "value": f"{MSG_BILLING}annual"},
        ]
        return {
            "current_node": "pricing",
            "_selected_plan": selected,
            "buttons": buttons,
            "messages": [AIMessage(content=f"Great choice! The {label} plan it is. How would you like to pay?")],
        }

    # Couldn't parse — re-show plan buttons
    buttons = []
    for p in ["standard", "plus", "pro"]:
        info = PRICING_PLANS[p]
        buttons.append({"label": f"{p.capitalize()} — ${info['monthly']}/mo", "value": f"{MSG_PLAN}{p}"})
    buttons.append({"label": "Not ready yet — skip", "value": f"{MSG_PLAN}skip"})
    return {
        "current_node": "pricing",
        "buttons": buttons,
        "messages": [AIMessage(content="Which plan works best for you?")],
    }


async def complete_node(state: OnboardingState) -> dict:
    """Generate final JSON output."""
    services_output = []
    for s in state.get("services", []):
        cat_name = s.get("category_name", "")
        existing = next((x for x in services_output if x["category"] == cat_name), None)
        if existing:
            existing["subcategories"].append({
                "name": s.get("subcategory_name", ""),
                "id": s.get("subcategory_id", 0),
            })
        else:
            services_output.append({
                "category": cat_name,
                "category_id": s.get("category_id", 0),
                "subcategories": [{
                    "name": s.get("subcategory_name", ""),
                    "id": s.get("subcategory_id", 0),
                }],
            })

    # Build clean service area output — compact region-based descriptor
    service_areas = state.get("service_areas", {})

    licence_info = state.get("licence_info", {})
    output = {
        "business_name": state.get("business_name", ""),
        "abn": state.get("abn", ""),
        "entity_type": state.get("entity_type", ""),
        "gst_registered": state.get("gst_registered", False),
        "licence": {
            "number": licence_info.get("licence_number", ""),
            "type": licence_info.get("licence_type", ""),
            "status": licence_info.get("status", ""),
            "expiry": licence_info.get("expiry_date", ""),
            "classes": state.get("licence_classes", []),
        } if licence_info else None,
        "services": services_output,
        "service_areas": {
            "base": {
                "suburb": service_areas.get("base_suburb", ""),
                "postcode": service_areas.get("base_postcode", state.get("business_postcode", "")),
                "lat": service_areas.get("base_lat", 0),
                "lng": service_areas.get("base_lng", 0),
            },
            "radius_km": service_areas.get("radius_km", 20),
            "regions_included": service_areas.get("regions_included", []),
            "regions_excluded": service_areas.get("regions_excluded", []),
            "barriers": service_areas.get("barriers", []),
            "travel_notes": service_areas.get("travel_notes", ""),
        },
        "contact_name": state.get("contact_name", ""),
        "contact_phone": state.get("contact_phone", ""),
        "profile": {
            "description": state.get("profile_description", ""),
            "years_in_business": state.get("years_in_business"),
            "logo": state.get("profile_logo", ""),
            "photos": state.get("profile_photos", []),
        },
        "subscription": {
            "plan": state.get("subscription_plan", ""),
            "billing": state.get("subscription_billing", ""),
            "price": state.get("subscription_price", ""),
        } if state.get("subscription_plan") and state.get("subscription_plan") != "skip" else None,
    }

    return {
        "current_node": "complete",
        "output_json": output,
        "confirmed": True,
        "profile_saved": True,
        "messages": [AIMessage(content="You're all set! Your profile is ready on Service Seeking — you'll start getting matched with relevant jobs in your area soon. Welcome aboard!")],
    }


# ────────── HELPERS ──────────

def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences — handle ```json ... ``` blocks
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    # Find first { and match its closing } using brace depth counting
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Fallback: first { to last }
    end = text.rfind("}")
    if end > start:
        return text[start:end + 1]
    return text


async def _confirm_business(abr: dict, state: dict) -> dict:
    """Confirm a business and enrich with licence data + web search."""
    business_name = abr.get("display_name", state.get("business_name_input", ""))
    legal_name = abr.get("legal_name", "") or business_name
    abn = abr.get("abn", "")
    postcode = abr.get("postcode", "")
    business_state = abr.get("state", "")

    # Gate: inactive/cancelled ABN
    abn_status = abr.get("status", "Active")
    if abn_status and abn_status != "Active":
        logger.warning(f"[BIZ] ABN {abn} status is '{abn_status}' — blocking onboarding")
        return {
            "current_node": "business_verification",
            "abr_results": [],
            "business_verified": False,
            "messages": [AIMessage(content=(
                f"It looks like the ABN for {business_name} is currently **{abn_status.lower()}** on the Australian Business Register. "
                f"You'll need an active ABN to join Service Seeking. "
                f"If you have a different business with an active ABN, I can help you set that one up."
            ))],
            "buttons": [
                {"label": "Try a different business", "value": MSG_RESTART_BIZ},
            ],
        }

    # Run licence + Brave + Google Places in parallel
    # Licence search: use legal_name (entity registration name) — that's what NSW Fair Trading has
    licence_search_name = legal_name

    # Route licence lookup by state
    async def _qbcc_as_browse():
        """Wrap sync QBCC CSV lookup to match nsw_licence_browse return shape."""
        result = qbcc_licence_lookup(abn, legal_name)
        if result:
            return {"results": [result], "count": 1}
        return {"results": [], "count": 0}

    t0 = time.time()
    if business_state == "QLD":
        licence_task = _qbcc_as_browse()
    elif business_state == "NSW":
        licence_task = nsw_licence_browse(licence_search_name)
    else:
        # WA, VIC, SA, TAS, ACT, NT — no licence API, rely on web extraction + self-report
        async def _no_licence():
            return {"results": [], "count": 0}
        licence_task = _no_licence()

    brave_location = f"{postcode} {business_state}" if postcode else business_state
    web_task = brave_web_search(f"{business_name} {brave_location} Australia")
    google_query_suffix = f"{postcode} {business_state}" if postcode else business_state
    google_task = google_places_search(business_name, google_query_suffix)
    licence_results, web_results, google_place = await asyncio.gather(
        licence_task, web_task, google_task,
    )
    t1 = time.time()
    logger.info(f"[BIZ] Licence + Brave + Google Places: {t1 - t0:.1f}s (parallel)")

    # Google Places retry: if no result, try with legal name / Pty Ltd variant
    # e.g. "Ssr Painters" → "SSR Painters Pty Ltd" (Google listing often uses the company name)
    if not google_place:
        retry_names = []
        if legal_name and legal_name.lower() != business_name.lower():
            retry_names.append(legal_name.title() if legal_name.isupper() else legal_name)
        if not any("pty" in n.lower() for n in [business_name] + retry_names):
            retry_names.append(f"{business_name} Pty Ltd")
        for retry_name in retry_names:
            google_place = await google_places_search(retry_name, google_query_suffix)
            if google_place:
                logger.info(f"[BIZ] Google Places retry found result with '{retry_name}'")
                break
        if not google_place:
            logger.info(f"[BIZ] Google Places: no result after retries")

    # NSW: if no results and name has apostrophe, retry without it
    search_name = licence_search_name
    if business_state == "NSW" and not licence_results.get("results") and ("'" in search_name or "\u2019" in search_name):
        search_name = search_name.replace("'", "").replace("\u2019", "")
        licence_results = await nsw_licence_browse(search_name)
        logger.info(f"[BIZ] Licence retry without apostrophe: {len(licence_results.get('results', []))} results")

    # Licence fallback for trusts/companies: legal name (e.g. "KENDOBAY PTY. LIMITED") won't
    # match a personal-name licence. Don't try trading name — too risky for false positives
    # (e.g. "Petes Plumbing" matches "Pete's Precise Plumbing Pty Ltd"). Instead, ask the user.
    _ENTITY_PATTERNS = re.compile(r'\b(pty|ltd|limited|trust|trustee|holdings|group)\b', re.IGNORECASE)
    if business_state in ("NSW", "QLD") and not licence_results.get("results") and _ENTITY_PATTERNS.search(legal_name):
        logger.info(f"[BIZ] Legal name '{legal_name}' is a company/trust — will ask user for licence holder name")
        state["_needs_licence_holder_name"] = True

    licence_source_label = {"QLD": "QBCC CSV", "NSW": "NSW Licence Browse"}.get(business_state, "")
    if licence_source_label:
        _trace(state, licence_source_label, t1 - t0,
               f"{len(licence_results.get('results', []))} licence matches for '{search_name}'",
               {"results": [
                   {"licensee": r.get("licensee"), "status": r.get("status"),
                    "licence_number": r.get("licence_number")}
                   for r in licence_results.get("results", [])[:5]
               ]})
    _trace(state, "Brave Web Search", t1 - t0,
           f"{len(web_results) if web_results else 0} web results",
           {"results": [
               {"title": r.get("title"), "url": r.get("url")}
               for r in (web_results or [])[:3]
           ]})

    # Validate Google Place is in the right state (common name businesses return wrong locations)
    _STATE_NAMES = {"NSW": "New South Wales", "VIC": "Victoria", "QLD": "Queensland",
                    "WA": "Western Australia", "SA": "South Australia", "TAS": "Tasmania",
                    "ACT": "Australian Capital Territory", "NT": "Northern Territory"}
    google_address = google_place.get("address", "")
    if google_place and business_state and google_address:
        state_name = _STATE_NAMES.get(business_state, business_state)
        if business_state not in google_address and state_name not in google_address:
            logger.warning(f"[GOOGLE] Place '{google_place.get('name', '')}' is in '{google_address}' — expected {business_state}. Discarding wrong-state result.")
            google_place = {}

    # Quality check: discard low-confidence Google results (wrong name + no useful data)
    if google_place:
        g_name = google_place.get("name", "").lower().strip()
        b_name = business_name.lower().strip()
        g_reviews = google_place.get("review_count", 0) or 0
        g_website = google_place.get("website", "")
        # Check name similarity: do the core words overlap?
        g_words = set(re.sub(r'[^\w\s]', '', g_name).split())
        b_words = set(re.sub(r'[^\w\s]', '', b_name).split())
        _STOP_WORDS = {"pty", "ltd", "limited", "the", "and", "of", "services", "service", "group", "australia"}
        g_words -= _STOP_WORDS
        b_words -= _STOP_WORDS
        common = g_words & b_words
        # Require meaningful overlap: at least 50% of the shorter name's words must match
        # (1 word out of 3 is not enough — "Smith Broughton Auctioneers" ≠ "Broughton Construction Company")
        if g_words and b_words:
            min_words = min(len(g_words), len(b_words))
            name_overlap = len(common) >= max(1, (min_words + 1) // 2)  # ceil(50%)
        else:
            name_overlap = g_name in b_name or b_name in g_name
        if not name_overlap:
            logger.warning(f"[GOOGLE] Name mismatch: '{google_place.get('name', '')}' doesn't match '{business_name}' (overlap: {len(common)}/{min_words if g_words and b_words else 0} words). Discarding.")
            google_place = {}

    # Discard Google Places data if business is listed as permanently closed
    # ABN is the source of truth — if it's Active, proceed. Just don't use stale Google data.
    google_status = google_place.get("business_status", "")
    if google_status == "CLOSED_PERMANENTLY":
        logger.info(f"[GOOGLE] '{google_place.get('name', '')}' is CLOSED_PERMANENTLY — discarding Google data (ABN is active)")
        google_place = {}

    # Extract Google rating + reviews from Places API
    google_rating = google_place.get("rating", 0.0)
    google_review_count = google_place.get("review_count", 0)
    google_reviews = google_place.get("reviews", [])

    _trace(state, "Google Places", t1 - t0,
           f"{google_rating}★ ({google_review_count} reviews)" if google_rating else "not found",
           {"name": google_place.get("name", ""), "rating": google_rating,
            "review_count": google_review_count, "website": google_place.get("website", ""),
            "reviews": len(google_reviews)})

    google_photos = google_place.get("photos", [])
    if google_rating:
        logger.info(f"[BIZ] Google: {google_rating}★ ({google_review_count} reviews), {len(google_reviews)} review snippets, {len(google_photos)} photos")

    # Pre-detect trade categories (needed for licence matching trade relevance)
    google_type = google_place.get("primary_type", "")
    google_types = google_place.get("types", [])
    pre_detected = _detect_categories(business_name, [],
                                      google_place.get("name", ""), google_type,
                                      google_types=google_types)
    if pre_detected:
        _trace(state, "Category Pre-Detection", 0,
               f"Pre-detected: {', '.join(pre_detected)} (before licence)",
               {"categories": pre_detected,
                "inputs": {"business_name": business_name,
                           "google_name": google_place.get("name", ""),
                           "google_type": google_type}})

    # Find the best licence match (current, matching name closely)
    licence_info = {}
    licence_classes = []
    licence_matches = licence_results.get("results", [])

    # Scrape website text for evidence keywords (runs in parallel with licence details)
    google_website = google_place.get("website", "")
    website_text_task = scrape_website_text(google_website) if google_website else None

    if business_state == "QLD" and licence_matches:
        # QBCC CSV lookup already returns the pre-matched result — no details call needed
        licence_info = licence_matches[0]
        licence_classes = [c["name"] for c in licence_info.get("classes", []) if c.get("active")]
        logger.info(f"[BIZ] QBCC licence #{licence_info.get('licence_number')} classes: {licence_classes}")
        _trace(state, "QBCC Licence", t1 - t0,
               f"Licence #{licence_info.get('licence_number')} — {', '.join(licence_classes)}",
               {"licence_number": licence_info.get("licence_number"),
                "status": licence_info.get("status"),
                "classes": licence_classes})
        # Await website text if started
        website_text = await website_text_task if website_text_task else ""
    else:
        # NSW (or other states with browse-style results): find best match then get details
        best_match, match_details = match_licence(licence_matches, search_name,
                                                  detected_categories=pre_detected,
                                                  return_details=True)
        _trace(state, "Licence Matching", 0,
               f"{'Matched: ' + match_details.get('winner', '?') if best_match else 'No match'} — {match_details.get('reason', '')}",
               match_details)

        if best_match:
            lid = best_match.get("licence_id", "")
            if lid:
                t2 = time.time()
                # Run licence details + website text scrape in parallel
                if website_text_task:
                    details, website_text = await asyncio.gather(
                        nsw_licence_details(lid), website_text_task,
                    )
                    website_text_task = None  # consumed
                else:
                    details = await nsw_licence_details(lid)
                    website_text = ""
                t3 = time.time()
                logger.info(f"[BIZ] Licence details + website scrape: {t3 - t2:.1f}s (parallel)")
                if not details.get("error"):
                    licence_info = details
                    licence_classes = [
                        c["name"] for c in details.get("classes", []) if c.get("active")
                    ]
                    logger.info(f"[BIZ] Licence #{details.get('licence_number')} classes: {licence_classes}")
                    _trace(state, "NSW Licence Details", t3 - t2,
                           f"Licence #{details.get('licence_number')} — {', '.join(licence_classes)}",
                           {"licence_number": details.get("licence_number"),
                            "status": details.get("status"),
                            "expiry": details.get("expiry_date"),
                            "classes": licence_classes})

        # Use expired licence for category signal when no current match found
        expired_match = match_details.get("expired_match")
        if not best_match and expired_match:
            expired_type = expired_match.get("licence_type", "")
            expired_status = expired_match.get("status", "")
            expired_expiry = expired_match.get("expiry_date", "")
            expired_number = expired_match.get("licence_number", "")
            logger.info(f"[BIZ] Expired licence found: #{expired_number} ({expired_type}) — {expired_status}, expired {expired_expiry}")
            # Store expired licence info so the AI can mention it
            licence_info = {
                "licence_number": expired_number,
                "licence_type": expired_type,
                "status": expired_status,
                "expiry_date": expired_expiry,
                "licence_source": "nsw",
                "_expired": True,
            }
            # Extract trade category from licence type (e.g. "Contractor Licence" → "Builder")
            _LICENCE_TYPE_TO_CATEGORY = {
                "contractor": "Builder", "builder": "Builder",
                "electrician": "Electrician", "electrical": "Electrician",
                "plumber": "Plumber", "plumbing": "Plumber",
                "trade": "Builder",  # generic trade licence
            }
            for kw, cat in _LICENCE_TYPE_TO_CATEGORY.items():
                if kw in expired_type.lower():
                    if cat not in pre_detected:
                        pre_detected.append(cat)
                    break
            _trace(state, "Expired Licence", 0,
                   f"#{expired_number} ({expired_type}) — {expired_status}, expired {expired_expiry}",
                   expired_match)

        # Await website text if it wasn't consumed in the licence parallel block
        if website_text_task:
            website_text = await website_text_task
        elif not best_match:
            website_text = await scrape_website_text(google_website) if google_website else ""
        # website_text is now set in all paths (licence+parallel, no-licence, no-website)

    # Extract contact person from licence associated parties
    contact_name = ""
    if licence_info:
        parties = licence_info.get("associated_parties", [])
        for p in parties:
            if p.get("party_type") == "Individual" and p.get("role") in ("Director", "Nominated Supervisor", "Partner", "Sole Trader"):
                contact_name = p.get("name", "")
                break

    # Extract phone: prefer Google Places, fall back to Brave regex
    contact_phone = google_place.get("phone", "")
    if not contact_phone and web_results:
        for r in web_results[:3]:
            desc = r.get("description", "")
            phone_match = re.search(r'(?:1[38]00\s?\d{3}\s?\d{3}|0[2-8]\d{2}\s?\d{3}\s?\d{3}|\(0[2-8]\)\s?\d{4}\s?\d{4})', desc)
            if phone_match:
                contact_phone = phone_match.group(0).strip()
                break

    if contact_name:
        logger.info(f"[BIZ] Contact person: {contact_name}")
    if contact_phone:
        logger.info(f"[BIZ] Contact phone: {contact_phone}")

    # Re-detect categories: keyword matching for name + licence (high signal),
    # then LLM classification for web presence (replaces brittle keyword scanning of website text)
    detected_categories = _detect_categories(business_name, licence_classes,
                                             "", "",  # skip Google/website keywords — LLM handles those
                                             max_categories=0)

    # LLM classification of web presence (Google + website text)
    # Only call if we have web data to classify AND keyword detection didn't find enough
    llm_categories = []
    is_trade = True
    if google_place or website_text:
        t_classify = time.time()
        classification = await classify_business_from_web(
            business_name,
            google_name=google_place.get("name", ""),
            google_type=google_type,
            google_reviews=google_reviews,
            website_text=website_text,
        )
        classify_time = time.time() - t_classify
        is_trade = classification.get("is_trade", True)
        llm_categories = classification.get("categories", [])
        _trace(state, "LLM Business Classifier", classify_time,
               f"{'Trade' if is_trade else 'NOT TRADE'}: {', '.join(llm_categories) if llm_categories else 'no categories'} — {classification.get('reason', '')}",
               {"is_trade": is_trade, "categories": llm_categories, "reason": classification.get("reason", "")})

        # Merge LLM categories with keyword-detected ones (deduplicated, keyword-detected first)
        seen = set(detected_categories)
        for cat in llm_categories:
            if cat in _SS_CATEGORIES and cat not in seen:
                detected_categories.append(cat)
                seen.add(cat)

    detected_category = detected_categories[0] if detected_categories else None
    if detected_categories:
        _trace(state, "Category Detection", 0,
               f"Detected: {', '.join(detected_categories)}",
               {"categories": detected_categories,
                "sources": {"keyword": _detect_categories(business_name, licence_classes, "", ""),
                            "llm": llm_categories}})

    # Web extraction: try to extract licence from Brave descriptions, then full website scan
    # Applies to any state/trade combo with patterns in _STATE_LICENCE_CONFIG
    # (NSW/QLD have entries for pest control + air con only — building/plumbing/electrical use API/CSV)
    if not licence_info and detected_categories:
        brave_text = " ".join(r.get("description", "") for r in (web_results or [])[:3])
        for try_cat in detected_categories:
            state_config = get_licence_config(business_state, try_cat)
            if not state_config:
                continue
            # Quick check: Brave search descriptions (already fetched, no extra HTTP call)
            web_licence = extract_licence_from_text(brave_text, try_cat, business_state)
            # Full website scan: licence numbers often buried deep in footers past the 5k text limit
            if not web_licence and google_website:
                t_scan = time.time()
                web_licence = await scan_website_for_licence(google_website, try_cat, business_state)
                scan_time = time.time() - t_scan
                if web_licence:
                    logger.info(f"[BIZ] {business_state} licence found via full website scan: #{web_licence['licence_number']} ({scan_time:.1f}s)")
            if web_licence:
                licence_info = web_licence
                licence_classes = [c["name"] for c in web_licence.get("classes", []) if c.get("active")]
                detected_category = try_cat
                logger.info(f"[BIZ] {business_state} licence extracted: #{web_licence.get('licence_number')} ({try_cat})")
                _trace(state, f"{business_state} Web Extraction", 0,
                       f"Extracted {try_cat} licence #{web_licence['licence_number']} from website/Brave text",
                       {"licence_number": web_licence["licence_number"], "trade": try_cat})
                break

    # Self-report routing: QLD ESO or any state with regulated trades without licence
    needs_licence_number = False
    licence_self_report = {}

    if not licence_info and detected_categories:
        # Try first detected category that has a licence config (one self-report is enough)
        for try_cat in detected_categories:
            if business_state == "QLD" and try_cat == "Electrician":
                # QLD electricians are licensed via ESO (not QBCC)
                needs_licence_number = True
                licence_self_report = {
                    "regulator": "Electrical Safety Office (ESO)",
                    "label": "ESO licence number",
                    "trade": "Electrician",
                    "state": "QLD",
                    "optional": False,
                    "default_classes": ["Electrical Work"],
                }
                break
            config = get_licence_config(business_state, try_cat)
            if config:
                needs_licence_number = True
                licence_self_report = {
                    "regulator": config["regulator"],
                    "label": config["label"],
                    "trade": try_cat,
                    "state": business_state,
                    "optional": config.get("optional", False),
                    "default_classes": config["default_classes"],
                }
                break

    if website_text:
        _trace(state, "Website Text Scrape", 0,
               f"{len(website_text)} chars from {google_website or 'website'}",
               {"url": google_website, "chars": len(website_text), "preview": website_text[:200]})

    return {
        "current_node": "business_verification",
        "business_name": business_name,
        "legal_name": legal_name,
        "abn": abn,
        "entity_type": abr.get("entity_type", ""),
        "gst_registered": abr.get("gst_registered", False),
        "business_postcode": postcode,
        "business_state": business_state,
        "business_verified": True,
        "licence_info": licence_info,
        "licence_classes": licence_classes,
        "_needs_licence_number": needs_licence_number,
        "_licence_self_report": licence_self_report,
        "web_results": web_results[:3] if web_results else [],
        "website_text": website_text if website_text else "",
        "contact_name": contact_name,
        "contact_phone": contact_phone,
        "google_rating": google_rating,
        "google_review_count": google_review_count,
        "google_reviews": google_reviews,
        "business_website": google_place.get("website", ""),
        "google_business_name": google_place.get("name", ""),
        "google_primary_type": google_place.get("primary_type", ""),
        "google_types": google_place.get("types", []),
        "google_photos": google_place.get("photos", []),
        "business_suburb": google_place.get("suburb", ""),
        "google_address": google_place.get("short_address", "") or google_place.get("address", ""),
        "abn_registration_date": abr.get("entity_start_date", ""),
        "_is_trade_business": is_trade,
        "_detected_categories": detected_categories,
        "messages": [AIMessage(content=f"Great, {business_name} is confirmed!")],
    }


def _get_last_human_message(messages: list) -> str | None:
    """Get the content of the last human message."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _format_conversation(messages: list, max_turns: int = 6) -> str:
    """Format recent conversation history for LLM context."""
    recent = messages[-(max_turns * 2):] if len(messages) > max_turns * 2 else messages
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content[:200]}")
    return "\n".join(lines) if lines else "(No conversation yet)"


def _format_abr_results(results: dict, search_term: str) -> str:
    """Format ABR results for display."""
    entries = [r for r in results.get("results", []) if r.get("status", "Active") == "Active"]

    if not entries:
        return f"I couldn't find a business matching '{search_term}' on the ABR. Could you try a different name, or enter your ABN directly?"

    # All results shown as cards — buttons handle the detail display
    label = "match" if len(entries) == 1 else f"{len(entries)} matches"
    msg = f"I found {label} for '{search_term}'. Which one is yours?"
    if len(entries) >= 5:
        msg += f" If you don't see your business, try adding your postcode (e.g. \"{search_term} 2088\")."
    return msg

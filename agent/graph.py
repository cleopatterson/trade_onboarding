"""LangGraph state machine for the Trade Onboarding wizard

CORE PRINCIPLE: No scripting.
Give the LLM context, goals, and guides — let it figure out the conversation.
No keyword matching, no canned responses, no rigid flows.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agent.state import OnboardingState
from agent.config import ANTHROPIC_API_KEY, MODEL_SMART, MODEL_FAST
from agent.tools import (
    abr_lookup, get_category_taxonomy_text,
    search_suburbs_by_postcode,
    get_suburbs_in_radius_grouped, get_regional_guide,
    find_subcategory_guide,
    nsw_licence_browse, nsw_licence_details,
    brave_web_search,
)


# ────────── MODELS ──────────

llm = ChatAnthropic(
    model=MODEL_SMART,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=2048,
    temperature=0.3,
)

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


# ────────── NODE FUNCTIONS ──────────

async def welcome_node(state: OnboardingState) -> dict:
    """Greet the user and ask for business name/ABN."""
    response = await llm_fast.ainvoke([
        SystemMessage(content="""You are the Service Seeking onboarding assistant. You help Australian tradies get set up on the platform.
You are warm, friendly, and speak in natural Australian English.

Write a welcome message that:
- Greets them warmly
- Briefly explains what's about to happen: you'll look up their business, figure out what services they offer, and sort out where they work — all in about 2 minutes
- Mentions you'll do most of the heavy lifting by pulling in their details automatically (ABN, licences, etc)
- Asks for their business name or ABN to kick things off
- Feels like a real person, not a corporate form. Keep it concise — 3-4 short sentences.
- If they include a postcode with their business name (e.g. "dans plumbing 2155") you can match them faster"""),
        HumanMessage(content="Hi, I'd like to get set up on Service Seeking."),
    ])

    return {
        "current_node": "welcome",
        "messages": [response],
    }


async def business_verification_node(state: OnboardingState) -> dict:
    """Look up and verify the business via ABR.

    This node has tool calls (ABR API) so it keeps some structure,
    but delegates conversation to the LLM.
    """
    messages = state.get("messages", [])
    last_msg = _get_last_human_message(messages)

    if not last_msg:
        return {"current_node": "business_verification"}

    # If no ABR results yet, do the lookup
    if not state.get("abr_results") and not state.get("business_verified"):
        clean = last_msg.strip().replace(" ", "")
        is_abn = clean.isdigit() and len(clean) >= 9
        search_type = "abn" if is_abn else "name"

        # Detect trailing postcode (e.g. "dans plumbing 2155")
        postcode_match = re.search(r'\b(\d{4})\s*$', last_msg.strip())
        search_term = last_msg
        user_postcode = None
        if postcode_match and not is_abn:
            user_postcode = postcode_match.group(1)
            search_term = last_msg[:postcode_match.start()].strip()
            print(f"[BIZ] Detected postcode {user_postcode} in input, searching for '{search_term}'")

        t0 = time.time()
        results = await abr_lookup(search_term, search_type)
        abr_results = results.get("results", [])
        _trace(state, "ABR Lookup", time.time() - t0,
               f"{len(abr_results)} results for '{search_term}'",
               {"search_type": search_type, "results": [
                   {"name": r.get("entity_name"), "abn": r.get("abn"), "postcode": r.get("postcode")}
                   for r in abr_results[:5]
               ]})

        # If user provided a postcode, filter results and auto-confirm if single match
        if user_postcode and abr_results:
            filtered = [r for r in abr_results if r.get("postcode") == user_postcode]
            if len(filtered) == 1:
                print(f"[BIZ] Single match in postcode {user_postcode}: {filtered[0].get('entity_name')}, auto-confirming")
                return await _confirm_business(filtered[0], state)
            elif filtered:
                # Multiple matches in postcode — show only those
                abr_results = filtered
                results["results"] = filtered
                results["count"] = len(filtered)
            # If no matches in postcode, fall through and show all results

        return {
            "current_node": "business_verification",
            "business_name_input": search_term,
            "abn_input": last_msg if is_abn else "",
            "abr_results": abr_results,
            "messages": [AIMessage(content=_format_abr_results(results, search_term))],
        }

    # We have ABR results — let the LLM interpret the user's response
    abr_results = state.get("abr_results", [])

    # Quick-match: if the message contains "Yes, it's" + an ABN from results, it's a button click confirmation
    selected_abr = None
    for r in abr_results:
        abn = r.get("abn", "")
        if abn and abn in last_msg and ("yes" in last_msg.lower() or "it's" in last_msg.lower()):
            print(f"[BIZ] Quick-match: ABN {abn} found in message, confirming directly")
            selected_abr = r
            break

    if not selected_abr:
        # Fall back to LLM classifier
        response = await llm_fast.ainvoke([
            SystemMessage(content=f"""You are the Service Seeking onboarding assistant. A tradie is verifying their business details.

ABR RESULTS ON FILE: {json.dumps(abr_results)}

The user has responded to the ABR results. Determine what they want:
- If they're confirming or selecting a business (yes, that's me, correct, selecting by name, "Yes, it's [NAME]", etc): respond with JUST the word CONFIRMED
- If they're rejecting ALL options (no, wrong, not me, none of these, etc): respond with JUST the word REJECTED
- If they're providing a new search term: respond with JUST the word NEWSEARCH

Respond with ONLY one word: CONFIRMED, REJECTED, or NEWSEARCH"""),
            HumanMessage(content=last_msg),
        ])

        intent = response.content.strip().upper()
        print(f"[BIZ] Classifier intent: {intent} for message: {last_msg[:80]}")

        if "CONFIRMED" in intent:
            # Find which result was selected
            selected_abr = abr_results[0] if abr_results else {}
            for r in abr_results:
                if r.get("abn", "") and r["abn"] in last_msg:
                    selected_abr = r
                    break

    if selected_abr:
        return await _confirm_business(selected_abr, state)

    if "REJECTED" in intent:
        return {
            "current_node": "business_verification",
            "abr_results": [],
            "business_verified": False,
            "messages": [AIMessage(content="No worries! Could you try a different business name, or enter your ABN directly?")],
        }

    # New search
    clean = last_msg.strip().replace(" ", "")
    is_abn = clean.isdigit() and len(clean) >= 9
    results = await abr_lookup(last_msg, "abn" if is_abn else "name")
    return {
        "current_node": "business_verification",
        "business_name_input": last_msg,
        "abr_results": results.get("results", []),
        "messages": [AIMessage(content=_format_abr_results(results, last_msg))],
    }


async def service_discovery_node(state: OnboardingState) -> dict:
    """Discover and map services through natural conversation.

    NO SCRIPTING. The LLM gets goals, guides, and context.
    It figures out the conversation.

    Model selection:
    - Turn 1 with licence data → Haiku (high-signal mapping, ~2s vs ~12s)
    - Turn 1 without licence data → Sonnet (needs inference from name)
    - Turn 2+ → Haiku with trimmed prompt (just update the list)
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
    licence_info = state.get("licence_info", {})

    is_follow_up = bool(services)  # Services already mapped = turn 2+

    # ── TURN 2+: Haiku with trimmed prompt ──
    if is_follow_up:
        model = llm_fast_json
        model_name = MODEL_FAST
        taxonomy = get_category_taxonomy_text()
        current_cats = list(set(s.get("category_name", "") for s in services))
        relevant_taxonomy = _get_relevant_taxonomy(taxonomy, current_cats)

        prompt = f"""You are updating a tradie's services on Service Seeking.

BUSINESS: {business_name}
CURRENT SERVICES: {json.dumps(services)}

RELEVANT TAXONOMY (for looking up IDs if adding new services):
{relevant_taxonomy}

The user responded to your gap question. Update the services list:
- If they want to add a service, add it with correct category_id/subcategory_id from the taxonomy
- If they want to remove something, remove it
- Keep all existing services unless they explicitly asked to remove them
- Set step_complete = true — you're done

Return a JSON object:
{{"response": "brief one-line confirmation of changes", "services": [full updated array with input, category_name, category_id, subcategory_name, subcategory_id, confidence], "buttons": [], "step_complete": true}}

Return ONLY the JSON object."""

        t_llm = time.time()
        response = await model.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_msg or "Looks good"),
        ])
        llm_time = time.time() - t_llm

    # ── TURN 1: Full prompt with taxonomy + guide ──
    else:
        model = llm_fast_json  # Haiku — web results + subcategory guide + business name provide enough signal
        model_name = MODEL_FAST

        guide = find_subcategory_guide(business_name)
        taxonomy = get_category_taxonomy_text()

        licence_context = ""
        if licence_classes:
            licence_context = f"\nNSW FAIR TRADING LICENCE CLASSES: {', '.join(licence_classes)}"
            if licence_info.get("licence_number"):
                licence_context += f"\nLicence #{licence_info['licence_number']} — Status: {licence_info.get('status', 'Unknown')}, Expiry: {licence_info.get('expiry_date', 'Unknown')}"
            if licence_info.get("compliance_clean") is False:
                licence_context += "\n⚠️ Compliance issues on record"

        web_context = ""
        if web_results:
            web_lines = [f"- {r.get('title', '')}: {r.get('url', '')}" for r in web_results[:3]]
            web_context = f"\nWEB PRESENCE:\n" + "\n".join(web_lines)

        conv_history = _format_conversation(messages, max_turns=4)

        static_context = f"""You are the Service Seeking onboarding assistant helping a tradie set up their services.

GOAL: Figure out what services this tradie offers and map them to the Service Seeking category taxonomy. Use the subcategory guide to ask smart questions about gaps — don't ask scripted questions, ask relevant ones based on what you know about this trade.

SUBCATEGORY GUIDE:
{guide[:4000] if guide else "No specific guide available for this trade."}

CATEGORY TAXONOMY:
{taxonomy[:6000]}

GUIDELINES:
- This flows directly from business confirmation — the conversation is already going. Don't re-introduce yourself.
- Be conversational and Australian. Keep it short — tradies are busy.
- Licence classes are your strongest signal — they tell you exactly what this tradie is licensed for. Web results and business name add context.
- Map as many services as you can in one go. The full detail is in the JSON services array — your response text is just a conversational summary. Keep it to a sentence or two: mention the total count and group names, not every individual service. No headers, no bullet points, no line breaks in the summary.
- Then ask ONE short gap question — a service they likely offer but you haven't mapped yet. Just the question, no preamble or explanation. Focus on actual trade services, not scheduling preferences (emergency call-outs, after-hours aren't service types in the taxonomy).
- When the tradie answers your gap question, update the list and wrap up. Two turns max — be decisive.
- Don't announce what you're doing, just do it. Keep your total response under 3 sentences.

Return a JSON object:
{{"response": "your conversational message", "services": [array of mapped services with input, category_name, category_id, subcategory_name, subcategory_id, confidence], "buttons": ["2-4 button options that let the tradie answer your gap question with a tap"], "step_complete": true/false}}

step_complete = true when: the tradie has confirmed or responded to the gap question, or hasn't raised issues with the mapped services.

Return ONLY the JSON object."""

        dynamic_context = f"""BUSINESS: {business_name}
SERVICES MAPPED SO FAR: {json.dumps(services) if services else "None yet"}
{licence_context}
{web_context}

CONVERSATION SO FAR:
{conv_history}"""

        t_llm = time.time()
        response = await model.ainvoke([
            SystemMessage(content=[
                {"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic_context},
            ]),
            HumanMessage(content=last_msg or "Let's set up my services"),
        ])
        llm_time = time.time() - t_llm

    # ── Parse response (shared by both paths) ──
    try:
        raw_json = _extract_json(response.content)
        if not raw_json or raw_json[0] != "{":
            print(f"[SVC] LLM returned plain text, wrapping: {response.content[:200]}")
            data = {
                "response": response.content,
                "services": services,
                "buttons": [],
                "step_complete": False,
            }
        else:
            data = json.loads(raw_json)
        new_services = data.get("services", services)
        message = data.get("response", "What services do you offer?")
        buttons = data.get("buttons", [])
        step_complete = data.get("step_complete", False)

        print(f"[SVC] user='{last_msg}' | model={model_name} | mapped={len(new_services)} | complete={step_complete}")
        _trace(state, "LLM: Service Discovery", llm_time,
               f"Mapped {len(new_services)} services, complete={step_complete}",
               {"model": model_name, "services_count": len(new_services),
                "step_complete": step_complete, "follow_up": is_follow_up})

        return {
            "current_node": "service_discovery",
            "services": new_services,
            "services_raw": last_msg or f"Inferred from: {business_name}",
            "services_confirmed": step_complete,
            "buttons": [{"label": b, "value": b} for b in buttons],
            "messages": [AIMessage(content=message)],
        }
    except Exception as e:
        print(f"[SVC ERROR] {e} | raw response: {response.content[:300]}")
        return {
            "current_node": "service_discovery",
            "messages": [AIMessage(content=f"What services does {business_name} offer? Just tell me in your own words.")],
        }


async def service_area_node(state: OnboardingState) -> dict:
    """Map service areas through natural, geography-aware conversation.

    NO SCRIPTING. The LLM gets goals, regional guides, and suburb data.
    It figures out the conversation — asks about barriers, corridors,
    congestion zones relevant to THIS tradie's location.

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

    last_msg = _get_last_human_message(messages)

    # Gather geographic context
    grouped = get_suburbs_in_radius_grouped(postcode, 20.0) if postcode else {}

    # Build region summary with suburb counts
    # Filter out regions with <3 suburbs — those are misclassified stray entries in the CSV
    region_list = ""
    if grouped.get("by_area"):
        region_lines = []
        for area, suburbs in sorted(grouped["by_area"].items(), key=lambda x: len(x[1]), reverse=True):
            if len(suburbs) < 3:
                continue
            sample = ", ".join([s["name"] for s in suburbs[:3]])
            region_lines.append(f"  - {area} ({len(suburbs)} suburbs, e.g. {sample})")
        region_list = "\n".join(region_lines)

    # Turn 1 always populates base_suburb, so its presence means we're on turn 2+
    is_follow_up = bool(service_areas.get("base_suburb"))
    model = llm_fast_json
    model_name = MODEL_FAST

    # ── TURN 2+: Trimmed prompt — just update include/exclude ──
    if is_follow_up:
        # Collect all region names from grouped data for exclude list
        all_regions = [area for area, suburbs in (grouped.get("by_area", {})).items() if len(suburbs) >= 3]

        prompt = f"""You are finalizing a tradie's service area on Service Seeking.

BASE: {grouped.get("base_suburb", "Unknown")} ({postcode})
ALL REGIONS WITHIN 20KM: {json.dumps(all_regions)}
CURRENT SELECTION: included={json.dumps(service_areas.get("regions_included", []))}, excluded={json.dumps(service_areas.get("regions_excluded", []))}

USER SAID: "{last_msg}"

Set regions_included to the regions they selected. Set regions_excluded to ALL remaining regions from the list above.
Set step_complete = true.

Return a JSON object:
{{"response": "Locked in! (or brief confirmation)", "service_areas": {{"base_suburb": "{grouped.get('base_suburb', '')}", "base_postcode": "{postcode}", "base_lat": {grouped.get('base_lat', 0)}, "base_lng": {grouped.get('base_lng', 0)}, "radius_km": 20, "regions_included": ["selected regions"], "regions_excluded": ["all other regions"], "barriers": [], "travel_notes": ""}}, "buttons": [], "step_complete": true}}

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

        static_context = f"""You are the Service Seeking onboarding assistant helping a tradie define their service area.

GOAL: Figure out which REGIONS this tradie covers. Real coverage isn't a perfect circle — it's a blob shaped by traffic, barriers, and preferences. Your job is to identify which regions they include and which they exclude.

REGIONS WITHIN 20KM OF BASE:
{region_list or "No region data available"}
Total: {grouped.get("total", 0)} suburbs across these regions

REGIONAL GUIDE (barriers, congestion, corridors):
{regional_guide[:4000] if regional_guide else "No regional guide available."}

GUIDELINES:
- This step flows directly from the service confirmation — the conversation is already going. Don't re-introduce yourself or rehash what you already know. The user's last message was about services, not areas.
- Be conversational and Australian. Keep it short — tradies are busy.
- Talk in terms of regions/areas, not individual suburbs
- Use the regional guide to understand natural boundaries, but don't explain the geography to the tradie — they already know it. Just ask which areas they cover.
- Present the nearby regions and ask which ones they cover. One simple question, no preamble. Offer buttons for likely groupings.
- When they tell you their areas, lock it in and move on. Two turns max — be decisive, don't ask for confirmation of what they just told you.
- Keep your total response under 2 sentences.

Return a JSON object:
{{"response": "your conversational message", "service_areas": {{"base_suburb": "", "base_postcode": "", "base_lat": 0, "base_lng": 0, "radius_km": 20, "regions_included": ["region names they cover"], "regions_excluded": ["region names within radius they don't cover"], "barriers": ["relevant barriers from regional guide"], "travel_notes": "brief note on coverage shape"}}, "buttons": ["2-4 options matching likely region groupings — a tap beats typing"], "step_complete": true/false}}

Use REAL region names from the grouped data above for regions_included and regions_excluded.
step_complete = true when the tradie has indicated which regions they cover.

Return ONLY the JSON object."""

        dynamic_context = f"""BUSINESS: {business_name}
BASE SUBURB: {grouped.get("base_suburb", "Unknown")} ({postcode})
STATE: {business_state}
CURRENT SERVICE AREA: Not set yet"""

        t_llm = time.time()
        response = await model.ainvoke([
            SystemMessage(content=[
                {"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic_context},
            ]),
            HumanMessage(content=last_msg or "Let's set up my service area"),
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
        print(f"[AREA] user='{last_msg}' | model={model_name} | included={included} | excluded={excluded} | complete={step_complete}")
        _trace(state, "LLM: Service Area", llm_time,
               f"{len(included)} regions included, {len(excluded)} excluded, complete={step_complete}",
               {"model": model_name, "regions_included": included,
                "regions_excluded": excluded, "step_complete": step_complete, "follow_up": is_follow_up})

        return {
            "current_node": "service_area",
            "location_raw": last_msg or f"Based on {postcode}",
            "service_areas": new_areas,
            "service_areas_confirmed": step_complete,
            "buttons": [{"label": b, "value": b} for b in buttons],
            "messages": [AIMessage(content=message)],
        }
    except Exception as e:
        print(f"[AREA ERROR] {e}")
        base = grouped.get("base_suburb", "your area")
        return {
            "current_node": "service_area",
            "messages": [AIMessage(content=f"Where do you typically work? You're based in {base} — do you mainly work locally or travel further afield?")],
        }


async def confirmation_node(state: OnboardingState) -> dict:
    """Show final summary and handle edits."""
    messages = state.get("messages", [])
    last_msg = _get_last_human_message(messages)

    business_name = state.get("business_name", "Unknown")
    abn = state.get("abn", "Unknown")
    entity_type = state.get("entity_type", "Unknown")
    services = state.get("services", [])
    service_areas = state.get("service_areas", {})
    services_text = ", ".join([s.get("subcategory_name", s.get("input", "")) for s in services])
    regions_included = service_areas.get("regions_included", [])
    regions_excluded = service_areas.get("regions_excluded", [])
    areas_text = ", ".join(regions_included) if regions_included else "Not set"
    travel_notes = service_areas.get("travel_notes", "")

    # Only classify intent if we're ALREADY in the confirmation node
    # (i.e. the summary has been shown and the user is responding to it).
    # On first entry (auto-chained from service_area), always show the summary.
    already_in_confirmation = state.get("current_node") == "confirmation"

    if last_msg and already_in_confirmation:
        # Fast path: structured removal from the confirmation UI
        if 'confirm and complete' in last_msg.lower() and ('remove services:' in last_msg.lower() or 'remove areas:' in last_msg.lower()):
            svc_match = re.search(r'Remove services?:\s*(.+?)(?:\.\s*|$)', last_msg, re.IGNORECASE)
            if svc_match:
                removed_svcs = [s.strip() for s in svc_match.group(1).split(',')]
                services = [s for s in services if s.get('subcategory_name', s.get('input', '')) not in removed_svcs]

            area_match = re.search(r'Remove areas?:\s*(.+?)(?:\.\s*|$)', last_msg, re.IGNORECASE)
            if area_match:
                removed_areas = [a.strip() for a in area_match.group(1).split(',')]
                current_included = service_areas.get('regions_included', [])
                current_excluded = service_areas.get('regions_excluded', [])
                service_areas['regions_included'] = [r for r in current_included if r not in removed_areas]
                service_areas['regions_excluded'] = current_excluded + [r for r in removed_areas if r not in current_excluded]

            return {
                "current_node": "confirmation",
                "services": services,
                "services_confirmed": True,
                "service_areas": service_areas,
                "service_areas_confirmed": True,
                "confirmed": True,
            }

        intent = await llm_fast.ainvoke([
            SystemMessage(content=f"""A tradie is reviewing their setup summary. Determine their intent.

SUMMARY:
- Business: {business_name} (ABN: {abn})
- Services: {services_text}
- Service areas: {areas_text}

If they want to confirm/complete: respond CONFIRMED
If they want to edit services: respond EDIT_SERVICES
If they want to edit service areas: respond EDIT_AREAS
If they want to edit business details: respond EDIT_BUSINESS

Respond with ONLY one word."""),
            HumanMessage(content=last_msg),
        ])

        intent_text = intent.content.strip().upper()

        if "CONFIRMED" in intent_text:
            return {"current_node": "confirmation", "confirmed": True}
        if "EDIT_SERVICES" in intent_text:
            current_services = [s.get("subcategory_name", s.get("input", "")) for s in services]
            buttons = [{"label": f"\u2715 {svc}", "value": f"Remove {svc}"} for svc in current_services]
            buttons.append({"label": "Done editing", "value": "Keep current services, confirm and complete"})
            return {
                "current_node": "confirmation",
                "services_confirmed": False,
                "buttons": buttons,
                "messages": [AIMessage(content="Tap any services to remove, or type to add more:")],
            }
        if "EDIT_AREAS" in intent_text:
            buttons = [{"label": f"\u2715 {area}", "value": f"Remove {area} from my areas"} for area in regions_included]
            buttons.append({"label": "Done editing", "value": "Keep current areas, confirm and complete"})
            return {
                "current_node": "confirmation",
                "service_areas_confirmed": False,
                "buttons": buttons,
                "messages": [AIMessage(content="Tap any areas to remove, or type to add more:")],
            }
        if "EDIT_BUSINESS" in intent_text:
            return {
                "current_node": "confirmation",
                "business_verified": False,
                "messages": [AIMessage(content="No worries — what's the correct business name or ABN?")],
            }

    # Show summary
    base_suburb = service_areas.get("base_suburb", "Unknown")
    radius = service_areas.get("radius_km", 20)

    contact_name = state.get("contact_name", "")
    contact_phone = state.get("contact_phone", "")

    # Derive trade from service categories (e.g. "Plumber", "Electrician")
    trades = list(dict.fromkeys(s.get("category_name", "") for s in services if s.get("category_name")))
    trade_text = ", ".join(trades) if trades else "Not set"

    summary = f"""Here's a summary of your setup:

- Business: {business_name}
- ABN: {abn}
- Trade: {trade_text}"""

    if contact_name:
        summary += f"\n- Contact: {contact_name}"
    if contact_phone:
        summary += f"\n- Phone: {contact_phone}"

    summary += f"""
- Services: {services_text}
- Based in: {base_suburb}
- Coverage: {areas_text} (within {radius}km)"""

    if regions_excluded:
        summary += f"\n- Excluding: {', '.join(regions_excluded)}"

    summary += f"""

Everything look good?"""

    return {
        "current_node": "confirmation",
        "messages": [AIMessage(content=summary)],
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
    }

    return {
        "current_node": "complete",
        "output_json": output,
        "confirmed": True,
        "messages": [AIMessage(content="You're all set! Your profile is ready on Service Seeking — you'll start getting matched with relevant jobs in your area soon. Welcome aboard!")],
    }


# ────────── HELPERS ──────────

def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Find the first { and last } to extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


async def _confirm_business(abr: dict, state: dict) -> dict:
    """Confirm a business and enrich with licence data + web search."""
    business_name = abr.get("entity_name", state.get("business_name_input", ""))
    abn = abr.get("abn", "")
    postcode = abr.get("postcode", "")
    business_state = abr.get("state", "")

    # Clean the search term (remove special chars that break the licence API)
    clean_name = business_name.replace("'", "").replace("'", "")

    # Run licence lookup and web search in parallel for enrichment
    t0 = time.time()
    licence_task = nsw_licence_browse(clean_name)
    web_task = brave_web_search(f"{business_name} {business_state} tradesperson")
    licence_results, web_results = await asyncio.gather(licence_task, web_task)
    t1 = time.time()
    print(f"[BIZ] Licence browse + web search: {t1 - t0:.1f}s (parallel)")

    _trace(state, "NSW Licence Browse", t1 - t0,
           f"{len(licence_results.get('results', []))} licence matches for '{clean_name}'",
           {"results": [
               {"licensee": r.get("licensee"), "status": r.get("status"),
                "licence_number": r.get("licence_number"), "suburb": r.get("suburb")}
               for r in licence_results.get("results", [])[:5]
           ]})
    _trace(state, "Brave Web Search", t1 - t0,
           f"{len(web_results) if web_results else 0} web results",
           {"results": [
               {"title": r.get("title"), "url": r.get("url")}
               for r in (web_results or [])[:3]
           ]})

    # Find the best licence match (current, matching name closely)
    licence_info = {}
    licence_classes = []
    licence_matches = licence_results.get("results", [])

    # Try exact match first, then partial
    best_match = None
    for lic in licence_matches:
        if lic.get("status") != "Current":
            continue
        licensee = lic.get("licensee", "").lower()
        if clean_name.lower() in licensee or licensee in clean_name.lower():
            best_match = lic
            break

    # If no name match, take first current result
    if not best_match:
        for lic in licence_matches:
            if lic.get("status") == "Current":
                best_match = lic
                break

    if best_match:
        lid = best_match.get("licence_id", "")
        if lid:
            t2 = time.time()
            details = await nsw_licence_details(lid)
            t3 = time.time()
            print(f"[BIZ] Licence details: {t3 - t2:.1f}s")
            if not details.get("error"):
                licence_info = details
                licence_classes = [
                    c["name"] for c in details.get("classes", []) if c.get("active")
                ]
                print(f"[BIZ] Licence #{details.get('licence_number')} classes: {licence_classes}")
                _trace(state, "NSW Licence Details", t3 - t2,
                       f"Licence #{details.get('licence_number')} — {', '.join(licence_classes)}",
                       {"licence_number": details.get("licence_number"),
                        "status": details.get("status"),
                        "expiry": details.get("expiry_date"),
                        "classes": licence_classes})

    # Extract contact person from licence associated parties
    contact_name = ""
    if licence_info:
        parties = licence_info.get("associated_parties", [])
        for p in parties:
            if p.get("party_type") == "Individual" and p.get("role") in ("Director", "Nominated Supervisor", "Partner", "Sole Trader"):
                contact_name = p.get("name", "")
                break

    # Extract phone from Brave web results
    contact_phone = ""
    if web_results:
        for r in web_results[:3]:
            desc = r.get("description", "")
            # Look for Australian phone patterns: 1300/1800, 04xx, (0x) xxxx
            phone_match = re.search(r'(?:1[38]00\s?\d{3}\s?\d{3}|0[24]\d{2}\s?\d{3}\s?\d{3}|\(0\d\)\s?\d{4}\s?\d{4})', desc)
            if phone_match:
                contact_phone = phone_match.group(0).strip()
                break

    if contact_name:
        print(f"[BIZ] Contact person: {contact_name}")
    if contact_phone:
        print(f"[BIZ] Contact phone: {contact_phone}")

    return {
        "current_node": "business_verification",
        "business_name": business_name,
        "abn": abn,
        "entity_type": abr.get("entity_type", ""),
        "gst_registered": abr.get("gst_registered", False),
        "business_postcode": postcode,
        "business_state": business_state,
        "business_verified": True,
        "licence_info": licence_info,
        "licence_classes": licence_classes,
        "web_results": web_results[:3] if web_results else [],
        "contact_name": contact_name,
        "contact_phone": contact_phone,
        "messages": [AIMessage(content=f"Great, {business_name} is confirmed!")],
    }


def _get_last_human_message(messages: list) -> str | None:
    """Get the content of the last human message."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _get_relevant_taxonomy(full_taxonomy: str, category_names: list[str]) -> str:
    """Extract taxonomy entries for the given categories (for trimmed turn 2 prompts)."""
    if not category_names:
        return full_taxonomy[:3000]

    lines = full_taxonomy.split("\n")
    relevant = []
    include = False
    for line in lines:
        # Category header lines don't start with "  -"
        if not line.startswith("  -"):
            include = any(cat.lower() in line.lower() for cat in category_names)
        if include:
            relevant.append(line)

    result = "\n".join(relevant)
    return result if result else full_taxonomy[:3000]


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
    entries = results.get("results", [])

    if not entries:
        return f"I couldn't find a business matching '{search_term}' on the ABR. Could you try a different name, or enter your ABN directly?"

    if len(entries) == 1:
        r = entries[0]
        gst = "Yes" if r.get("gst_registered") else "No"
        location = f"{r.get('state', '')}" + (f" {r.get('postcode', '')}" if r.get('postcode') else "")
        return f"""I found a match on the ABR:

- Business: {r.get('entity_name', 'Unknown')}
- ABN: {r.get('abn', '')}
- Type: {r.get('entity_type', 'Unknown')}
- GST Registered: {gst}
- Location: {location}

Is this your business?"""

    # Multiple results — don't list them in text, buttons will handle selection
    return f"I found {len(entries)} matches for '{search_term}'. Which one is yours?"

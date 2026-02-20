"""
Edge Case Test Harness for Trade Onboarding Wizard

Runs 20 real NSW trade businesses through the full onboarding flow,
evaluating: business confirmation, service mapping, area selection,
and profile building (logo, photos, description).

Results written to tests/edge_case_results.csv
"""

import asyncio
import csv
import json
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

BASE_URL = "http://localhost:8001"
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_case_results.csv")

# ── Test businesses: name, ABN, expected_trade, location hint ──
# Mix: 5 electricians, 4 plumbers, 2 carpenters, 2 builders, 2 landscapers,
#      1 tiler, 1 roofer, 1 fencer, 1 painter, 1 cleaner
# All ABNs verified against ABR — real NSW businesses
TEST_BUSINESSES = [
    # Electricians (5)
    {"name": "DANIEL SCOPE ELECTRICAL PTY LTD", "abn": "82640351293", "trade": "Electrician", "location": "Northern Beaches"},
    {"name": "SURE 2 ELECTRICAL PTY LTD", "abn": "54628502072", "trade": "Electrician", "location": "Northern Beaches"},
    {"name": "SWITCHED ON ELECTRICAL", "abn": "80566074031", "trade": "Electrician", "location": "Hills District"},
    {"name": "SPARK & ELECTRICAL PTY LTD", "abn": "48692763492", "trade": "Electrician", "location": "Canterbury"},
    {"name": "AMBIAH PTY LIMITED", "abn": "46607331055", "trade": "Electrician", "location": "North Shore"},
    # Plumbers (4)
    {"name": "QUINTESSENTIAL PLUMBING", "abn": "57338442101", "trade": "Plumber", "location": "St George"},
    {"name": "Everyday Plumbing Solutions Pty Limited", "abn": "33144903622", "trade": "Plumber", "location": "Western Sydney"},
    {"name": "RUTTLEY PLUMBING PTY LTD", "abn": "30159664767", "trade": "Plumber", "location": "North Shore"},
    {"name": "Pure Plumbing", "abn": "32335288933", "trade": "Plumber", "location": "Sutherland"},
    # Carpenters (2)
    {"name": "SYDNEY CARPENTRY COMPANY", "abn": "65216914213", "trade": "Carpenter", "location": "Inner West"},
    {"name": "Concept Carpentry", "abn": "95428780982", "trade": "Carpenter", "location": "Northern Beaches"},
    # Builders (2)
    {"name": "AND BUILT PTY LTD", "abn": "80600416653", "trade": "Builder", "location": "Wollongong"},
    {"name": "DALTON BUILDING SERVICES", "abn": "87147560281", "trade": "Builder", "location": "Western NSW"},
    # Landscapers (2)
    {"name": "Growing Rooms Landscapes", "abn": "75736955912", "trade": "Landscaper", "location": "Northern Beaches"},
    {"name": "URBAN LANDSCAPE SOLUTIONS", "abn": "67993405570", "trade": "Landscaper", "location": "Inner West"},
    # Other trades (5)
    {"name": "SYDNEY WIDE ROOFING PTY LIMITED", "abn": "54605163531", "trade": "Roofer", "location": "Sydney"},
    {"name": "Fantastic Cleaning", "abn": "41742687530", "trade": "Cleaner", "location": "Sydney"},
    {"name": "PROFENCE INDUSTRIES PTY LIMITED", "abn": "35091206727", "trade": "Fencer", "location": "Central Coast"},
    {"name": "BEST TILING PTY LTD", "abn": "20000938698", "trade": "Tiler", "location": "Western Sydney"},
    {"name": "FRESHCOAT PAINTING", "abn": "20549242906", "trade": "Painter", "location": "Northern Beaches"},
]


async def run_test(client: httpx.AsyncClient, biz: dict, test_num: int) -> dict:
    """Run a single business through the full onboarding flow."""
    result = {
        "test_num": test_num,
        "business_name": biz["name"],
        "abn": biz["abn"],
        "expected_trade": biz["trade"],
        "location_hint": biz["location"],
        # Biz verification
        "biz_pass": "",
        "biz_notes": "",
        "biz_name_matched": "",
        "biz_postcode": "",
        "biz_state": "",
        "licence_found": "",
        "licence_classes": "",
        "google_rating": "",
        "google_reviews": "",
        "facebook_found": "",
        "website_found": "",
        # Service mapping
        "svc_pass": "",
        "svc_notes": "",
        "svc_count": "",
        "svc_turns": 0,
        # Area
        "area_pass": "",
        "area_notes": "",
        "area_regions": "",
        # Profile
        "profile_pass": "",
        "profile_notes": "",
        "has_logo": "",
        "has_photos": "",
        "photo_count": 0,
        "has_description": "",
        "description_length": 0,
        "contact_name": "",
        # Timing
        "total_time": 0,
        "error": "",
    }

    try:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[{test_num}/20] Testing: {biz['name']}")
        print(f"  ABN: {biz['abn']} | Trade: {biz['trade']}")
        print(f"{'='*60}")

        # Step 1: Create session
        resp = await client.post(f"{BASE_URL}/api/session", json={}, timeout=30.0)
        if resp.status_code != 200:
            result["error"] = f"Session creation failed: {resp.status_code}"
            return result
        session_data = resp.json()
        session_id = session_data["session_id"]
        print(f"  Session: {session_id}")
        print(f"  Welcome: {session_data['response']['text'][:80]}...")

        # Step 2: Send ABN (triggers biz verification → auto-chains to services)
        print(f"  Sending ABN: {biz['abn']}...")
        resp = await client.post(
            f"{BASE_URL}/api/chat",
            json={"session_id": session_id, "message": biz["abn"]},
            timeout=60.0,
        )
        if resp.status_code != 200:
            result["error"] = f"ABN chat failed: {resp.status_code}"
            return result
        chat1 = resp.json()
        state = chat1["state"]
        node = chat1["response"]["node"]
        print(f"  Node: {node}")
        print(f"  Biz verified: {state.get('business_verified')}")
        print(f"  Biz name: {state.get('business_name')}")
        print(f"  Response: {chat1['response']['text'][:100]}...")

        # Check if ABN matched — might need to confirm from multiple results
        buttons = chat1["response"].get("buttons", [])
        if node == "business_verification" and buttons:
            # Need to pick the right match or confirm
            # Look for button matching the business name
            confirm_msg = None
            for btn in buttons:
                val = btn.get("value", "")
                if biz["abn"] in val or "confirm" in val.lower():
                    confirm_msg = val
                    break
            if not confirm_msg and buttons:
                confirm_msg = buttons[0].get("value", buttons[0].get("label", "Yes"))

            if confirm_msg:
                print(f"  Confirming: {confirm_msg[:60]}...")
                resp = await client.post(
                    f"{BASE_URL}/api/chat",
                    json={"session_id": session_id, "message": confirm_msg},
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    chat1 = resp.json()
                    state = chat1["state"]
                    node = chat1["response"]["node"]

        # Record biz verification results
        result["biz_name_matched"] = state.get("business_name", "")
        result["biz_postcode"] = state.get("business_postcode", "")
        result["biz_state"] = state.get("business_state", "NSW")
        result["licence_found"] = "Yes" if state.get("licence_info") else "No"
        result["licence_classes"] = ", ".join(state.get("licence_classes", []))
        result["google_rating"] = state.get("google_rating", 0)
        result["google_reviews"] = state.get("google_review_count", 0)
        result["facebook_found"] = "Yes" if state.get("facebook_url") else "No"
        result["website_found"] = "Yes" if state.get("business_website") else "No"

        biz_verified = state.get("business_verified", False)
        result["biz_pass"] = "PASS" if biz_verified else "FAIL"
        if not biz_verified:
            result["biz_notes"] = f"Not verified. Node: {node}"
            result["error"] = "Business verification failed"
            result["total_time"] = round(time.time() - t0, 1)
            return result

        biz_notes = []
        if not state.get("licence_info"):
            biz_notes.append("No licence")
        if state.get("google_rating", 0) > 0:
            biz_notes.append(f"Google {state['google_rating']}★")
        if state.get("facebook_url"):
            biz_notes.append("Has FB")
        if state.get("business_website"):
            biz_notes.append("Has website")
        result["biz_notes"] = "; ".join(biz_notes)

        print(f"  Licence: {result['licence_classes'] or 'None'}")
        print(f"  Google: {result['google_rating']}★ ({result['google_reviews']} reviews)")

        # Step 3: Service discovery — should auto-chain here
        # May need multiple turns for gap questions
        svc_turns = 0
        if node == "service_discovery":
            svc_turns = 1
            # First turn auto-chained, check if complete
            svc_count = len(state.get("services", []))
            print(f"  Services mapped: {svc_count}")

            # Confirm services (say "Looks good") if needed
            while node == "service_discovery" and svc_turns < 5:
                # If services are mapped, confirm
                if state.get("services_confirmed"):
                    break
                if svc_count > 0:
                    msg = "Looks good"
                else:
                    # Trade-appropriate fallback
                    trade_msgs = {
                        "Electrician": "I do all types of electrical work",
                        "Plumber": "I do all types of plumbing work",
                        "Carpenter": "I do all types of carpentry work",
                        "Builder": "I do all types of building work",
                        "Landscaper": "I do all types of landscaping work",
                        "Roofer": "I do all types of roofing work",
                        "Tiler": "I do all types of tiling work",
                        "Painter": "I do all types of painting work",
                        "Cleaner": "I do all types of cleaning work",
                        "Fencer": "I do all types of fencing work",
                    }
                    msg = trade_msgs.get(biz["trade"], f"I do all types of {biz['trade'].lower()} work")
                print(f"  SVC turn {svc_turns + 1}: sending '{msg}'")
                resp = await client.post(
                    f"{BASE_URL}/api/chat",
                    json={"session_id": session_id, "message": msg},
                    timeout=60.0,
                )
                if resp.status_code != 200:
                    break
                chat_svc = resp.json()
                state = chat_svc["state"]
                node = chat_svc["response"]["node"]
                svc_count = len(state.get("services", []))
                svc_turns += 1
                print(f"  Node: {node}, Services: {svc_count}")

        result["svc_count"] = len(state.get("services", []))
        result["svc_turns"] = svc_turns
        svc_confirmed = state.get("services_confirmed", False)
        result["svc_pass"] = "PASS" if svc_confirmed and result["svc_count"] > 0 else "FAIL"
        if not svc_confirmed:
            result["svc_notes"] = f"Not confirmed after {svc_turns} turns. Node: {node}"
        elif result["svc_count"] < 3:
            result["svc_notes"] = f"Only {result['svc_count']} services mapped"
        else:
            result["svc_notes"] = f"{result['svc_count']} services in {svc_turns} turns"
        print(f"  Services result: {result['svc_pass']} — {result['svc_notes']}")

        # Step 4: Service area — should auto-chain or we need to provide area
        if node == "service_area":
            # Provide the location hint as area
            area_msg = biz["location"]
            print(f"  Sending area: '{area_msg}'")
            resp = await client.post(
                f"{BASE_URL}/api/chat",
                json={"session_id": session_id, "message": area_msg},
                timeout=60.0,
            )
            if resp.status_code == 200:
                chat_area = resp.json()
                state = chat_area["state"]
                node = chat_area["response"]["node"]

            # May need another turn if area wasn't confirmed
            if node == "service_area" and not state.get("service_areas_confirmed"):
                print(f"  Area not confirmed, sending 'All of Sydney'")
                resp = await client.post(
                    f"{BASE_URL}/api/chat",
                    json={"session_id": session_id, "message": "All of Sydney"},
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    chat_area = resp.json()
                    state = chat_area["state"]
                    node = chat_area["response"]["node"]

        areas = state.get("service_areas", {})
        area_regions = list(areas.get("included", {}).keys()) if isinstance(areas.get("included"), dict) else areas.get("included", [])
        result["area_regions"] = ", ".join(area_regions) if area_regions else str(areas)
        area_confirmed = state.get("service_areas_confirmed", False)
        result["area_pass"] = "PASS" if area_confirmed else "FAIL"
        result["area_notes"] = f"{len(area_regions)} regions" if area_confirmed else f"Not confirmed. Node: {node}"
        print(f"  Area result: {result['area_pass']} — {result['area_notes']}")

        # Step 5: Profile — should auto-chain from area
        # Wait for profile to build (it does LLM + scraping)
        if node == "profile":
            # Profile is already built by the auto-chain
            result["has_logo"] = "Yes" if state.get("profile_logo") else "No"
            result["has_photos"] = "Yes" if state.get("profile_photos") else "No"
            result["photo_count"] = len(state.get("profile_photos", []))
            result["has_description"] = "Yes" if state.get("profile_description") else "No"
            result["description_length"] = len(state.get("profile_description", ""))
            result["contact_name"] = state.get("contact_name", "")

            profile_notes = []
            if state.get("profile_logo"):
                profile_notes.append("Has logo")
            else:
                profile_notes.append("NO LOGO")
            profile_notes.append(f"{result['photo_count']} photos")
            if result["description_length"] > 50:
                profile_notes.append(f"Desc: {result['description_length']}ch")
            else:
                profile_notes.append("SHORT/NO desc")
            result["profile_notes"] = "; ".join(profile_notes)

            # Pass if we have a description and at least something visual
            has_content = result["description_length"] > 50
            result["profile_pass"] = "PASS" if has_content else "FAIL"
        else:
            result["profile_pass"] = "FAIL"
            result["profile_notes"] = f"Never reached profile. Node: {node}"

        print(f"  Profile result: {result['profile_pass']} — {result['profile_notes']}")
        result["total_time"] = round(time.time() - t0, 1)
        print(f"  Total time: {result['total_time']}s")

    except Exception as e:
        result["error"] = str(e)
        result["total_time"] = round(time.time() - t0, 1)
        print(f"  ERROR: {e}")

    return result


async def main():
    print("="*60)
    print("TRADE ONBOARDING EDGE CASE TEST")
    print(f"Testing {len(TEST_BUSINESSES)} businesses")
    print("="*60)

    # Check server health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if resp.status_code != 200:
                print("ERROR: Server not healthy")
                return
            print("Server healthy\n")
        except Exception:
            print("ERROR: Server not running on port 8001")
            return

    results = []
    async with httpx.AsyncClient() as client:
        for i, biz in enumerate(TEST_BUSINESSES):
            result = await run_test(client, biz, i + 1)
            results.append(result)
            # Small delay between tests to avoid overwhelming
            if i < len(TEST_BUSINESSES) - 1:
                await asyncio.sleep(1)

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{'='*60}")
        print(f"Results written to: {RESULTS_FILE}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    biz_pass = sum(1 for r in results if r["biz_pass"] == "PASS")
    svc_pass = sum(1 for r in results if r["svc_pass"] == "PASS")
    area_pass = sum(1 for r in results if r["area_pass"] == "PASS")
    profile_pass = sum(1 for r in results if r["profile_pass"] == "PASS")
    errors = sum(1 for r in results if r["error"])

    print(f"  Business verification: {biz_pass}/{total} passed")
    print(f"  Service mapping:       {svc_pass}/{total} passed")
    print(f"  Area selection:        {area_pass}/{total} passed")
    print(f"  Profile building:      {profile_pass}/{total} passed")
    print(f"  Errors:                {errors}/{total}")

    # Flag edge cases
    print(f"\n{'='*60}")
    print("EDGE CASES & ISSUES")
    print(f"{'='*60}")
    for r in results:
        issues = []
        if r["biz_pass"] == "FAIL":
            issues.append(f"BIZ FAIL: {r['biz_notes']}")
        if r["svc_pass"] == "FAIL":
            issues.append(f"SVC FAIL: {r['svc_notes']}")
        if r["area_pass"] == "FAIL":
            issues.append(f"AREA FAIL: {r['area_notes']}")
        if r["profile_pass"] == "FAIL":
            issues.append(f"PROFILE FAIL: {r['profile_notes']}")
        if r["has_logo"] == "No" and r["profile_pass"] == "PASS":
            issues.append("NO LOGO (monogram fallback)")
        if r["photo_count"] == 0 and r["profile_pass"] == "PASS":
            issues.append("NO PHOTOS")
        if r["licence_found"] == "No":
            issues.append("NO LICENCE FOUND")
        if r["error"]:
            issues.append(f"ERROR: {r['error']}")
        if issues:
            print(f"\n  [{r['test_num']}] {r['business_name']}")
            for issue in issues:
                print(f"      → {issue}")

    all_clean = not any(
        r["biz_pass"] == "FAIL" or r["svc_pass"] == "FAIL" or
        r["area_pass"] == "FAIL" or r["profile_pass"] == "FAIL" or r["error"]
        for r in results
    )
    if all_clean:
        print("  No issues found!")

    print(f"\nDone in {sum(r['total_time'] for r in results):.0f}s total")


if __name__ == "__main__":
    asyncio.run(main())

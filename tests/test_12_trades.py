#!/usr/bin/env python3
"""
Integration test: Service discovery pipeline for 12 trades.

Exercises the key functions that service_discovery_node uses:
  1. Category detection (_detect_category / _detect_categories)
  2. ABN lookup (abr_lookup - real API call)
  3. Initial service mapping (compute_initial_services - tiered)
  4. Related category suggestions (suggest_related_categories)
  5. Cluster groups (get_filtered_cluster_groups)
  6. Subcategory guide (find_subcategory_guide)

Run with: ./venv/bin/python tests/test_12_trades.py
"""

import asyncio
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import (
    _detect_category,
    _detect_categories,
    abr_lookup,
    compute_initial_services,
    suggest_related_categories,
    get_filtered_cluster_groups,
    find_subcategory_guide,
)

# ── Test businesses ──────────────────────────────────────────────────────────

BUSINESSES = [
    {"label": "Electrician",    "name": "101 Electrical Services Pty Ltd",                                        "abn": "66669666871", "state": "NSW"},
    {"label": "Handyman",       "name": "handyman Zhengui Gong",                                                  "abn": "99733633772", "state": "NSW"},
    {"label": "Plumber",        "name": "Tonks Plumbing Drainage & Gasfitting,Roofing, LPG",                      "abn": "81094119645", "state": "NSW"},
    {"label": "Painter",        "name": "TOMEH PAINTING & RENDERING PTY LTD",                                     "abn": "60635439888", "state": "VIC"},
    {"label": "Cleaner",        "name": "Brisbane bond clean",                                                     "abn": "34861382706", "state": "QLD"},
    {"label": "Gardener",       "name": "The Tree Muskateers",                                                     "abn": "12297861822", "state": "NSW"},
    {"label": "Concreter",      "name": "All Render and Concrete Group",                                           "abn": "33169064377", "state": "NSW"},
    {"label": "Bathroom Reno",  "name": "The Big Budget Homes Pty Ltd - Professional Bathroom Renovations",        "abn": "47607660373", "state": "QLD"},
    {"label": "Upholsterer",    "name": "A & E Upholstery Pty Ltd",                                               "abn": "52608234761", "state": "VIC"},
    {"label": "Kitchen Reno",   "name": "ELBALQA JOINERY PTY LTD",                                                "abn": "25636616101", "state": "NSW"},
    {"label": "Carpenter",      "name": "Gosford Carpentry & Building Services",                                   "abn": "58354557206", "state": "NSW"},
    {"label": "Builder",        "name": "Deep Island Construction Pty Ltd",                                        "abn": "81660532892", "state": "QLD"},
]

SEPARATOR = "=" * 80
SUBSEP = "-" * 60


async def test_one_business(biz: dict, index: int) -> str:
    """Run all 6 steps for one business and return formatted report."""
    lines = []
    name = biz["name"]
    abn = biz["abn"]
    label = biz["label"]

    lines.append(SEPARATOR)
    lines.append(f"  #{index+1}  {label.upper()}: {name}")
    lines.append(f"  ABN: {abn} | State: {biz['state']}")
    lines.append(SEPARATOR)

    # ── Step 1: Category Detection ───────────────────────────────────────
    lines.append("")
    lines.append("STEP 1: Category Detection")
    lines.append(SUBSEP)

    single_cat = _detect_category(name, None, None, None)
    multi_cats = _detect_categories(name, None, None, None)

    lines.append(f"  _detect_category()   -> {single_cat or '(none)'}")
    lines.append(f"  _detect_categories() -> {multi_cats or '(none)'}")

    if not single_cat and not multi_cats:
        lines.append("  ** Non-tiered trade: no keyword match in business name")

    # ── Step 2: ABN Lookup ───────────────────────────────────────────────
    lines.append("")
    lines.append("STEP 2: ABN Lookup")
    lines.append(SUBSEP)

    try:
        abr = await abr_lookup(abn, search_type="abn")
        results = abr.get("results", [])
        if results:
            r = results[0]
            lines.append(f"  Entity name : {r.get('entity_name', '?')}")
            lines.append(f"  Legal name  : {r.get('legal_name', '?')}")
            lines.append(f"  Entity type : {r.get('entity_type', '?')}")
            lines.append(f"  GST         : {'Yes' if r.get('gst_registered') else 'No'}")
            lines.append(f"  Status      : {r.get('status', '?')}")
            lines.append(f"  State       : {r.get('state', '?')}")
        elif abr.get("error"):
            lines.append(f"  ERROR: {abr['error']}")
        else:
            lines.append("  No results returned")
    except Exception as e:
        lines.append(f"  EXCEPTION: {e}")

    # ── Step 3: Initial Service Mapping (Tiered) ─────────────────────────
    lines.append("")
    lines.append("STEP 3: Initial Service Mapping (Tiered)")
    lines.append(SUBSEP)

    svc_result = compute_initial_services(
        business_name=name,
        licence_classes=[],
        google_business_name=None,
        google_primary_type=None,
        google_reviews=[],
        web_results=[],
        website_text="",
    )

    tiered = svc_result.get("tiered", False)
    lines.append(f"  Tiered?        : {'Yes' if tiered else 'No'}")

    category_names = []
    specialist_gaps = []

    if tiered:
        services = svc_result.get("services", [])
        category_names = svc_result.get("category_names", [])
        specialist_gaps = svc_result.get("specialist_gaps", [])

        lines.append(f"  Categories     : {', '.join(category_names)}")
        lines.append(f"  Pre-mapped     : {len(services)} services")

        # Break down by confidence
        high = [s for s in services if s.get("confidence") == "high"]
        evidence = [s for s in services if s.get("confidence") == "evidence"]
        licence = [s for s in services if s.get("confidence") == "licence"]
        lines.append(f"    Core (high)  : {len(high)}")
        lines.append(f"    Evidence     : {len(evidence)}")
        lines.append(f"    Licence      : {len(licence)}")
        lines.append(f"  Specialist gaps: {len(specialist_gaps)}")

        if services:
            lines.append(f"  Sample services: {', '.join(s['subcategory_name'] for s in services[:5])}")
        if specialist_gaps:
            lines.append(f"  Sample gaps    : {', '.join(g['subcategory_name'] for g in specialist_gaps[:5])}")
    else:
        lines.append("  ** Not tiered — would fall back to full taxonomy + LLM mapping")

    # ── Step 4: Related Category Suggestions ─────────────────────────────
    lines.append("")
    lines.append("STEP 4: Related Category Suggestions")
    lines.append(SUBSEP)

    if category_names:
        suggestions = suggest_related_categories(category_names)
        if suggestions:
            for s in suggestions:
                lines.append(f"  - {s['category']} ({s['pct']}% co-occurrence)")
        else:
            lines.append("  No related categories suggested")
    else:
        # For non-tiered trades, try with the label as a fallback
        suggestions = suggest_related_categories([label])
        if suggestions:
            for s in suggestions:
                lines.append(f"  - {s['category']} ({s['pct']}% co-occurrence)")
        else:
            lines.append("  No related categories (no detected category to base on)")

    # ── Step 5: Cluster Groups ───────────────────────────────────────────
    lines.append("")
    lines.append("STEP 5: Cluster Groups")
    lines.append(SUBSEP)

    if tiered and specialist_gaps:
        clusters = get_filtered_cluster_groups(
            gaps=specialist_gaps,
            business_name=name,
            licence_classes=None,
            google_business_name="",
            google_primary_type="",
        )
        if clusters:
            lines.append(f"  {len(clusters)} cluster(s):")
            for cl in clusters:
                svc_names = [s["name"] for s in cl["services"]]
                lines.append(f"    [{cl['label']}] ({len(svc_names)} services)")
                for sn in svc_names:
                    lines.append(f"      - {sn}")
        else:
            lines.append("  No clusters (all gaps may be unclustered)")
    else:
        lines.append("  Skipped (not tiered or no specialist gaps)")

    # ── Step 6: Subcategory Guide ────────────────────────────────────────
    lines.append("")
    lines.append("STEP 6: Subcategory Guide")
    lines.append(SUBSEP)

    guide = find_subcategory_guide(name)
    if guide:
        lines.append(f"  Guide found: {len(guide)} chars")
        # Show first 100 chars as preview
        preview = guide[:100].replace("\n", " ").strip()
        lines.append(f"  Preview: {preview}...")
    else:
        lines.append("  No guide found for this trade")

    lines.append("")
    return "\n".join(lines)


async def main():
    print()
    print("=" * 80)
    print("  SERVICE DISCOVERY PIPELINE — 12 TRADE INTEGRATION TEST")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Run sequentially to avoid hammering ABR API
    all_reports = []
    for i, biz in enumerate(BUSINESSES):
        report = await test_one_business(biz, i)
        all_reports.append(report)
        print(report)
        # Small delay between ABR calls
        if i < len(BUSINESSES) - 1:
            await asyncio.sleep(0.3)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print(SEPARATOR)
    print("  SUMMARY")
    print(SEPARATOR)
    print()
    print(f"  {'#':<3} {'Trade':<16} {'Category Detected':<25} {'Tiered?':<8} {'Services':<10} {'Gaps':<6} {'Guide?'}")
    print(f"  {'-'*3} {'-'*16} {'-'*25} {'-'*8} {'-'*10} {'-'*6} {'-'*6}")

    for i, biz in enumerate(BUSINESSES):
        name = biz["name"]
        cat = _detect_category(name, None, None, None) or "(none)"
        cats = _detect_categories(name, None, None, None)

        svc_result = compute_initial_services(
            business_name=name,
            licence_classes=[],
            google_business_name=None,
            google_primary_type=None,
            google_reviews=[],
            web_results=[],
            website_text="",
        )
        tiered = svc_result.get("tiered", False)
        n_services = len(svc_result.get("services", []))
        n_gaps = len(svc_result.get("specialist_gaps", []))
        guide = find_subcategory_guide(name)
        has_guide = "Yes" if guide else "No"

        cat_display = ", ".join(cats) if cats else "(none)"

        print(f"  {i+1:<3} {biz['label']:<16} {cat_display:<25} {'Yes' if tiered else 'No':<8} {n_services:<10} {n_gaps:<6} {has_guide}")

    print()
    print(f"  Test completed at {time.strftime('%H:%M:%S')}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

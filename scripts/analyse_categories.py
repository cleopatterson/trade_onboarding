#!/usr/bin/env python3
"""Analyse SS business export CSV to compute category co-occurrence data.

Reads a CSV with an `industry` column (comma/pipe/semicolon-separated categories
per row), computes co-occurrence percentages, and outputs related_categories.json.

Usage:
    python scripts/analyse_categories.py data/businesses.csv
    python scripts/analyse_categories.py data/businesses.csv --threshold 20 --cap 4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def parse_categories(raw: str) -> list[str]:
    """Split a raw industry string into individual category names."""
    # Try common delimiters: pipe, semicolon, comma
    # Pipe first (most unambiguous), then semicolon, then comma
    if "|" in raw:
        parts = raw.split("|")
    elif ";" in raw:
        parts = raw.split(";")
    else:
        parts = raw.split(",")
    return [p.strip() for p in parts if p.strip()]


def build_cooccurrence(rows: list[list[str]]) -> dict[str, Counter]:
    """Build co-occurrence counts: for each category, count how many businesses
    also list each other category."""
    # Count how many businesses each category appears in
    category_counts: Counter = Counter()
    cooccurrence: dict[str, Counter] = {}

    for cats in rows:
        unique = list(dict.fromkeys(cats))  # dedupe preserving order
        for cat in unique:
            category_counts[cat] += 1
            if cat not in cooccurrence:
                cooccurrence[cat] = Counter()
            for other in unique:
                if other != cat:
                    cooccurrence[cat][other] += 1

    return cooccurrence, category_counts


def compute_related(
    cooccurrence: dict[str, Counter],
    category_counts: Counter,
    threshold_pct: int = 15,
    cap: int = 5,
) -> dict[str, list[dict]]:
    """Convert raw co-occurrence counts to percentage-based related categories.

    For each category, the percentage is: (co-occurrence count / category count) * 100.
    Only includes categories above the threshold, capped at `cap` per category.
    """
    result = {}
    for cat, others in sorted(cooccurrence.items()):
        cat_count = category_counts[cat]
        if cat_count < 10:
            continue  # skip very rare categories (noisy data)

        related = []
        for other, count in others.most_common():
            pct = round(count / cat_count * 100)
            if pct >= threshold_pct:
                related.append({"category": other, "pct": pct})

        if related:
            # Sort by pct desc, cap
            related.sort(key=lambda x: x["pct"], reverse=True)
            result[cat] = related[:cap]

    return result


def detect_industry_column(fieldnames: list[str]) -> str | None:
    """Auto-detect the industry/category column name."""
    candidates = ["industry", "industries", "category", "categories", "trade", "trades"]
    for name in fieldnames:
        if name.lower().strip() in candidates:
            return name
    # Fuzzy: any column containing "industr" or "categor"
    for name in fieldnames:
        nl = name.lower()
        if "industr" in nl or "categor" in nl:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyse SS business CSV for category co-occurrence"
    )
    parser.add_argument("csv_path", help="Path to the SS business export CSV")
    parser.add_argument(
        "--threshold", type=int, default=15,
        help="Minimum co-occurrence percentage to include (default: 15)"
    )
    parser.add_argument(
        "--cap", type=int, default=5,
        help="Maximum related categories per trade (default: 5)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: resources/related_categories.json)"
    )
    parser.add_argument(
        "--column", default=None,
        help="Column name for industry/category (auto-detected if omitted)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "resources" / "related_categories.json"
    )

    # Read CSV
    rows: list[list[str]] = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        col_name = args.column or detect_industry_column(reader.fieldnames or [])
        if not col_name:
            print(f"Error: Could not detect industry column. Columns: {reader.fieldnames}", file=sys.stderr)
            print("Use --column to specify explicitly.", file=sys.stderr)
            sys.exit(1)
        print(f"Using column: '{col_name}'")

        for row in reader:
            raw = row.get(col_name, "").strip()
            if not raw:
                skipped += 1
                continue
            cats = parse_categories(raw)
            if len(cats) >= 1:
                rows.append(cats)

    print(f"Loaded {len(rows)} businesses ({skipped} skipped, no industry)")

    # Build co-occurrence
    cooccurrence, category_counts = build_cooccurrence(rows)
    print(f"Found {len(category_counts)} unique categories")

    # Compute related
    related = compute_related(cooccurrence, category_counts, args.threshold, args.cap)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(related, f, indent=2)
    print(f"Wrote {output_path} ({len(related)} categories with related entries)")

    # Summary stats
    print("\n── Summary ──")
    avg_related = sum(len(v) for v in related.values()) / max(len(related), 1)
    print(f"Average related categories: {avg_related:.1f}")

    # Top 10 most connected
    top = sorted(related.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("\nTop 10 most connected categories:")
    for cat, rels in top:
        names = ", ".join(f"{r['category']} ({r['pct']}%)" for r in rels[:3])
        print(f"  {cat} ({len(rels)} related): {names}")

    # Multi-category businesses stats
    multi = [r for r in rows if len(r) >= 2]
    print(f"\nMulti-category businesses: {len(multi)}/{len(rows)} ({len(multi)/max(len(rows),1)*100:.0f}%)")


if __name__ == "__main__":
    main()

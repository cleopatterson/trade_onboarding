"""Unit tests for core tool functions in agent/tools.py.

Tests parsing logic and deterministic helpers only — no HTTP calls.
"""
import json
import pytest
from agent.tools import (
    _parse_jsonp_response,
    compute_service_gaps,
    get_suburbs_in_radius_grouped,
    find_subcategory_guide,
    get_regional_guide,
    search_suburbs_by_postcode,
)


# ────────── _parse_jsonp_response ──────────

class TestParseJsonpResponse:
    """Tests for ABR JSONP parsing."""

    def test_abn_lookup_single_result(self):
        jsonp = 'c({"Abn":"51824753556","EntityName":"SMITH PLUMBING PTY LTD","EntityTypeName":"Australian Private Company","Gst":"2001-07-01","AddressState":"NSW","AddressPostcode":"2093","AbnStatus":"Active","EntityStartDate":"2010-03-15"})'
        result = _parse_jsonp_response(jsonp, "abn")
        assert result["count"] == 1
        assert result["results"][0]["abn"] == "51824753556"
        assert result["results"][0]["entity_name"] == "SMITH PLUMBING PTY LTD"
        assert result["results"][0]["gst_registered"] is True
        assert result["results"][0]["state"] == "NSW"
        assert result["results"][0]["postcode"] == "2093"
        assert result["results"][0]["status"] == "Active"

    def test_abn_lookup_no_result(self):
        jsonp = 'c({"Abn":"","Message":"No matching record found"})'
        result = _parse_jsonp_response(jsonp, "abn")
        assert result["count"] == 0
        assert result["results"] == []
        assert "error" in result

    def test_name_search_deduplicates_by_abn(self):
        jsonp = 'c({"Names":[{"Abn":"11111111111","Name":"SMITH PTY LTD","NameType":"Entity Name","State":"NSW","Postcode":"2000","Score":100},{"Abn":"11111111111","Name":"Smith Plumbing","NameType":"Business Name","State":"NSW","Postcode":"2000","Score":95}]})'
        result = _parse_jsonp_response(jsonp, "name")
        assert result["count"] == 1
        # Business Name should be preferred over Entity Name
        assert result["results"][0]["entity_name"] == "Smith Plumbing"

    def test_name_search_multiple_abns(self):
        jsonp = 'c({"Names":[{"Abn":"11111111111","Name":"Smith A","NameType":"Entity Name","State":"NSW","Postcode":"2000","Score":100},{"Abn":"22222222222","Name":"Smith B","NameType":"Entity Name","State":"VIC","Postcode":"3000","Score":90}]})'
        result = _parse_jsonp_response(jsonp, "name")
        assert result["count"] == 2

    def test_invalid_jsonp_returns_error(self):
        result = _parse_jsonp_response("not valid jsonp", "abn")
        assert result["count"] == 0
        assert "error" in result

    def test_malformed_json_inside_callback(self):
        result = _parse_jsonp_response("c({bad json})", "abn")
        assert result["count"] == 0
        assert "error" in result


# ────────── compute_service_gaps ──────────

class TestComputeServiceGaps:
    """Tests for deterministic service gap computation."""

    def test_no_services_with_matching_business_name(self):
        """Should return all subcategories for the matched trade."""
        gaps = compute_service_gaps([], "Dan's Plumbing Pty Ltd")
        assert len(gaps) > 0
        assert all(g.get("subcategory_id") for g in gaps)
        assert all(g.get("subcategory_name") for g in gaps)

    def test_no_match_returns_empty(self):
        """Unknown business name should return no gaps."""
        gaps = compute_service_gaps([], "Acme Corporation")
        assert gaps == []

    def test_mapped_services_reduce_gaps(self):
        """Mapping some services should reduce the gap count."""
        all_gaps = compute_service_gaps([], "Smith Electrical Services")
        if not all_gaps:
            pytest.skip("No electrician category in taxonomy")
        first_gap = all_gaps[0]
        mapped = [{
            "category_name": first_gap["category_name"],
            "category_id": first_gap["category_id"],
            "subcategory_name": first_gap["subcategory_name"],
            "subcategory_id": first_gap["subcategory_id"],
        }]
        remaining = compute_service_gaps(mapped, "Smith Electrical Services")
        assert len(remaining) == len(all_gaps) - 1

    def test_licence_class_detection(self):
        """Should detect trade from licence classes when name doesn't match."""
        gaps = compute_service_gaps([], "JONES, MATTHEW PAUL", licence_classes=["Electrician"])
        assert len(gaps) > 0

    def test_google_name_detection(self):
        """Should detect trade from Google Places business name."""
        gaps = compute_service_gaps([], "STACEY, MATTHEW GREGORY",
                                     google_business_name="Stacey Electrical")
        assert len(gaps) > 0

    def test_google_type_detection(self):
        """Should detect trade from Google Places primary type."""
        gaps = compute_service_gaps([], "Some Company Pty Ltd",
                                     google_primary_type="electrician")
        assert len(gaps) > 0

    def test_existing_services_override_business_name(self):
        """If services are already mapped to a category, use that over business name."""
        # Map a plumbing service for a business named "electrical"
        mapped = [{
            "category_name": "Plumber",
            "category_id": 999,
            "subcategory_name": "Hot Water Systems",
            "subcategory_id": 888,
        }]
        gaps = compute_service_gaps(mapped, "Smith Electrical Services")
        # Should return plumbing gaps, not electrical
        if gaps:
            assert gaps[0]["category_name"] == "Plumber"


# ────────── get_suburbs_in_radius_grouped ──────────

class TestGetSuburbsInRadiusGrouped:
    """Tests for suburb radius grouping."""

    def test_known_postcode_returns_base(self):
        """Postcode 2093 (Balgowlah, NSW) should return base info."""
        result = get_suburbs_in_radius_grouped("2093", 5.0)
        assert result.get("base_suburb") is not None
        assert result.get("base_lat", 0) != 0
        assert result.get("base_lng", 0) != 0

    def test_unknown_postcode_returns_empty(self):
        result = get_suburbs_in_radius_grouped("0000", 20.0)
        assert result.get("base_suburb") is None
        assert result.get("total") == 0

    def test_larger_radius_returns_more_suburbs(self):
        small = get_suburbs_in_radius_grouped("2093", 5.0)
        large = get_suburbs_in_radius_grouped("2093", 20.0)
        assert large.get("total", 0) >= small.get("total", 0)

    def test_results_grouped_by_area(self):
        result = get_suburbs_in_radius_grouped("2093", 20.0)
        by_area = result.get("by_area", {})
        if by_area:
            for area, suburbs in by_area.items():
                assert isinstance(suburbs, list)
                for s in suburbs:
                    assert "name" in s
                    assert "postcode" in s


# ────────── find_subcategory_guide ──────────

class TestFindSubcategoryGuide:
    """Tests for trade keyword → guide file matching."""

    def test_plumber_match(self):
        guide = find_subcategory_guide("Dan's Plumbing Pty Ltd")
        assert len(guide) > 100  # Should return substantial content

    def test_electrician_match(self):
        guide = find_subcategory_guide("Smith Electrical Services")
        assert len(guide) > 100

    def test_no_match(self):
        guide = find_subcategory_guide("Acme Corporation")
        assert guide == ""

    def test_case_insensitive(self):
        guide1 = find_subcategory_guide("SMITH PLUMBING")
        guide2 = find_subcategory_guide("smith plumbing")
        assert guide1 == guide2


# ────────── get_regional_guide ──────────

class TestGetRegionalGuide:
    """Tests for state code → regional guide mapping."""

    def test_nsw_returns_sydney_guide(self):
        guide = get_regional_guide("NSW")
        assert len(guide) > 100

    def test_vic_returns_melbourne_guide(self):
        guide = get_regional_guide("VIC")
        assert len(guide) > 100

    def test_unknown_state_returns_empty(self):
        guide = get_regional_guide("NT")
        assert guide == ""

    def test_case_insensitive(self):
        guide1 = get_regional_guide("nsw")
        guide2 = get_regional_guide("NSW")
        assert guide1 == guide2

    def test_caching(self):
        """Subsequent calls should return from cache."""
        guide1 = get_regional_guide("QLD")
        guide2 = get_regional_guide("QLD")
        assert guide1 is guide2  # Same object = cached

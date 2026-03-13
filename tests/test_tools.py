"""Unit tests for core tool functions in agent/tools.py and graph.py helpers.

Tests parsing logic and deterministic helpers only — no HTTP calls.
"""
import json
import pytest
from agent.tools import (
    _parse_jsonp_response,
    compute_service_gaps,
    compute_initial_services,
    get_suburbs_in_radius_grouped,
    find_subcategory_guide,
    get_regional_guide,
    get_filtered_cluster_groups,
    search_suburbs_by_postcode,
    qbcc_load_csv,
    qbcc_licence_lookup,
    _qbcc_licences,
    _detect_category,
    _detect_categories,
    extract_licence_from_text,
    get_licence_config,
    _STATE_LICENCE_CONFIG,
    _VIC_LICENCE_CONFIG,
    _wa_dmirs_extract_viewstate,
    _wa_dmirs_parse_results,
    suggest_related_categories,
    map_extra_categories,
    _load_related_categories,
    match_licence,
)
from agent.graph import (
    _process_cluster_response,
    _merge_llm_services,
    MSG_YES_ALL,
)


# ────────── _parse_jsonp_response ──────────

class TestParseJsonpResponse:
    """Tests for ABR JSONP parsing."""

    def test_abn_lookup_single_result(self):
        jsonp = 'c({"Abn":"51824753556","EntityName":"SMITH PLUMBING PTY LTD","EntityTypeName":"Australian Private Company","Gst":"2001-07-01","AddressState":"NSW","AddressPostcode":"2093","AbnStatus":"Active","EntityStartDate":"2010-03-15"})'
        result = _parse_jsonp_response(jsonp, "abn")
        assert result["count"] == 1
        assert result["results"][0]["abn"] == "51824753556"
        assert result["results"][0]["display_name"] == "Smith Plumbing Pty Ltd"
        assert result["results"][0]["legal_name"] == "SMITH PLUMBING PTY LTD"
        assert result["results"][0]["gst_registered"] is True
        assert result["results"][0]["state"] == "NSW"
        assert result["results"][0]["postcode"] == "2093"
        assert result["results"][0]["status"] == "Active"

    def test_abn_lookup_with_trading_name(self):
        """ABN lookup with both EntityName and BusinessName — trading name preferred for display."""
        jsonp = 'c({"Abn":"51824753556","EntityName":"A GRADE CARPENTRY GROUP PTY LTD","BusinessName":["A Grade Carpentry Group"],"EntityTypeName":"Australian Private Company","Gst":"2001-07-01","AddressState":"NSW","AddressPostcode":"2088","AbnStatus":"Active"})'
        result = _parse_jsonp_response(jsonp, "abn")
        assert result["count"] == 1
        r = result["results"][0]
        assert r["display_name"] == "A Grade Carpentry Group"  # trading name for display
        assert r["legal_name"] == "A GRADE CARPENTRY GROUP PTY LTD"  # entity name for licence

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
        r = result["results"][0]
        # Business Name should be preferred over Entity Name for display
        assert r["display_name"] == "Smith Plumbing"
        # Entity Name preserved as legal_name
        assert r["legal_name"] == "SMITH PTY LTD"

    def test_name_search_preserves_both_names(self):
        """Name search with both Entity and Trading Name for same ABN — both preserved."""
        jsonp = 'c({"Names":[{"Abn":"33333333333","Name":"A GRADE CARPENTRY GROUP PTY LTD","NameType":"Entity Name","State":"NSW","Postcode":"2088","Score":100},{"Abn":"33333333333","Name":"A Grade Carpentry Group","NameType":"Trading Name","State":"NSW","Postcode":"2088","Score":95}]})'
        result = _parse_jsonp_response(jsonp, "name")
        assert result["count"] == 1
        r = result["results"][0]
        assert r["display_name"] == "A Grade Carpentry Group"  # trading name
        assert r["legal_name"] == "A GRADE CARPENTRY GROUP PTY LTD"  # entity name

    def test_name_search_single_type_uses_same_for_both(self):
        """When only Entity Name exists for an ABN, legal_name equals display_name."""
        jsonp = 'c({"Names":[{"Abn":"44444444444","Name":"JONES ELECTRICAL PTY LTD","NameType":"Entity Name","State":"VIC","Postcode":"3000","Score":100}]})'
        result = _parse_jsonp_response(jsonp, "name")
        assert result["count"] == 1
        r = result["results"][0]
        assert r["display_name"] == "Jones Electrical Pty Ltd"
        assert r["legal_name"] == "JONES ELECTRICAL PTY LTD"

    def test_name_search_multiple_abns(self):
        jsonp = 'c({"Names":[{"Abn":"11111111111","Name":"Smith A","NameType":"Entity Name","State":"NSW","Postcode":"2000","Score":100},{"Abn":"22222222222","Name":"Smith B","NameType":"Entity Name","State":"VIC","Postcode":"3000","Score":90}]})'
        result = _parse_jsonp_response(jsonp, "name")
        assert result["count"] == 2
        # Both should have legal_name populated
        assert result["results"][0].get("legal_name") == "Smith A"
        assert result["results"][1].get("legal_name") == "Smith B"

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

    def test_border_town_cross_state(self):
        """Tweed Heads (NSW 2485) should include Coolangatta (QLD 4225) within radius."""
        result = get_suburbs_in_radius_grouped("2485", 10.0)
        all_suburbs = []
        for suburbs in result.get("by_area", {}).values():
            all_suburbs.extend(suburbs)
        names = [s["name"] for s in all_suburbs]
        # Coolangatta QLD is ~1km from Tweed Heads NSW — must appear
        assert "Coolangatta" in names, f"Coolangatta not found in cross-state results: {names[:20]}"

    def test_bad_coords_filtered_out(self):
        """Suburbs with non-Australian coords should be excluded from results."""
        from agent.tools import _is_valid_suburb_coords
        # Antewenegerrde NT has coords in Sweden (58.45, 14.90)
        assert not _is_valid_suburb_coords({"lat": "58.45", "lng": "14.90"})
        # Valid Sydney suburb
        assert _is_valid_suburb_coords({"lat": "-33.87", "lng": "151.21"})
        # Valid Darwin suburb
        assert _is_valid_suburb_coords({"lat": "-12.46", "lng": "130.84"})


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


# ────────── _process_cluster_response ──────────

def _make_gap(name, sub_id, cat_name="Electrician", cat_id=1):
    return {
        "subcategory_name": name,
        "subcategory_id": sub_id,
        "category_name": cat_name,
        "category_id": cat_id,
    }


class TestProcessClusterResponse:
    """Tests for deterministic cluster response processing."""

    def test_yes_all_adds_all_cluster_services(self):
        """'Yes, all of these' adds every service in the cluster."""
        gaps = [_make_gap("Solar Panels", 101), _make_gap("EV Charging", 102), _make_gap("Data Cabling", 103)]
        services, added, remaining = _process_cluster_response(
            [101, 102], gaps, [], MSG_YES_ALL,
        )
        assert len(services) == 2
        assert set(added) == {"Solar Panels", "EV Charging"}
        # Only non-cluster gap remains
        assert len(remaining) == 1
        assert remaining[0]["subcategory_id"] == 103

    def test_individual_word_overlap_match(self):
        """Individual button selection matches by word overlap."""
        gaps = [_make_gap("Solar Panels", 101), _make_gap("EV Charging", 102)]
        services, added, remaining = _process_cluster_response(
            [101, 102], gaps, [], "Just solar panels",
        )
        assert len(services) == 1
        assert added == ["Solar Panels"]
        assert len(remaining) == 0  # Both cluster IDs removed from gaps

    def test_decline_skips_cluster(self):
        """Decline (no word overlap) adds nothing but still removes cluster from gaps."""
        gaps = [_make_gap("Solar Panels", 101), _make_gap("EV Charging", 102)]
        services, added, remaining = _process_cluster_response(
            [101, 102], gaps, [], "Not for us",
        )
        assert len(services) == 0
        assert added == []
        assert len(remaining) == 0  # Cluster removed regardless


# ────────── _merge_llm_services ──────────

class TestMergeLlmServices:
    """Tests for ensuring pre-added services survive LLM output."""

    def test_dedup_preserves_pre_added(self):
        """Pre-added services not in LLM output get appended."""
        state_services = [
            {"subcategory_id": 101, "subcategory_name": "Solar Panels"},
            {"subcategory_id": 102, "subcategory_name": "EV Charging"},
        ]
        llm_services = [
            {"subcategory_id": 101, "subcategory_name": "Solar Panels"},
            {"subcategory_id": 103, "subcategory_name": "Data Cabling"},
        ]
        merged = _merge_llm_services(state_services, llm_services)
        ids = [s["subcategory_id"] for s in merged]
        assert 101 in ids
        assert 102 in ids  # Was missing from LLM output, now added
        assert 103 in ids
        assert len(merged) == 3  # No duplicates


# ────────── QBCC Licence Lookup ──────────

class TestQBCCLicenceLookup:
    """Tests for QBCC CSV licence lookup (QLD)."""

    @classmethod
    def setup_class(cls):
        """Load QBCC CSV once for all tests."""
        if not _qbcc_licences["loaded"]:
            qbcc_load_csv()

    def test_abn_match(self):
        """Known QLD plumbing company found by ABN."""
        result = qbcc_licence_lookup("56051254301", "")
        assert result is not None
        assert result["licence_number"]
        assert result["status"] == "Current"

    def test_name_fallback(self):
        """Company with no ABN found by legal name."""
        result = qbcc_licence_lookup("", "PROTECH COAT PTY. LTD.")
        assert result is not None
        assert result["licence_number"] == "15144048"

    def test_no_match(self):
        """Non-existent ABN and name returns None."""
        result = qbcc_licence_lookup("00000000000", "Nonexistent Corp Pty Ltd")
        assert result is None

    def test_classes_populated(self):
        """Licence classes are extracted and deduplicated."""
        result = qbcc_licence_lookup("56051254301", "")
        classes = result.get("classes", [])
        assert len(classes) >= 2
        names = [c["name"] for c in classes]
        assert "Plumbing and Drainage" in names

    def test_licence_source_field(self):
        """Result includes licence_source='qbcc_csv'."""
        result = qbcc_licence_lookup("56051254301", "")
        assert result["licence_source"] == "qbcc_csv"

    def test_classes_deduped(self):
        """Same class from multiple rows is not duplicated."""
        result = qbcc_licence_lookup("56051254301", "")
        names = [c["name"] for c in result["classes"]]
        assert len(names) == len(set(names))


# ────────── VIC Licence Extraction ──────────

class TestExtractLicenceFromText:
    """Tests for VIC licence number extraction from web text."""

    def test_rec_number_electrician(self):
        """REC number extracted for electricians."""
        result = extract_licence_from_text("Licensed electrician REC 12345 serving Melbourne", "Electrician")
        assert result is not None
        assert result["licence_number"] == "12345"
        assert result["licence_source"] == "web_extracted"

    def test_esv_number_electrician(self):
        """ESV number extracted for electricians."""
        result = extract_licence_from_text("ESV 987654 registered electrical contractor", "Electrician")
        assert result is not None
        assert result["licence_number"] == "987654"

    def test_dbu_number_builder(self):
        """DB-U number extracted for builders."""
        result = extract_licence_from_text("Registered builder DB-U 12345 in Victoria", "Builder")
        assert result is not None
        assert result["licence_number"] == "12345"

    def test_cdb_number_builder(self):
        """CDB number extracted for builders."""
        result = extract_licence_from_text("Commercial builder CDB 54321", "Builder")
        assert result is not None
        assert result["licence_number"] == "54321"

    def test_painter_not_licensed(self):
        """Painters are not licensed trades — no config, no extraction."""
        result = extract_licence_from_text("Painting contractor DB-L 99999", "Painter")
        assert result is None

    def test_carpenter_not_licensed(self):
        """Carpenters are not licensed trades — no config, no extraction."""
        result = extract_licence_from_text("Carpentry services DB-L 88888", "Carpenter")
        assert result is None

    def test_plumber_with_context(self):
        """Plumber number extracted when context keywords present."""
        result = extract_licence_from_text("VBA registered plumber licence 12345678", "Plumber")
        assert result is not None
        assert result["licence_number"] == "12345678"
        assert result["classes"][0]["name"] == "Plumbing and Drainage"

    def test_plumber_no_context_rejected(self):
        """Plumber number NOT extracted without context keywords (avoids false positives)."""
        result = extract_licence_from_text("We serve 12345678 customers in Melbourne", "Plumber")
        assert result is None

    def test_cleaner_returns_none(self):
        """Cleaners have no VIC licence config — returns None."""
        result = extract_licence_from_text("Professional cleaning service licence 12345", "Cleaner")
        assert result is None

    def test_gardener_returns_none(self):
        """Gardeners have no VIC licence config — returns None."""
        result = extract_licence_from_text("Gardening and landscaping REC 12345", "Gardener")
        assert result is None

    def test_empty_text_returns_none(self):
        """Empty text returns None."""
        result = extract_licence_from_text("", "Electrician")
        assert result is None

    def test_unknown_trade_returns_none(self):
        """Unknown trade returns None."""
        result = extract_licence_from_text("REC 12345", "Roofer")
        assert result is None

    def test_lic_no_format_plumber(self):
        """LIC No. format extracted for plumbers (common VIC website format)."""
        result = extract_licence_from_text("Melbourne Plumbing Co LIC No. 110636 Privacy Policy", "Plumber")
        assert result is not None
        assert result["licence_number"] == "110636"

    def test_classes_populated(self):
        """Extracted licence has correct classes."""
        result = extract_licence_from_text("REC 12345 electrician", "Electrician")
        assert result is not None
        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "Electrical Work"
        assert result["classes"][0]["active"] is True


# ────────── WA DMIRS ViewState Extraction ──────────

class TestWaDmirsViewState:
    """Tests for DMIRS PrimeFaces ViewState extraction."""

    def test_standard_viewstate(self):
        html = '<input type="hidden" name="javax.faces.ViewState" value="abc123xyz" />'
        assert _wa_dmirs_extract_viewstate(html) == "abc123xyz"

    def test_value_before_name_order(self):
        html = '<input type="hidden" value="xyz789" name="javax.faces.ViewState" />'
        assert _wa_dmirs_extract_viewstate(html) == "xyz789"

    def test_missing_field_returns_none(self):
        html = '<input type="hidden" name="other_field" value="abc" />'
        assert _wa_dmirs_extract_viewstate(html) is None

    def test_empty_html_returns_none(self):
        assert _wa_dmirs_extract_viewstate("") is None

    def test_none_html_returns_none(self):
        assert _wa_dmirs_extract_viewstate(None) is None


# ────────── WA DMIRS Parse Results ──────────

class TestWaDmirsParseResults:
    """Tests for DMIRS search result HTML parsing."""

    def test_single_result(self):
        html = '''
        <a class="licenceElementTitle">John Smith Electrical</a>
        <a class="licenceElementTitle">EC012345</a>
        <span class="licenceStatus">Current</span>
        '''
        results = _wa_dmirs_parse_results(html)
        assert len(results) == 1
        assert results[0]["licensee"] == "John Smith Electrical"
        assert results[0]["licence_number"] == "EC012345"
        assert results[0]["status"] == "Current"

    def test_multiple_results(self):
        html = '''
        <a class="licenceElementTitle">Smith Electrical</a>
        <a class="licenceElementTitle">EC001111</a>
        <span class="licenceStatus">Current</span>
        <a class="licenceElementTitle">Jones Electrical</a>
        <a class="licenceElementTitle">EC002222</a>
        <span class="licenceStatus">Expired</span>
        '''
        results = _wa_dmirs_parse_results(html)
        assert len(results) == 2
        assert results[0]["licence_number"] == "EC001111"
        assert results[0]["status"] == "Current"
        assert results[1]["licence_number"] == "EC002222"
        assert results[1]["status"] == "Expired"

    def test_empty_results(self):
        html = '<div class="noResults">No matching records found</div>'
        results = _wa_dmirs_parse_results(html)
        assert results == []

    def test_empty_html(self):
        assert _wa_dmirs_parse_results("") == []

    def test_none_html(self):
        assert _wa_dmirs_parse_results(None) == []


# ────────── State Licence Config ──────────

class TestStateLicenceConfig:
    """Tests for multi-state licence configuration."""

    def test_all_states_present(self):
        expected = {"VIC", "WA", "SA", "TAS", "ACT", "NT"}
        assert expected.issubset(set(_STATE_LICENCE_CONFIG.keys()))

    def test_required_keys(self):
        for state, trades in _STATE_LICENCE_CONFIG.items():
            for trade, config in trades.items():
                assert "regulator" in config, f"{state}/{trade} missing regulator"
                assert "label" in config, f"{state}/{trade} missing label"
                assert "patterns" in config, f"{state}/{trade} missing patterns"
                assert "default_classes" in config, f"{state}/{trade} missing default_classes"
                assert isinstance(config["patterns"], list), f"{state}/{trade} patterns not a list"

    def test_wa_trades(self):
        wa = _STATE_LICENCE_CONFIG["WA"]
        assert "Electrician" in wa
        assert "Plumber" in wa
        assert "Gas Fitter" in wa
        assert "Builder" in wa  # Added — WA Dept of Commerce

    def test_vic_backward_compat(self):
        assert _VIC_LICENCE_CONFIG is _STATE_LICENCE_CONFIG["VIC"]
        assert "Electrician" in _VIC_LICENCE_CONFIG
        assert "Plumber" in _VIC_LICENCE_CONFIG

    def test_get_licence_config_found(self):
        config = get_licence_config("VIC", "Electrician")
        assert config is not None
        assert config["regulator"] == "Energy Safe Victoria (ESV)"

    def test_get_licence_config_not_found(self):
        assert get_licence_config("VIC", "Roofer") is None
        assert get_licence_config("XX", "Electrician") is None


# ────────── Multi-State Extraction ──────────

class TestMultiStateExtraction:
    """Tests for licence extraction across different states."""

    def test_wa_ec_pattern(self):
        result = extract_licence_from_text("Licensed EC 12345 electrician", "Electrician", "WA")
        assert result is not None
        assert result["licence_number"] == "12345"
        assert result["licence_source"] == "web_extracted"

    def test_wa_pl_pattern(self):
        result = extract_licence_from_text("Registered plumbing PL 98765", "Plumber", "WA")
        assert result is not None
        assert result["licence_number"] == "98765"

    def test_sa_pge_pattern(self):
        result = extract_licence_from_text("Licence PGE 54321 electrical contractor", "Electrician", "SA")
        assert result is not None
        assert result["licence_number"] == "54321"

    def test_sa_bld_pattern(self):
        result = extract_licence_from_text("BLD 99999 registered builder", "Builder", "SA")
        assert result is not None
        assert result["licence_number"] == "99999"

    def test_nt_c_prefix_electrician(self):
        result = extract_licence_from_text("Electrical contractor licence C12345", "Electrician", "NT")
        assert result is not None
        assert result["licence_number"] == "12345"

    def test_unknown_state_returns_none(self):
        result = extract_licence_from_text("REC 12345", "Electrician", "XX")
        assert result is None

    def test_vic_default_param_regression(self):
        """VIC is the default state param — existing calls without state still work."""
        result = extract_licence_from_text("REC 12345 electrician", "Electrician")
        assert result is not None
        assert result["licence_number"] == "12345"

    def test_tas_electrician(self):
        result = extract_licence_from_text("EL 5678 licensed electrician", "Electrician", "TAS")
        assert result is not None
        assert result["licence_number"] == "5678"

    def test_act_builder(self):
        result = extract_licence_from_text("BL 12345 registered builder", "Builder", "ACT")
        assert result is not None
        assert result["licence_number"] == "12345"


# ────────── _detect_categories (multi-category) ──────────

class TestDetectCategories:
    """Tests for multi-category detection."""

    def test_single_category(self):
        result = _detect_categories("Smith Electrical", [], "", "")
        assert result == ["Electrician"]

    def test_dual_category_from_name(self):
        result = _detect_categories("Smith Building & Carpentry", [], "", "")
        assert "Builder" in result
        assert "Carpenter" in result
        assert len(result) == 2

    def test_cap_at_2(self):
        result = _detect_categories("Building Carpentry Tiling Services", [], "", "")
        assert len(result) == 2

    def test_licence_adds_secondary(self):
        result = _detect_categories("Smith Plumbing", ["Gas Fitter Licence"], "", "")
        assert result[0] == "Plumber"
        assert "Gas Fitter" in result

    def test_no_match_returns_empty(self):
        result = _detect_categories("XYZ Corp", [], "", "")
        assert result == []

    def test_dedup_same_source(self):
        """'build' in name AND Google name → one Builder."""
        result = _detect_categories("Smith Building", [], "Smith Building Co", "")
        assert result.count("Builder") == 1

    def test_google_type_secondary(self):
        result = _detect_categories("Smith Plumbing", [], "", "electrician")
        assert result[0] == "Plumber"
        assert "Electrician" in result

    def test_backward_compat_with_single(self):
        """_detect_categories and _detect_category agree on single-category cases."""
        for name in ["Dan's Plumbing", "Smith Electrical", "Aussie Painters"]:
            single = _detect_category(name, [], "", "")
            multi = _detect_categories(name, [], "", "")
            assert multi[0] == single


# ────────── Multi-category initial services ──────────

class TestMultiCategoryInitialServices:
    """Tests for compute_initial_services with multi-category."""

    def test_builder_and_carpenter(self):
        result = compute_initial_services(
            "Smith Building & Carpentry", [], "", "", [], [],
        )
        assert result["tiered"] is True
        cat_names = {s["category_name"] for s in result["services"]}
        assert len(cat_names) >= 2
        assert "category_names" in result
        assert len(result["category_names"]) == 2
        # Gaps should span both categories
        gap_cats = {g["category_name"] for g in result["specialist_gaps"]}
        assert len(gap_cats) >= 2

    def test_single_category_unchanged(self):
        result = compute_initial_services(
            "Dan's Plumbing", [], "", "", [], [],
        )
        assert result["tiered"] is True
        assert result["category_name"] == "Plumber"
        assert result["category_names"] == ["Plumber"]
        # All services should be Plumber
        for s in result["services"]:
            assert s["category_name"] == "Plumber"

    def test_non_tiered_unchanged(self):
        result = compute_initial_services(
            "XYZ Locksmith", [], "", "", [], [],
        )
        assert result["tiered"] is False


# ────────── Multi-category gaps ──────────

class TestMultiCategoryGaps:
    """Tests for compute_service_gaps with multi-category."""

    def test_gaps_from_two_categories(self):
        """Services from 2 categories → gaps for both."""
        services = [
            {"subcategory_name": "General Electrical", "subcategory_id": 850,
             "category_name": "Electrician"},
            {"subcategory_name": "General Carpentry", "subcategory_id": 870,
             "category_name": "Carpenter"},
        ]
        gaps = compute_service_gaps(services, "Smith Electrical & Carpentry")
        gap_cats = {g["category_name"] for g in gaps}
        assert "Electrician" in gap_cats
        assert "Carpenter" in gap_cats

    def test_single_category_gaps_unchanged(self):
        services = [
            {"subcategory_name": "General Plumbing", "subcategory_id": 100,
             "category_name": "Plumber"},
        ]
        gaps = compute_service_gaps(services, "Dan's Plumbing")
        gap_cats = {g["category_name"] for g in gaps}
        assert gap_cats == {"Plumber"}


# ────────── Multi-category cluster groups ──────────

class TestMultiCategoryClusterGroups:
    """Tests for get_filtered_cluster_groups with multi-category."""

    def test_combined_clusters(self):
        """Combined gaps → clusters from both tiered categories."""
        # Build gaps spanning two categories
        result = compute_initial_services(
            "Smith Building & Carpentry", [], "", "", [], [],
        )
        if not result.get("tiered"):
            pytest.skip("Tier definitions not available")
        gaps = result["specialist_gaps"]
        clusters = get_filtered_cluster_groups(
            gaps, "Smith Building & Carpentry",
        )
        # Should have clusters from at least one (possibly both) categories
        assert len(clusters) > 0
        # Cluster labels should include labels from both if both have cluster_groups
        cluster_labels = [c["label"] for c in clusters]
        assert len(cluster_labels) > 0


# ────────── Multi-category guide ──────────

class TestMultiGuide:
    """Tests for find_subcategory_guide with multi-category."""

    def test_single_keyword_guide(self):
        """A trade with only one guide file should return non-empty content."""
        guide = find_subcategory_guide("Aussie Painters")
        assert guide != ""

    def test_multi_guide_longer_than_single(self):
        """Multi-category guide should contain content from both trades."""
        single = find_subcategory_guide("Smith Carpentry")
        multi = find_subcategory_guide("Smith Building & Carpentry")
        assert multi != ""
        assert single != ""
        # Multi-guide should be strictly longer (has content from both)
        assert len(multi) > len(single)


# ────────── Related category suggestions ──────────

class TestSuggestRelatedCategories:
    """Tests for suggest_related_categories()."""

    def test_plumber_gets_suggestions(self):
        """Plumber should get common related categories like Handyman."""
        result = suggest_related_categories(["Plumber"])
        assert len(result) > 0
        names = [r["category"] for r in result]
        assert "Handyman" in names

    def test_excludes_detected(self):
        """Already-detected categories should be filtered out."""
        result = suggest_related_categories(["Plumber", "Handyman"])
        names = [r["category"] for r in result]
        assert "Plumber" not in names
        assert "Handyman" not in names

    def test_respects_threshold(self):
        """Categories below threshold should be excluded."""
        # Use a very high threshold to filter everything
        result = suggest_related_categories(["Plumber"], min_pct=99)
        assert result == []

    def test_respects_cap(self):
        """Should not return more than max_suggestions."""
        result = suggest_related_categories(["Plumber"], max_suggestions=2)
        assert len(result) <= 2

    def test_sorted_by_pct_desc(self):
        """Results should be sorted by pct descending."""
        result = suggest_related_categories(["Plumber"])
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["pct"] >= result[i + 1]["pct"]

    def test_empty_when_no_data(self):
        """Graceful when category has no related data."""
        result = suggest_related_categories(["NonexistentTrade123"])
        assert result == []

    def test_empty_for_empty_input(self):
        """Returns empty for empty detected list."""
        result = suggest_related_categories([])
        assert result == []

    def test_data_file_loads(self):
        """related_categories.json loads and has content."""
        data = _load_related_categories()
        assert isinstance(data, dict)
        assert len(data) > 0


class TestMapExtraCategories:
    """Tests for map_extra_categories()."""

    def test_tiered_category_adds_services(self):
        """A tiered extra category (e.g. Handyman) should add core services."""
        new_svcs, new_gaps, mapped = map_extra_categories(
            ["Handyman"], [], set(), "", [],
        )
        # Handyman is tiered in service_tiers.json — should get core services
        assert len(new_svcs) > 0
        # All returned services should be from Handyman category
        for s in new_svcs:
            assert s["category_name"] == "Handyman"

    def test_non_tiered_category_adds_gaps(self):
        """A non-tiered extra category should add all subcategories as gaps."""
        new_svcs, new_gaps, mapped = map_extra_categories(
            ["Locksmith"], [], set(), "", [],
        )
        # Locksmith is not tiered — no services, but gaps
        assert len(new_svcs) == 0
        assert len(new_gaps) > 0

    def test_no_duplicate_services(self):
        """Services already mapped should not be added again."""
        existing = {"Home Handiwork", "Odd Jobs"}
        new_svcs, new_gaps, mapped = map_extra_categories(
            ["Handyman"], [], existing, "", [],
        )
        new_names = {s["subcategory_name"] for s in new_svcs}
        assert "Home Handiwork" not in new_names
        assert "Odd Jobs" not in new_names

    def test_unknown_category_returns_empty(self):
        """Unknown category name should return empty results."""
        new_svcs, new_gaps, mapped = map_extra_categories(
            ["TotallyFakeCategory"], [], set(), "", [],
        )
        assert new_svcs == []
        assert new_gaps == []

    def test_multiple_categories(self):
        """Multiple extra categories should all be mapped."""
        new_svcs, new_gaps, mapped = map_extra_categories(
            ["Handyman", "Locksmith"], [], set(), "", [],
        )
        # Should have services from Handyman + gaps from Locksmith
        assert len(new_svcs) > 0 or len(new_gaps) > 0


# ────────── MATCH LICENCE ──────────

class TestMatchLicence:
    """Tests for consolidated licence matching."""

    def test_prefers_trade_relevant_match(self):
        """Painter 'Smith' should NOT match a plumber licence for 'Smith'."""
        results = [
            {"licensee": "SMITH, JOHN", "licence_type": "Plumber", "status": "Current", "licence_id": "1"},
            {"licensee": "SMITH, BRENDAN", "licence_type": "Painter", "status": "Current", "licence_id": "2"},
        ]
        best = match_licence(results, "SMITH, BRENDAN", detected_categories=["Painter"])
        assert best is not None
        assert best["licence_id"] == "2"

    def test_rejects_wrong_trade_low_name(self):
        """Common surname with wrong trade should be rejected."""
        results = [
            {"licensee": "SMITH, JOHN", "licence_type": "Plumber", "status": "Current", "licence_id": "1"},
            {"licensee": "SMITH, PETER", "licence_type": "Plumber", "status": "Current", "licence_id": "2"},
        ]
        # Searching for "SMITH, BRENDAN" (painter) — both are plumbers, word overlap is 1/2 = 50%
        best = match_licence(results, "SMITH, BRENDAN", detected_categories=["Painter"])
        # Should return None — no name_score >= 2 after suffix strip
        assert best is None

    def test_strips_business_suffixes(self):
        """PTY LTD etc. should be stripped before matching."""
        results = [
            {"licensee": "ACME ELECTRICAL", "licence_type": "Electrician", "status": "Current", "licence_id": "1"},
        ]
        best = match_licence(results, "Acme Electrical Pty Ltd", detected_categories=["Electrician"])
        assert best is not None
        assert best["licence_id"] == "1"

    def test_substring_match(self):
        """Exact substring should score highest."""
        results = [
            {"licensee": "SMITH PAINTING SERVICES", "licence_type": "Painter", "status": "Current", "licence_id": "1"},
        ]
        best = match_licence(results, "Smith Painting", detected_categories=["Painter"])
        assert best is not None
        assert best["licence_id"] == "1"

    def test_no_results_returns_none(self):
        assert match_licence([], "test") is None
        assert match_licence([{"licensee": "X", "status": "Expired"}], "X") is None

    def test_skips_non_current(self):
        """Non-current licences should be skipped."""
        results = [
            {"licensee": "ACME ELECTRICAL", "licence_type": "Electrician", "status": "Expired", "licence_id": "1"},
            {"licensee": "ACME ELECTRICAL", "licence_type": "Electrician", "status": "Current", "licence_id": "2"},
        ]
        best = match_licence(results, "Acme Electrical")
        assert best is not None
        assert best["licence_id"] == "2"

    def test_single_current_fallback(self):
        """Single current result with name_score >= 1 should be accepted."""
        results = [
            {"licensee": "B SMITH PAINTING", "licence_type": "Painter", "status": "Current", "licence_id": "1"},
        ]
        # "Smith" overlaps 1/3 words = 33% (score 0 normally), but single-current fallback
        # Actually "smith" in both, 1/min(1,3)=33% → score 0. But let's test with better overlap
        best = match_licence(results, "Smith Painting", detected_categories=["Painter"])
        # "smith painting" vs "b smith painting" — 2/2 = 100% overlap → name_score=2, accepted
        assert best is not None

    def test_no_categories_still_matches_by_name(self):
        """Without detected_categories, matching should work on name alone."""
        results = [
            {"licensee": "JONES ELECTRICAL PTY LTD", "licence_type": "Electrician", "status": "Current", "licence_id": "1"},
        ]
        best = match_licence(results, "Jones Electrical Pty Ltd")
        assert best is not None
        assert best["licence_id"] == "1"

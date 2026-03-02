"""State definition for the Trade Onboarding wizard"""
from typing import TypedDict
from langchain_core.messages import BaseMessage


class OnboardingState(TypedDict):
    # Session
    session_id: str
    current_node: str

    # Conversation
    messages: list[BaseMessage]

    # Business Verification (from ABR)
    business_name_input: str
    abn_input: str
    abr_results: list[dict]
    business_name: str
    legal_name: str
    abn: str
    entity_type: str
    gst_registered: bool
    business_verified: bool
    business_postcode: str
    business_state: str

    # Licence Enrichment (from NSW Fair Trading, QBCC CSV, WA DMIRS, or web extraction)
    licence_info: dict
    licence_classes: list[str]
    _needs_licence_number: bool       # QLD/VIC: prompt for licence self-report
    _licence_self_report: dict         # data-driven self-report context (regulator, label, etc.)

    # Web Enrichment (from Brave Search + website scrape)
    web_results: list[dict]
    website_text: str               # scraped text from business website (for evidence keywords)

    # Service Discovery
    services_raw: str
    services: list[dict]
    services_confirmed: bool
    _svc_turn: int
    _specialist_gap_ids: list[int]     # persisted specialist gap IDs from tiered mapping
    _pending_cluster_ids: list[int]    # subcategory IDs asked about last turn (for deterministic processing)

    # Service Areas
    location_raw: str
    service_areas: dict
    service_areas_confirmed: bool

    # Contact (extracted from licence + web data)
    contact_name: str
    contact_phone: str

    # Completion
    confirmed: bool
    output_json: dict

    # Profile Builder
    abn_registration_date: str
    years_in_business: int
    profile_description: str
    profile_description_draft: str
    profile_logo: str
    profile_photos: list[str]
    profile_saved: bool
    profile_intro: str

    # Social / Reviews / Web
    google_rating: float              # e.g. 4.8
    google_review_count: int          # e.g. 47
    google_reviews: list[dict]        # [{text: "...", rating: 5}, ...]
    facebook_url: str                 # e.g. "https://facebook.com/SmithPlumbing"
    business_website: str             # from Google Places websiteUri
    google_business_name: str         # display name from Google Places
    google_primary_type: str          # e.g. "electrician", "plumber"

    # Pricing / Subscription
    pricing_shown: bool
    subscription_plan: str        # "standard" | "plus" | "pro" | "skip" | ""
    subscription_billing: str     # "monthly" | "quarterly" | "annual" | ""
    subscription_price: str       # e.g. "$79/mo"
    _selected_plan: str           # guard for pricing node turn 2 vs turn 3

    # UI / Auto-chain
    buttons: list[dict]           # node-generated button options
    _auto_chained: bool           # suppress stale user messages during auto-chain

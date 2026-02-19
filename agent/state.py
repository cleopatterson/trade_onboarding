"""State definition for the Trade Onboarding wizard"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class OnboardingState(TypedDict):
    # Session
    session_id: str
    current_node: str

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]

    # Business Verification (from ABR)
    business_name_input: str
    abn_input: str
    abr_results: list[dict]
    business_name: str
    abn: str
    entity_type: str
    gst_registered: bool
    business_verified: bool
    business_postcode: str
    business_state: str

    # Licence Enrichment (from NSW Fair Trading)
    licence_info: dict
    licence_classes: list[str]

    # Web Enrichment (from Brave Search)
    web_results: list[dict]

    # Service Discovery
    services_raw: str
    services: list[dict]
    services_confirmed: bool

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

    # Pricing / Subscription
    pricing_shown: bool
    subscription_plan: str        # "standard" | "plus" | "pro" | "skip" | ""
    subscription_billing: str     # "monthly" | "quarterly" | "annual" | ""
    subscription_price: str       # e.g. "$79/mo"

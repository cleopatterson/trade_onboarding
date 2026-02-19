# Trade Onboarding Wizard - Product Requirements Document

## 1. System Architecture

### 1.1 Overview

The Trade Onboarding Wizard is a Python/FastAPI service running a LangGraph agent that guides tradies through Service Seeking business registration via a conversational web interface.

```
User (Browser)
    │
    ▼
┌─────────────────────────────────┐
│  FastAPI Server (port 8001)     │
│  ├── POST /api/chat             │  ← JSON responses + LLM-generated buttons
│  ├── POST /api/session          │  ← Create new session
│  ├── GET  /api/session/:id      │  ← Resume session
│  ├── GET  /api/session/:id/result│ ← Final JSON output
│  ├── GET  /static/*             │  ← Landing + wizard HTML/JS
│  └── GET  /health               │  ← Health check
└─────────────┬───────────────────┘
              │
    ┌─────────┴──────────────────┐
    ▼                            ▼
┌──────────────┐    ┌───────────────────────┐
│ State Machine│    │  External APIs         │
│ (7 nodes)    │    │  ├── ABR JSON API      │
│              │    │  ├── NSW Fair Trading   │
│ Auto-chains  │    │  │   (OAuth2 + REST)    │
│ between steps│    │  ├── Brave Search API   │
│              │    │  ├── SS Category Taxonomy│
│              │    │  └── Suburbs CSV (15K+) │
└──────────────┘    └───────────────────────┘
```

### 1.2 File Structure (Actual)

```
Trade_onboarding/
├── docs/
│   ├── MASTERPLAN.md           # Vision and strategy
│   ├── PRD.md                  # This file
│   └── TASKS.md                # Implementation log and roadmap
│
├── agent/
│   ├── graph.py                # State machine — 7 nodes, no-scripting principle
│   ├── state.py                # OnboardingState TypedDict
│   ├── tools.py                # ABR, NSW Fair Trading, Brave Search, geo helpers
│   └── config.py               # Environment config, model IDs, API keys
│
├── server/
│   ├── __init__.py
│   └── app.py                  # FastAPI — endpoints, auto-chaining, button logic
│
├── web/
│   └── landing.html            # Landing page + wizard modal (all-in-one HTML/JS/CSS)
│
├── resources/
│   ├── suburbs.csv                    # 15,761 AU suburbs (id, name, state, postcode, lat, lng, area, region)
│   ├── subcategories.json             # SS category taxonomy (dict keyed by category)
│   ├── sydney_regions.md              # Barriers, congestion, corridors, high-demand areas
│   ├── melbourne_regions.md
│   ├── brisbane_regions.md
│   ├── perth_regions.md
│   ├── plumber-subcategory-guide.md   # Gaps-to-watch-for tables per trade
│   ├── electrician-subcategory-guide.md
│   ├── cleaner-subcategory-guide.md
│   ├── gardener-subcategory-guide.md
│   ├── plumbing_subcategories.md
│   ├── electrical_subcategories.md
│   ├── painter_subcategories.md
│   └── carpentry_subcategories.md
│
├── venv/                       # Python 3.9 virtualenv
├── requirements.txt
├── .env                        # API keys (ABR, NSW Fair Trading, Brave, Anthropic)
└── .env.example
```

## 2. Conversation Flow

### 2.1 State Machine

```
                    ┌─────────┐
                    │ WELCOME │
                    └────┬────┘
                         │ User provides business name
                         ▼
              ┌──────────────────────┐
              │ BUSINESS_VERIFICATION │◄──┐
              └──────────┬───────────┘   │
                         │               │ User says "no, that's not me"
                    ┌────┴────┐          │
                    │ ABR     │          │
                    │ Lookup  │──────────┘
                    └────┬────┘
                         │ User confirms business
                         ▼
              ┌──────────────────────┐
              │  SERVICE_DISCOVERY   │◄──┐
              └──────────┬───────────┘   │
                         │               │ User wants to add/change
                    ┌────┴────┐          │
                    │ Category│          │
                    │ Mapping │──────────┘
                    └────┬────┘
                         │ User confirms services
                         ▼
              ┌──────────────────────┐
              │  SERVICE_AREA_MAPPING │◄──┐
              └──────────┬────────────┘   │
                         │                │ User wants to adjust
                    ┌────┴────┐           │
                    │ Location│           │
                    │ Parser  │───────────┘
                    └────┬────┘
                         │ User confirms areas
                         ▼
              ┌──────────────────────┐
              │       PROFILE        │
              │ (preview + edit)     │
              └──────────┬───────────┘
                         │ User publishes profile
                         ▼
              ┌──────────────────────┐
              │       PRICING        │
              │ (plan + billing)     │──── User skips → straight to complete
              └──────────┬───────────┘
                         │ User selects plan + billing
                         ▼
                  ┌──────────────┐
                  │   COMPLETE   │
                  │ (output JSON)│
                  └──────────────┘
```

### 2.2 Node Specifications

#### WELCOME
- **Purpose:** Greet user and capture business name
- **Model:** Haiku 4.5 (fast, simple task)
- **Prompt:** Friendly Australian greeting, ask for business name or ABN
- **Output:** `business_name_input` or `abn_input` captured in state
- **Buttons:** None (free text input)

#### BUSINESS_VERIFICATION
- **Purpose:** Find and confirm the business via ABR, then enrich with licence + web data
- **Model:** Haiku 4.5 (classifier for intent: CONFIRMED/REJECTED/NEWSEARCH)
- **Tools:** `abr_lookup`, `nsw_licence_browse`, `nsw_licence_details`, `brave_web_search`
- **Flow:**
  1. Call ABR JSON API with business name or ABN
  2. If single match → present details card, ask "Is this your business?"
  3. If multiple matches → show as clickable buttons (ABN in value for identification)
  4. If no match → ask to try different name or enter ABN directly
  5. Quick-match: if button click contains ABN from results, confirm instantly (no LLM needed)
  6. **On confirm → parallel enrichment:** NSW Fair Trading licence lookup + Brave web search
  7. Licence details provide `licenceClasses` (e.g. Plumber, Gasfitter, Drainer)
  8. Web results provide website/Facebook/reviews context
- **Output:** `business_name`, `abn`, `entity_type`, `business_verified`, `licence_classes`, `licence_info`, `web_results`
- **Buttons:** Data-driven from ABR results (generated by server, not LLM)

#### SERVICE_DISCOVERY
- **Purpose:** Understand what services the business provides and map to SS categories
- **Model:** Sonnet 4.5 (needs strong NLU + taxonomy mapping)
- **Context:** Licence classes, web results, subcategory guide, full SS taxonomy, conversation history
- **Flow (no scripting — LLM decides the conversation):**
  1. If licence classes available → map services from licence data (high confidence)
  2. Otherwise → infer from business name + ask
  3. Use subcategory guide to ask smart gap questions (e.g. "do you do hot water systems?")
  4. 2-3 rounds max, then wrap up
  5. Returns JSON with {response, services, buttons, step_complete}
- **Output:** `services[]` with `{input, category_name, category_id, subcategory_name, subcategory_id, confidence}`
- **Buttons:** LLM-generated, contextual (e.g. "Yes, all hot water types" / "Just electric")

#### SERVICE_AREA
- **Purpose:** Determine where the business operates using geography-aware conversation
- **Model:** Sonnet 4.5 (needs geo reasoning)
- **Context:** Regional guide (barriers, congestion, corridors), grouped suburb data, conversation history
- **Flow (no scripting — LLM decides the conversation):**
  1. Load suburbs within 20km radius, grouped by area
  2. Use regional guide to ask about barriers (e.g. Spit Bridge), corridors, high-demand areas
  3. Offer "I work everywhere in [city]" as easy exit
  4. 2-3 rounds max, present refined area, confirm
  5. Returns JSON with {response, service_areas, buttons, step_complete}
- **Output:** `service_areas` with `{suburbs, postcodes, base_suburb, areas_covered, areas_avoided, travel_notes}`
- **Buttons:** LLM-generated, contextual (e.g. "Yes, I cross the Spit Bridge" / "I stick to the Beaches")

#### PROFILE
- **Purpose:** Generate polished profile preview with LLM description, website images, and editable fields
- **Model:** Haiku 4.5 (generates business description + intro)
- **Flow:**
  1. LLM generates business description and conversational intro from collected data
  2. `discover_business_website()` infers domain, scrapes logo + photos
  3. `ai_filter_photos()` classifies scraped images as WORK/SKIP via Haiku vision
  4. Profile preview rendered: SS blue hero, about section, services/areas (editable via pencil toggle), gallery, CTA
  5. User can edit description, remove services/areas, upload photos/logo
  6. "Publish My Profile" sends `__SAVE_PROFILE__:` payload with edits
- **Output:** `profile_description`, `profile_intro`, `profile_logo`, `profile_photos`, `profile_saved`
- **Buttons:** None (interactive profile card with publish CTA)

#### PRICING
- **Purpose:** Recommend subscription plan based on service area coverage
- **Model:** None (deterministic, data-driven)
- **Flow:**
  1. Recommend plan based on region count (≤2 → Standard $49, ≤5 → Plus $79, >5 → Pro $119)
  2. User selects plan or skips
  3. If plan selected, offer billing frequency (monthly/quarterly/annual with discounts)
  4. Skip sets `subscription_plan: "skip"` and auto-chains to complete
- **Output:** `subscription_plan`, `subscription_billing`, `subscription_price`
- **Buttons:** Plan buttons (`__PLAN__:standard`, etc.) then billing buttons (`__BILLING__:quarterly`, etc.)

#### COMPLETE
- **Purpose:** Generate final output and confirm onboarding
- **Model:** None (deterministic)
- **Flow:**
  1. Generate structured JSON output (includes licence data if available)
  2. Display success message with next steps
- **Output:** Final JSON payload with business, licence, services, service_areas

## 3. State Schema

```python
class OnboardingState(TypedDict):
    # Session
    session_id: str
    current_node: str
    messages: Annotated[list[BaseMessage], add_messages]

    # Business Verification (from ABR)
    business_name_input: str          # Raw user input
    abn_input: str                    # If provided directly
    abr_results: list[dict]           # Raw ABR search results
    business_name: str                # Confirmed business name
    abn: str                          # Confirmed ABN
    entity_type: str                  # e.g. "Business Name", "Australian Private Company"
    gst_registered: bool
    business_verified: bool
    business_postcode: str            # From ABR
    business_state: str               # e.g. "NSW", "VIC"

    # Licence Enrichment (from NSW Fair Trading)
    licence_info: dict                # Full licence details (number, status, expiry, compliance)
    licence_classes: list[str]        # e.g. ["Plumber And Roof Plumber", "Gasfitter", "Drainer"]

    # Web Enrichment (from Brave Search)
    web_results: list[dict]           # [{title, url, description}] — business website, Facebook, etc.

    # Service Discovery
    services_raw: str                 # Raw user description or "Inferred from: {name}"
    services: list[dict]              # Mapped services
    # Each: {input, category_name, category_id, subcategory_name, subcategory_id, confidence}
    services_confirmed: bool

    # Service Areas
    location_raw: str                 # Raw user description
    service_areas: dict               # {suburbs, postcodes, base_suburb, radius_km, areas_covered, areas_avoided, travel_notes}
    service_areas_confirmed: bool

    # Contact (extracted from licence + web data)
    contact_name: str                 # From NSW licence associatedParties
    contact_phone: str                # From Brave search descriptions (regex)

    # Profile Builder
    abn_registration_date: str        # From ABR — used to calculate years in business
    years_in_business: int
    profile_description: str          # LLM-generated business description
    profile_description_draft: str    # Draft before user edits
    profile_intro: str                # LLM-generated conversational intro
    profile_logo: str                 # Base64 or URL
    profile_photos: list[str]         # Base64 or URLs
    profile_saved: bool

    # Pricing / Subscription
    pricing_shown: bool
    subscription_plan: str            # "standard" | "plus" | "pro" | "skip" | ""
    subscription_billing: str         # "monthly" | "quarterly" | "annual" | ""
    subscription_price: str           # e.g. "$79/mo"

    # Completion
    confirmed: bool
    output_json: dict                 # Final structured output (includes licence data)
```

## 4. Agent Tools

### 4.1 abr_lookup

Search the Australian Business Register for business details.

```python
@tool
def abr_lookup(
    search_term: str,
    search_type: Literal["name", "abn"] = "name"
) -> dict:
    """
    Search the ABR for business details.

    Args:
        search_term: Business name or ABN to search for
        search_type: Whether searching by name or ABN

    Returns:
        {
            "results": [
                {
                    "abn": "12345678901",
                    "entity_name": "Smith Painting Pty Ltd",
                    "entity_type": "Australian Private Company",
                    "gst_registered": true,
                    "state": "NSW",
                    "postcode": "2095",
                    "status": "Active",
                    "business_names": ["Smith Painting", "Smith & Co Painters"]
                }
            ],
            "count": 1
        }
    """
```

**ABR API Details (JSON endpoint — simpler than XML):**
- Name search: `GET https://abr.business.gov.au/json/MatchingNames.aspx?name=...&maxResults=5&callback=c&guid=GUID`
- ABN lookup: `GET https://abr.business.gov.au/json/AbnDetails.aspx?abn=...&callback=c&guid=GUID`
- Response: JSONP wrapper `c({...})` → strip callback, parse JSON
- GUID: free registration at abr.business.gov.au/Tools/WebServices
- Returns: ABN, entity name, entity type, state, postcode, GST status

### 4.2 nsw_licence_browse / nsw_licence_details

Look up tradie licences on the NSW Fair Trading Trades Register.

**Swagger spec:** `https://apinsw.onegov.nsw.gov.au/api/swagger/spec/25`

```python
async def nsw_licence_browse(search_term: str) -> dict:
    """Browse the NSW Trades Register by business name.
    Returns: licenceID, licensee, licenceNumber, licenceType, status, suburb, postcode, expiryDate"""

async def nsw_licence_details(licence_id: str) -> dict:
    """Get full details for a specific licence.
    Returns: licence classes (e.g. Plumber And Roof Plumber, Gasfitter, Drainer),
    conditions, compliance history, ABN, ACN, associated parties"""
```

**API Details:**
- OAuth2: `GET https://api.onegov.nsw.gov.au/oauth/client_credential/accesstoken?grant_type=client_credentials` (note: GET not POST)
- Browse: `GET https://api.onegov.nsw.gov.au/tradesregister/v1/browse?searchText=...`
- Details: `GET https://api.onegov.nsw.gov.au/tradesregister/v1/details?licenceid=...`
- Verify: `GET https://api.onegov.nsw.gov.au/tradesregister/v1/verify?licenceNumber=...`
- Auth: `Authorization: Bearer {token}` + `apikey: {api_key}` headers on every request
- Token valid ~12 hours, cached in-memory
- Free tier: 2,500 calls/month
- **NSW only** — no data for VIC/QLD/WA/SA businesses

**Key value:** The `licenceClasses` from the details endpoint tell us *exactly* what a tradie is licensed for (e.g. "Plumber And Roof Plumber", "Drainer", "Gasfitter", "Lp Gasfitter"). This feeds directly into service discovery so the LLM can map services with high confidence instead of guessing from the business name.

### 4.3 brave_web_search

Search the web for business information using Brave Search API.

```python
async def brave_web_search(query: str, count: int = 5) -> list[dict]:
    """Search the web for business website, Facebook, reviews.
    Returns: [{title, url, description}, ...]"""
```

**API Details:**
- Endpoint: `GET https://api.search.brave.com/res/v1/web/search?q=...&count=5&country=AU`
- Auth: `X-Subscription-Token: {api_key}` header
- Used during business confirmation to find web presence
- Results passed to service_discovery LLM as additional context

### 4.4 Geo Helpers (implemented)

```python
def get_suburbs_in_radius_grouped(base_postcode, radius_km) -> dict   # Suburbs grouped by area
def get_regional_guide(state_code) -> str                               # Regional guide markdown
def find_subcategory_guide(business_name) -> str                        # Trade-specific guide
def get_category_taxonomy_text() -> str                                 # Full SS taxonomy as text
def search_suburbs_by_postcode(postcode) -> list                        # Suburbs for a postcode
```

### 4.5 duplicate_checker (not yet implemented)

Check if a business is already registered on Service Seeking. Requires SS API integration.

## 5. Prompt Design

All prompts are inline in `agent/graph.py` (no separate prompts file).

### 5.1 System Prompt

```
You are the Service Seeking onboarding assistant. You help Australian tradies
get set up on the Service Seeking marketplace.

You are friendly, efficient, and speak in Australian English. Keep responses
short and conversational — tradies are busy people registering between jobs
on their phone.

Rules:
- One question at a time
- Always confirm what you've understood before moving on
- Keep it under 3 sentences per response where possible
- Use Australian terminology: "tradies", "ABN", "suburbs"
- Be warm but efficient — respect their time
```

### 5.2 Node Prompts

Each node has a focused inline prompt in `agent/graph.py` that:
- Describes the node's purpose and goals
- Provides context (licence classes, taxonomy, regional guides, conversation history)
- Defines JSON response structure
- Specifies tone and response length

Prompts are co-located with the node functions — no separate prompts file.

## 6. Model Strategy

| Node | Model | Reasoning |
|------|-------|-----------|
| welcome | Haiku 4.5 | Simple greeting, fast (~0.5s) |
| business_verification | Haiku 4.5 | Intent classification (CONFIRMED/REJECTED/NEWSEARCH) |
| service_discovery | Haiku 4.5 | Taxonomy mapping + gap questions (licence classes provide strong signal) |
| service_area | Haiku 4.5 | Geo interpretation with regional context (regional guides provide structure) |
| profile | Haiku 4.5 | Business description + intro generation |
| pricing | None | Deterministic — data-driven plan recommendation |
| complete | None | Deterministic — no LLM call |

**Prompt Caching:** Enable for category taxonomy and suburb data — these are large, static contexts that benefit from ~90% cost reduction on cache hits.

## 7. Web Interface

### 7.1 Layout

```
┌─────────────────────────────────────────┐
│  Service Seeking  ·  Business Registration │
├─────────────────────────────────────────┤
│                                         │
│  ● Business ─── ● Services ─── ● Areas ─── ○ Done  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░  │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│  🤖 G'day! I'm here to help you get    │
│     set up on Service Seeking.          │
│                                         │
│     What's the name of your business?   │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ Smith Painting Pty Ltd          │    │
│  └─────────────────────────────────┘    │
│                                         │
│  🤖 I found a match on the ABR:        │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ Smith Painting Pty Ltd          │    │
│  │ ABN: 12 345 678 901             │    │
│  │ Type: Australian Private Company│    │
│  │ GST: Registered                 │    │
│  │ Location: Manly, NSW 2095       │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [Yes, that's me]  [No, that's not me]  │
│                                         │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │ Type your message...        [→] │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 7.2 Interface Features

- **Progress bar:** Green bar flush at modal top (no step labels)
- **Question counter:** "QUESTION 1", "QUESTION 2" etc. in blue
- **Contextual buttons:** LLM-generated options that directly answer the current question
- **Info box:** Blue-accented card for ABR results and confirmation summary
- **Service chips:** Blue pill-shaped tags showing mapped services
- **Magic stars loading:** Blue sparkle animation during processing
- **Active/inactive submit:** Button changes from muted to blue when input has text
- **Mobile responsive:** Full-screen modal on mobile, centered 700px max on desktop

### 7.3 Design System (matched to Concierge wizard)

- **Interactive colour:** `#0066cc` / `#298AE3` (blue) — buttons, links, focus, labels
- **Success/progress:** `#1FC759` / `#00BB4D` (green) — progress bar, confirm button, check icon
- **Typography:** `-apple-system, ...` system font, 18px/500 question text, `letter-spacing: -0.018em`
- **Buttons:** `2px solid #EDEDED`, hover `#167DD5` border + `#f8faff` bg, slide right 5px
- **Confirm button:** Green border/bg variant for final confirmation
- **Input:** 54px height, `#EDEDED` border, focus `#167DD5`
- **Info box:** `#f8faff` bg, `4px solid #0066cc` left border
- **Animations:** fadeInUp on questions, skeleton shimmer on loading, magic stars sparkle

## 8. API Endpoints

### POST /api/session
Create a new onboarding session.

**Request:**
```json
{}
```
No body required — contact details are extracted automatically from licence and web data.

**Response:**
```json
{
  "session_id": "uuid-here",
  "created_at": "2026-02-13T10:00:00Z"
}
```

### POST /api/chat
Send a message and receive a JSON response (auto-chains through steps).

**Request:**
```json
{
  "session_id": "uuid-here",
  "message": "Smith Painting"
}
```

**Response:** JSON (may contain multiple chained responses)
```json
{
  "response": "G'day! I found a match on the ABR...",
  "node": "business_verification",
  "buttons": [
    {"label": "Yes, that's me", "value": "Yes, that's me"},
    {"label": "No, that's not me", "value": "No, that's not me"}
  ],
  "progress": 25,
  "done": false
}
```

> **Planned:** SSE streaming for token-by-token output (see TASKS.md Phase 5).

### GET /api/session/:id
Resume an existing session (returns conversation history + current state).

### GET /api/session/:id/result
Get the final structured output for a completed session.

**Response:**
```json
{
  "status": "complete",
  "result": {
    "business_name": "Smith Painting Pty Ltd",
    "abn": "12345678901",
    "entity_type": "Australian Private Company",
    "gst_registered": true,
    "services": [
      {
        "category": "Painting",
        "category_id": 42,
        "subcategories": [
          {"name": "Interior Painting", "id": 301},
          {"name": "Exterior Painting", "id": 302}
        ]
      }
    ],
    "service_areas": {
      "postcodes": ["2095", "2088", "2093"],
      "suburbs": [
        {"name": "Manly", "postcode": "2095", "state": "NSW"},
        {"name": "Mosman", "postcode": "2088", "state": "NSW"}
      ]
    },
    "contact_name": "John Smith",
    "contact_phone": "0412 345 678"
  }
}
```

### GET /api/health
Health check endpoint.

## 9. Session Storage

**Current:** In-memory Python dict (`sessions = {}` in `server/app.py`). Sessions are lost on server restart.

**Planned:** Persistent storage for production (see TASKS.md Phase 6). Per-session JSONL logging in `logs/` provides replay capability in the meantime.

## 10. Output Format

Final structured output generated on completion:

```json
{
  "business_name": "A.C. SMITH PLUMBING PTY. LTD.",
  "abn": "70058396295",
  "entity_type": "Australian Private Company",
  "gst_registered": true,
  "licence": {
    "number": "42092C",
    "type": "Contractor Licence",
    "status": "Current",
    "expiry": "10/02/2027",
    "classes": ["Plumber And Roof Plumber", "Drainer", "Gasfitter", "Lp Gasfitter"]
  },
  "services": [
    {
      "category": "Plumber",
      "category_id": 146,
      "subcategories": [
        {"name": "General Plumbing", "id": 2329},
        {"name": "Gas Fitting", "id": 2321},
        {"name": "Drain Repairs", "id": 2320},
        {"name": "Clogged and Blocked Drain", "id": 2322},
        {"name": "Pipe Repair", "id": 2327},
        {"name": "Roof Plumbing", "id": 2349}
      ]
    }
  ],
  "service_areas": {
    "suburbs": [
      {"name": "Bowraville", "postcode": "2449", "state": "NSW"},
      {"name": "Nambucca Heads", "postcode": "2448", "state": "NSW"}
    ],
    "postcodes": ["2449", "2448"],
    "base_suburb": "Bowraville",
    "radius_km": 20,
    "areas_covered": ["Nambucca Valley"],
    "areas_avoided": [],
    "travel_notes": ""
  },
  "contact_name": "Shane Matthew Howison",
  "contact_phone": "0412 345 678"
}
```

## 11. Error Handling

| Error | User Experience |
|-------|----------------|
| ABR API timeout/error | "I'm having trouble reaching the business register. Could you give me your ABN directly?" |
| No ABR match | "I couldn't find that business. Could you try a different name, or enter your ABN?" |
| Service mapping uncertain | "I'm not 100% sure about this one — did you mean [option A] or [option B]?" |
| Unknown location | "I'm not sure where that is — could you give me a suburb name or postcode?" |
| Duplicate business found | "It looks like [Business Name] might already be on Service Seeking. Would you like to log in to your existing account instead?" |
| Session expired | "Looks like we lost our place. Let me start fresh — what's your business name?" |

## 12. Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...          # Claude API access
ABR_GUID=xxxxxxxx-xxxx-xxxx-xxxx     # Australian Business Register JSON API key

# NSW Fair Trading Trades API (OAuth2)
NSW_TRADES_API_KEY=...                # API key (client_id)
NSW_TRADES_API_SECRET=...             # API secret
NSW_TRADES_AUTH_HEADER=Basic ...      # Base64(key:secret) for OAuth token request

# Brave Search API
BRAVE_SEARCH_API_KEY=...              # For web search enrichment

# Optional
PORT=8001                             # Server port (8001 to avoid Concierge on 8000)
```

## 13. Performance Targets

| Metric | Target |
|--------|--------|
| First response (welcome) | <1 second |
| ABR lookup + response | <2 seconds |
| Service mapping + response | <2 seconds |
| Location parsing + response | <2 seconds |
| Total conversation turns | 6-10 |
| Total completion time | <2 minutes |

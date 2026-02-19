# Trade Onboarding Wizard - Tasks

## Status Key
- **Done** — Completed and tested
- **In Progress** — Currently being worked on
- **Planned** — Scoped and ready to build
- **Future** — Ideas, not yet scoped

---

## Phase 0: Project Setup ✅
> Foundation — repo, dependencies, basic server

- [x] Initialise Python project with `requirements.txt`
- [x] Set up FastAPI server with health check endpoint
- [x] Set up LangGraph-inspired state machine (direct node dispatch)
- [x] Create `.env` with environment variables
- [x] Copy `suburbs.csv` from Trade Agent `resources/` (15,761 suburbs)
- [x] Source `subcategories.json` SS category taxonomy
- [x] Copy subcategory guides (plumber, electrician, cleaner, gardener, painter, carpentry)
- [x] Copy regional guides (sydney, melbourne, brisbane, perth)
- [x] Create landing page + wizard HTML/JS
- [x] Git init + initial commit

---

## Phase 1: Business Verification Flow ✅
> ABR lookup, business confirmation, enrichment

- [x] Register for ABR API GUID (`f746a5d8-...`)
- [x] Implement `abr_lookup` tool — search by name (JSON API)
- [x] Implement `abr_lookup` tool — search by ABN (JSON API)
- [x] Parse ABR JSONP responses into clean dict
- [x] Handle ABR edge cases: no results, multiple results, inactive ABN
- [x] Build `welcome` node — Haiku greeting + capture business name/ABN
- [x] Build `business_verification` node — present results, confirm identity
- [x] Quick-match: button clicks with ABN bypass LLM classifier
- [x] LLM classifier fallback (Haiku): CONFIRMED / REJECTED / NEWSEARCH
- [x] Handle "no, that's not me" → re-search flow
- [x] ABR results as clickable buttons (multi-result shows business names, single result shows Yes/No)
- [x] Wire up state transitions: welcome → business_verification → service_discovery (auto-chain)

### Phase 1b: Licence + Web Enrichment ✅
> NSW Fair Trading API + Brave Search — enrich business profile on confirmation

- [x] Register for NSW Fair Trading Trades API (api.nsw.gov.au, Product 25)
- [x] OAuth2 token endpoint — `GET /oauth/client_credential/accesstoken` (token cached 12h)
- [x] Implement `nsw_licence_browse` — search by business name (`/tradesregister/v1/browse`)
- [x] Implement `nsw_licence_details` — get licence classes, conditions, compliance (`/tradesregister/v1/details`)
- [x] Handle apostrophes in business names (clean before search)
- [x] Match best licence result (current status, name match)
- [x] Register for Brave Search API
- [x] Implement `brave_web_search` — find business website, Facebook, reviews
- [x] Parallel enrichment: licence + web search run concurrently on business confirmation
- [x] Feed `licence_classes` into service discovery node
- [x] Feed `web_results` into service discovery node
- [x] Include licence data in final output JSON
- [x] Test with real NSW licensed plumber — confirmed licence classes: Plumber And Roof Plumber, Drainer, Gasfitter, Lp Gasfitter

---

## Phase 2: Service Discovery Flow ✅
> LLM-driven service mapping with guides, licence classes, and taxonomy

- [x] Load SS category taxonomy text for LLM context
- [x] Implement `find_subcategory_guide` — match trade keywords to guide files
- [x] Build `service_discovery` node — NO SCRIPTING, LLM gets goals + guides + context
- [x] LLM uses licence classes to map services with high confidence
- [x] LLM uses subcategory guide to ask smart gap questions (e.g. hot water, gas fitting)
- [x] LLM-generated contextual buttons at every step (not hardcoded)
- [x] JSON response format: {response, services, buttons, step_complete}
- [x] Auto-chain: business_verified → service_discovery
- [x] Handle 2-3 rounds max, then wrap up
- [x] Test with varied trades — plumbing, painting keywords confirmed

---

## Phase 3: Service Area Mapping ✅
> Geography-aware, guide-driven service area conversation

- [x] Build suburb lookup from `suburbs.csv` (name, postcode, lat/lng, area, region)
- [x] Implement `get_suburbs_in_radius_grouped` — suburbs within radius, grouped by area
- [x] Implement `get_regional_guide` — load regional guide markdown (barriers, corridors, congestion)
- [x] Haversine distance calculation
- [x] Build `service_area` node — NO SCRIPTING, LLM gets goals + regional guide + grouped suburbs
- [x] LLM asks about barriers (e.g. Spit Bridge), high-demand areas, travel willingness
- [x] LLM-generated contextual buttons
- [x] JSON response format: {response, service_areas, buttons, step_complete}
- [x] Auto-chain: services_confirmed → service_area
- [x] Regional guides for Sydney, Melbourne, Brisbane, Perth

---

## Phase 4: Profile Builder + Output ✅
> Profile preview, editable services/areas, LLM description, image scraping

- [x] Build `profile` node — LLM-generated description + intro, website image scraping
- [x] Website discovery: infer business domain from name, try common AU TLDs
- [x] Scrape logo + photos from business website (og:image, logo patterns, large images)
- [x] AI image filter: Haiku vision classifies scraped photos as WORK/SKIP
- [x] Social media scraping: Facebook/Instagram og:image for logo/photo fallback
- [x] Profile builder UI: SS blue gradient hero, white card sections on gray background
- [x] Hero: logo upload, business name, contact, suburb/postcode, establishment year, trust badges
- [x] Editable About section: LLM-generated description in textarea with char counter
- [x] Editable services: pencil toggle → tap to strikethrough, hint text, checkmark when done
- [x] Editable areas: same pencil toggle UX as services
- [x] Licences card: green rows with shield icons, 2-column grid
- [x] Gallery card: photo grid with add/remove, drag-drop upload
- [x] CTA card: "Publish My Profile" with SS blue gradient + green button
- [x] Upload endpoint: `POST /api/upload` for logo + work photos (base64, max 5MB)
- [x] Save sends JSON payload: `{description, removed_services, removed_areas}`
- [x] Confirmation step removed — editing absorbed into profile preview
- [x] Auto-chain: service_areas_confirmed → profile → pricing → complete
- [x] Conversational AI intro on profile page (LLM-generated, varies per session)
- [x] Build `complete` node — generate final structured JSON with profile data
- [x] Output JSON includes: business, ABN, licence, services, service_areas, contact, profile, subscription
- [x] Result endpoint: `GET /api/session/:id/result`
- [x] Pricing/subscription node — data-driven plan recommendation after profile publish
- [x] 3-turn pricing flow: plan selection → billing frequency → confirmation (or skip)
- [x] Plan recommendation based on region count (≤2→Standard, ≤5→Plus, >5→Pro)
- [x] Subscription data in output JSON (plan, billing, price) — omitted when skipped

---

## Phase 5: Polish + Optimisation
> Performance, UX, edge cases

- [x] Prompt caching for category taxonomy and suburb data (`cache_control: {"type": "ephemeral"}`)
- [x] Progress bar in wizard UI (green bar flush at modal top)
- [x] OAuth token pre-warming on server startup
- [x] API call tracing in browser dev tools (color-coded console groups)
- [x] Per-session JSONL logging with replay endpoints (`GET /api/logs`, `GET /api/logs/{id}`)
- [x] Postcode-in-search auto-confirm (e.g. "dans plumbing 2155" skips button step)
- [x] Compact service summary (one-line format, counts only when >1)
- [x] Markdown bold rendering in chat UI
- [x] Removed step nav, kept progress bar only
- [x] Friendly welcome message with process overview and timing
- [x] Gap questions on new paragraph, contextual "Question N" label
- [x] Smart gap questions: must ask about services NOT already mapped
- [x] Buttons must directly answer the question (no generic "Looks good")
- [x] Contact extraction: person name from NSW licence parties, phone from Brave search
- [x] Trade field in confirmation summary (derived from service categories)
- [x] Removed "Type: Entity Name" and "Notes" from summary (not relevant)
- [x] Removed contact_email/contact_phone collection (redirect to proper login)
- [x] Changed "registration" language to "onboarding/setup" throughout
- [x] Edit flow from confirmation preserves existing data (add/remove, not restart)
- [x] Button leak fix: stale buttons cleared between auto-chain steps
- [x] CSS matched to Concierge wizard (blue interactive, green success, typography, spacing)
- [x] Fixed junk excluded regions (state filter + min 3 suburbs threshold)
- [x] LLM JSON fallback when plain text returned
- [x] Performance optimisation: Haiku for all nodes, parallel auto-chain (21s → 6s, 11s → 5s)
- [x] Prompt de-scripting: removed turn-by-turn scripts, replaced with guidance
- [x] ~~Editable confirmation screen~~ (replaced by profile builder — services/areas editable there)
- [x] Progressive loading: magic stars + API activity steps for key transitions
- [x] Completion screen fix: service areas showing regions instead of "0 suburbs"
- [x] CSS refinements: info-box spacing, question text line-height/weight/smoothing
- [ ] Streaming SSE implementation (currently JSON responses)
- [ ] Mobile responsiveness testing
- [ ] Edge cases: very long service lists, unusual business types, non-trade businesses
- [ ] Handle non-NSW businesses gracefully (no licence data, inform user)
- [ ] LangSmith tracing integration
- [ ] Conversation quality review (20+ test sessions)

---

## Phase 6: Integration + Deployment
> SS platform integration, production deployment

- [ ] SS API integration — create business profile on completion
- [ ] Duplicate checker (SS API) — check if ABN already registered
- [ ] Embeddable iframe variant
- [x] Railway deployment (Procfile + .python-version, live at trade-onboarding.up.railway.app)
- [x] Git repo: github.com/cleopatterson/trade_onboarding
- [x] Environment variable documentation (.env.example)
- [ ] Dockerfile + production config
- [ ] API security (rate limiting, input validation)
- [ ] Session persistence (currently in-memory dict)
- [ ] Monitoring + alerting setup
- [ ] Analytics dashboard (completion rates, drop-off, timing)

---

## Future Ideas
> Not scoped, revisit later

- [ ] Licence verification for other states (VIC, QLD, WA, SA — need separate APIs)
- [x] ~~Photo upload for business logo~~ (done in profile builder)
- [ ] Insurance details capture
- [x] ~~Portfolio/previous work links~~ (gallery upload in profile builder)
- [ ] Multi-language support
- [ ] Voice input (Whisper)
- [ ] A/B test against traditional form
- [ ] Referral tracking
- [ ] Progressive profiling (capture more details over first week)
- [ ] Use Brave Search to auto-populate business description from website

---

## Build Log

### Feb 13, 2026 — Initial Build
- Set up Python/FastAPI project with LangGraph-inspired state machine
- Implemented full 6-node conversation flow (welcome → business_verification → service_discovery → service_area → confirmation → complete)
- ABR JSON API integration (replaced planned XML approach — simpler)
- Real ABR GUID obtained and tested with live data
- No-scripting principle applied: LLM gets context + goals + guides, figures out the conversation
- Subcategory guides drive smart trade-specific questions
- Regional guides drive smart geography-aware questions
- Auto-chaining between steps (no dead "ok" turns)
- LLM-generated contextual buttons at every step (Concierge pattern)
- ABR results as clickable buttons with ABN identification

### Feb 13, 2026 — Licence + Web Enrichment
- NSW Fair Trading Trades API integrated (OAuth2 + REST)
- Swagger spec: https://apinsw.onegov.nsw.gov.au/api/swagger/spec/25
- Token endpoint is GET (not POST) — key finding
- Browse + Details endpoints return licence classes (e.g. Plumber, Gasfitter, Drainer)
- Brave Search API integrated for web presence lookup
- Both run in parallel on business confirmation
- Licence classes feed directly into service discovery — LLM maps services with high confidence from actual licence data
- Tested with real NSW plumber: A.C. Smith Plumbing → Licence #42092C → classes: Plumber And Roof Plumber, Drainer, Gasfitter, Lp Gasfitter → 6 services auto-mapped

### Feb 17, 2026 — Postcode Detection + UX Polish
- Postcode-in-search: "dans plumbing 2155" auto-confirms business and skips button step
- Compact service summary: one-line format instead of full list ("18 services — plumbing (8), hot water (3)...")
- Removed step nav dots/labels, kept green progress bar flush at modal top
- Per-session JSONL logging with `/api/logs` endpoints for session replay
- OAuth token pre-warming on server startup (eliminates cold-start delay)
- API call tracing in browser dev tools: console.group with color-coded entries for LLM/API calls

### Feb 18, 2026 — Contact Extraction + Edit Flow + CSS Overhaul
- Contact name extraction from NSW licence `associatedParties` (Director/Nominated Supervisor)
- Contact phone extraction from Brave search descriptions via regex
- Removed contact_email/contact_phone fields (will redirect to proper login)
- Changed "registration" to "onboarding/setup" throughout prompts
- Friendlier welcome message: process overview, ~2 minute estimate, postcode tip
- Gap questions must be about services NOT already mapped (fixed contradictions)
- Buttons must directly answer the gap question (removed generic "Looks good")
- Service area: TURN 2 confirms without re-asking "does that sound right?"
- Confirmation summary: added Trade field, removed Type/Notes, tightened table
- Edit flow from confirmation now preserves data — shows current list, asks what to add/remove
- Button leak fix: `_merge()` helper clears stale buttons between auto-chained nodes
- Fixed `[] or fallback` bug (empty buttons treated as falsy)
- CSS matched to Concierge wizard: blue (#0066cc) for interactive elements, green for progress/success
- Typography: 18px/500 question text, -0.018em letter-spacing, 14px/600 question labels
- Input: 54px height, #EDEDED borders, active/inactive submit button states
- Magic stars loading: blue glow effect instead of green
- Fixed junk regions in service areas (NT suburbs with Sydney coords, wrong area labels)
- State filter + min 3 suburbs threshold for region grouping
- LLM JSON fallback: wraps plain text in default structure when JSON parsing fails
- Service counts in brackets only shown when category has >1 service

### Feb 18, 2026 — Performance, Prompt Review, Edit UX, Deployment
- **Performance**: Switched all nodes to Haiku (removed Sonnet dependency entirely)
  - Business confirm → services: 21s → ~6s (Haiku + web results provide enough signal)
  - Service confirm → areas: 11s → ~5s (parallel auto-chain via asyncio.gather)
  - Persistent httpx client for API connection reuse
  - Reduced max_tokens from 4096 to 2048
- **Prompt de-scripting**: Removed turn-by-turn scripts from service_discovery + service_area prompts
  - Replaced with guidance: "keep it short — tradies are busy", "under 2-3 sentences"
  - Service area prompt tells LLM not to explain geography the tradie already knows
  - Auto-chain context: "this flows from service confirmation, don't re-introduce yourself"
- **Editable confirmation screen**: Complete redesign of confirmation step
  - Services + areas rendered as toggleable chips (tap to cross off, tap to restore)
  - Business info shown as static summary box
  - Single "Looks good" button collects all removals into one message
  - Backend handles structured removal message directly (no LLM round trip)
- **Progressive loading**: Replaced skeleton with magic stars + API activity steps
  - Business confirm: "Checking NSW trade licences" → "Searching web" → "Mapping services"
  - Service confirm: "Updating service list" → "Loading regions" → "Mapping coverage area"
  - Steps appear with stagger animation, spinner → checkmark progression
  - Skeleton preserved for all other transitions
- **Completion screen fix**: Service areas showed "0 suburbs" — was looking for `suburbs` array instead of `regions_included`
- **CSS**: Info-box padding/line-height tightened, question text weight 400, letter-spacing fixes
- **Deployment**: Git init, pushed to github.com/cleopatterson/trade_onboarding, deployed on Railway
  - Procfile: `uvicorn server.app:app --host 0.0.0.0 --port $PORT`
  - Live at: https://trade-onboarding.up.railway.app

### Feb 19, 2026 — Profile Builder + Confirmation Removal
- **Profile Builder**: New `profile` node replaces raw completion screen with polished profile preview
  - SS blue gradient hero (#298AE3) with logo upload, business name, trust badges, suburb + est. year
  - White card sections on gray (#F8FAFC) background: Licences, About, Services, Areas, Gallery, CTA
  - LLM-generated business description (Haiku) + conversational intro message
  - JSON prompt output with markdown code fence stripping (`\`\`\`json` → parsed)
  - Services + areas editable via pencil toggle (tap to strikethrough, hint text, checkmark icon)
  - Photo gallery with upload, remove buttons, drag-drop support
  - "Publish My Profile" CTA sends JSON payload with description + removed services/areas
- **Website Image Scraping**: Domain discovery + AI image filter
  - `discover_business_website()`: infers domain from business name, tries .com.au/.au/.net.au
  - Always runs in parallel with Brave scrape — discovered domain takes priority
  - `ai_filter_photos()`: downloads candidate images, sends batch to Haiku vision
  - Classifies each image as WORK (real job photos) or SKIP (logos, banners, stock)
  - Size thresholds: skip <2KB (icons) and >2MB, download up to 8 candidates
  - Directory site filtering: mylocaltrades, hipages, etc. added to skip list
- **Confirmation Step Removed**: Flow now goes SERVICE_AREA → PROFILE → COMPLETE
  - Saves one step in the wizard (faster onboarding)
  - Services/areas editing moved to profile preview (pencil toggle)
  - `determine_node()` and both auto-chain paths updated to skip confirmation
  - `confirmed` flag set automatically during auto-chain
- **Service Discovery Improvements**
  - Gap questions: 2-3 per session instead of 1 (more thorough service coverage)
  - Safety cap: forces `services_confirmed` after 4 turns (prevents infinite gap loops)
  - Turn 2+ prompt includes subcategory guide + licence classes for better follow-up questions
  - Declined gap questions trigger immediate wrap-up (no more stuck loops)
- **Licence Search Fix**: Apostrophe handling improved
  - Searches with original business name first (e.g. "DAN'S PLUMBING")
  - Only retries without apostrophe if 0 results (was always stripping before)
- **Loading Screen Fix**: Profile building animation now shows correctly
  - Combined `isServiceConfirm || isAreaConfirm` into single trigger
  - Handles parallel path (gap question → services confirm → area confirm → profile build)
  - 4-step animation: finalising services → pulling images → generating description → preparing preview
- **Design**: SS blue (#298AE3) for hero + CTA, existing green (#1FC759) for publish button
  - 2-column grid for licences and service areas
  - Shield icon for licences, map pin for areas
  - Responsive: grids collapse to single column on mobile

### Feb 19, 2026 — Pricing/Subscription Node
- **Pricing Node**: New `pricing_node` between profile publish and completion
  - Data-driven (no LLM call) — recommends plan based on service area region count
  - 3 plans: Standard ($49/mo, 10km), Plus ($79/mo, 20km), Pro ($119/mo, 50km)
  - Quarterly (20% off) and annual (40% off) billing options
  - 3-turn flow: plan recommendation → billing frequency → confirmation
  - Skip path: "Not ready yet" sets `subscription_plan: "skip"`, proceeds to complete
  - Guard logic: `_selected_plan` distinguishes plan selection turn from billing turn
- **Auto-chain**: profile_saved → pricing → complete (both normal + parallel paths)
- **State**: 4 new fields: `pricing_shown`, `subscription_plan`, `subscription_billing`, `subscription_price`
- **Output JSON**: `subscription` object included when plan selected (omitted on skip)
- **Frontend**: Progress bar updated to 6 steps, pricing uses standard chat Q&A layout

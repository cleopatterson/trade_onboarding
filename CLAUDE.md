# Trade Onboarding Wizard

## Project Overview
Conversational AI wizard that guides Australian tradies through Service Seeking business onboarding in under 2 minutes. Replaces the static multi-step registration form with a chat interface.

## Quick Start
```bash
# Activate virtualenv (Python 3.9)
source venv/bin/activate

# Start server (port 8001)
python -m server.app

# Open in browser
open http://localhost:8001
```

## Architecture
- **Server:** FastAPI (`server/app.py`) — endpoints, auto-chaining, session management
- **Agent:** LangGraph-inspired state machine (`agent/graph.py`) — 6 nodes, prompts inline, no-scripting principle
- **Tools:** ABR, NSW Fair Trading, Brave Search, geo helpers (`agent/tools.py`)
- **Frontend:** Single-file HTML/JS/CSS (`web/landing.html`) — landing page + wizard modal
- **Models:** Claude Sonnet 4.5 (service/area mapping), Claude Haiku 4.5 (welcome, classifiers)

## Key Files
| File | Purpose |
|------|---------|
| `server/app.py` | FastAPI server, auto-chaining, session state, button logic, logging |
| `agent/graph.py` | State machine nodes: welcome, business_verification, service_discovery, service_area, confirmation, complete |
| `agent/tools.py` | ABR lookup, NSW licence browse/details, Brave search, suburb grouping, regional guides |
| `agent/config.py` | Environment config, model IDs, API keys |
| `agent/state.py` | OnboardingState TypedDict |
| `web/landing.html` | Landing page + wizard modal (all-in-one) |
| `resources/suburbs.csv` | 15,761 AU suburbs (name, state, postcode, lat, lng, area, region) |
| `resources/subcategories.json` | SS category taxonomy |
| `resources/*-subcategory-guide.md` | Trade-specific gap question guides (plumber, electrician, cleaner, gardener) |
| `resources/*_subcategories.md` | Subcategory reference lists (plumbing, electrical, painter, carpentry) |
| `resources/*_regions.md` | Regional guides (sydney, melbourne, brisbane, perth) |

## Design Principles
1. **No scripting** — prompts provide context/goals/guides, LLM figures out the conversation
2. **Australian English** — "tradies", "ABN", "suburbs", mate-like tone
3. **Two turns max** per step — be decisive, don't over-ask
4. **Auto-chain** — steps flow seamlessly (business → services → areas → confirm → complete)
5. **Blue interactive, green success** — wizard uses blue (#0066cc) for actions, green (#1FC759) for progress/confirm only

## Conversation Flow
```
WELCOME → BUSINESS_VERIFICATION → SERVICE_DISCOVERY → SERVICE_AREA → CONFIRMATION → COMPLETE
                                                                          ↑         |
                                                                          └─ Edit ──┘
```

## External APIs
ABR JSON API, NSW Fair Trading Trades API (OAuth2), Brave Search API. See `docs/PRD.md` Section 4 for details, `.env.example` for required keys.

## Development Notes
- Server runs on port **8001** (Concierge uses 8000)
- Python venv at `./venv/bin/python` — system Python won't have dependencies
- OAuth token pre-warmed on startup for NSW Trades API
- Per-session JSONL logging in `logs/` directory
- API call traces visible in browser dev tools console (color-coded)
- Prompt caching enabled for taxonomy + guides (large static context)
- LLM JSON fallback: if LLM returns plain text, wraps in default structure
- Suburbs CSV has some bad data: state filter + min 3 suburbs threshold filters junk regions
- Contact name extracted from NSW licence `associatedParties`, phone from Brave search descriptions
- Edit flow from confirmation preserves existing data (doesn't wipe) — asks what to add/remove

## Docs
- `docs/MASTERPLAN.md` — vision, design principles, success metrics, project relationships
- `docs/PRD.md` — technical spec: state schema, API details, node specs, model strategy
- `docs/TASKS.md` — phase tracking + build log

# Trade Onboarding Wizard - Masterplan

## Vision

Replace the static Service Seeking business registration form with a conversational AI wizard that guides tradies through signup in under 2 minutes. The wizard verifies their business against the Australian Business Register, discovers their services through natural language, and maps their service areas — all through a chat interface that feels like texting a mate, not filling out government paperwork.

This is the **supply-side onboarding** complement to the Concierge (demand-side job intake). Together they complete the AI-native marketplace experience.

## Why This Matters

**Current state:** Tradies land on a registration page, face a multi-step form with dropdowns, category pickers, and postcode entry. Drop-off is high. Many select wrong categories or incomplete service areas, leading to poor lead matching downstream.

**With the wizard:**
- Business details are auto-populated from ABR — no manual entry
- Services are described in plain English and mapped to the SS taxonomy
- Service areas are described naturally ("Northern Beaches", "within 20km of Manly") and converted to postcodes
- The result is a higher-quality, more complete business profile from day one

## Design Principles

### 1. Conversational, Not Forms
One question at a time. No dropdowns, no multi-select panels. The tradie types naturally and the AI figures out the structure.

### 2. Australian English
"Tradies" not "contractors". "ABN" not "tax ID". "Suburbs" not "neighborhoods". Match how real Australian business owners talk.

### 3. Verify, Don't Trust
Always confirm ABR results and service mappings with the user. Show what was understood and let them correct it. Never silently assume.

### 4. Fast and Focused
Target: 3-5 exchanges per section, under 2 minutes total. No unnecessary questions. If ABR gives us the business name and entity type, don't ask again.

### 5. Mobile-First
Tradies register on their phones. The interface must work perfectly on small screens with one-thumb interaction.

### 6. No Scripting
**This is a core principle for all AI interactions.** Do not provide scripts or rigid conversation flows for the agent. When you give an AI a script, it follows the script instead of thinking. Instead, provide context, goals, and guidelines — then let the agent figure out how to have the conversation. Provide the agent with subcategory guides, regional guides, and taxonomy data as context. Describe what needs to be achieved, not what to say. No keyword matching, no canned responses, no `if user says X then respond with Y`.

### 7. Graceful Degradation
If ABR is down, allow manual entry. If service mapping is uncertain, show options and ask. Never dead-end the conversation.

## Target Users

### Primary: Australian Tradies
- Painters, plumbers, electricians, builders, handymen, landscapers
- Small business owners (sole traders and small companies)
- Typically registering on mobile between jobs
- Want to get listed and start receiving leads quickly
- May not know Service Seeking's exact category names

### Secondary: Service Seeking Operations Team
- Need complete, accurate business profiles
- Want higher registration completion rates
- Need to reduce manual data cleanup
- Review onboarding metrics weekly

## Architecture Overview

Python/FastAPI server running a LangGraph-inspired state machine with 6 nodes, a chat wizard frontend, and external API integrations (ABR, NSW Fair Trading, Brave Search). See `docs/PRD.md` for detailed architecture diagrams, file structure, and API specs.

## Relationship to Other Projects

| Project | Side | Purpose |
|---------|------|---------|
| **Concierge** | Demand | Customers posting jobs → AI-guided job briefs |
| **Trade Agent (Baz)** | Supply | Tradies managing leads → AI-assisted quoting |
| **Trade Onboarding** | Supply | Tradies registering → AI-guided business profiles |

The onboarding wizard shares:
- **From Concierge:** LangGraph architecture, wizard UI pattern, conversation state management
- **From Trade Agent:** Suburb/geo data, category taxonomy, Australian English tone, mobile-first design

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Completion rate | >80% | Registrations completed / started |
| Time to complete | <2 minutes | First message to final confirmation |
| ABR match rate | >90% | Businesses found on first search |
| Service mapping accuracy | >95% | Correct categories confirmed by user |
| Service area accuracy | >90% | Correct postcodes confirmed by user |
| Profile completeness | 100% | All required fields populated |
| First response time | <2 seconds | Server response latency |

## Weekly Review

- Review with Ollie + SS team (align with existing Friday cadence)
- Key dashboards: completion funnel, drop-off points, ABR match rates
- Review edge cases: failed lookups, service mapping misses, location parsing errors
- Compare profile quality vs traditional form registrations

## Current Implementation Status (Feb 18, 2026)

The prototype is **fully functional end-to-end** with significant UX polish:
1. **ABR Business Lookup** — real ABR JSON API with live GUID, postcode-in-search auto-confirm
2. **NSW Fair Trading Licence Lookup** — OAuth2 (pre-warmed), licence classes + contact name extraction
3. **Brave Web Search** — business website/Facebook, phone number extraction via regex
4. **Service Discovery** — LLM-driven with licence classes, subcategory guides, taxonomy, compact summaries
5. **Service Area Mapping** — geography-aware with regional guides, state-filtered suburb grouping
6. **Confirmation + Output** — trade field, edit flow (add/remove without restart), structured JSON
7. **Dev Tools** — API call tracing in browser console, per-session JSONL logging, replay endpoints

All steps auto-chain (business confirm → services → areas → confirmation → complete).
Buttons are LLM-generated and contextual at every step.
CSS matched to Concierge wizard (blue interactive, green success).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ABR API downtime | Can't verify businesses | Fallback to mock data + manual ABN entry |
| NSW Trades API only covers NSW | No licence data for other states | Graceful degradation — infer from business name instead |
| NSW Trades API rate limit (2,500/month) | Throttled at ~800 onboardings/month | Monitor usage, upgrade to premium if needed |
| Service mapping errors | Wrong categories assigned | Always confirm with user, allow corrections |
| Location ambiguity | Wrong service areas | Show suburb list for confirmation, allow editing |
| Duplicate registrations | Wasted effort | Check SS database before completing (not yet implemented) |
| Turn time ~12s on confirmation | Slow UX | Licence + web search run in parallel; could add loading state |

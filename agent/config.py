"""Configuration for Trade Onboarding Wizard"""
import logging
import os
import sys
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ABR_GUID = os.getenv("ABR_GUID", "")
PORT = int(os.getenv("PORT", "8001"))

# NSW Fair Trading Trades API
NSW_TRADES_API_KEY = os.getenv("NSW_TRADES_API_KEY", "")
NSW_TRADES_API_SECRET = os.getenv("NSW_TRADES_API_SECRET", "")
NSW_TRADES_AUTH_HEADER = os.getenv("NSW_TRADES_AUTH_HEADER", "")

# Brave Search API
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")

# Google Places API (for Google Business Profile ratings/reviews)
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# Service Seeking API (for existing user profile improvement)
SS_API_TOKEN = os.getenv("SS_API_TOKEN", "") or os.getenv("SERVICE_SEEKING_API_TOKEN", "")
SS_API_URL = os.getenv("SS_API_URL", "") or os.getenv("SERVICE_SEEKING_API_URL", "")
SS_API_BASIC_AUTH = os.getenv("SS_API_BASIC_AUTH", "") or os.getenv("SERVICE_SEEKING_API_BASIC_AUTH", "")

# CORS — comma-separated allowed origins (default: localhost only)
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", f"http://localhost:{PORT}").split(",") if o.strip()
]

# Model IDs
MODEL_FAST = "claude-haiku-4-5-20251001"


# ────────── Startup Validation ──────────

def validate_env():
    """Check required env vars on startup. Fail fast if Anthropic key is missing."""
    if not ANTHROPIC_API_KEY:
        logger.critical("ANTHROPIC_API_KEY is not set. Cannot start.")
        sys.exit(1)

    optional = {
        "ABR_GUID": ABR_GUID,
        "NSW_TRADES_API_KEY": NSW_TRADES_API_KEY,
        "BRAVE_SEARCH_API_KEY": BRAVE_SEARCH_API_KEY,
        "GOOGLE_PLACES_API_KEY": GOOGLE_PLACES_API_KEY,
    }
    missing = [k for k, v in optional.items() if not v]
    if missing:
        logger.warning(f"Optional keys not set: {', '.join(missing)} — related features will be disabled")

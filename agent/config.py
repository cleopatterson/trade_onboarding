"""Configuration for Trade Onboarding Wizard"""
import os
from dotenv import load_dotenv

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

# Model IDs
MODEL_SMART = "claude-sonnet-4-5-20250929"
MODEL_FAST = "claude-haiku-4-5-20251001"

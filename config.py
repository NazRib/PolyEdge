"""
Configuration for Polymarket Edge trading system.
Copy this to config.py and adjust to your preferences.
"""

# ─── API Endpoints ───────────────────────────────────────
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"

# ─── Trading Parameters ─────────────────────────────────
BANKROLL = 1000.0           # Total capital allocation in USD
KELLY_FRACTION = 0.25       # Fraction of Kelly to use (0.25 = quarter Kelly)
MIN_EDGE = 0.05             # Minimum edge (5%) to consider a trade
MAX_POSITION_PCT = 0.10     # Max % of bankroll in any single position
STOP_LOSS_PCT = 0.50        # Exit if position loses this % of entry value

# ─── Market Filters ─────────────────────────────────────
MIN_VOLUME_24H = 1000       # Minimum 24h volume in USD
MIN_LIQUIDITY = 5000        # Minimum total liquidity in USD
PRICE_RANGE = (0.05, 0.95)  # Only trade markets in this probability range
MAX_DAYS_TO_RESOLUTION = 90 # Skip markets resolving too far out
MIN_DAYS_TO_RESOLUTION = 1  # Skip markets resolving too soon (can't exit)

# ─── Probability Estimation ─────────────────────────────
LLM_CALIBRATION_OFFSET = -0.03  # LLMs tend to be slightly overconfident
ENSEMBLE_WEIGHTS = {
    "base_rate": 0.30,
    "llm_estimate": 0.40,
    "market_momentum": 0.15,
    "whale_signal": 0.15,
}

# ─── Data Storage ────────────────────────────────────────
DATA_DIR = "data"
TRADES_FILE = "data/paper_trades.json"
CALIBRATION_FILE = "data/calibration_log.json"
MARKETS_CACHE = "data/markets_cache.json"

# ─── Logging ─────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ─── LLM Provider Configuration ─────────────────────────
# Default provider: "claude" or "gpt"
LLM_PROVIDER = "claude"

# Claude (Anthropic) — set ANTHROPIC_API_KEY env var
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# GPT-5.4 (Azure AI Foundry Responses API)
# Set these env vars:
#   AZURE_OPENAI_API_KEY      — your Azure OpenAI API key
#   AZURE_OPENAI_ENDPOINT     — e.g. https://your-resource.openai.azure.com
#   AZURE_OPENAI_DEPLOYMENT   — deployment name (default: gpt-5.4-pro)
AZURE_OPENAI_DEPLOYMENT = "gpt-5.4-pro"
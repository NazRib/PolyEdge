# Polymarket Edge — Personal Prediction Market Trading System

A quantitative trading system for Polymarket prediction markets that combines
LLM-powered probability estimation, multi-source context enrichment, whale
tracking, and Kelly Criterion position sizing to identify and exploit mispricings.

**Current phase:** Live validation and paper trading (Phase 5).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      POLYMARKET EDGE                            │
├──────────┬───────────────┬──────────────┬──────────┬───────────┤
│ Market   │ Context       │ Probability  │ Position │ Paper     │
│ Scanner  │ Enricher      │ Estimator    │ Sizer    │ Trader    │
│          │               │ (Ensemble)   │ (Kelly)  │ & Tracker │
│          ├───────────────┤              │          │           │
│          │ • News (Claude│              │          │           │
│          │   + web search│              │          │           │
│          │ • Kalshi xref │              │          │           │
│          │ • FRED econ   │              │          │           │
│          │ • Related mkts│              │          │           │
│          │ • Whale tracker│             │          │           │
├──────────┴───────────────┴──────────────┴──────────┴───────────┤
│                 Polymarket API Client Layer                      │
│          (Gamma API + CLOB API + Data API)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
polymarket_edge/
├── run_pipeline.py              # Main entry point — all run modes
├── config.py                    # Trading parameters and filters
├── core/
│   ├── api_client.py            # Unified Polymarket API client (Gamma + CLOB + Data)
│   ├── market_scanner.py        # Scans and scores tradeable opportunities
│   ├── context_enricher.py      # Multi-source enrichment pipeline
│   ├── llm_estimator.py         # Claude-powered probability estimation + calibration
│   ├── probability.py           # Ensemble framework, Bayesian updating, calibration tracking
│   ├── kelly.py                 # Kelly Criterion position sizing (fractional Kelly)
│   ├── paper_trader.py          # Paper trading engine with P&L and Brier score tracking
│   └── whale_profiler.py        # Behavioral profiling of top traders
├── strategies/
│   ├── edge_detector.py         # Basic pipeline (no enrichment)
│   └── enriched_edge_detector.py # Full pipeline with enrichment + live LLM
├── backtest_llm.py              # Backtesting framework for strategy comparison
└── data/                        # Generated at runtime
    ├── paper_trades.json
    ├── calibration_log.json
    ├── markets_cache.json
    └── whale_profiles.json      # Persistent whale behavioral profiles
```

## Quick Start

```bash
# 1. Install dependencies
pip install requests numpy pandas scipy

# 2. Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."    # Required for live LLM estimation + news search
export FRED_API_KEY="your-fred-key"      # Required for economic indicators from FRED

# 3. Run a demo with simulated data (no API keys needed)
python run_pipeline.py --demo

# 4. Scan live markets without LLM calls
python run_pipeline.py --scan-only

# 5. Run the full enriched pipeline with live Claude estimation
python run_pipeline.py --enriched --live

# 6. Check paper trading performance
python run_pipeline.py --report
```

## Run Modes

| Command | What it does |
|---|---|
| `--demo` | Synthetic markets, no API calls — shows the system end-to-end |
| `--scan-only` | Scans Polymarket for top candidates, prints scores |
| `--enriched` | Enriched pipeline with simulated LLM (no Claude API needed) |
| `--enriched --live` | Full pipeline: news search, whale tracking, live Claude estimation |
| `--enriched --whale-profiles` | Enriched pipeline with profile-aware whale signals |
| `--enriched --live --whale-profiles` | Full pipeline with profiled whales + live Claude |
| `--profile-whales` | Build/refresh whale behavioral profiles (~5-8 min) |
| `--whale-report` | Print whale profiler report (strategy breakdown, top signals) |
| `--whale-backtest` | A/B backtest: profiled vs naive whale signal (synthetic data) |
| `--enrich-demo` | Demos the enrichment sources on sample markets |
| `--report` | Prints paper trading P&L and calibration report |
| *(no flags)* | Basic pipeline without context enrichment |

## How It Works

### 1. Market Scanner (`market_scanner.py`)
Fetches active markets from the Gamma API and scores them on four dimensions:
liquidity (can you get in and out?), edge potential (is the price in an
interesting range?), timing (resolving soon enough to matter, not so soon you
can't exit), and order book imbalance (are informed traders accumulating?).
Markets below the minimum score threshold are filtered out.

### 2. Context Enrichment (`context_enricher.py`)
For each candidate market, five enrichment sources run in sequence:

- **News search** — Uses Claude with the web_search tool to find recent news relevant to the market question. Returns headlines, key facts, and a sentiment signal. Includes exponential backoff retry on rate limits.
- **Kalshi cross-platform pricing** — Searches for the same or similar market on Kalshi. A significant price difference signals potential mispricing.
- **FRED economic indicators** — Pulls relevant macroeconomic data (Fed Funds rate, CPI, unemployment, GDP, etc.) for economics-related markets.
- **Related markets** — Finds other Polymarket markets in the same event or with keyword overlap, checking for pricing consistency.
- **Whale tracker** — Fetches the Polymarket leaderboard (top profitable traders) once per pipeline run, then for each market checks if any of these whales hold positions. Cross-references holder wallets against the whale registry to produce signals with direction, size, and trader credibility (lifetime PnL). When whale profiles are enabled (see below), signals are enriched with strategy type, category-specific win rates, and credibility scores.

### 2b. Whale Profiler (`whale_profiler.py`) — optional
Builds rich behavioral profiles for top traders by analyzing their positions across categories. Run periodically with `--profile-whales` to build/refresh profiles. Profiles persist to `data/whale_profiles.json` and accumulate over time.

For each whale, the profiler computes: position count and sizing distributions, YES/NO balance, category concentration (Herfindahl index), win rates (overall and per-category), and a strategy classification:

- **CONVICTION** — Few large positions, high average size. Most informative signal.
- **SPECIALIST** — Concentrated in one category (60%+ of positions). High value in their specialty, low value elsewhere.
- **DIVERSIFIED** — Many positions across categories. Moderate signal.
- **MARKET_MAKER** — Very high position count, balanced YES/NO, high volume/PnL ratio. Signal is noise — these wallets provide liquidity, not conviction. Heavily discounted.

### 3. Probability Estimation (`llm_estimator.py` + `probability.py`)
An ensemble of estimators produces a final probability for each market:

- **Enriched LLM estimator** (45% weight, live mode) — Feeds all enrichment data plus market context into a structured Claude prompt. The LLM reasons through base rates, news impact, cross-platform signals, and whale positioning before outputting a calibrated probability. A post-hoc calibration layer corrects for systematic LLM overconfidence.
- **Base rate estimator** (15%) — Anchors on the current market price as a prior.
- **Momentum estimator** (20%) — Detects recent price trends from CLOB price history.
- **Book imbalance estimator** (15%) — Measures asymmetry in the order book.
- **Whale tracker estimator** (10%) — Basic mode: weights whale positions by PnL and size. Profiled mode: uses `profiled_whale_estimator` which weights by category-specific credibility, strategy type, and win rate. Market makers are discounted; conviction traders in their specialty category get the strongest weight.

### 4. Position Sizing (`kelly.py`)
Applies fractional Kelly Criterion given the estimated probability, market
price, confidence level, and bankroll. Enforces maximum position sizes and
minimum edge requirements. Outputs a concrete dollar amount and side (YES/NO)
for each trade.

### 5. Paper Trading (`paper_trader.py`)
Logs all signals and tracks hypothetical P&L as markets resolve. Maintains
a Brier score calibration log to measure whether the system's probability
estimates are actually well-calibrated over time.

## Configuration

Edit `config.py` to adjust trading parameters:

- `BANKROLL` — Total capital allocation ($1,000 default)
- `KELLY_FRACTION` — Fraction of full Kelly to use (0.25 = quarter Kelly, conservative)
- `MIN_EDGE` — Minimum edge to consider a trade (0.05 = 5%)
- `MAX_POSITION_PCT` — Max percentage of bankroll in any single position
- `MIN_VOLUME_24H` / `MIN_LIQUIDITY` — Liquidity filters
- `PRICE_RANGE` — Only trade markets in this probability range (default 5%–95%)
- `MAX_DAYS_TO_RESOLUTION` / `MIN_DAYS_TO_RESOLUTION` — Time horizon filters

## Key Concepts

- **Edge**: The difference between your estimated probability and the market price. A 5% edge on a 50¢ market means you think the true probability is 55%.
- **Brier Score**: Measures calibration quality — 0.25 is a coin flip, below 0.20 is decent, below 0.15 is good.
- **Kelly Criterion**: The mathematically optimal bet size given your edge and the odds. Full Kelly is aggressive; fractional Kelly (0.25x) trades off growth rate for lower variance.
- **Whale signal**: When traders with strong historical track records take large positions, it may indicate private information or superior analysis. The profiled version distinguishes conviction traders (informative) from market makers (noise) and weights signals by category-specific credibility.

## API Dependencies

| API | Required for | Auth |
|---|---|---|
| Polymarket Gamma | Market discovery | None (public) |
| Polymarket CLOB | Pricing, order books, price history | None (public, read-only) |
| Polymarket Data | Leaderboard, holders, positions | None (public) |
| Anthropic Claude | LLM estimation + news search | `ANTHROPIC_API_KEY` env var |
| FRED | Economic indicators | `FRED_API_KEY` env var |
| Kalshi | Cross-platform pricing | None (public) |

## Rate Limiting

The pipeline includes rate limit handling at multiple levels:

- **Polymarket APIs** — 0.5s throttle between requests via the client layer.
- **Claude API** — Exponential backoff retry (up to 3 attempts) on 429 responses, in both the news search enricher and the LLM estimator. A 1.5s inter-market delay in the pipeline loop prevents rate limiting in the first place.
- **Kalshi / FRED** — Per-request throttling with graceful error handling.
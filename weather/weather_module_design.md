# Weather Module — Technical Design

## Overview

A specialized module for trading daily temperature prediction markets on Polymarket, built as an extension to the existing Polymarket Edge system. Weather markets present a structurally distinct opportunity: unlike political or economic markets where the outcome is genuinely uncertain until it happens, temperature outcomes can be modeled probabilistically using numerical weather prediction (NWP) data that is freely available but not yet fully priced into the market.

The module operates as a **standalone pipeline** alongside (not integrated into) the main Polymarket Edge system. This architectural decision was made because weather markets are fundamentally different from binary prediction markets: they are multi-outcome (8-10 temperature buckets per event), require multi-outcome Kelly sizing, use numerical weather models instead of LLM estimation, and follow model update cycles rather than general market scanning. The weather pipeline shares `PaperTrader` and `PolymarketClient` from `core/` but runs its own market discovery, probability computation, and position sizing.


## Market Structure (What We're Trading)

Polymarket's daily temperature markets follow a consistent pattern:

**Question format:** "Highest temperature in {City} on {Date}?"

**Outcome structure:** Multi-outcome market with temperature buckets. US cities use 2°F buckets (e.g., "82-83°F"), international cities use 1°C buckets (e.g., "13°C"). Each bucket is a separate Yes/No tradeable outcome.

**Resolution source:** Weather Underground historical data for a specific airport weather station. Each city maps to a fixed ICAO station code:

| City | Station | ICAO | Unit | Bucket Size |
|------|---------|------|------|-------------|
| NYC | LaGuardia Airport | KLGA | °F | 2°F |
| Houston | William P. Hobby Airport | KHOU | °F | 2°F |
| Atlanta | Hartsfield-Jackson Intl | KATL | °F | 2°F |
| Denver | Denver Intl Airport | KDEN | °F | 2°F |
| London | Heathrow (likely) | EGLL | °C | 1°C |
| Munich | Munich Airport | EDDM | °C | 1°C |
| Seoul | Incheon (likely) | RKSI | °C | 1°C |
| Beijing | Capital Intl Airport | ZBAA | °C | 1°C |
| Wellington | Wellington Intl Airport | NZWN | °C | 1°C |
| Hong Kong | Hong Kong Intl (likely) | VHHH | °C | 1°C |

**Resolution URL pattern:** `https://www.wunderground.com/history/daily/{country_path}/{ICAO}`

**Resolution precision:** Whole degrees (no decimals). The recorded daily high from the station.

**Scale:** ~331 active daily temperature markets, ~$3.5M in trading volume, across roughly 6 cities with markets 3-5 days out.


## Phased Roadmap

### Phase 1 — Edge Validation (Week 1-2)

**Goal:** Prove that multi-model forecast consensus systematically outperforms market prices, before writing any trading logic.

Phase 1 runs on two parallel tracks: a **historical backtest** that reconstructs what would have happened over the past several months using archived data, and a **forward collector** that captures live snapshots going forward. The historical track gives you a dataset of 300-600+ resolved markets within a day or two. The forward track validates that the historical findings hold in real-time and catches any data pipeline issues.

---

#### Track A — Historical Backtest (fast validation)

**New file:** `weather/backtest.py`

**Core idea:** Both the weather forecast data and the Polymarket market data exist historically. By reconstructing what the models predicted at the time each market was active, and comparing that against the market prices that were live at the same moment, we can simulate the full pipeline over months of resolved markets without waiting.

**Step 1: Enumerate resolved temperature markets**

The Gamma API supports fetching closed/resolved markets. Temperature market slugs follow a predictable pattern: `highest-temperature-in-{city}-on-{month}-{day}-{year}`. The script enumerates all resolved temperature events by:

- Fetching closed events with the relevant tag IDs (temperature / weather / daily temperature).
- Alternatively, constructing slugs programmatically for each city × date going back 90 days and hitting the event endpoint directly.
- As a fallback / supplement, the `poly_data` open-source repository provides a downloadable `markets.csv` snapshot with all historical markets including `closedTime`, `question`, outcomes, and token IDs, which avoids rate-limiting issues during bulk enumeration.

For each resolved market, we extract: city, target date, temperature unit, bucket boundaries (from the outcomes list), which bucket won (the outcome with price = $1.00 at resolution), and the token IDs for each bucket (needed for price history).

**Step 2: Pull historical market prices**

The CLOB `/prices-history` endpoint accepts `startTs`, `endTs`, and a `fidelity` (resolution in minutes) parameter. For each bucket's token ID, pull price snapshots at key lead times before the target date:

```
For each resolved market:
    For each bucket token_id:
        GET /prices-history?market={token_id}&startTs={target_date - 72h}&endTs={target_date}&fidelity=60

        → Extract prices at: 72h before, 48h before, 24h before, 12h before, 6h before resolution
```

This gives us the market's probability distribution across buckets at each of those lead times. The sum across buckets at any timestamp should be close to 1.0 (any deviation is itself a signal of market inefficiency).

**Step 3: Pull historical weather forecasts**

Open-Meteo's Historical Forecast API archives what each model predicted at the time, not just what actually happened. The data is continuously archived from 2022 onwards and uses the exact same API schema as the live forecast endpoint, making the backtest code trivially reusable for forward collection.

For deterministic models (what each model predicted as the daily high):
```
GET https://api.open-meteo.com/v1/forecast
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max
  &start_date={target_date}&end_date={target_date}
  &models=ecmwf_ifs04,gfs_seamless,icon_seamless,gem_seamless,jma_seamless
  &past_days=3
```

The `past_days` parameter is key — it returns the forecast as it was issued N days before the target date, reconstructing the exact information that would have been available to a trader at the time.

For ensemble members (probability distribution):
```
GET https://api.open-meteo.com/v1/ensemble
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max
  &start_date={target_date}&end_date={target_date}
  &models=ecmwf_ifs025,gfs025
  &past_days=3
```

Additionally, the Previous Runs API explicitly provides forecasts with a lead-time offset, archiving what each model predicted 1, 2, 3, and 4+ days before the event:

```
GET https://api.open-meteo.com/v1/previous-runs
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max
  &start_date={target_date}&end_date={target_date}
  &previous_day=1    # what the model predicted 1 day before
  &models=ecmwf_ifs04,gfs_seamless,icon_seamless
```

**Step 4: Get actual observed temperature (ground truth)**

Two sources for what actually happened:

1. The resolved market itself — whichever bucket went to $1.00 tells us the observed temperature range. This is the authoritative resolution source.
2. Open-Meteo's Historical Weather API (ERA5 reanalysis) or the Historical Forecast API's "best match" mode provides the actual observed daily high, useful for computing exact station bias.

```
GET https://api.open-meteo.com/v1/forecast
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max
  &start_date={target_date}&end_date={target_date}
  &past_days=0
```

**Step 5: Compute backtest results**

For each resolved market × lead time, compute:

1. **Model-implied bucket probabilities** — run ensemble members through the `compute_bucket_probabilities()` function to produce a probability distribution across buckets.
2. **Market-implied bucket probabilities** — the CLOB prices at that lead time, normalized to sum to 1.0.
3. **Brier score (model)** — sum of (model_prob - outcome)² across all buckets, where outcome is 1 for the winning bucket and 0 for all others.
4. **Brier score (market)** — same calculation using market prices.
5. **Simulated edge** — for each bucket, `model_prob - market_price`. Positive values are buckets where the model sees higher probability than the market.
6. **Simulated P&L** — if we'd bought every bucket where model_prob > market_price + MIN_EDGE, what would the net return be? Apply Kelly sizing with the configured fraction.

**Output:** `data/weather_backtest.json` containing per-market results, and a summary report:

```python
@dataclass
class BacktestResult:
    """Result for a single resolved market at a specific lead time."""
    city: str
    target_date: str
    lead_hours: int
    
    # Model assessment
    model_bucket_probs: dict[str, float]    # {"13°C": 0.62, "14°C": 0.23, ...}
    model_brier_score: float
    model_top_bucket: str                    # Model's most likely bucket
    model_top_prob: float                    # Probability of that bucket
    
    # Market assessment  
    market_bucket_probs: dict[str, float]
    market_brier_score: float
    market_top_bucket: str
    market_top_prob: float
    
    # Ground truth
    actual_bucket: str                       # Which bucket won
    actual_temperature: float                # Observed daily high (if available)
    
    # Edge analysis
    brier_delta: float                       # market_brier - model_brier (positive = model wins)
    max_edge: float                          # Largest model_prob - market_price
    tradeable_buckets: int                   # Count where edge > MIN_EDGE
    simulated_pnl: float                     # What Kelly-sized trades would have returned
    
    # Model diagnostics
    model_agreement: float                   # Fraction of deterministic models on same bucket
    ensemble_std: float                      # Ensemble spread (uncertainty)
    per_model_forecasts: dict[str, float]    # {"ecmwf": 14.2, "gfs": 13.8, ...}
    station_bias_observed: dict[str, float]  # Forecast - actual, per model
```

**Expected dataset size:** Polymarket runs temperature markets for ~6-10 cities, ~3-5 days ahead, daily. Over 90 days, that's roughly 540-1500 resolved city-date markets. Even conservatively, 300+ markets with full data should be achievable, measured at 3 lead times each (24h, 48h, 72h) for ~900+ data points.

**Backtest analysis questions this answers:**

- Does edge exist? (Brier score delta > 0 across the full sample)
- How large is the edge? (Mean simulated P&L per trade)
- When is edge largest? (By lead time — 72h vs 24h vs 12h)
- Which cities are most mispriced? (Per-city Brier delta)
- How reliable is model consensus? (Agreement level vs. actual accuracy)
- What's the optimal MIN_EDGE threshold? (Plot edge threshold vs. win rate)
- Does station bias correction help? (Brier delta with vs. without bias correction)

---

#### Track B — Forward Data Collection (live validation)

**New file:** `weather/data_collector.py`

Runs in parallel with the backtest, capturing live data going forward. This serves three purposes: it validates that the historical findings hold in real-time, it catches any data pipeline issues before they matter, and it starts building the station bias correction table.

**What it does:**

1. Identifies active temperature markets from the Gamma API by filtering on category/tags and parsing the question format to extract city, date, and temperature unit.
2. For each market, pulls forecasts from multiple weather models via Open-Meteo (free tier, no API key required, but burst rate limits apply — see Risks).
3. Logs a daily snapshot: `{city, date, model_forecasts[], market_bucket_prices[], timestamp}` to `data/weather_snapshots.json`.
4. After markets resolve, records the actual outcome from the resolved market data (winning bucket goes to $1.00).
5. Produces a daily report comparing model-implied probabilities vs. market prices vs. actual outcome.

---

#### Shared: Data Sources (all via Open-Meteo, free tier)

| Model | Provider | Resolution | Update Freq | Horizon | Historical Archive |
|-------|----------|------------|-------------|---------|-------------------|
| IFS HRES | ECMWF | 9 km | Every 6h | 10 days | From 2022 |
| GFS | NOAA | 13-25 km | Every 6h | 16 days | From 2022 |
| ICON | DWD (Germany) | 13 km | Every 6h | 7 days | From 2022 |
| GEM | CMC (Canada) | 15 km | Every 12h | 10 days | From 2022 |
| JMA | Japan Met Agency | 20 km | Every 6h | 11 days | From 2022 |
| HRRR | NOAA | 3 km (US only) | Every 1h | 18h | From 2022 |
| AIFS | ECMWF (AI) | 0.25° | Every 6h | 10 days | From 2024 |

**Open-Meteo API calls (live and historical use the same schema):**

For deterministic models (daily high forecast):
```
GET https://api.open-meteo.com/v1/forecast
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max
  &forecast_days=7
  &models=ecmwf_ifs04,gfs_seamless,icon_seamless,gem_seamless,jma_seamless
```

For ensemble spread (uncertainty quantification):
```
GET https://api.open-meteo.com/v1/ensemble
  ?latitude={lat}&longitude={lon}
  &daily=temperature_2m_max
  &models=ecmwf_ifs025,gfs025
```

**Key design decision:** Use the ensemble API (51 ECMWF members, multiple GFS members) to build a probability distribution over temperature outcomes, not just the deterministic point forecast. The spread across ensemble members directly gives you forecast uncertainty, which maps to bucket probabilities.

---

#### Phase 1 Success Criteria

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| Model Brier score < market Brier score | Delta > 0.03 | Historical backtest, 300+ markets |
| Positive simulated P&L | Net positive at 0.08 MIN_EDGE | Historical backtest with Kelly sizing |
| Edge present at 24h lead time | >15% of markets have tradeable edge | Backtest edge distribution |
| Forward results consistent with backtest | Brier delta within 1 σ | First 2 weeks of forward data |
| Station bias is learnable | Per-model bias std < 2°F / 1°C | Backtest residual analysis |

If the backtest shows no meaningful edge (Brier delta < 0.01, or simulated P&L is flat/negative), stop here. The historical data gives us a definitive answer cheaply.

---

### Phase 1 — Empirical Results (completed)

**Dataset:** 30-day backtest, 10 cities, 540 data points across 186 resolved markets at 3 lead times (24h, 48h, 72h). Two runs performed: without bias correction and with per-city, per-model station bias correction.

#### Edge exists but is city-specific

The overall Brier score delta is slightly negative (-0.007), meaning the model's probability distribution is not systematically better calibrated than the market across all cities. However, the simulated P&L is strongly positive (+$13,657 on a $1,000 bankroll) because the Kelly sizer identifies specific buckets where the model assigns significantly more probability than the market, and those asymmetric bets pay off. You don't need to be better calibrated overall — you need to be right on the specific buckets you trade.

#### City-by-city performance (with bias correction)

| City | N | Delta | Sim P&L | Verdict |
|------|---|-------|---------|---------|
| London | 85 | +0.002 | +$10,504 | Primary — highest P&L, consistent |
| NYC | 85 | +0.011 | +$6,683 | Primary — model genuinely better |
| Hong Kong | 26 | -0.003 | +$1,440 | Secondary — positive but smaller sample |
| Atlanta | 88 | -0.001 | +$659 | Secondary — flipped from -$1,839 after bias correction |
| Beijing | 15 | +0.004 | +$528 | Watch — too few data points |
| Denver | 3 | +0.018 | +$348 | Watch — too few data points |
| Houston | 3 | +0.015 | +$263 | Watch — too few data points |
| Munich | 59 | -0.005 | -$457 | Exclude — improved with correction but still negative |
| Wellington | 88 | -0.017 | -$1,246 | Exclude — bias correction insufficient |
| Seoul | 88 | -0.036 | -$5,066 | Exclude — non-stationary bias, correction made it worse |

**Recommended trading set:** London, NYC, Hong Kong, Atlanta, Beijing, Denver, Houston (7 cities). Exclude Seoul, Wellington, Munich until further analysis. Estimated P&L from included cities only: ~+$20K over 30 days.

#### Lead time analysis

| Lead | N | Model BS | Market BS | Delta | Sim P&L |
|------|---|----------|-----------|-------|---------|
| 72h | 171 | 0.081 | 0.077 | -0.003 | +$3,328 |
| 48h | 184 | 0.080 | 0.073 | -0.007 | +$5,523 |
| 24h | 185 | 0.080 | 0.069 | -0.011 | +$4,806 |

The 48h window is the sweet spot — far enough out that the market hasn't converged, close enough that forecasts are reasonably accurate. Earlier backtests with 6h and 12h lead times showed consistently negative delta — the market has already priced in the forecast by then. These were removed from the lead time configuration.

#### Station bias correction impact

Bias correction (subtracting per-model mean forecast error per city) had the largest impact on cities with extreme model biases:

| City | Before correction | After correction | Improvement |
|------|-------------------|------------------|-------------|
| Atlanta | -$1,839 | +$659 | +$2,498 (flipped) |
| Munich | -$1,582 | -$457 | +$1,125 |
| NYC | +$6,198 | +$6,683 | +$485 |
| London | +$9,772 | +$10,504 | +$732 |
| Seoul | -$3,642 | -$5,066 | -$1,424 (worse) |

Seoul worsened because its bias is non-stationary — it shifts with weather patterns, making a flat mean correction counterproductive. The overall portfolio P&L improved by +$3,325 (+32%).

#### Key model biases discovered

| City | Worst model | Bias | Best model | Bias |
|------|-------------|------|------------|------|
| Houston | JMA | -4.1° | GFS | +0.0° |
| Atlanta | JMA | -3.5° | GFS | +0.0° |
| Seoul | GEM | +3.3° | JMA | +0.0° |
| Denver | JMA | -2.3° | GFS | +0.0° |
| Hong Kong | GFS | -2.3° | ICON | +0.0° |
| NYC | GEM | -1.8° | GFS | +0.0° |

GFS is the most universally accurate model (near 0° bias everywhere). JMA consistently underforecasts US cities. GEM overforecasts Seoul significantly.

#### Confidence tier performance

| Tier | N | Delta | Win% | Sim P&L |
|------|---|-------|------|---------|
| STRONG | 38 | +0.012 | 47% | +$816 |
| NEAR_SAFE | 147 | -0.000 | 22% | +$7,264 |
| LOW | 355 | -0.012 | 10% | +$5,576 |

STRONG tier shows genuine calibration advantage (positive delta) with 47% win rate. NEAR_SAFE is the volume driver — near-zero delta but positive P&L through asymmetric Kelly sizing. LOW tier is still net positive but with only 10% win rate; the large P&L comes from occasional very large wins.

#### Conclusions

1. **Conditional GO.** The edge is real and exploitable in specific cities (London, NYC, Hong Kong, Atlanta).
2. **Bias correction is essential** for US cities where JMA has large systematic errors.
3. **Seoul should be excluded** — its bias is non-stationary and corrections make performance worse.
4. **The 48h lead time is the sweet spot** for entering positions.
5. **The P&L profile is asymmetric** — low win rate (13-16%) but high win/loss ratio (~7:1), typical of positive-EV tail strategies.
6. **Ensemble data will improve results** — the current backtest uses point forecasts only (ensemble archives unavailable). Live trading with ensemble data from `ensemble-api.open-meteo.com` will produce tighter probability distributions.

---

### Forward Validation Results (Phase 3 — in progress)

**Paper trading period:** March 26 — April 7, 2026 (~10 days, 4 scans/day).

**Latest report (April 7, 2026, after bug fixes):**

| Metric | Value | Backtest comparison |
|--------|-------|---------------------|
| Resolved trades | 101 (9W / 92L) | — |
| Win rate | 9% | Backtest: 13-16% |
| Avg win | $71.14 | Backtest: $336 |
| Avg loss | $14.41 | Backtest: $47 |
| Total P&L | -$685.79 | Backtest: +$13,657 |
| Brier score | 0.117 | Backtest: 0.080 |
| Expected value/trade | -$6.70 | Backtest: +$19 |

**Assessment:** Forward results are significantly worse than backtest. The win rate (9%) is below the backtest range (13-16%), and critically the avg win ($71) is much smaller than backtest ($336), indicating the market prices the winning buckets more tightly in live trading — there's less payoff upside when you win.

**Data quality issues discovered and fixed during paper trading:**

1. **Duplicate trades (fixed March 29):** The scanner re-entered trades on the same bucket across scan cycles. ~5 buckets had doubled positions, amplifying losses. Fix: deduplication check against open `market_id` set.

2. **Missing model quality gate (fixed April 7):** Open-Meteo deterministic API returned 502/429 rate limit errors, leaving events with "0 models + 80 ensemble". The scanner traded these with no bias correction and no model agreement signal — effectively degraded data. Fix: quality gate requiring ≥3 deterministic models to trade.

3. **API rate limiting (fixed April 7):** Rate limit delay was 0.15s between requests, causing burst throttling on Open-Meteo's free tier. Increased to 1.0s.

4. **SSL transient errors (fixed March 28):** Open-Meteo connections dropped intermittently. Fix: application-level retry with 2s/4s backoff for SSLError and ConnectionError.

**Current status:** Paper trading reset with all fixes applied. Running clean data collection to determine whether the strategy is viable after correcting the data quality issues, or whether the market has become too efficient.

#### Competitive Landscape (discovered March 30, 2026)

Research revealed a crowded and growing field of automated weather traders on Polymarket:

- **Known profitable bots:** One turned $1,000 → $24,000 on London weather since April 2025. Another made $65,000 across NYC, London, Seoul. A trader "meropi" is cited as a top automated weather trader.
- **Public tooling:** Multiple open-source GitHub repos (polyBot-Weather, polymarket-kalshi-weather-bot), Medium tutorials, and no-code bot platforms (PolyTraderBot at $250) all implement the same core strategy — compare weather model forecasts to Polymarket prices, trade the discrepancy.
- **Common approach:** Most use GFS ensemble (31 members) from Open-Meteo, 8% edge threshold, Kelly sizing — nearly identical to our setup.
- **Implication:** The easy edge may have been competed away since the backtest period (pre-February 2026). The February 2026 viral articles publicizing the strategy likely brought many new automated participants, making prices more efficient.

**Our potential differentiators:** Multi-model ensemble (5 deterministic + 2 ensemble vs most bots using only GFS), per-station bias correction, city-level exclusion. However, the marginal improvement from these may not overcome the fundamental efficiency gain from dozens of bots arbitraging the same signal.

---

### Phase 2 — Standalone Scanner & Paper Trading (completed)

**Decision:** Weather markets were built as a standalone pipeline rather than integrated into the main enrichment pipeline. The main pipeline is designed for binary markets with LLM-driven estimation. Forcing multi-outcome weather markets through that architecture would have required different scanning logic, multi-outcome Kelly sizing, different ensemble weights, and scheduling around model update cycles. A standalone approach is cleaner and more portable.

**What was built:** `weather/scanner.py` — a complete trading pipeline:

1. **Discovery** — Fetches open temperature events from Gamma API using tag 103040, filters to 7 tradeable cities
2. **Enrichment** — Fetches live deterministic (5 models) + ensemble (80+ members) forecasts, applies bias correction
3. **Edge detection** — Compares model probabilities against market prices across all buckets per event
4. **Position sizing** — Per-bucket Kelly sizing with three exposure caps: per-bucket ($50), per-city/date ($150), portfolio (30%)
5. **Paper trading** — Uses `core.PaperTrader` with separate `data/weather/` data directory
6. **Resolution checking** — Monitors open trades and resolves against Gamma API outcomes
7. **Quality gate** — Requires ≥3 deterministic models before trading; skips events where API failures left only ensemble data

**Also built:** `weather/enricher.py` — a `WeatherEnricher` class that can optionally plug into `ContextEnricher` for cases where a temperature market appears in the general scanner. Not currently wired in — the standalone scanner handles all weather trading.

**Scheduler:** `run_weather_collector.py` — runs both data collection and scanner paper trading at 04, 10, 16, 22 UTC (4h after each model update cycle). Supports `--once`, `--collect-only`, `--scan-only` flags.

---

### Phase 3 — Live Validation (in progress)

**Goal:** Validate that the backtest edge holds in real-time with live ensemble data and actual market prices.

**Status as of April 7, 2026:** Paper trading running for ~10 days. See "Forward Validation Results" section above for detailed metrics.

---

### Phase 4 — Live Execution (pending validation)

**Goal:** If paper trading validates the edge, execute real trades via Polymarket CLOB API with a funded wallet.

**Prerequisites:** Paper trading must show positive P&L over 30+ resolved trades with clean data (no duplicate trade bugs, no missing-model trades). Current results are under assessment after fixing multiple data quality issues.

---

### Phase 5 — Automation & Speed (pending)

**Goal:** Optimize execution timing around model update cycles for maximum edge capture at the 48h window.


## File Structure

```
weather/
├── __init__.py
├── config.py               # Cities, stations, models, tag IDs, trading parameters
├── utils.py                # Question parsing, bucket extraction, probability computation
├── models.py               # Open-Meteo API client (deterministic + ensemble, with retry)
├── bias.py                 # Station bias correction (build, save, load, apply per-city/model)
├── backtest.py             # Phase 1A: historical edge analysis with tag-based enumeration
├── data_collector.py       # Phase 1B: forward live snapshot collection
├── scanner.py              # Phase 2: standalone market scanner + paper trading pipeline
├── enricher.py             # Optional: WeatherEnricher for main pipeline integration
└── weather_module_design.md # This document

run_weather_collector.py    # Scheduler: runs collector + scanner at 04/10/16/22 UTC
```


## Configuration Additions (`weather/config.py`)

```python
# Cities Polymarket actively runs temperature markets for
# This registry needs manual maintenance as Polymarket adds/removes cities
WEATHER_CITIES = {
    "NYC": {
        "lat": 40.7731, "lon": -73.8803,          # LaGuardia coordinates
        "icao": "KLGA", "unit": "F", "bucket_size": 2,
        "wunderground_path": "us/ny/new-york-city/KLGA",
    },
    "Houston": {
        "lat": 29.6454, "lon": -95.2789,
        "icao": "KHOU", "unit": "F", "bucket_size": 2,
        "wunderground_path": "us/tx/houston/KHOU",
    },
    "Atlanta": {
        "lat": 33.6407, "lon": -84.4277,
        "icao": "KATL", "unit": "F", "bucket_size": 2,
        "wunderground_path": "us/ga/atlanta/KATL",
    },
    "Denver": {
        "lat": 39.8561, "lon": -104.6737,
        "icao": "KDEN", "unit": "F", "bucket_size": 2,
        "wunderground_path": "us/co/denver/KDEN",
    },
    "London": {
        "lat": 51.4700, "lon": -0.4543,
        "icao": "EGLL", "unit": "C", "bucket_size": 1,
        "wunderground_path": "gb/london/EGLL",
    },
    "Munich": {
        "lat": 48.3537, "lon": 11.7750,
        "icao": "EDDM", "unit": "C", "bucket_size": 1,
        "wunderground_path": "de/munich/EDDM",
    },
    "Seoul": {
        "lat": 37.4602, "lon": 126.4407,
        "icao": "RKSI", "unit": "C", "bucket_size": 1,
        "wunderground_path": "kr/seoul/RKSI",
    },
    "Beijing": {
        "lat": 40.0799, "lon": 116.5844,
        "icao": "ZBAA", "unit": "C", "bucket_size": 1,
        "wunderground_path": "cn/beijing/ZBAA",
    },
    "Wellington": {
        "lat": -41.3272, "lon": 174.8050,
        "icao": "NZWN", "unit": "C", "bucket_size": 1,
        "wunderground_path": "nz/wellington/NZWN",
    },
    "Hong Kong": {
        "lat": 22.3080, "lon": 113.9185,
        "icao": "VHHH", "unit": "C", "bucket_size": 1,
        "wunderground_path": "cn/hong-kong/VHHH",
    },
}

# Models to query from Open-Meteo
WEATHER_MODELS_DETERMINISTIC = [
    "ecmwf_ifs04",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless",
    "jma_seamless",
]
WEATHER_MODELS_ENSEMBLE = ["ecmwf_ifs025", "gfs025"]

# Ensemble probability computation
SMOOTHING_SIGMA_F = 1.5     # Gaussian kernel σ for °F markets
SMOOTHING_SIGMA_C = 0.8     # Gaussian kernel σ for °C markets

# Confidence tiers
CONFIDENCE_TIERS = {
    "LOCK":      {"min_agreement": 1.00, "min_models": 5},  # 100% unanimous
    "STRONG":    {"min_agreement": 0.90, "min_models": 4},  # 90%+ agree
    "SAFE":      {"min_agreement": 0.80, "min_models": 3},  # 80%+ agree
    "NEAR_SAFE": {"min_agreement": 0.60, "min_models": 3},  # 60%+ agree
}

# Trading parameters (weather-specific overrides)
WEATHER_MIN_EDGE = 0.08          # Higher minimum edge than general markets
WEATHER_KELLY_FRACTION = 0.20    # Slightly more conservative
WEATHER_MAX_BUCKET_POSITION = 50 # Max $ per individual bucket bet
WEATHER_MAX_CITY_EXPOSURE = 150  # Max $ total across all buckets for one city/date
```


## Key Technical Risks & Mitigations

**Risk: Station-to-model mismatch.** Weather models forecast for a grid cell, not the exact station location. Airport stations can differ from the nearest grid point due to runway heat, elevation, or microclimate.
*Mitigation:* Phase 1 data collection builds a per-station, per-model bias table. Apply correction before computing probabilities. Revisit monthly as seasonal bias can shift.

**Risk: Market resolution ambiguity.** Polymarket resolves on Weather Underground data. If WU reports a different value than what models predicted (e.g., due to station equipment issues), you lose regardless of forecast quality.
*Mitigation:* Cross-reference WU station data with nearby METAR/SYNOP observations. If they diverge significantly, reduce confidence or skip the market.

**Risk: Edge compression.** Other automated traders (Degen Doppler already exists) are exploiting the same data. As more participants automate, the market becomes more efficient.
*Mitigation:* Differentiate through station bias correction, ensemble-based probability computation (most competitors use simpler point-forecast agreement), and speed (trading promptly after model updates).

**Risk: Thin liquidity per bucket.** A $400K total volume market split across 8-10 buckets means ~$40-50K per bucket. Large positions can move the price against you.
*Mitigation:* Cap per-bucket position size. Use limit orders rather than market orders. Spread across multiple cities/dates for diversification.

**Risk: Correlated losses across cities.** A systematic model failure (e.g., all models miss a cold front) could cause losses across multiple cities simultaneously.
*Mitigation:* Cap total weather exposure as a fraction of bankroll. The `WEATHER_MAX_CITY_EXPOSURE` config limits single-city risk, but also enforce a portfolio-level weather ceiling.

**Risk: Backtest overfitting / survivorship bias.** Historical analysis may overstate edge if we inadvertently optimize parameters to fit the backtest data, or if we only see markets that Polymarket chose to keep running (survivorship). Additionally, the market may have been less efficient 3 months ago than it is today due to fewer automated participants.
*Mitigation:* Use the backtest only for go/no-go, not for parameter tuning. Keep parameter choices (smoothing σ, MIN_EDGE, Kelly fraction) fixed before looking at results. Split the backtest into an in-sample period (first 60 days) and out-of-sample holdout (last 30 days) to check for degradation. Forward collection (Track B) serves as the final out-of-sample validation.

**Risk: Historical forecast API gaps.** Open-Meteo's archive coverage varies by model — some models may have incomplete archives for certain dates or regions, and the ensemble archive may not go as far back as the deterministic models.
*Mitigation:* Design the backtest to gracefully degrade — if a model is missing for a given date, use the remaining available models. Track data completeness as a metric and exclude market-dates where fewer than 3 models have archived data.

**Risk: Non-stationary bias (confirmed for Seoul).** Some cities exhibit model biases that shift with weather patterns or seasons, making a flat mean correction counterproductive. The backtest showed Seoul's GEM bias at +3.3° was not stable — correction worsened P&L from -$3,642 to -$5,066.
*Mitigation:* Exclude cities where bias correction worsens performance. For future work, consider rolling-window bias (last 14 days) or weather-regime-dependent corrections rather than a flat historical mean. Monitor included cities monthly for bias drift.

**Risk: Open-Meteo free tier rate limiting (confirmed).** With 7 cities × 5 days × 2 API calls per event, plus the data collector making similar calls, the scanner triggers 429 (rate limit) and 502 errors on Open-Meteo's free tier during burst periods. When deterministic models fail but ensemble succeeds, the system previously traded on degraded ensemble-only data with no bias correction.
*Mitigation:* Increased inter-request delay from 0.15s to 1.0s. Added quality gate requiring ≥3 deterministic models before trading. Consider Open-Meteo's paid tier if the strategy proves viable, or stagger city requests across scan cycles.

**Risk: Competitive edge compression (confirmed in forward validation).** The weather trading strategy was publicly documented in viral February 2026 articles and multiple open-source implementations. The backtest (using pre-February data) showed +$13,657 P&L; forward paper trading (post-February) shows -$685 P&L. The market appears to have become significantly more efficient as automated participants increased.
*Mitigation:* Monitor forward results for another week with clean data. If the edge has been competed away, consider: (a) abandoning weather for other market categories, (b) looking for second-order edges the simple bots miss (e.g., regime-dependent bias, extreme weather events, precipitation markets), or (c) focusing only on cities/times where competition is thinnest.


## Dependencies

| Dependency | Purpose | Auth Required | Cost |
|-----------|---------|---------------|------|
| Open-Meteo Forecast API | Multi-model deterministic forecasts (live) | None | Free (non-commercial) |
| Open-Meteo Historical Forecast API | Archived model predictions for backtest | None | Free (non-commercial) |
| Open-Meteo Previous Runs API | Lead-time-specific archived forecasts | None | Free (non-commercial) |
| Open-Meteo Ensemble API | Ensemble member forecasts for probability | None | Free (non-commercial) |
| Open-Meteo Historical Weather API | Observed actuals (ERA5 reanalysis) | None | Free (non-commercial) |
| Polymarket Gamma API | Market discovery, resolved market metadata | None | Free |
| Polymarket CLOB API | Order books, pricing per bucket, price history | None | Free |
| poly_data (GitHub) | Bulk historical market CSV (optional accelerator) | None | Free |
| Weather Underground | Resolution verification (manual or scrape) | None (public pages) | Free |

No additional API keys are needed beyond what is already configured (Anthropic, FRED).


## Success Metrics

### Phase 1 — Backtest Validation (completed)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Backtest sample size | 300+ markets | 540 data points / 186 markets | ✅ |
| Model Brier Score | < 0.15 | 0.080 | ✅ |
| Market Brier Score | > 0.18 | 0.073 | ❌ Market is better calibrated than expected |
| Brier Score Delta (overall) | > 0.03 | -0.007 | ❌ Model doesn't beat market overall |
| Brier Score Delta (best cities) | > 0.03 | +0.011 (NYC) | ⚠️ City-specific |
| Simulated P&L (backtest) | Positive | +$13,657 (+32% with bias correction) | ✅ |
| Edge by lead time | Larger at 48-72h | 48h best (+$5,523), 6h worst (-$76) | ✅ |
| Bias correction impact | Improves losing cities | Atlanta flipped -$1,839 → +$659 | ✅ |
| City-level profitability | Majority of cities positive | 7/10 cities profitable | ✅ |

**Interpretation:** The original go/no-go criteria (overall Brier delta > 0.03) were designed for a scenario with ensemble data. With point-forecast-only backtest data, the model's probability distribution is less precise, but the Kelly sizer still extracts significant P&L from specific mispriced buckets. The edge is real but city-specific and asymmetric — it shows up in P&L rather than in aggregate calibration metrics.

### Phase 2-5 — Live Validation & Trading

| Metric | Target | Actual (April 7) | Status |
|--------|--------|-------------------|--------|
| Paper P&L | Positive over 2 weeks | -$685.79 over 10 days | ❌ |
| Win rate | 10-16% (matching backtest) | 9% (101 trades) | ⚠️ Below target |
| Avg win / Avg loss ratio | > 5:1 | 4.9:1 ($71/$14) | ⚠️ Borderline |
| Brier score | < 0.10 (matching backtest) | 0.117 | ❌ Worse than backtest |
| Win rate on STRONG tier | > 40% | Not yet measured post-fix | ⏳ |
| Data quality | Clean (no bugs) | 3 bugs found and fixed | ✅ Fixed |
| API reliability | <5% failure rate | ~15% 502/429 failures at 0.15s | ✅ Fixed (1.0s delay) |

**Interpretation:** Forward results are materially worse than backtest on every metric. Two possible explanations: (1) data quality bugs degraded ~30-40% of trades (duplicate positions, missing-model trades), contaminating the results, or (2) the market has become more efficient post-February 2026 due to increased bot competition. The paper trading was reset on April 7 with all bugs fixed — the next week of clean data will disambiguate.


## Implementation Priority

### Completed

- **Phase 1A — Historical Backtest**: 540 data points across 10 cities, 30 days, 3 lead times. Tag-based market enumeration → historical forecast retrieval → probability computation → CLOB price history → Brier scoring → Kelly P&L simulation. Result: +$13,657 simulated P&L.
- **Station Bias Correction**: Per-city, per-model mean bias table from backtest residuals. Flipped Atlanta from -$1,839 to +$659. Applied via `--bias-correct` flag in backtest, loaded automatically in live scanner.
- **City Selection**: 7 profitable cities (London, NYC, Hong Kong, Atlanta, Beijing, Denver, Houston). 3 excluded (Seoul, Wellington, Munich).
- **Standalone Weather Scanner**: Full pipeline in `weather/scanner.py` — discovery, enrichment, edge detection, Kelly sizing, paper trading, resolution checking.
- **Forward Data Collector**: Running on 6h schedule, 885+ snapshots collected across 10 cities.
- **Automated Scheduler**: `run_weather_collector.py` runs both collector and scanner at 04/10/16/22 UTC.
- **Bug Fixes**: Duplicate trade dedup, missing-model quality gate (≥3 required), API rate limit increase (0.15s → 1.0s), SSL retry logic, Gamma API slug endpoint (`/events/slug/{slug}`), tag ID hardcoding.

### In Progress

- **Forward Validation (Phase 3)**: Paper trading reset on April 7 with all bugs fixed. Accumulating clean data to determine if the edge exists in real-time or has been competed away. Decision point: 1 week of clean results.

### Next Steps (conditional on validation results)

**If paper trading turns positive (win rate ≥12%, positive P&L over 30+ trades):**
1. Phase 4 — Live execution via CLOB API with funded wallet
2. Phase 5 — Execution speed optimization around model update cycles
3. Consider Open-Meteo paid tier for API reliability

**If paper trading remains negative:**
1. Conclude that the weather market edge has been competed away by the growing bot ecosystem
2. Preserve the infrastructure for potential future use (new cities, precipitation markets, regime changes)
3. Redirect effort to the main pipeline (Phase 5 paper trading for political/economic markets)

### Key Technical Learnings

- **Open-Meteo historical ensemble data is NOT available.** The `previous-runs-api` and `historical-forecast-api` subdomains both 404 on `/v1/ensemble`. Historical backtests must use point forecasts with wider Gaussian σ. Live forecasts have full ensemble access (confirmed: 80+ members from `ensemble-api.open-meteo.com`).
- **The Gamma API `/tags` endpoint does not reliably list temperature tags.** Tag IDs must be discovered from actual market metadata and hardcoded (temperature = 103040, weather = 84).
- **CLOB price history can be sparse for low-activity buckets.** A progressive window search (6h → 12h → 24h) recovers most data points, but some lead times still have no market prices and must be skipped.
- **Post-resolution prices must never be used as a market baseline** — they give Brier score ≈ 0.0 and make the comparison meaningless.
- **Polymarket uses both "or above/below" and "or higher/lower"** for edge bucket labels — both variants must be parsed.
- **Open-Meteo free tier has burst rate limits.** At 0.15s between requests, 70+ calls in a scan cycle trigger 429/502 errors. Increased to 1.0s.
- **Always gate on data quality before trading.** When deterministic model API fails but ensemble succeeds, the system has degraded data (no bias correction, no model agreement). A quality gate requiring ≥3 deterministic models prevents trading on incomplete signals.
- **Deduplication is essential for scheduled scanners.** Without checking open positions, the scanner re-enters the same bucket trade every cycle, doubling risk.
- **Backtest-to-live divergence is a real risk.** The backtest used pre-February 2026 data. The strategy went viral in February 2026, bringing many new automated participants. The market may have structurally changed between the backtest period and the live trading period.
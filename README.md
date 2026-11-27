# agentfund

An AI-run mini asset manager. Specialized agents do the work of a real firm, and every decision passes through deterministic, auditable controls.

**Research question:** how much do LLM trading agents' backtest results degrade once look-ahead contamination (the model having seen the historical data in training) is removed?

> **Paper trading only.** The final product trades Alpaca's paper account: simulated money against real, current market prices. The code cannot reach a real-money account (see [Safety](#safety-paper-only)), and it has no code for bank accounts, transfers or funding.

> Status: **all 6 thin-slice build steps done** (the agent layer is still one analyst running on an offline stand-in). Everything runs offline on a synthetic market, so the pipeline and tests work without keys. Real prices come from Yahoo Finance and paper trades go to a local play-money account, so no keys or accounts are needed.

## What works now

| Build step | Module | What it does |
|---|---|---|
| 1. Point-in-time data | `agentfund/data` | `PointInTimeStore` keeps full history; decision code only gets a `PITView` frozen at a date, which raises `LookAheadError` on any request for later data. Fundamentals are keyed by **filed date**, not period end, and restatements appear only once refiled. The universe keeps delisted names (no survivorship bias). DuckDB and CSV loaders. |
| 2. Backtester + leakage tests | `agentfund/backtest` | Event-driven daily engine: decide at the close, fill at the **next open**, pay commission + half spread + square-root impact, delisted names liquidated with a haircut. Metrics include the probabilistic and **deflated Sharpe ratio**. `leakage.py` scrambles all data after a cutoff and requires decisions and the equity curve up to that cutoff to be bit-for-bit unchanged. |
| 3. Analyst agent | `agentfund/agents` | Builds a data pack from the `PITView` (price stats, filed fundamentals, point-in-time filing excerpts) and asks Claude for a structured `ResearchNote` (rating, conviction, thesis, evidence). `anonymize=True` replaces the ticker/name with an alias, rebases prices to 100 and turns dates into offsets: the lever for measuring contamination. Every LLM call is recorded with tokens, cost and latency. |
| 4. Risk veto layer | `agentfund/risk` | `RiskOfficer` approves, resizes or vetoes every proposal in a fixed rule order: kill switch (checked daily), tradeability, long-only, min price, position cap, sector cap, gross cap, liquidity (ADV participation), turnover. `RiskLimits` is frozen and fingerprinted; `tightened()` refuses any change that loosens a limit, so agents can ask for more caution but never less. |
| 5. Paper trading + decision log | `agentfund/broker`, `agentfund/paper`, `agentfund/audit` | Real daily prices from Yahoo Finance (or Alpaca), then the same strategy → risk officer path places paper orders in a local play-money account (or Alpaca's paper account). Dry run by default. Every cycle and order goes into an append-only JSONL log where each record carries the SHA-256 of the previous one; `verify-log` detects any edit, deletion or reordering. |

| 6. Dashboard | `agentfund/dashboard` | `python -m agentfund dashboard` serves http://localhost:8050: paper account value, positions and P&L, every strategy's backtest, and the decision log, which updates as new entries are written (Server-Sent Events). Read-only. |
| Portfolio construction | `agentfund/strategies/portfolio.py` | Equal weight, inverse volatility, risk parity (equal risk contribution) and long-only minimum variance, both on a Ledoit-Wolf shrunk covariance matrix, compared head to head. |
| Walk-forward testing | `agentfund/backtest/walkforward.py` | Picks a variant using only the trailing 3 years, then scores it on the next 6 months, window by window. Reports out-of-sample Sharpe next to the hindsight-best variant, plus the deflated Sharpe across all variants tried. |
| Factor attribution | `agentfund/backtest/attribution.py` | Regresses strategy returns on the Fama-French 5 factors + momentum (free from Ken French's data library) with Newey-West t-stats: how much is alpha, how much is factor exposure. |

Strategies: `equal_weight`, `momentum_12_1`, `inverse_vol`, `risk_parity`, `min_variance`, and `analyst_named` / `analyst_anon` (agent-driven).

## Quick start

```bash
make setup         # uv venv + editable install
make test          # 60 tests, including the look-ahead guard
python -m agentfund compare
python -m agentfund leakage --strategy momentum_12_1
python -m agentfund backtest --strategy analyst_anon
python -m agentfund research SYN010 --as-of 2021-03-31 --anonymize
python -m agentfund contamination
python -m agentfund verify-log runs/<run>/decisions.jsonl
python -m agentfund walkforward                    # strategy choice judged out of sample
python -m agentfund factors                        # download Fama-French factors
python -m agentfund --data data/prices.duckdb attribution
python -m agentfund dashboard                      # http://localhost:8050
```

### Paper trading (no account or keys needed)

Prices come from Yahoo Finance (`yfinance`) and trades go to a **local paper account**: a JSON file on your computer (`runs/paper/local_account.json`) holding $100,000 of play money. It never connects to a broker or a bank.

**The daily routine is one command**, run after the close (4:30pm New York time or later):

```bash
python -m agentfund daily
```

It downloads the latest prices, fills the last run's orders at that morning's open (real prices, same costs as the backtest), checks the drawdown kill switch, rebalances on the first run of each month, and prints a short report. It's safe to run any time and any number of times: during market hours it waits, and with no new prices (weekend, holiday, second run) it does nothing. Every run is written to the hash-chained decision log.

| Command | What it does |
|---|---|
| `python -m agentfund daily` | The routine above |
| `python -m agentfund daily --dry-run` | Shows what it would do, places nothing |
| `python -m agentfund daily --strategy min_variance` | Switches strategy (remembered); rebalances into it at this run. Default: `risk_parity` |
| `python -m agentfund pause` / `resume` | Stop / restart placing new orders (prices and fills keep updating) |
| `python -m agentfund paper status` | Balance, positions, P&L |
| `python -m agentfund dashboard` | http://localhost:8050 |

### Running it automatically (macOS)

```bash
python -m agentfund schedule install     # every weekday at 5:00pm New York time, in your local time
python -m agentfund schedule status      # when it runs, plus the last lines of runs/paper/daily.log
python -m agentfund schedule remove
```

macOS doesn't let background jobs read the Downloads, Desktop or Documents folders, so keep the project in your home folder (for example `~/agentfund`). If the Mac is asleep at 5pm, the job runs when it wakes.

yfinance is free but unofficial (it can rate-limit or break; `daily` then carries on with the last saved prices), and today's 50 large caps (`agentfund/data/universes.py`) carry survivorship bias over past years. That's fine for paper trading today, but not for research results.

**Optional: Alpaca.** `fetch --source alpaca` and `paper ... --broker alpaca` use Alpaca's free data and paper account instead (set `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` to paper keys).

### Safety: paper only

- The default local paper account is a file. It contains no network code at all (a test checks this).
- The trading URL is a hard-coded constant (`https://paper-api.alpaca.markets`). There is no option to change it, and the client refuses Alpaca's live URL, including through the `APCA_API_BASE_URL` environment variable.
- Before any order, the client checks the account number Alpaca returns starts with `PA` (paper) and refuses otherwise.
- The client only implements account, positions, clock and order calls. A test fails if a method mentioning transfers, banks, ACH, wires, funding, deposits or withdrawals is ever added.
- Orders are a dry run unless you pass `--submit`.

Each `backtest` writes `runs/<timestamp>-<strategy>/` with `equity.csv`, `trades.csv`, `summary.json` and the hash-chained `decisions.jsonl`.

### Using Claude

Without credentials the analyst uses a deterministic offline stub (clearly marked `[offline stub]` in its thesis). Add `--use-claude` and set `ANTHROPIC_API_KEY` to call `claude-opus-5-5` with adaptive thinking, structured outputs and server-side refusal fallback. Notes are cached in `runs/note_cache`, so reruns are free. A full monthly backtest covering 15 names over 8 years is roughly 1,400 calls; start with `--start` / `--coverage` to keep the first runs small.

### Using real data

`--data path/to/folder` loads `prices.csv` (`date,ticker,open,high,low,close,volume`), `universe.csv` (`ticker,name,sector,start_date,end_date`, **including delisted names**) and optional `fundamentals.csv` (`ticker,period_end,filed_date,...`). `--data file.duckdb` loads a saved store.

## Why the leakage test matters

`tests/test_leakage.py` runs in CI. It contains a deliberately leaky strategy (`PeekingMomentum`) and asserts that the harness catches it, and asserts that every real strategy, the analyst pipeline and the engine's equity curve are invariant to scrambled future data. If a change introduces look-ahead bias anywhere in the decision path, CI fails.

Note what this does and does not cover: the harness proves the *code* never sees the future. It cannot stop a *model* from remembering it. That residual is exactly what `contamination` (named vs anonymized) is designed to measure, and it only means something with `--use-claude` on real historical data.

## Roadmap

Next, now that the thin slice runs end to end:

- EDGAR loader for 10-K/10-Q/transcripts into `DocumentStore`
- PM vs skeptic debate agents producing an investment memo, and an LLM risk agent in front of `RiskOfficer`
- Rust core (PyO3) for the engine loop and a limit-order-book execution simulator
- MCP server exposing portfolio, risk and research tools
- Agent evals suite (rating accuracy vs forward returns, evidence faithfulness, calibration)
- Optional: agent-based market simulation (herding, bubbles)

Before building the contamination study, review the existing literature on LLM look-ahead bias in financial forecasting to find the gap this project fills.

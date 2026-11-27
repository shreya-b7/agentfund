"""Command-line entry point: ``python -m agentfund <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .agents import AnalystAgent, CallLedger, default_llm
from .audit import DecisionLog, verify
from .backtest import Backtester, summarize
from .backtest.leakage import check_backtest, check_strategy
from .data import PointInTimeStore, generate_market
from .data.store import load_csv_dir, load_duckdb, save_duckdb
from .strategies import STRATEGIES, AnalystStrategy


def _load(path: str | None, seed: int) -> PointInTimeStore:
    if not path:
        return generate_market(seed=seed)
    p = Path(path)
    return load_duckdb(p) if p.suffix == ".duckdb" else load_csv_dir(p)


def _run_dir(root: str, name: str) -> Path:
    d = Path(root) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt(summary: dict) -> str:
    pct = {"cagr", "vol", "max_drawdown", "psr", "deflated_sharpe"}
    return "  ".join(f"{k}={v:.1%}" if k in pct else f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items())


def _analyst(args, ledger: CallLedger, anonymize: bool) -> AnalystAgent:
    llm = default_llm(ledger, offline=not args.use_claude)
    return AnalystAgent(llm, anonymize=anonymize, cache_dir=args.cache)


def _strategy(args, store, ledger):
    if args.strategy in ("analyst_named", "analyst_anon"):
        return AnalystStrategy(_analyst(args, ledger, args.strategy == "analyst_anon"), coverage=args.coverage)
    return STRATEGIES[args.strategy]()


def cmd_synth(args) -> None:
    store = generate_market(n_tickers=args.tickers, seed=args.seed)
    save_duckdb(store, args.out)
    print(f"wrote {args.out}: {store.prices['ticker'].nunique()} tickers, {len(store.dates)} days (synthetic)")


def cmd_backtest(args) -> None:
    store = _load(args.data, args.seed)
    ledger = CallLedger()
    strat = _strategy(args, store, ledger)
    out = _run_dir(args.out, strat.name)
    log = DecisionLog(out / "decisions.jsonl")
    res = Backtester(store, strat, log=log).run(args.start, args.end)
    summ = summarize(res, n_trials=args.trials)
    res.equity.to_csv(out / "equity.csv")
    res.trades.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"strategy": strat.name, **summ, "llm": ledger.totals(), "log_head": log.head}, indent=2))
    print(f"{strat.name}: {_fmt(summ)}")
    if ledger.records:
        print(f"llm: {ledger.totals()}")
    print(f"run saved to {out} (decision log head {log.head[:16]})")


def cmd_compare(args) -> None:
    store = _load(args.data, args.seed)
    rows = {}
    for name, cls in STRATEGIES.items():
        rows[name] = summarize(Backtester(store, cls()).run(args.start, args.end), n_trials=len(STRATEGIES))
    df = pd.DataFrame(rows).T[["cagr", "vol", "sharpe", "max_drawdown", "annual_turnover", "cost_drag_bps_per_year", "deflated_sharpe"]]
    print(df.round(3).to_string())


def cmd_research(args) -> None:
    store = _load(args.data, args.seed)
    ledger = CallLedger()
    note = _analyst(args, ledger, args.anonymize).research(store.view(args.as_of), args.ticker)
    print(note.model_dump_json(indent=2))
    print(f"llm: {ledger.totals()}", file=sys.stderr)


def cmd_leakage(args) -> None:
    store = _load(args.data, args.seed)
    cls = STRATEGIES[args.strategy]
    for label, rep in (
        ("decisions", check_strategy(lambda s: cls(), store, args.cutoff)),
        ("equity", check_backtest(lambda s: cls(), store, args.cutoff)),
    ):
        print(f"{label}: {'PASS' if rep.passed else 'FAIL'} - {rep.detail} ({rep.checked_dates} dates)")
        if not rep.passed:
            sys.exit(1)


def cmd_contamination(args) -> None:
    """Named vs anonymized analyst over the same dates: the look-ahead contamination gap."""
    store = _load(args.data, args.seed)
    results = {}
    for anon in (False, True):
        ledger = CallLedger()
        strat = AnalystStrategy(_analyst(args, ledger, anon), coverage=args.coverage)
        res = Backtester(store, strat).run(args.start, args.end)
        results[strat.name] = {**summarize(res, n_trials=2), "llm_cost_usd": ledger.totals()["cost_usd"]}
    df = pd.DataFrame(results).T
    print(df[["cagr", "sharpe", "max_drawdown", "deflated_sharpe", "llm_cost_usd"]].round(3).to_string())
    gap = results["analyst_named"]["sharpe"] - results["analyst_anon"]["sharpe"]
    print(f"\nSharpe gap (named - anonymized): {gap:+.3f}")
    if not args.use_claude:
        print("note: offline stub ignores names, so the gap is ~0 by construction. Use --use-claude on real data.")


PRICES_DB = "data/prices.duckdb"


def cmd_fetch(args) -> None:
    if args.source == "alpaca":
        from .data.alpaca import alpaca_store

        store = alpaca_store(start=args.start)
    else:
        from .data.yahoo import yahoo_store

        store = yahoo_store(start=args.start)
    save_duckdb(store, args.out)
    print(f"wrote {args.out}: {store.prices['ticker'].nunique()} tickers, "
          f"{store.dates[0].date()} to {store.dates[-1].date()} ({args.source}, adjusted daily prices)")


def _paper_client(args, store):
    if args.broker == "alpaca":
        from .broker import AlpacaPaperClient

        return AlpacaPaperClient()
    from .paper import LocalPaperBroker

    return LocalPaperBroker(store)


def cmd_paper(args) -> None:
    from .paper import PaperTrader

    if not Path(args.prices).exists():
        sys.exit(f"no price file at {args.prices}. Run: python -m agentfund fetch")
    store = load_duckdb(args.prices)
    client = _paper_client(args, store)
    acct = client.connect()  # local: fills pending orders; Alpaca: refuses anything but a paper account
    print(f"PAPER account {acct['account_number']}: equity ${float(acct['equity']):,.2f}, "
          f"cash ${float(acct['cash']):,.2f}  (simulated money, prices to {store.dates[-1].date()})")
    if args.action == "status":
        for p in client.positions():
            print(f"  {p['symbol']:6s} {float(p['qty']):>10.4f} sh  ${float(p['market_value']):>12,.2f}  "
                  f"P&L ${float(p['unrealized_pl']):>10,.2f}")
        pending = client.open_orders()
        if pending:
            print(f"{len(pending)} orders waiting to fill at the next open")
        return
    ledger = CallLedger()
    res = PaperTrader(client, store, _strategy(args, store, ledger)).run_cycle(submit=args.submit)
    print(f"as of {res.as_of} close: risk verdict {res.decision.verdict.value}, "
          f"{len(res.decision.breaches)} limit adjustments")
    for o in res.orders:
        size = f"${o.notional:,.2f}" if o.notional is not None else f"{o.qty} sh"
        print(f"  {o.side.upper():4s} {o.symbol:6s} {size}")
    if res.skipped_reason:
        print(f"not sent: {res.skipped_reason}")
    elif args.submit:
        print(f"placed {len(res.submitted)} paper orders; they fill at the next trading day's open")
    else:
        print("dry run: nothing placed. Add --submit to place these paper orders.")
    print("logged to runs/paper/decisions.jsonl")


def cmd_walkforward(args) -> None:
    from .backtest.walkforward import momentum_grid, strategy_grid, walk_forward

    store = _load(args.data, args.seed)
    grid = momentum_grid() if args.grid == "momentum" else strategy_grid()
    print(f"backtesting {len(grid)} variants, then picking one per window using only past data...")
    wf = walk_forward(store, grid, train_days=args.train, test_days=args.test)
    print(wf.choices.round(2).to_string(index=False))
    s = wf.summary()
    print(f"\nout-of-sample Sharpe {s['oos_sharpe']:.2f}  (hindsight pick '{wf.in_sample_best}' had "
          f"{s['hindsight_best_sharpe']:.2f})")
    print(f"out-of-sample max drawdown {s['oos_max_drawdown']:.1%}, deflated Sharpe {s['oos_deflated_sharpe']:.1%} "
          f"across {len(grid)} variants")


def cmd_factors(args) -> None:
    from .backtest.attribution import download_factors

    df = download_factors(args.out)
    print(f"wrote {args.out}: {len(df)} days of Fama-French 5 factors + momentum, "
          f"{df.index[0].date()} to {df.index[-1].date()}")


def cmd_attribution(args) -> None:
    from .backtest.attribution import attribute, load_factors

    if not Path(args.factors).exists():
        sys.exit(f"no factor file at {args.factors}. Run: python -m agentfund factors")
    store = _load(args.data, args.seed)
    factors = load_factors(args.factors)
    rows = {}
    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    for name in names:
        res = Backtester(store, STRATEGIES[name]()).run(args.start, args.end)
        a = attribute(res.returns.iloc[1:], factors)
        rows[name] = {"alpha/yr": a["alpha_annual"], "alpha_t": a["alpha_t"], **a["betas"], "R2": a["r2"]}
    print(f"daily regression {a['start']} to {a['end']} ({a['days']} days), Newey-West t-stats")
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nalpha = return the factors don't explain. |alpha_t| > 2 is the usual bar for 'probably real'.")


def cmd_daily(args) -> None:
    from .paper import PaperTrader
    from .paper.daily import format_report, load_config, market_is_open, run_daily, save_config

    print(f"--- {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")
    if args.strategy:
        cfg = load_config()
        if cfg.get("strategy") != args.strategy:
            save_config(strategy=args.strategy)
            state = Path("runs/paper/state.json")
            if state.exists():  # rebalance into the new strategy at the next run
                st = json.loads(state.read_text())
                st.pop("last_rebalance_month", None)
                state.write_text(json.dumps(st, indent=2))
            print(f"strategy set to {args.strategy}; it takes over at this run's rebalance")
    strategy = load_config()["strategy"]

    if market_is_open():
        print("The market is open, so today's prices aren't final. Nothing to do until after 4:30pm New York time.")
        return
    if not args.no_fetch:
        from .data.yahoo import yahoo_store

        try:
            store = yahoo_store()
            save_duckdb(store, args.prices)
        except Exception as e:  # keep going on yesterday's prices rather than crash a scheduled run
            print(f"price download failed ({type(e).__name__}: {e}); using the last saved prices")
    if not Path(args.prices).exists():
        sys.exit(f"no price file at {args.prices}; price download failed and there is no earlier copy")
    store = load_duckdb(args.prices)
    args.strategy = strategy
    args.use_claude, args.cache, args.coverage = False, None, 15
    client = _paper_client(args, store)
    trader = PaperTrader(client, store, _strategy(args, store, CallLedger()))
    print(format_report(run_daily(trader, submit=not args.dry_run), strategy))


def cmd_pause(args) -> None:
    from .paper.daily import PAUSE_PATH

    PAUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_PATH.write_text(datetime.now().isoformat())
    from .audit import DecisionLog

    DecisionLog("runs/paper/decisions.jsonl").append("paper_pause", {"paused": True})
    print("Paused: daily runs still update prices and fill waiting orders, but place no new orders.")


def cmd_resume(args) -> None:
    from .paper.daily import PAUSE_PATH

    PAUSE_PATH.unlink(missing_ok=True)
    from .audit import DecisionLog

    DecisionLog("runs/paper/decisions.jsonl").append("paper_pause", {"paused": False})
    print("Resumed: the next daily run trades normally.")


def cmd_schedule(args) -> None:
    from .paper import schedule

    project = Path.cwd()
    if args.action == "install":
        hour = minute = None
        if args.time:
            hour, minute = (int(x) for x in args.time.split(":"))
        try:
            print(schedule.install(project, hour, minute))
        except RuntimeError as e:
            sys.exit(str(e))
    elif args.action == "remove":
        print(schedule.remove())
    else:
        print(schedule.status(project))


def cmd_dashboard(args) -> None:
    from .dashboard import serve

    serve(port=args.port, prices=args.prices)


def cmd_verify(args) -> None:
    ok, msg = verify(args.path)
    print(("OK: " if ok else "TAMPERED: ") + msg)
    sys.exit(0 if ok else 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agentfund", description=__doc__)
    p.add_argument("--data", help="DuckDB file or CSV folder (default: synthetic market)")
    p.add_argument("--seed", type=int, default=7, help="seed for the synthetic market")
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_opts(sp):
        sp.add_argument("--use-claude", action="store_true", help="call Claude (needs ANTHROPIC_API_KEY); default is the offline stub")
        sp.add_argument("--cache", default="runs/note_cache", help="research note cache folder")
        sp.add_argument("--coverage", type=int, default=15, help="names the analyst covers each month")

    s = sub.add_parser("synth", help="write a synthetic market to DuckDB")
    s.add_argument("--out", default="data/market.duckdb")
    s.add_argument("--tickers", type=int, default=60)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("backtest", help="run one strategy and save a run folder")
    s.add_argument("--strategy", default="momentum_12_1", choices=[*STRATEGIES, "analyst_named", "analyst_anon"])
    s.add_argument("--start"), s.add_argument("--end")
    s.add_argument("--trials", type=int, default=1, help="variants tried so far (for the deflated Sharpe)")
    s.add_argument("--out", default="runs")
    llm_opts(s)
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("compare", help="baselines side by side")
    s.add_argument("--start"), s.add_argument("--end")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("research", help="one analyst research note")
    s.add_argument("ticker")
    s.add_argument("--as-of", required=True)
    s.add_argument("--anonymize", action="store_true")
    llm_opts(s)
    s.set_defaults(func=cmd_research)

    s = sub.add_parser("leakage", help="future-perturbation leakage test for a strategy")
    s.add_argument("--strategy", default="momentum_12_1", choices=list(STRATEGIES))
    s.add_argument("--cutoff", default="2019-06-28")
    s.set_defaults(func=cmd_leakage)

    s = sub.add_parser("contamination", help="named vs anonymized analyst backtest")
    s.add_argument("--start"), s.add_argument("--end")
    llm_opts(s)
    s.set_defaults(func=cmd_contamination)

    s = sub.add_parser("fetch", help="download real daily prices (Yahoo Finance by default)")
    s.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    s.add_argument("--start", default="2015-01-01")
    s.add_argument("--out", default=PRICES_DB)
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("paper", help="paper trading with simulated money (local account by default)")
    s.add_argument("action", choices=["status", "rebalance"])
    s.add_argument("--broker", default="local", choices=["local", "alpaca"],
                   help="local: play-money account in a file on this computer; alpaca: Alpaca paper account")
    s.add_argument("--prices", default=PRICES_DB, help="price file written by fetch")
    s.add_argument("--strategy", default="inverse_vol", choices=[*STRATEGIES, "analyst_named", "analyst_anon"])
    s.add_argument("--submit", action="store_true", help="actually send orders to the paper account")
    llm_opts(s)
    s.set_defaults(func=cmd_paper)

    s = sub.add_parser("walkforward", help="walk-forward test: pick variants on past data, judge on the next period")
    s.add_argument("--grid", default="strategies", choices=["strategies", "momentum"])
    s.add_argument("--train", type=int, default=756, help="training window in trading days (756 = 3 years)")
    s.add_argument("--test", type=int, default=126, help="test window in trading days (126 = 6 months)")
    s.set_defaults(func=cmd_walkforward)

    s = sub.add_parser("factors", help="download Fama-French factors from Ken French's data library")
    s.add_argument("--out", default="data/ff_factors.csv")
    s.set_defaults(func=cmd_factors)

    s = sub.add_parser("attribution", help="Fama-French attribution of strategy returns")
    s.add_argument("--strategy", default="all", choices=["all", *STRATEGIES])
    s.add_argument("--factors", default="data/ff_factors.csv")
    s.add_argument("--start"), s.add_argument("--end")
    s.set_defaults(func=cmd_attribution)

    s = sub.add_parser("daily", help="the daily routine: prices, fills, monthly rebalance, report (run after the close)")
    s.add_argument("--strategy", choices=list(STRATEGIES), help="switch strategy (remembered for later runs)")
    s.add_argument("--dry-run", action="store_true", help="show what it would do without placing orders")
    s.add_argument("--no-fetch", action="store_true", help="use the saved prices instead of downloading")
    s.add_argument("--broker", default="local", choices=["local", "alpaca"])
    s.add_argument("--prices", default=PRICES_DB)
    s.set_defaults(func=cmd_daily)

    s = sub.add_parser("schedule", help="run 'daily' automatically every weekday (macOS)")
    s.add_argument("action", choices=["install", "remove", "status"])
    s.add_argument("--time", help="local time HH:MM (default: 5:00pm New York time)")
    s.set_defaults(func=cmd_schedule)

    sub.add_parser("pause", help="stop placing new paper orders").set_defaults(func=cmd_pause)
    sub.add_parser("resume", help="start placing paper orders again").set_defaults(func=cmd_resume)

    s = sub.add_parser("dashboard", help="local web dashboard at http://localhost:8050 (read-only)")
    s.add_argument("--port", type=int, default=8050)
    s.add_argument("--prices", default=PRICES_DB)
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("verify-log", help="verify a hash-chained decision log")
    s.add_argument("path")
    s.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

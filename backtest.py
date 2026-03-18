"""
TradingAgents Backtesting Script

Runs the multi-agent trading system across historical dates, tracks decisions
against actual price movements, calculates performance metrics, and lets the
agents learn from their results via reflect_and_remember.

Usage:
    python backtest.py --ticker NVDA --start 2024-01-15 --end 2024-06-15
    python backtest.py --ticker AAPL --start 2024-03-01 --end 2024-09-01 --interval weekly
    python backtest.py --ticker TSLA --start 2024-01-15 --end 2024-06-15 --position-size 10000
    python backtest.py --tickers NVDA AAPL TSLA --start 2024-01-15 --end 2024-06-15
    python backtest.py --ticker NVDA --start 2024-01-15 --end 2024-06-15 --resume
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()


def get_trading_dates(ticker: str, start: str, end: str, interval: str = "biweekly") -> list[str]:
    """Get actual trading dates from market data within the given range."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        print(f"Error: No market data found for {ticker} between {start} and {end}")
        sys.exit(1)

    all_dates = [d.strftime("%Y-%m-%d") for d in data.index]

    step = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}.get(interval, 10)
    return all_dates[::step]


def get_price_on_date(price_data: pd.DataFrame, date_str: str, direction: str = "forward") -> Optional[float]:
    """Get the closing price on or near a given date."""
    target = pd.Timestamp(date_str)

    if target in price_data.index:
        close = price_data.loc[target, "Close"]
        return float(close.iloc[0]) if hasattr(close, "iloc") else float(close)

    if direction == "forward":
        future = price_data[price_data.index > target]
        if not future.empty:
            close = future.iloc[0]["Close"]
            return float(close.iloc[0]) if hasattr(close, "iloc") else float(close)
    else:
        past = price_data[price_data.index < target]
        if not past.empty:
            close = past.iloc[-1]["Close"]
            return float(close.iloc[0]) if hasattr(close, "iloc") else float(close)

    return None


def calculate_buy_and_hold(price_data: pd.DataFrame, start_date: str, end_date: str,
                           position_size: float) -> dict:
    """Calculate buy-and-hold benchmark returns for the same period."""
    start_price = get_price_on_date(price_data, start_date, "forward")
    end_price = get_price_on_date(price_data, end_date, "backward")

    if not start_price or not end_price:
        return {"strategy": "buy_and_hold", "pnl": 0, "return_pct": 0}

    shares = position_size / start_price
    pnl = shares * (end_price - start_price)
    return_pct = ((end_price - start_price) / start_price) * 100

    return {
        "strategy": "buy_and_hold",
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 2),
    }


def calculate_metrics(trades: list[dict]) -> dict:
    """Calculate performance metrics from trade history."""
    if not trades:
        return {}

    executed = [t for t in trades if t["decision"] in ("BUY", "SELL")]
    if not executed:
        return {
            "total_decisions": len(trades),
            "decisions": {d: sum(1 for t in trades if t["decision"] == d)
                         for d in ("BUY", "SELL", "HOLD")},
            "executed_trades": 0,
        }

    pnls = [t["pnl"] for t in executed]
    returns = [t["return_pct"] for t in executed]
    wins = [p for p in pnls if p > 0]

    total_return = sum(returns)
    avg_return = total_return / len(returns) if returns else 0

    # Sharpe-like ratio (simplified — annualized assuming biweekly trades)
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = var_r ** 0.5
        sharpe = (mean_r / std_r) * (26 ** 0.5) if std_r > 0 else 0
    else:
        sharpe = 0

    # Max drawdown from cumulative PnL
    cumulative = []
    running = 0
    for p in pnls:
        running += p
        cumulative.append(running)

    peak = cumulative[0]
    max_dd = 0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Consecutive wins/losses
    max_consec_wins = max_consec_losses = curr_wins = curr_losses = 0
    for p in pnls:
        if p > 0:
            curr_wins += 1
            curr_losses = 0
        else:
            curr_losses += 1
            curr_wins = 0
        max_consec_wins = max(max_consec_wins, curr_wins)
        max_consec_losses = max(max_consec_losses, curr_losses)

    return {
        "total_decisions": len(trades),
        "decisions": {d: sum(1 for t in trades if t["decision"] == d)
                      for d in ("BUY", "SELL", "HOLD")},
        "executed_trades": len(executed),
        "win_rate": round(len(wins) / len(executed) * 100, 1),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl_per_trade": round(sum(pnls) / len(pnls), 2),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "total_return_pct": round(total_return, 2),
        "avg_return_pct": round(avg_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(profit_factor, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "cumulative_pnl": [round(c, 2) for c in cumulative],
    }


def print_report(ticker: str, trades: list[dict], metrics: dict,
                 benchmark: Optional[dict] = None):
    """Print a formatted backtest report."""
    print("\n" + "=" * 80)
    print(f"  BACKTEST REPORT: {ticker}")
    print("=" * 80)

    if not trades:
        print("  No trades were executed.")
        return

    # Trade log
    print(f"\n  {'Date':<12} {'Decision':<8} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Return':>8} {'Cum P&L':>12}")
    print("  " + "-" * 76)
    cum_pnl = 0
    for t in trades:
        entry = f"${t['entry_price']:.2f}" if t["entry_price"] else "   N/A"
        exit_p = f"${t['exit_price']:.2f}" if t["exit_price"] else "   N/A"
        if t["decision"] in ("BUY", "SELL"):
            cum_pnl += t["pnl"]
            pnl = f"${t['pnl']:+,.2f}"
            ret = f"{t['return_pct']:+.1f}%"
            cum = f"${cum_pnl:+,.2f}"
        else:
            pnl = "     --"
            ret = "   --"
            cum = f"${cum_pnl:+,.2f}"
        print(f"  {t['date']:<12} {t['decision']:<8} {entry:>10} {exit_p:>10} {pnl:>12} {ret:>8} {cum:>12}")

    # Summary
    print("\n  " + "-" * 76)
    print("  PERFORMANCE SUMMARY")
    print("  " + "-" * 76)
    decisions = metrics.get("decisions", {})
    print(f"  Total Decisions:       {metrics.get('total_decisions', 0)}")
    print(f"    BUY: {decisions.get('BUY', 0)}  |  SELL: {decisions.get('SELL', 0)}  |  HOLD: {decisions.get('HOLD', 0)}")

    if metrics.get("executed_trades", 0) > 0:
        print(f"  Win Rate:              {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor:         {metrics['profit_factor']}")
        print(f"  Total P&L:             ${metrics['total_pnl']:+,.2f}")
        print(f"  Avg P&L/Trade:         ${metrics['avg_pnl_per_trade']:+,.2f}")
        print(f"  Best Trade:            ${metrics['best_trade']:+,.2f}")
        print(f"  Worst Trade:           ${metrics['worst_trade']:+,.2f}")
        print(f"  Total Return:          {metrics['total_return_pct']:+.2f}%")
        print(f"  Avg Return/Trade:      {metrics['avg_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']}")
        print(f"  Max Drawdown:          ${metrics['max_drawdown']:+,.2f}")
        print(f"  Max Consec Wins:       {metrics['max_consecutive_wins']}")
        print(f"  Max Consec Losses:     {metrics['max_consecutive_losses']}")

    # Buy-and-hold comparison
    if benchmark:
        print(f"\n  " + "-" * 76)
        print("  VS BUY-AND-HOLD BENCHMARK")
        print("  " + "-" * 76)
        print(f"  Buy & Hold Return:     {benchmark['return_pct']:+.2f}%  (${benchmark['pnl']:+,.2f})")
        if metrics.get("executed_trades", 0) > 0:
            alpha = metrics["total_return_pct"] - benchmark["return_pct"]
            print(f"  Agent Alpha:           {alpha:+.2f}%")
            verdict = "OUTPERFORMED" if alpha > 0 else ("UNDERPERFORMED" if alpha < 0 else "MATCHED")
            print(f"  Verdict:               Agent {verdict} buy-and-hold")

    # Equity curve (text sparkline)
    cum_pnl_list = metrics.get("cumulative_pnl", [])
    if len(cum_pnl_list) > 1:
        print(f"\n  " + "-" * 76)
        print("  EQUITY CURVE (cumulative P&L)")
        print("  " + "-" * 76)
        _print_ascii_chart(cum_pnl_list)

    print("=" * 80 + "\n")


def _print_ascii_chart(values: list[float], width: int = 60, height: int = 12):
    """Print a simple ASCII chart of values."""
    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0:
        print("  (flat)")
        return

    for row in range(height, -1, -1):
        threshold = min_val + (val_range * row / height)
        label = f"  ${threshold:>+10,.0f} |"
        line = ""
        for v in values:
            if v >= threshold:
                line += "#"
            else:
                line += " "
        if row == height or row == 0 or row == height // 2:
            print(f"{label}{line}")
        else:
            print(f"              |{line}")

    print(f"              +{''.join(['-' for _ in values])}")


def load_checkpoint(results_file: Path) -> list[dict]:
    """Load previously saved trades for resuming an interrupted backtest."""
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            return data.get("trades", [])
    return []


def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "biweekly",
    position_size: float = 10000,
    hold_days: int = 5,
    llm_provider: str = "anthropic",
    deep_model: str = "claude-sonnet-4-5",
    quick_model: str = "claude-haiku-4-5",
    debate_rounds: int = 1,
    analysts: Optional[list[str]] = None,
    learn: bool = True,
    resume: bool = False,
):
    """Run a full backtest.

    Args:
        ticker: Stock ticker symbol
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        interval: Decision frequency — 'weekly', 'biweekly', 'monthly'
        position_size: Dollar amount per trade
        hold_days: Trading days to hold before measuring outcome
        llm_provider: LLM provider name
        deep_model: Model for deep reasoning
        quick_model: Model for quick tasks
        debate_rounds: Number of bull/bear debate rounds
        analysts: Which analysts to use (default: all four)
        learn: Whether agents should learn from results via reflect_and_remember
        resume: If True, skip dates already processed in a previous run
    """
    if analysts is None:
        analysts = ["market", "social", "news", "fundamentals"]

    print(f"\nStarting backtest: {ticker} from {start_date} to {end_date}")
    print(f"  Interval: {interval} | Position: ${position_size:,.0f} | Hold: {hold_days} days")
    print(f"  LLM: {llm_provider}/{deep_model} | Debate rounds: {debate_rounds}")
    print(f"  Analysts: {', '.join(analysts)} | Learning: {'ON' if learn else 'OFF'}")
    if resume:
        print("  Resume mode: ON")
    print()

    # Get decision dates
    decision_dates = get_trading_dates(ticker, start_date, end_date, interval)
    print(f"  {len(decision_dates)} decision points identified\n")

    if not decision_dates:
        print("No trading dates found. Check your date range.")
        return

    # Fetch full price history (with buffer for exit prices)
    buffer_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=45)).strftime("%Y-%m-%d")
    price_data = yf.download(ticker, start=start_date, end=buffer_end, progress=False)

    if price_data.empty:
        print(f"Error: Could not fetch price data for {ticker}")
        return

    # Calculate buy-and-hold benchmark
    benchmark = calculate_buy_and_hold(price_data, start_date, end_date, position_size)
    print(f"  Buy & Hold benchmark: {benchmark['return_pct']:+.2f}% (${benchmark['pnl']:+,.2f})\n")

    # Resume support — load previous trades
    results_dir = Path(f"backtest_results/{ticker}")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"backtest_{start_date}_to_{end_date}.json"

    trades = []
    completed_dates = set()
    if resume:
        trades = load_checkpoint(results_file)
        completed_dates = {t["date"] for t in trades}
        if completed_dates:
            print(f"  Resuming: {len(completed_dates)} dates already completed, skipping...\n")

    # Set up the trading agents
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = llm_provider
    config["deep_think_llm"] = deep_model
    config["quick_think_llm"] = quick_model
    config["max_debate_rounds"] = debate_rounds

    ta = TradingAgentsGraph(
        selected_analysts=analysts,
        debug=False,
        config=config,
    )

    start_time = time.time()

    for i, date in enumerate(decision_dates):
        if date in completed_dates:
            continue

        elapsed = time.time() - start_time
        remaining_count = len(decision_dates) - i
        if i > len(completed_dates) and elapsed > 0:
            done_count = i - len(completed_dates)
            avg_time = elapsed / done_count
            eta_min = (remaining_count * avg_time) / 60
            eta_str = f" | ETA: {eta_min:.0f}min"
        else:
            eta_str = ""

        print(f"  [{i+1}/{len(decision_dates)}] Analyzing {ticker} on {date}...{eta_str}", end=" ", flush=True)

        try:
            _, decision = ta.propagate(ticker, date)
        except Exception as e:
            print(f"ERROR: {e}")
            trades.append({
                "date": date,
                "decision": "ERROR",
                "entry_price": None,
                "exit_price": None,
                "pnl": 0,
                "return_pct": 0,
                "error": str(e),
            })
            # Save checkpoint after each trade for resume support
            _save_results(results_file, ticker, trades, {}, benchmark)
            continue

        decision = decision.strip().upper()
        if decision not in ("BUY", "SELL", "HOLD"):
            decision = "HOLD"

        entry_price = get_price_on_date(price_data, date, "forward")

        # Get exit price (hold_days trading days later)
        entry_idx = None
        for idx, d in enumerate(price_data.index):
            if d >= pd.Timestamp(date):
                entry_idx = idx
                break

        exit_price = None
        if entry_idx is not None:
            exit_idx = min(entry_idx + hold_days, len(price_data) - 1)
            close_val = price_data.iloc[exit_idx]["Close"]
            exit_price = float(close_val.iloc[0]) if hasattr(close_val, "iloc") else float(close_val)

        # Calculate P&L
        pnl = 0.0
        return_pct = 0.0
        if entry_price and exit_price and decision in ("BUY", "SELL"):
            shares = position_size / entry_price
            if decision == "BUY":
                pnl = shares * (exit_price - entry_price)
                return_pct = ((exit_price - entry_price) / entry_price) * 100
            elif decision == "SELL":
                pnl = shares * (entry_price - exit_price)
                return_pct = ((entry_price - exit_price) / entry_price) * 100

        trade = {
            "date": date,
            "decision": decision,
            "entry_price": round(entry_price, 2) if entry_price else None,
            "exit_price": round(exit_price, 2) if exit_price else None,
            "pnl": round(pnl, 2),
            "return_pct": round(return_pct, 2),
        }
        trades.append(trade)

        indicator = "+" if pnl > 0 else ("-" if pnl < 0 else "~")
        print(f"{decision} | P&L: ${pnl:+.2f} ({return_pct:+.1f}%) {indicator}")

        # Let agents learn from this trade
        if learn and decision != "HOLD":
            try:
                ta.reflect_and_remember(pnl)
            except Exception as e:
                print(f"    (reflection failed: {e})")

        # Save checkpoint after each trade
        _save_results(results_file, ticker, trades, {}, benchmark)

    # Final metrics and report
    elapsed_total = time.time() - start_time
    metrics = calculate_metrics(trades)
    metrics["elapsed_seconds"] = round(elapsed_total, 1)

    print_report(ticker, trades, metrics, benchmark)

    _save_results(results_file, ticker, trades, metrics, benchmark)
    print(f"  Results saved to {results_file}")
    print(f"  Total time: {elapsed_total / 60:.1f} minutes\n")

    return trades, metrics


def _save_results(results_file: Path, ticker: str, trades: list[dict],
                  metrics: dict, benchmark: dict):
    """Save results to JSON file."""
    with open(results_file, "w") as f:
        json.dump({
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "trades": trades,
            "metrics": metrics,
            "benchmark": benchmark,
        }, f, indent=2)


def run_multi_ticker(tickers: list[str], **kwargs):
    """Run backtests across multiple tickers and produce a comparison."""
    all_results = {}

    for ticker in tickers:
        print(f"\n{'#' * 80}")
        print(f"  TICKER: {ticker}")
        print(f"{'#' * 80}")

        result = run_backtest(ticker=ticker, **kwargs)
        if result:
            trades, metrics = result
            all_results[ticker] = {"trades": trades, "metrics": metrics}

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("  MULTI-TICKER COMPARISON")
        print("=" * 80)
        print(f"\n  {'Ticker':<8} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'Return':>9} {'Sharpe':>8}")
        print("  " + "-" * 55)
        for t, r in all_results.items():
            m = r["metrics"]
            print(f"  {t:<8} {m.get('executed_trades', 0):>7} "
                  f"{m.get('win_rate', 0):>6.1f}% "
                  f"${m.get('total_pnl', 0):>+10,.2f} "
                  f"{m.get('total_return_pct', 0):>+8.2f}% "
                  f"{m.get('sharpe_ratio', 0):>8.2f}")
        print("=" * 80 + "\n")

    # Save combined results
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)
    combined_file = results_dir / f"comparison_{'_'.join(tickers)}.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Combined results saved to {combined_file}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --ticker NVDA --start 2024-01-15 --end 2024-06-15
  python backtest.py --tickers NVDA AAPL TSLA --start 2024-01-15 --end 2024-06-15
  python backtest.py --ticker NVDA --start 2024-01-15 --end 2024-06-15 --resume
  python backtest.py --ticker NVDA --start 2024-01-15 --end 2024-06-15 --interval weekly --hold-days 10
        """
    )
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol")
    parser.add_argument("--tickers", type=str, nargs="+", default=None,
                        help="Multiple stock tickers for comparison backtests")
    parser.add_argument("--start", type=str, default="2024-01-15", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-06-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="biweekly",
                        choices=["daily", "weekly", "biweekly", "monthly"],
                        help="Decision frequency")
    parser.add_argument("--position-size", type=float, default=10000,
                        help="Dollar amount per trade (default: $10,000)")
    parser.add_argument("--hold-days", type=int, default=5,
                        help="Trading days to hold before measuring outcome (default: 5)")
    parser.add_argument("--llm-provider", type=str, default="anthropic",
                        help="LLM provider (anthropic, openai, google, etc.)")
    parser.add_argument("--deep-model", type=str, default="claude-sonnet-4-5",
                        help="Deep thinking model (default: claude-sonnet-4-5)")
    parser.add_argument("--quick-model", type=str, default="claude-haiku-4-5",
                        help="Quick thinking model (default: claude-haiku-4-5)")
    parser.add_argument("--debate-rounds", type=int, default=1,
                        help="Number of bull/bear debate rounds (1, 3, or 5)")
    parser.add_argument("--analysts", type=str, nargs="+",
                        default=["market", "social", "news", "fundamentals"],
                        help="Which analysts to use")
    parser.add_argument("--no-learn", action="store_true",
                        help="Disable agent learning from results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous interrupted backtest")
    args = parser.parse_args()

    common_kwargs = dict(
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        position_size=args.position_size,
        hold_days=args.hold_days,
        llm_provider=args.llm_provider,
        deep_model=args.deep_model,
        quick_model=args.quick_model,
        debate_rounds=args.debate_rounds,
        analysts=args.analysts,
        learn=not args.no_learn,
        resume=args.resume,
    )

    if args.tickers:
        run_multi_ticker(tickers=args.tickers, **common_kwargs)
    elif args.ticker:
        run_backtest(ticker=args.ticker, **common_kwargs)
    else:
        # Default to NVDA if nothing specified
        run_backtest(ticker="NVDA", **common_kwargs)


if __name__ == "__main__":
    main()

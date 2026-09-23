"""Run market scans independently of the Streamlit browser session.

Examples:
  python run_market_scheduler.py --once
  python run_market_scheduler.py --time 16:00
"""
import argparse
import time
from datetime import datetime

from analysis import indicators
from analysis.market_scanner import scan_market
from data.market_data import provider
from data.scan_store import save_scan
from signals.engine import technical_signal


def run_once():
    results = scan_market(provider, indicators, technical_signal)
    payload = save_scan(results)
    counts = {name: group["count"] for name, group in payload["groups"].items()}
    print(f"Scan completed at {payload['scanned_at']}: {counts}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run scheduled market scans.")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit.")
    parser.add_argument("--time", default="", help="Local 24-hour time, for example 16:00.")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between schedule checks.")
    args = parser.parse_args()
    if args.once:
        run_once()
        return
    if not args.time:
        parser.error("Provide --once or --time HH:MM.")
    datetime.strptime(args.time, "%H:%M")
    last_run_date = None
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == args.time and last_run_date != now.date():
            run_once()
            last_run_date = now.date()
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.telemetry.config import load_settings, tiled_schedule
from src.telemetry.validation import validate


def render_table(report) -> str:
    lines = ["STATUS  CHECK                         MESSAGE"]
    lines.append("-" * 90)
    for check in report.checks:
        message = (check.message or "").replace("\n", " | ")[:68]
        lines.append(f"{check.status:<7} {check.name:<29} {message}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw SentinelAI telemetry dataset")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--settings", type=Path, default=ROOT / "config" / "settings.yaml")
    parser.add_argument("--interval-sec", type=int, default=None)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "evaluations")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    interval_sec = args.interval_sec or int(settings.get("telemetry", {}).get("collection_interval_sec", 30))
    schedule = tiled_schedule(settings)

    report = validate(args.root, interval_sec=interval_sec, schedule=schedule)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.report_dir / f"validation_report_{stamp}.json"
    out_path.write_text(report.to_json(), encoding="utf-8")

    print(render_table(report))
    print("-" * 90)
    print(f"overall: {report.status()}   ({report.metric_rows} metric rows, {report.log_rows} log rows, "
          f"{report.incident_rows} incidents)")
    print(f"report -> {out_path}")
    print("\nmetric distributions (mean/std/min/p50/max):")
    for col, stats in report.distributions.items():
        print(f"  {col:<22} {stats['mean']:>9} {stats['std']:>9} {stats['min']:>9} "
              f"{stats['p50']:>9} {stats['max']:>9}")


if __name__ == "__main__":
    main()
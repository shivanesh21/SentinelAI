from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.risk import RiskEngine
from src.evidence import EvidenceAssembler, evidence_to_json
from src.telemetry.config import load_settings


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    engine = RiskEngine.load(ROOT / "models", settings)

    from src.features.store import FeatureStore
    store = FeatureStore(ROOT / "data" / "features")
    test = store.load_split("test", ROOT / "data" / "splits")

    assembler = EvidenceAssembler(
        engine,
        threshold=0.40,  # WARNING level
        feature_columns=engine.feature_columns,
        top_k_features=5,
    )

    evidence_list = assembler.assemble(test)

    print("=" * 80)
    print(f"EVIDENCE PACKAGES GENERATED: {len(evidence_list)} (risk >= WARNING)")
    print("=" * 80)

    # Show first 3 evidence packages
    for i, ev in enumerate(evidence_list[:3]):
        print(f"\n{'='*80}")
        print(f"EVIDENCE #{i+1}")
        print(f"{'='*80}")
        print(evidence_to_json(ev))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    by_level = {}
    for ev in evidence_list:
        by_level[ev.risk_level] = by_level.get(ev.risk_level, 0) + 1
    for level in ["NORMAL", "WARNING", "ANOMALOUS", "CRITICAL"]:
        if level in by_level:
            print(f"  {level}: {by_level[level]}")


if __name__ == "__main__":
    main()
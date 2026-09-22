from __future__ import annotations

from pathlib import Path

import yaml

from .simulator.failure_scenarios import ScenarioSpec
from .simulator.service import ServiceSimulator


def load_settings(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_services(path: str | Path) -> list[ServiceSimulator]:
    config = load_settings(path)
    return [
        ServiceSimulator(
            name=item["name"],
            replicas=item.get("replicas", 1),
            profile=item.get("profile", {}),
        )
        for item in config.get("services", [])
    ]


def tiled_schedule(config: dict) -> list[dict]:
    schedule = config.get("simulation", {}).get("scenario_schedule", [])
    repeats = int(config.get("simulation", {}).get("schedule_repeats", 1))
    period = float(config.get("simulation", {}).get("schedule_period_min", 0))
    if repeats <= 1 or period <= 0:
        return schedule
    tiles = []
    for r in range(repeats):
        for item in schedule:
            clone = dict(item)
            clone["start_min"] = float(item["start_min"]) + r * period
            tiles.append(clone)
    return tiles


def load_scenarios(path: str | Path) -> list[ScenarioSpec]:
    config = load_settings(path)
    return [
        ScenarioSpec(
            service=item["service"],
            scenario_type=item["type"],
            start_offset_min=float(item["start_min"]),
            duration_min=float(item["duration_min"]),
            intensity=item.get("intensity", 0.7),
        )
        for item in tiled_schedule(config)
    ]


def default_duration(path: str | Path) -> float:
    config = load_settings(path)
    simulation = config.get("simulation", {})
    repeats = int(simulation.get("schedule_repeats", 1))
    period = float(simulation.get("schedule_period_min", 0))
    duration = float(simulation.get("duration_min", 720))
    if repeats > 1 and period > 0:
        return max(duration, period * repeats)
    return duration
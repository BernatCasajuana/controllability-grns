#!/usr/bin/env python3
"""Attractor-specific GRN control using succession diagrams and stable motifs.

This script is intentionally limited to control tasks:
- build succession diagram and attractor seeds
- choose a target attractor
- compute BioBaLM succession-control interventions
- compute PyStableMotifs minimal/internal drivers
- summarize and export control candidates
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

import biobalm as balm
from biobalm import control as balm_control

# Suppress optional executable warnings printed by PyStableMotifs at import time.
with contextlib.redirect_stderr(io.StringIO()):
    import pystablemotifs as sm


RULES = {
    "rule": """COUP, !FGF8\nEMX2, !SP8\nFGF8, SP8\nPAX6, SP8\nSP8, FGF8""",
    "rule_2": "COUP, EMX2\nEMX2, !FGF8\nFGF8, SP8\nPAX6, !EMX2\nSP8, FGF8",
    "next_r": "COUP,   !SP8&!PAX6&COUP | !FGF8 | EMX2\nEMX2,   !SP8 | !FGF8 | EMX2\nFGF8,   PAX6 | !COUP\nPAX6,   !EMX2\nSP8,    !EMX2\n",
}


def load_rule_text(rule_name: str, rule_text_file: str | None) -> str:
    if rule_text_file:
        return Path(rule_text_file).read_text(encoding="utf-8")

    if rule_name not in RULES:
        available = ", ".join(sorted(RULES))
        raise ValueError(f"Unknown rule name '{rule_name}'. Available: {available}")

    return RULES[rule_name]


def attractor_seeds_dataframe(rule_text: str) -> tuple[balm.SuccessionDiagram, dict, pd.DataFrame]:
    """Build succession diagram and return expanded attractor seeds as a dataframe."""
    sd = balm.SuccessionDiagram.from_rules(rule_text)
    sd.build()

    seeds = sd.expanded_attractor_seeds()
    flat = {key: value[0] for key, value in seeds.items() if value}

    df = pd.DataFrame.from_dict(flat, orient="index").sort_index()
    df.index.name = "attractor_id"
    return sd, seeds, df


def normalize_balm_interventions(interventions: list) -> list[dict]:
    """Convert BioBaLM intervention paths into unique merged control dictionaries."""
    controls = []
    seen = set()

    for intervention in interventions:
        for control_path in intervention.control:
            merged = {}
            consistent = True
            for override in control_path:
                for node, value in override.items():
                    if node in merged and merged[node] != value:
                        consistent = False
                        break
                    merged[node] = value
                if not consistent:
                    break

            if consistent:
                merged = dict(sorted(merged.items()))
                key = tuple(merged.items())
                if key not in seen:
                    seen.add(key)
                    controls.append(merged)

    controls.sort(key=lambda item: (len(item), sorted(item.items())))
    return controls


def controls_to_table(method_name: str, controls: list[dict]) -> pd.DataFrame:
    if not controls:
        return pd.DataFrame(
            [{"method": method_name, "control": "(none found)", "size": None}]
        )

    return pd.DataFrame(
        [
            {
                "method": method_name,
                "control": ", ".join(f"{k}={v}" for k, v in control.items()),
                "size": len(control),
            }
            for control in controls
        ]
    )


def min_control_size(controls: list[dict]) -> int | None:
    return min((len(control) for control in controls), default=None)


def best_control(controls: list[dict]) -> dict:
    return min(controls, key=len) if controls else {}


def control_overlap_dataframe(best_controls: dict[str, dict]) -> pd.DataFrame:
    names = list(best_controls.keys())
    rows = []

    for a_name, b_name in combinations(names, 2):
        a_nodes = set(best_controls[a_name].keys())
        b_nodes = set(best_controls[b_name].keys())

        union = a_nodes | b_nodes
        overlap = a_nodes & b_nodes

        rows.append(
            {
                "pair": f"{a_name} vs {b_name}",
                "overlap_nodes": ", ".join(sorted(overlap)) if overlap else "",
                "overlap_size": len(overlap),
                "jaccard": (len(overlap) / len(union)) if union else None,
            }
        )

    return pd.DataFrame(rows)


def build_target_state(row: pd.Series) -> dict[str, int]:
    """Create a partial-state target dict while skipping missing values."""
    target = {}
    for node, value in row.items():
        if pd.isna(value):
            continue
        target[node] = int(value)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute attractor-specific stable motif control candidates.")
    parser.add_argument("--rule-name", default="next_r", choices=sorted(RULES.keys()))
    parser.add_argument("--rule-text-file", default=None, help="Optional path to a custom .bnet rule file.")
    parser.add_argument("--target-attractor-id", type=int, default=None)
    parser.add_argument("--strategy", default="internal", help="BioBaLM succession_control strategy.")
    parser.add_argument("--out-dir", default="outputs/control")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule_text = load_rule_text(args.rule_name, args.rule_text_file)
    sd, _, attractor_df = attractor_seeds_dataframe(rule_text)

    if attractor_df.empty:
        raise RuntimeError("No attractor seeds found in succession diagram.")

    if args.target_attractor_id is None:
        target_attractor_id = int(attractor_df.index[0])
    else:
        target_attractor_id = int(args.target_attractor_id)

    if target_attractor_id not in attractor_df.index:
        valid_ids = ", ".join(str(int(i)) for i in attractor_df.index)
        raise ValueError(f"target-attractor-id {target_attractor_id} not found. Valid IDs: {valid_ids}")

    target_state = build_target_state(attractor_df.loc[target_attractor_id])

    balm_interventions = balm_control.succession_control(sd, target_state, strategy=args.strategy)
    balm_controls = normalize_balm_interventions(balm_interventions)

    primes = sm.format.bnet_text2primes(rule_text)
    psm_minimal = [dict(sorted(c.items())) for c in sm.drivers.minimal_drivers(target_state, primes)]
    psm_internal = [dict(sorted(c.items())) for c in sm.drivers.internal_drivers(target_state, primes)]

    control_df = pd.concat(
        [
            controls_to_table("BioBaLM succession_control", balm_controls),
            controls_to_table("PyStableMotifs minimal_drivers", psm_minimal),
            controls_to_table("PyStableMotifs internal_drivers", psm_internal),
        ],
        ignore_index=True,
    )

    size_summary = pd.DataFrame(
        {
            "set": [
                "BioBaLM succession control (best)",
                "PyStableMotifs minimal_drivers (best)",
                "PyStableMotifs internal_drivers (best)",
            ],
            "size": [
                min_control_size(balm_controls),
                min_control_size(psm_minimal),
                min_control_size(psm_internal),
            ],
        }
    )

    best_controls = {
        "Best BioBaLM": best_control(balm_controls),
        "Best minimal_drivers": best_control(psm_minimal),
        "Best internal_drivers": best_control(psm_internal),
    }
    overlap_df = control_overlap_dataframe(best_controls)

    attractor_path = out_dir / "attractor_seeds.csv"
    candidates_path = out_dir / "control_candidates.csv"
    size_path = out_dir / "control_size_summary.csv"
    overlap_path = out_dir / "best_control_overlaps.csv"
    details_path = out_dir / "target_and_best_controls.json"

    attractor_df.to_csv(attractor_path)
    control_df.to_csv(candidates_path, index=False)
    size_summary.to_csv(size_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    details_payload = {
        "target_attractor_id": target_attractor_id,
        "target_state": target_state,
        "best_controls": best_controls,
    }
    details_path.write_text(json.dumps(details_payload, indent=2), encoding="utf-8")

    print(f"Using rule set: {args.rule_name if not args.rule_text_file else 'custom file'}")
    print(f"Number of attractor seeds: {len(attractor_df)}")
    print(f"Selected attractor ID: {target_attractor_id}")
    print(f"Target partial state: {target_state}")
    print(f"Best BioBaLM control: {best_controls['Best BioBaLM']}")
    print(f"Best PyStableMotifs minimal_drivers: {best_controls['Best minimal_drivers']}")
    print(f"Best PyStableMotifs internal_drivers: {best_controls['Best internal_drivers']}")

    print(f"Saved attractor seeds: {attractor_path}")
    print(f"Saved control candidates: {candidates_path}")
    print(f"Saved control size summary: {size_path}")
    print(f"Saved best-control overlaps: {overlap_path}")
    print(f"Saved target/control JSON: {details_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Structural analysis for inferred Boolean GRNs.

This script is intentionally limited to network structure analysis:
- directed graph construction from Boolean rules
- FVS computations (AEON + exact enumeration for small graphs)
- structural driver/sensor upper bounds
- feedback-loop (cycle) analysis
- network visualization
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from biodivine_aeon import BooleanNetwork


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


def build_bn_and_graph(rule_text: str) -> tuple[BooleanNetwork, nx.DiGraph]:
    """Parse bnet rules into an AEON network and a NetworkX DiGraph."""
    bn = BooleanNetwork.from_bnet(rule_text)
    graph = nx.DiGraph()

    for name in bn.variable_names():
        graph.add_node(name)

    for reg in bn.regulations():
        src = bn.get_variable_name(reg["source"])
        tgt = bn.get_variable_name(reg["target"])
        graph.add_edge(src, tgt, sign=reg.get("sign"), essential=reg.get("essential", None))

    return bn, graph


def minimum_fvs_sets_exact(graph: nx.DiGraph, max_nodes_exact: int = 18) -> list[set[str]]:
    """Return all minimum-cardinality FVS solutions for small/medium graphs."""
    n_nodes = graph.number_of_nodes()
    nodes = list(graph.nodes())

    if nx.is_directed_acyclic_graph(graph):
        return [set()]

    if n_nodes > max_nodes_exact:
        # Greedy fallback for larger graphs where exact enumeration is expensive.
        g_work = graph.copy()
        picked: set[str] = set()
        while not nx.is_directed_acyclic_graph(g_work):
            score = {
                node: g_work.in_degree(node) * g_work.out_degree(node)
                for node in g_work.nodes()
            }
            to_remove = max(score, key=score.get)
            picked.add(to_remove)
            g_work.remove_node(to_remove)
        return [picked]

    for k in range(1, n_nodes + 1):
        solutions: list[set[str]] = []
        for subset in combinations(nodes, k):
            reduced = graph.copy()
            reduced.remove_nodes_from(subset)
            if nx.is_directed_acyclic_graph(reduced):
                solutions.append(set(subset))
        if solutions:
            return solutions

    return [set(nodes)]


def structural_control_bounds(rule_text: str, max_nodes_exact: int = 18) -> dict:
    """Compute structural upper bounds from source/sink nodes and minimum FVSs."""
    bn, graph = build_bn_and_graph(rule_text)

    source_nodes = sorted([n for n in graph.nodes if graph.in_degree(n) == 0])
    sink_nodes = sorted([n for n in graph.nodes if graph.out_degree(n) == 0])

    fvs_forward_aeon = sorted(bn.get_variable_name(v) for v in bn.feedback_vertex_set())
    fvs_forward_all = [sorted(s) for s in minimum_fvs_sets_exact(graph, max_nodes_exact=max_nodes_exact)]
    fvs_reverse_all = [
        sorted(s)
        for s in minimum_fvs_sets_exact(graph.reverse(copy=True), max_nodes_exact=max_nodes_exact)
    ]

    fvs_forward_choice = fvs_forward_all[0] if fvs_forward_all else []
    fvs_reverse_choice = fvs_reverse_all[0] if fvs_reverse_all else []

    driver_upper_bound = sorted(set(source_nodes) | set(fvs_forward_choice))
    sensor_upper_bound = sorted(set(sink_nodes) | set(fvs_reverse_choice))

    metrics = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_sources": len(source_nodes),
        "n_sinks": len(sink_nodes),
        "min_fvs_size_forward": len(fvs_forward_choice),
        "min_fvs_size_reverse": len(fvs_reverse_choice),
        "driver_upper_bound_size": len(driver_upper_bound),
        "sensor_upper_bound_size": len(sensor_upper_bound),
    }

    return {
        "bn": bn,
        "graph": graph,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "fvs_forward_aeon": fvs_forward_aeon,
        "fvs_forward_all": fvs_forward_all,
        "fvs_reverse_all": fvs_reverse_all,
        "driver_upper_bound": driver_upper_bound,
        "sensor_upper_bound": sensor_upper_bound,
        "metrics": metrics,
    }


def feedback_loops_dataframe(graph: nx.DiGraph, max_cycles: int = 2000) -> tuple[pd.DataFrame, bool]:
    """Enumerate directed cycles and return a tabular summary."""
    rows = []
    truncated = False

    for i, cycle in enumerate(nx.simple_cycles(graph), start=1):
        if i > max_cycles:
            truncated = True
            break

        cycle_edges = []
        for idx, src in enumerate(cycle):
            tgt = cycle[(idx + 1) % len(cycle)]
            cycle_edges.append(f"{src}->{tgt}")

        rows.append(
            {
                "cycle_index": i,
                "cycle_length": len(cycle),
                "cycle_nodes": " -> ".join(cycle + [cycle[0]]),
                "cycle_edges": ", ".join(cycle_edges),
            }
        )

    if not rows:
        df = pd.DataFrame(columns=["cycle_index", "cycle_length", "cycle_nodes", "cycle_edges"])
        return df, truncated

    df = pd.DataFrame(rows).sort_values(["cycle_length", "cycle_index"]).reset_index(drop=True)
    return df, truncated


def build_fvs_dataframe(struct: dict) -> pd.DataFrame:
    rows = [
        {
            "solution_type": "forward_min_fvs",
            "nodes": ", ".join(nodes),
            "size": len(nodes),
        }
        for nodes in struct["fvs_forward_all"]
    ]
    rows.extend(
        {
            "solution_type": "reverse_min_fvs",
            "nodes": ", ".join(nodes),
            "size": len(nodes),
        }
        for nodes in struct["fvs_reverse_all"]
    )
    return pd.DataFrame(rows)


def plot_structural_overview(graph: nx.DiGraph, struct: dict, output_path: Path) -> None:
    node_role = {}
    for node in graph.nodes:
        in_driver = node in struct["driver_upper_bound"]
        in_sensor = node in struct["sensor_upper_bound"]
        in_fvs = any(node in s for s in struct["fvs_forward_all"])

        if in_driver and in_sensor:
            node_role[node] = "driver+sensor"
        elif in_driver:
            node_role[node] = "driver-bound"
        elif in_sensor:
            node_role[node] = "sensor-bound"
        elif in_fvs:
            node_role[node] = "fvs-only"
        else:
            node_role[node] = "other"

    role_color = {
        "driver+sensor": "#ff7f0e",
        "driver-bound": "#1f77b4",
        "sensor-bound": "#2ca02c",
        "fvs-only": "#9467bd",
        "other": "#bdbdbd",
    }

    pos = nx.spring_layout(graph, seed=7)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=axes[0],
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        edge_color="#4d4d4d",
    )
    for role, color in role_color.items():
        nodes = [n for n, r in node_role.items() if r == role]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                ax=axes[0],
                node_color=color,
                node_size=1300,
                edgecolors="white",
                linewidths=1.5,
                label=role,
            )

    nx.draw_networkx_labels(graph, pos, ax=axes[0], font_size=10, font_weight="bold")
    axes[0].set_title("GRN with structural roles")
    axes[0].axis("off")
    axes[0].legend(loc="lower left", frameon=False)

    bound_sizes = pd.Series(
        {
            "|Driver upper bound|": len(struct["driver_upper_bound"]),
            "|Sensor upper bound|": len(struct["sensor_upper_bound"]),
            "|Forward min FVS|": len(struct["fvs_forward_all"][0]) if struct["fvs_forward_all"] else 0,
            "|Reverse min FVS|": len(struct["fvs_reverse_all"][0]) if struct["fvs_reverse_all"] else 0,
        }
    )

    axes[1].bar(
        bound_sizes.index,
        bound_sizes.values,
        color=["#1f77b4", "#2ca02c", "#d62728", "#17becf"],
    )
    axes[1].set_ylabel("Number of nodes")
    axes[1].set_title("Structural control/observation bounds")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute structural GRN metrics and FVS-based bounds.")
    parser.add_argument("--rule-name", default="next_r", choices=sorted(RULES.keys()))
    parser.add_argument("--rule-text-file", default=None, help="Optional path to a custom .bnet rule file.")
    parser.add_argument("--max-nodes-exact", type=int, default=18)
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--out-dir", default="outputs/structure")
    parser.add_argument("--plot-file", default="structural_overview.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule_text = load_rule_text(args.rule_name, args.rule_text_file)
    struct = structural_control_bounds(rule_text, max_nodes_exact=args.max_nodes_exact)
    feedback_df, feedback_truncated = feedback_loops_dataframe(struct["graph"], max_cycles=args.max_cycles)

    metrics_df = pd.DataFrame(struct["metrics"], index=[0]).T.rename(columns={0: "value"})
    metrics_df.index.name = "metric"
    metrics_df.loc["n_feedback_loops", "value"] = int(feedback_df.shape[0])
    metrics_df.loc["feedback_loop_truncated", "value"] = int(feedback_truncated)

    fvs_df = build_fvs_dataframe(struct)

    metrics_path = out_dir / "metrics.csv"
    fvs_path = out_dir / "fvs_solutions.csv"
    loops_path = out_dir / "feedback_loops.csv"
    bounds_path = out_dir / "bounds.json"

    metrics_df.reset_index().to_csv(metrics_path, index=False)
    fvs_df.to_csv(fvs_path, index=False)
    feedback_df.to_csv(loops_path, index=False)

    bounds_payload = {
        "source_nodes": struct["source_nodes"],
        "sink_nodes": struct["sink_nodes"],
        "fvs_forward_aeon": struct["fvs_forward_aeon"],
        "fvs_forward_all": struct["fvs_forward_all"],
        "fvs_reverse_all": struct["fvs_reverse_all"],
        "driver_upper_bound": struct["driver_upper_bound"],
        "sensor_upper_bound": struct["sensor_upper_bound"],
        "feedback_loop_count": int(feedback_df.shape[0]),
        "feedback_loop_truncated": feedback_truncated,
    }
    bounds_path.write_text(json.dumps(bounds_payload, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_path = out_dir / args.plot_file
        plot_structural_overview(struct["graph"], struct, plot_path)
        print(f"Saved structural plot: {plot_path}")

    print(f"Using rule set: {args.rule_name if not args.rule_text_file else 'custom file'}")
    print(f"Source nodes: {struct['source_nodes']}")
    print(f"Sink nodes: {struct['sink_nodes']}")
    print(f"Driver upper bound: {struct['driver_upper_bound']}")
    print(f"Sensor upper bound: {struct['sensor_upper_bound']}")
    print(f"Forward AEON minimum FVS: {struct['fvs_forward_aeon']}")
    print(f"Number of feedback loops reported: {feedback_df.shape[0]}")
    if feedback_truncated:
        print(f"Feedback-loop list truncated at max-cycles={args.max_cycles}")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved FVS solutions: {fvs_path}")
    print(f"Saved feedback loops: {loops_path}")
    print(f"Saved structural bounds JSON: {bounds_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import pandas as pd
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize evaluation metrics.")
    parser.add_argument("--metrics_path", type=str, default="reports/metrics.json")
    parser.add_argument("--output_path", type=str, default="reports/metrics_comparison.png")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data["rows"])

    # Validate required columns
    for col in ("method", "recall@10", "avg_latency_ms"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in metrics data. Available columns: {list(df.columns)}")

    # Dynamic color generation supporting any number of methods
    color_cycle = list(TABLEAU_COLORS.values())
    colors = [color_cycle[i % len(color_cycle)] for i in range(len(df))]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: Recall@10
    bars1 = axes[0].bar(df["method"], df["recall@10"], color=colors)
    axes[0].set_title("Recall@10 (Higher is Better)")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Chart 2: Average Latency
    bars2 = axes[1].bar(df["method"], df["avg_latency_ms"], color=colors)
    axes[1].set_title("Avg Latency (ms) (Lower is Better)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path)
    print(f"Saved visualization to {args.output_path}")

if __name__ == "__main__":
    main()

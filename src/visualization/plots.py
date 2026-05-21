from __future__ import annotations

import csv
from pathlib import Path


SUMMARY_FIELDS = [
    "decoder",
    "grasp_success_rate",
    "average_return",
    "action_mse",
    "trajectory_smoothness",
    "inference_latency_ms",
    "number_of_inference_steps",
    "gripper_timing_error",
    "final_object_lift_height",
]


def write_comparison_csv(summaries: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in SUMMARY_FIELDS})


def write_comparison_plot(summaries: list[dict], path) -> None:
    """Save a compact benchmark summary chart."""
    if not summaries:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    decoders = [s["decoder"] for s in summaries]
    metrics = [
        ("grasp_success_rate", "Success rate"),
        ("average_return", "Average return"),
        ("action_mse", "Action MSE"),
        ("inference_latency_ms", "Latency (ms)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [float(s.get(key, 0.0)) for s in summaries]
        ax.bar(decoders, values, color=colors[: len(decoders)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3g}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    fig.suptitle("VLA Action Decoder Benchmark Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

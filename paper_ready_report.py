import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def safe_mkdir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path: str, fieldnames: Sequence[str], rows: List[Dict]) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def fmt(x: float) -> str:
    return f"{x:.4f}"


def maybe_path(path: str) -> Optional[str]:
    return path if os.path.exists(path) else None


def process_phase1(input_dir: str, output_dir: str, plt, dpi: int, plot_ext: str, generated: List[str]) -> None:
    phase1_dir = os.path.join(input_dir, "phase1")
    main_path = maybe_path(os.path.join(phase1_dir, "main_results_table.json"))
    if not main_path:
        return

    data = load_json(main_path)
    if not isinstance(data, list) or not data:
        return

    rows = []
    for item in data:
        rows.append(
            {
                "dataset": item["dataset"],
                "baseline_srcc": item["baseline"]["srcc"],
                "baseline_plcc": item["baseline"]["plcc"],
                "patch_weighted_srcc": item["patch_weighted"]["srcc"],
                "patch_weighted_plcc": item["patch_weighted"]["plcc"],
                "delta_srcc": item["delta"]["srcc"],
                "delta_plcc": item["delta"]["plcc"],
                "p_bootstrap_srcc": item["significance"]["paired_bootstrap_p_srcc"],
                "p_bootstrap_plcc": item["significance"]["paired_bootstrap_p_plcc"],
                "p_ttest_preds": item["significance"]["paired_ttest_p_preds"],
            }
        )

    out_csv = os.path.join(output_dir, "phase1_main_table.csv")
    write_csv(out_csv, list(rows[0].keys()), rows)
    generated.append(out_csv)

    md_rows = [
        [
            r["dataset"],
            fmt(r["baseline_srcc"]),
            fmt(r["baseline_plcc"]),
            fmt(r["patch_weighted_srcc"]),
            fmt(r["patch_weighted_plcc"]),
            fmt(r["delta_srcc"]),
            fmt(r["delta_plcc"]),
        ]
        for r in rows
    ]
    out_md = os.path.join(output_dir, "phase1_main_table.md")
    write_markdown_table(
        out_md,
        ["Dataset", "Baseline SRCC", "Baseline PLCC", "Patch SRCC", "Patch PLCC", "ΔSRCC", "ΔPLCC"],
        md_rows,
    )
    generated.append(out_md)

    labels = [r["dataset"] for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    base_srcc = [r["baseline_srcc"] for r in rows]
    patch_srcc = [r["patch_weighted_srcc"] for r in rows]
    base_plcc = [r["baseline_plcc"] for r in rows]
    patch_plcc = [r["patch_weighted_plcc"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, base_srcc, width=width, label="Baseline")
    ax.bar(x + width / 2, patch_srcc, width=width, label="Patch-weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("SRCC")
    ax.set_title("Phase 1: SRCC Comparison")
    ax.legend()
    fig.tight_layout()
    out_plot = os.path.join(output_dir, f"phase1_srcc_comparison.{plot_ext}")
    fig.savefig(out_plot, dpi=dpi)
    plt.close(fig)
    generated.append(out_plot)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, base_plcc, width=width, label="Baseline")
    ax.bar(x + width / 2, patch_plcc, width=width, label="Patch-weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("PLCC")
    ax.set_title("Phase 1: PLCC Comparison")
    ax.legend()
    fig.tight_layout()
    out_plot = os.path.join(output_dir, f"phase1_plcc_comparison.{plot_ext}")
    fig.savefig(out_plot, dpi=dpi)
    plt.close(fig)
    generated.append(out_plot)


def _heatmap(values: Dict[Tuple[int, int], float], x_labels: List[int], y_labels: List[int], title: str, out_path: str, plt, dpi: int) -> None:
    grid = np.full((len(y_labels), len(x_labels)), np.nan, dtype=float)
    y_index = {v: i for i, v in enumerate(y_labels)}
    x_index = {v: i for i, v in enumerate(x_labels)}
    for (xv, yv), score in values.items():
        grid[y_index[yv], x_index[xv]] = score

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Patch size")
    ax.set_ylabel("Window size")
    ax.set_title(title)
    for y in range(len(y_labels)):
        for x in range(len(x_labels)):
            if not np.isnan(grid[y, x]):
                ax.text(x, y, f"{grid[y, x]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="SRCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def process_phase2(input_dir: str, output_dir: str, plt, dpi: int, plot_ext: str, generated: List[str]) -> None:
    phase2_dir = os.path.join(input_dir, "phase2")

    threshold_path = maybe_path(os.path.join(phase2_dir, "threshold_ablation_tid2013.json"))
    if threshold_path:
        rows = sorted(load_json(threshold_path), key=lambda x: x["percent_features_to_keep"])
        out_csv = os.path.join(output_dir, "phase2_threshold_ablation.csv")
        write_csv(out_csv, list(rows[0].keys()), rows)
        generated.append(out_csv)
        out_md = os.path.join(output_dir, "phase2_threshold_ablation.md")
        write_markdown_table(
            out_md,
            ["Percent", "SRCC", "PLCC"],
            [[fmt(r["percent_features_to_keep"]), fmt(r["srcc"]), fmt(r["plcc"])] for r in rows],
        )
        generated.append(out_md)

        xs = [r["percent_features_to_keep"] for r in rows]
        srcc = [r["srcc"] for r in rows]
        plcc = [r["plcc"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, srcc, marker="o", label="SRCC")
        ax.plot(xs, plcc, marker="o", label="PLCC")
        ax.set_xlabel("Percent features to keep")
        ax.set_ylabel("Correlation")
        ax.set_title("Phase 2: Threshold Ablation (TID2013)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_plot = os.path.join(output_dir, f"phase2_threshold_ablation.{plot_ext}")
        fig.savefig(out_plot, dpi=dpi)
        plt.close(fig)
        generated.append(out_plot)

    sensitivity_path = maybe_path(os.path.join(phase2_dir, "patch_window_sensitivity_tid2013.json"))
    if sensitivity_path:
        rows = load_json(sensitivity_path)
        out_csv = os.path.join(output_dir, "phase2_patch_window_sensitivity.csv")
        write_csv(out_csv, list(rows[0].keys()), rows)
        generated.append(out_csv)
        values = {(int(r["patch_size"]), int(r["window_size"])): float(r["srcc"]) for r in rows}
        patch_sizes = sorted({int(r["patch_size"]) for r in rows})
        window_sizes = sorted({int(r["window_size"]) for r in rows})
        out_plot = os.path.join(output_dir, f"phase2_patch_window_heatmap_srcc.{plot_ext}")
        _heatmap(values, patch_sizes, window_sizes, "Phase 2: Patch/Window Sensitivity SRCC", out_plot, plt, dpi)
        generated.append(out_plot)

    weight_map_path = maybe_path(os.path.join(phase2_dir, "weight_map_ablation_tid2013.json"))
    if weight_map_path:
        rows = load_json(weight_map_path)
        out_csv = os.path.join(output_dir, "phase2_weight_map_ablation.csv")
        write_csv(out_csv, list(rows[0].keys()), rows)
        generated.append(out_csv)
        out_md = os.path.join(output_dir, "phase2_weight_map_ablation_top5.md")
        ranked = sorted(rows, key=lambda x: x["srcc"], reverse=True)[:5]
        write_markdown_table(
            out_md,
            ["Weight source", "Aggregation", "Temp", "SRCC", "PLCC"],
            [[r["weight_source"], r["aggregation"], str(r["temperature"]), fmt(r["srcc"]), fmt(r["plcc"])] for r in ranked],
        )
        generated.append(out_md)


def process_phase3(input_dir: str, output_dir: str, plt, dpi: int, plot_ext: str, generated: List[str]) -> None:
    phase3_dir = os.path.join(input_dir, "phase3")

    geom_path = maybe_path(os.path.join(phase3_dir, "geometric_robustness_summary.json"))
    if geom_path:
        data = load_json(geom_path)
        rows = [
            {"method": "baseline", "srcc": data["baseline"]["srcc"]},
            {"method": "patch_weighted", "srcc": data["patch_weighted"]["srcc"]},
            {"method": "ssim", "srcc": data["ssim"]["srcc"]},
        ]
        out_csv = os.path.join(output_dir, "phase3_geometric_robustness.csv")
        write_csv(out_csv, ["method", "srcc"], rows)
        generated.append(out_csv)
        out_md = os.path.join(output_dir, "phase3_geometric_robustness.md")
        write_markdown_table(out_md, ["Method", "SRCC"], [[r["method"], fmt(r["srcc"])] for r in rows])
        generated.append(out_md)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([r["method"] for r in rows], [r["srcc"] for r in rows])
        ax.set_ylabel("SRCC")
        ax.set_title("Phase 3: Geometric Robustness")
        fig.tight_layout()
        out_plot = os.path.join(output_dir, f"phase3_geometric_robustness.{plot_ext}")
        fig.savefig(out_plot, dpi=dpi)
        plt.close(fig)
        generated.append(out_plot)

    complexity_path = maybe_path(os.path.join(phase3_dir, "complexity_analysis.json"))
    if complexity_path:
        rows = load_json(complexity_path)
        out_csv = os.path.join(output_dir, "phase3_complexity_analysis.csv")
        write_csv(out_csv, list(rows[0].keys()), rows)
        generated.append(out_csv)
        out_md = os.path.join(output_dir, "phase3_complexity_analysis.md")
        write_markdown_table(
            out_md,
            ["Device", "Method", "Avg sec/pair"],
            [[r["device"], r["method"], fmt(r["avg_seconds_per_pair"])] for r in rows],
        )
        generated.append(out_md)

        labels = [f'{r["method"]}-{r["device"]}' for r in rows]
        vals = [r["avg_seconds_per_pair"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, vals)
        ax.set_ylabel("Seconds per pair")
        ax.set_title("Phase 3: Complexity Analysis")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        out_plot = os.path.join(output_dir, f"phase3_complexity_analysis.{plot_ext}")
        fig.savefig(out_plot, dpi=dpi)
        plt.close(fig)
        generated.append(out_plot)

    cross_path = maybe_path(os.path.join(phase3_dir, "cross_dataset_generalization.json"))
    if cross_path:
        data = load_json(cross_path)
        rows = [{"dataset": "TID2013 (tuning set)", "srcc": data["best"]["srcc"], "plcc": data["best"]["plcc"]}]
        rows.extend({"dataset": r["dataset"], "srcc": r["srcc"], "plcc": r["plcc"]} for r in data.get("tests", []))
        out_csv = os.path.join(output_dir, "phase3_cross_dataset_generalization.csv")
        write_csv(out_csv, ["dataset", "srcc", "plcc"], rows)
        generated.append(out_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables/plots from experiment_results.")
    parser.add_argument("--input-dir", type=str, default="experiment_results")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("matplotlib is required for plot generation. Install with: pip install matplotlib") from exc

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(input_dir, "paper_ready"))
    safe_mkdir(output_dir)
    generated: List[str] = []

    process_phase1(input_dir, output_dir, plt, args.dpi, args.plot_format, generated)
    process_phase2(input_dir, output_dir, plt, args.dpi, args.plot_format, generated)
    process_phase3(input_dir, output_dir, plt, args.dpi, args.plot_format, generated)

    if not generated:
        print(f"No compatible result files found under {input_dir}.")
        return

    print("Generated artifacts:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()

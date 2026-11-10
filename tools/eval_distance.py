"""Evaluate ground distance estimation accuracy against tape-measured ground truth.

Reads a ground-truth CSV (point_label, X_true_m, Y_true_m, d_true_m) and
per-frame measurement log CSV(s) produced by run_live.py --log-csv.
Computes per-position median estimates, MAE, MRE, accuracy (1 - MRE),
range-bucket breakdown (0-4, 4-8, 8-12 m), and side-by-side comparison of
Planar Homography vs Pinhole Baseline. Saves CSV tables and plot to results/.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vision.calibration import CalibrationData
from vision.config import load_config
from vision.geometry import assumed_height_for_class, pinhole_distance


RANGE_BUCKETS: list[tuple[str, float, float]] = [
    ("0-4m", 0.0, 4.0),
    ("4-8m", 4.0, 8.0),
    ("8-12m", 8.0, 12.0),
    (">12m", 12.0, math.inf),
]


def load_ground_truth_csv(path: str | Path) -> pd.DataFrame:
    """Load ground-truth CSV supporting both Appendix A and short column names."""
    df = pd.read_csv(path)
    rename_map = {
        "X_true": "X_true_m",
        "Y_true": "Y_true_m",
        "d_true": "d_true_m",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = {"point_label", "X_true_m", "Y_true_m", "d_true_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth CSV missing required columns: {sorted(missing)}")
    df["point_label"] = df["point_label"].astype(str).str.strip()
    return df


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def load_log_csv(path: str | Path) -> pd.DataFrame:
    """Load a per-frame measurement log CSV created by MeasurementCsvLogger."""
    df = pd.read_csv(path)
    for col in ("t", "X", "Y", "distance", "box_h_px", "pinhole_distance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "in_range" in df.columns:
        df["in_range"] = df["in_range"].apply(_parse_bool)
    if "truncated" in df.columns:
        df["truncated"] = df["truncated"].apply(_parse_bool)
    return df


def _trim_stable_frames(df: pd.DataFrame, trim_fraction: float) -> pd.DataFrame:
    """Trim start/end transient frames from a stationary trial log."""
    if len(df) < 5 or trim_fraction <= 0.0:
        return df
    df_sorted = df.sort_values("t").reset_index(drop=True)
    n_trim = int(len(df_sorted) * trim_fraction)
    if n_trim > 0 and (len(df_sorted) - 2 * n_trim) >= 1:
        return df_sorted.iloc[n_trim : len(df_sorted) - n_trim]
    return df_sorted


def _fill_pinhole_if_needed(
    df: pd.DataFrame,
    focal_px: float | None,
    real_height_m: float | None,
) -> pd.Series:
    """Use logged pinhole_distance if present, or compute from box_h_px and focal_px."""
    if "pinhole_distance" in df.columns and df["pinhole_distance"].notna().any():
        return df["pinhole_distance"]
    if focal_px is not None and focal_px > 0 and real_height_m is not None and "box_h_px" in df.columns:
        return df["box_h_px"].apply(
            lambda h: pinhole_distance(float(h), real_height_m, focal_px)
            if pd.notna(h) and float(h) > 0
            else np.nan
        )
    return pd.Series(np.nan, index=df.index, dtype=float)


def evaluate_positions(
    gt_df: pd.DataFrame,
    position_logs: dict[str, pd.DataFrame],
    target_cls: str | None = None,
    trim_fraction: float = 0.1,
    focal_px: float | None = None,
    real_height_m: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate homography and pinhole distance estimates across ground-truth positions.

    Returns:
        (per_position_df, summary_df, buckets_df)
    """
    rows: list[dict[str, Any]] = []

    for _, gt_row in gt_df.iterrows():
        label = str(gt_row["point_label"])
        x_true = float(gt_row["X_true_m"])
        y_true = float(gt_row["Y_true_m"])
        d_true = float(gt_row["d_true_m"])

        log_df = position_logs.get(label)
        if log_df is None or log_df.empty:
            continue

        filtered = log_df.copy()
        if target_cls is not None and "cls" in filtered.columns:
            filtered = filtered[filtered["cls"] == target_cls]

        # Keep rows with valid numeric homography distance (bottom-truncated rows have NaN distance)
        filtered = filtered[filtered["distance"].notna()]
        if filtered.empty:
            continue

        # If multiple tracks exist in same frame, pick closest/lowest-ID or primary track;
        # trimming stabilizes stationary stand-still measurements.
        stable = _trim_stable_frames(filtered, trim_fraction=trim_fraction)
        pin_series = _fill_pinhole_if_needed(stable, focal_px=focal_px, real_height_m=real_height_m)

        d_homo_med = float(stable["distance"].median())
        x_est_med = float(stable["X"].median()) if "X" in stable.columns and stable["X"].notna().any() else np.nan
        y_est_med = float(stable["Y"].median()) if "Y" in stable.columns and stable["Y"].notna().any() else np.nan

        homo_abs_err = abs(d_homo_med - d_true)
        homo_rel_err = homo_abs_err / d_true if d_true > 0 else np.nan

        valid_pin = pin_series.dropna()
        if not valid_pin.empty:
            d_pin_med = float(valid_pin.median())
            pin_abs_err = abs(d_pin_med - d_true)
            pin_rel_err = pin_abs_err / d_true if d_true > 0 else np.nan
        else:
            d_pin_med = np.nan
            pin_abs_err = np.nan
            pin_rel_err = np.nan

        rows.append({
            "point_label": label,
            "n_frames": int(len(stable)),
            "X_true_m": x_true,
            "Y_true_m": y_true,
            "d_true_m": d_true,
            "X_est_med_m": x_est_med,
            "Y_est_med_m": y_est_med,
            "d_homography_med_m": d_homo_med,
            "homography_abs_err_m": homo_abs_err,
            "homography_rel_err": homo_rel_err,
            "d_pinhole_med_m": d_pin_med,
            "pinhole_abs_err_m": pin_abs_err,
            "pinhole_rel_err": pin_rel_err,
        })

    per_pos_df = pd.DataFrame(rows)
    if per_pos_df.empty:
        return per_pos_df, pd.DataFrame(), pd.DataFrame()

    # Summary: Homography vs Pinhole side by side
    summary_rows = []
    for method, abs_col, rel_col in [
        ("homography", "homography_abs_err_m", "homography_rel_err"),
        ("pinhole", "pinhole_abs_err_m", "pinhole_rel_err"),
    ]:
        valid = per_pos_df[per_pos_df[abs_col].notna()]
        if valid.empty:
            summary_rows.append({
                "method": method,
                "n_positions": 0,
                "mae_m": np.nan,
                "mre": np.nan,
                "accuracy": np.nan,
                "max_err_m": np.nan,
            })
        else:
            mae = float(valid[abs_col].mean())
            mre = float(valid[rel_col].mean())
            summary_rows.append({
                "method": method,
                "n_positions": int(len(valid)),
                "mae_m": mae,
                "mre": mre,
                "accuracy": 1.0 - mre,
                "max_err_m": float(valid[abs_col].max()),
            })
    summary_df = pd.DataFrame(summary_rows)

    # Range bucket breakdown
    bucket_rows = []
    for b_name, lo, hi in RANGE_BUCKETS:
        sub = per_pos_df[(per_pos_df["d_true_m"] >= lo) & (per_pos_df["d_true_m"] < hi)]
        if sub.empty:
            continue
        h_mae = float(sub["homography_abs_err_m"].mean())
        h_mre = float(sub["homography_rel_err"].mean())
        pin_sub = sub[sub["pinhole_abs_err_m"].notna()]
        p_mae = float(pin_sub["pinhole_abs_err_m"].mean()) if not pin_sub.empty else np.nan
        p_mre = float(pin_sub["pinhole_rel_err"].mean()) if not pin_sub.empty else np.nan
        bucket_rows.append({
            "range_bucket": b_name,
            "n_positions": int(len(sub)),
            "homography_mae_m": h_mae,
            "homography_mre": h_mre,
            "homography_accuracy": 1.0 - h_mre,
            "pinhole_mae_m": p_mae,
            "pinhole_mre": p_mre,
            "pinhole_accuracy": (1.0 - p_mre) if pd.notna(p_mre) else np.nan,
        })
    buckets_df = pd.DataFrame(bucket_rows)

    return per_pos_df, summary_df, buckets_df


def save_distance_plot(per_pos_df: pd.DataFrame, out_png: Path, title_prefix: str = "") -> None:
    """Save 2-panel comparison plot (Estimated vs True distance, Relative Error by Position)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Estimated vs True distance
    ax0 = axes[0]
    d_true = per_pos_df["d_true_m"].values
    max_d = max(float(np.max(d_true)), float(per_pos_df["d_homography_med_m"].max())) * 1.1
    ax0.plot([0, max_d], [0, max_d], "k--", label="Ideal (y = x)")
    ax0.scatter(d_true, per_pos_df["d_homography_med_m"], color="tab:blue", s=60, label="Homography")
    if per_pos_df["d_pinhole_med_m"].notna().any():
        pin_valid = per_pos_df[per_pos_df["d_pinhole_med_m"].notna()]
        ax0.scatter(
            pin_valid["d_true_m"],
            pin_valid["d_pinhole_med_m"],
            color="tab:orange",
            marker="^",
            s=60,
            label="Pinhole baseline",
        )
    ax0.set_xlabel("True Distance (m)")
    ax0.set_ylabel("Estimated Median Distance (m)")
    ax0.set_title(f"{title_prefix}Estimated vs True Distance".strip())
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # Right panel: Relative Error (%) per point
    ax1 = axes[1]
    labels = per_pos_df["point_label"].tolist()
    x_idx = np.arange(len(labels))
    width = 0.38
    ax1.bar(
        x_idx - width / 2,
        per_pos_df["homography_rel_err"] * 100.0,
        width=width,
        color="tab:blue",
        label="Homography MRE (%)",
    )
    if per_pos_df["pinhole_rel_err"].notna().any():
        ax1.bar(
            x_idx + width / 2,
            per_pos_df["pinhole_rel_err"].fillna(0.0) * 100.0,
            width=width,
            color="tab:orange",
            label="Pinhole MRE (%)",
        )
    ax1.axhline(10.0, color="red", linestyle=":", label="10% target")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("Relative Error (%)")
    ax1.set_title(f"{title_prefix}Per-Position Relative Error".strip())
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _gather_logs_from_args(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Collect per-position DataFrames from --logs, --logs-dir, or --log + --segments."""
    position_logs: dict[str, list[pd.DataFrame]] = {}

    if args.logs:
        for item in args.logs:
            if "=" not in item:
                raise ValueError(f"--logs items must be label=path.csv, got: {item}")
            lbl, path_str = item.split("=", 1)
            df = load_log_csv(path_str)
            position_logs.setdefault(lbl.strip(), []).append(df)

    if args.logs_dir:
        log_dir = Path(args.logs_dir)
        gt_df = load_ground_truth_csv(args.gt)
        labels = gt_df["point_label"].tolist()
        for csv_path in sorted(log_dir.glob("*.csv")):
            stem = csv_path.stem
            matched_label = next(
                (lbl for lbl in sorted(labels, key=len, reverse=True) if stem == lbl or stem.startswith(f"{lbl}_")),
                None,
            )
            if matched_label is not None:
                df = load_log_csv(csv_path)
                position_logs.setdefault(matched_label, []).append(df)

    if args.log:
        combined = load_log_csv(args.log)
        if args.segments:
            seg_df = pd.read_csv(args.segments)
            for _, row in seg_df.iterrows():
                lbl = str(row["point_label"]).strip()
                t0, t1 = float(row["t_start"]), float(row["t_end"])
                sub = combined[(combined["t"] >= t0) & (combined["t"] <= t1)]
                if not sub.empty:
                    position_logs.setdefault(lbl, []).append(sub)
        elif "point_label" in combined.columns:
            for lbl, sub in combined.groupby("point_label"):
                position_logs.setdefault(str(lbl).strip(), []).append(sub)
        else:
            raise ValueError("--log requires either --segments <segments.csv> or a point_label column in the log")

    return {lbl: pd.concat(dfs, ignore_index=True) for lbl, dfs in position_logs.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distance estimation logs against tape ground truth.")
    parser.add_argument("--gt", type=str, required=True, help="Ground-truth CSV (point_label, X_true_m, Y_true_m, d_true_m).")
    parser.add_argument("--logs", nargs="*", default=None, help="Explicit label=path.csv pairs.")
    parser.add_argument("--logs-dir", type=str, default=None, help="Directory of log CSVs named <point_label>*.csv.")
    parser.add_argument("--log", type=str, default=None, help="Single combined log CSV (with --segments or point_label column).")
    parser.add_argument("--segments", type=str, default=None, help="CSV mapping point_label, t_start, t_end into --log.")
    parser.add_argument("--cls", type=str, default="person", help="Target object class to evaluate (person|two_wheeler|car|all).")
    parser.add_argument("--calibration", type=str, default=None, help="Calibration JSON path (supplies focal_px if not logged).")
    parser.add_argument("--focal-px", type=float, default=None, help="Override focal length in pixels for pinhole baseline.")
    parser.add_argument("--trim", type=float, default=0.1, help="Fraction of frames to trim at start/end of trial (default 0.1).")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for CSVs and plots.")
    parser.add_argument("--prefix", type=str, default=None, help="Output file prefix (default: distance_<cls>).")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to default config.")
    parser.add_argument("--local-config", type=str, default="config/local.yaml", help="Path to local config.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(default_path=args.config, local_path=args.local_config)
    gt_df = load_ground_truth_csv(args.gt)

    target_cls = None if args.cls.lower() == "all" else args.cls
    prefix = args.prefix or f"distance_{args.cls}"

    focal_px = args.focal_px
    if focal_px is None:
        cal_path = (
            Path(args.calibration)
            if args.calibration is not None
            else Path(f"data/calibration/{cfg.camera.camera_id}.json")
        )
        if cal_path.is_file():
            calib = CalibrationData.load_json(cal_path)
            focal_px = calib.focal_px

    real_h = assumed_height_for_class(target_cls, cfg.pinhole_baseline) if target_cls else None
    position_logs = _gather_logs_from_args(args)

    per_pos_df, summary_df, buckets_df = evaluate_positions(
        gt_df=gt_df,
        position_logs=position_logs,
        target_cls=target_cls,
        trim_fraction=args.trim,
        focal_px=focal_px,
        real_height_m=real_h,
    )

    if per_pos_df.empty:
        print("No matching valid distance logs found for any ground-truth position.")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pos_csv = out_dir / f"{prefix}_per_position.csv"
    summary_csv = out_dir / f"{prefix}_summary.csv"
    buckets_csv = out_dir / f"{prefix}_buckets.csv"
    plot_png = out_dir / f"{prefix}_plot.png"

    per_pos_df.to_csv(per_pos_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    buckets_df.to_csv(buckets_csv, index=False)
    save_distance_plot(per_pos_df, plot_png, title_prefix=f"[{args.cls}] ")

    print("\n" + "=" * 72)
    print(f"DISTANCE EVALUATION REPORT ({args.cls})")
    print("=" * 72)
    print("\nPer-Position Results (Median):")
    print(per_pos_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nSummary Comparison (Homography vs Pinhole):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not buckets_df.empty:
        print("\nError vs Range Bucket:")
        print(buckets_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nSaved outputs to {out_dir}/ ({per_pos_csv.name}, {summary_csv.name}, {buckets_csv.name}, {plot_png.name})")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

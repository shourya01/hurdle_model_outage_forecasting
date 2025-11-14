#!/usr/bin/env python
"""
Two-stage outage forecaster with dataset-aware splits.

Key differences from train_outage.py:
* Splits follow the L + mT rule per dataset (Michigan or California).
* Weather is read directly from the NetCDF file unless --use-weather CSV is supplied.
* Forecasts stay within the NetCDF timeline (no extrapolation beyond the file).
* Outputs are limited to test forecasts CSV plus a calibration PDF; optional per-county
  plots can be enabled via --plot-results.
* Lookback is fixed at 48 timesteps for both datasets; weather covariates are restricted
  to those common across Michigan and California so the model travels cleanly between them.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------- constants ------------------------------------ #

DATASET_CONFIG = {
    "michigan": {"nc_file": "michigan.nc", "m_val": 5, "m_test": 5},
    "california": {"nc_file": "california.nc", "m_val": 30, "m_test": 30},
}

COMMON_WEATHER_FEATURES = [
    "blh",
    "cape",
    "d2m",
    "gust",
    "hcc",
    "ishf",
    "lcc",
    "lsm",
    "max_10si",
    "mcc",
    "pres",
    "pwat",
    "sdlwrf",
    "sdswrf",
    "slhtf",
    "sp",
    "sulwrf",
    "suswrf",
    "t2m",
    "tcc",
    "tcolw",
    "tp",
    "u10",
    "v10",
]

MAX_HAZARD_FEATURES = len(COMMON_WEATHER_FEATURES)
MAX_LOOKBACK_POINTS = 48
LAG_TEMPLATE = list(range(1, MAX_LOOKBACK_POINTS + 1))
OUT_ROLL_SUM_WINDOWS = [6, 12, 24]
OUT_ROLL_MAX_WINDOWS = [6, 12, 24]
WEATHER_ROLL_MAX_WINDOWS = [6, 12, 24]
WEATHER_ROLL_SUM_WINDOWS = [6, 12, 24]
CLIP_QUANTILES = (0.005, 0.995)
MIN_STD = 1e-6
EPS = 1e-8


# ---------------------------- helpers -------------------------------------- #

def decode_coord(values: Sequence) -> List[str]:
    decoded = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            decoded.append(item.decode("utf-8").strip())
        else:
            decoded.append(str(item).strip())
    return decoded


def compute_steps_per_day(timestamps: np.ndarray) -> int:
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to infer frequency.")
    delta = (timestamps[1] - timestamps[0]).astype("timedelta64[m]").astype(int)
    if delta <= 0:
        raise ValueError("Non-positive timestep detected.")
    return int(round((24 * 60) / delta))


def compute_split_plan(
    num_steps: int,
    lookback_steps: int,
    lookahead_steps: int,
    m_val: int,
    m_test: int,
) -> Dict[str, Tuple[int, int]]:
    test_len = lookback_steps + m_test * lookahead_steps
    val_len = lookback_steps + m_val * lookahead_steps
    test_start = num_steps - test_len
    val_start = test_start - val_len
    if val_start <= 0:
        raise ValueError("Not enough history to carve out train|val|test splits.")
    if test_start <= 0:
        raise ValueError("Test split overlaps the start of the dataset.")
    plan = {
        "train": (0, val_start),
        "val": (val_start, test_start),
        "test": (test_start, num_steps),
        "val_eval": (val_start + lookback_steps, test_start),
        "test_eval": (test_start + lookback_steps, num_steps),
    }
    if plan["val_eval"][0] >= plan["val_eval"][1]:
        raise ValueError("Validation evaluation window is empty; adjust m_val.")
    if plan["test_eval"][0] >= plan["test_eval"][1]:
        raise ValueError("Test evaluation window is empty; adjust m_test.")
    return plan


def build_lag_list(lookback_steps: int) -> List[int]:
    lags = [lag for lag in LAG_TEMPLATE if lag <= lookback_steps]
    if not lags:
        lags = [1]
    return lags


def select_hazard_features(
    outages: np.ndarray,
    weather: np.ndarray,
    train_range: Tuple[int, int],
    feature_names: Sequence[str],
    max_features: int = MAX_HAZARD_FEATURES,
) -> List[int]:
    start, end = train_range
    out_flat = outages[start:end].reshape(-1)
    valid_mask = np.isfinite(out_flat)
    rankings: List[Tuple[int, float]] = []
    for idx in range(weather.shape[2]):
        feat_flat = weather[start:end, :, idx].reshape(-1)
        mask = valid_mask & np.isfinite(feat_flat)
        if mask.sum() < 2:
            rankings.append((idx, 0.0))
            continue
        corr = np.corrcoef(out_flat[mask], feat_flat[mask])[0, 1]
        rankings.append((idx, abs(corr)))
    rankings.sort(key=lambda item: (-item[1], feature_names[item[0]]))
    return [idx for idx, _ in rankings[:max_features]]


def load_weather_override(
    csv_path: Path,
    feature_names: Sequence[str],
    timestamps: np.ndarray,
    counties: Sequence[str],
) -> Dict[Tuple[int, int], np.ndarray]:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "county" not in df.columns:
        raise ValueError("--use-weather CSV must contain 'timestamp' and 'county' columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["county"] = df["county"].astype(str).str.zfill(5)
    ts_map = {pd.Timestamp(ts): idx for idx, ts in enumerate(pd.to_datetime(timestamps))}
    county_map = {cid: idx for idx, cid in enumerate(counties)}
    feat_map = {name: idx for idx, name in enumerate(feature_names)}
    overrides: Dict[Tuple[int, int], np.ndarray] = {}
    for row in df.itertuples(index=False):
        ts_idx = ts_map.get(row.timestamp)
        county_idx = county_map.get(row.county)
        if ts_idx is None or county_idx is None:
            continue
        vector = np.full(len(feature_names), np.nan, dtype=np.float32)
        for feat, feat_idx in feat_map.items():
            if feat not in df.columns:
                continue
            value = getattr(row, feat)
            vector[feat_idx] = np.float32(value)
        overrides[(ts_idx, county_idx)] = vector
    return overrides


def apply_weather_overrides(
    base_weather: np.ndarray,
    overrides: Dict[Tuple[int, int], np.ndarray],
) -> np.ndarray:
    if not overrides:
        return base_weather
    weather = base_weather.copy()
    for (ts_idx, county_idx), vec in overrides.items():
        valid = np.isfinite(vec)
        if not valid.any():
            continue
        weather[ts_idx, county_idx, valid] = vec[valid]
    return weather


def trig_features(timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hours = timestamps.hour + timestamps.minute / 60.0
    dow = timestamps.dayofweek
    hour_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    return hour_sin, hour_cos, dow_sin, dow_cos


def count_valid(outages: np.ndarray, time_indices: np.ndarray) -> int:
    total = 0
    for t in time_indices:
        total += int(np.isfinite(outages[t]).sum())
    return total


def compute_run_length_since_nonzero(
    outages: np.ndarray,
    t: int,
    loc: int,
    lookback_steps: int,
) -> float:
    if t == 0:
        return float("nan")
    length = 0
    seen = False
    for lag in range(1, lookback_steps + 1):
        idx = t - lag
        if idx < 0:
            return float("nan") if not seen else float(length)
        val = outages[idx, loc]
        if not np.isfinite(val):
            val = 0.0
        seen = True
        if val > 0:
            return float(length)
        length += 1
    return float(length)


    rows: List[np.ndarray] = []
    targets: List[float] = []
    occ: List[int] = []
    meta: Optional[List[Tuple[int, int]]] = [] if return_meta else None
    for t in time_indices:
        hist_len = min(t, lookback_steps)
        for loc in range(num_locations):
            target_val = outages[t, loc]
            if not np.isfinite(target_val):
                continue
            feature_row = np.empty(total_dim, dtype=np.float32)
            pos = 0
            feature_row[pos] = float(hist_len)
            pos += 1
            lag_values: Dict[int, float] = {}
            for lag in active_lags:
                idx = t - lag
                if idx >= 0:
                    lag_val = outages[idx, loc]
                    val = float(0.0 if not np.isfinite(lag_val) else lag_val)
                else:
                    val = np.nan
                feature_row[pos] = val
                lag_values[lag] = val
                pos += 1
            for window in active_out_sum:
                idx_start = t - window
                if idx_start >= 0:
                    vals = outages[idx_start:t, loc]
                    val = float(np.nansum(np.where(np.isfinite(vals), vals, 0.0)))
                else:
                    val = np.nan
                feature_row[pos] = val
                pos += 1
            for window in active_out_max:
                idx_start = t - window
                if idx_start >= 0:
                    vals = outages[idx_start:t, loc]
                    if vals.size:
                        finite_vals = np.where(np.isfinite(vals), vals, np.nan)
                        val = float(np.nanmax(finite_vals))
                    else:
                        val = np.nan
                else:
                    val = np.nan
                feature_row[pos] = val
                pos += 1
            feature_row[pos] = compute_run_length_since_nonzero(outages, t, loc, lookback_steps)
            pos += 1
            out_lag1_val = lag_values.get(1, np.nan)
            hazard_lag_vals: Dict[str, float] = {}
            for feat_idx, feat_name in zip(hazard_idx, hazard_names):
                if t >= 1:
                    lag1 = weather[t - 1, loc, feat_idx]
                    feature_row[pos] = float(lag1) if np.isfinite(lag1) else np.nan
                else:
                    lag1 = np.nan
                    feature_row[pos] = np.nan
                hazard_lag_vals[feat_name] = feature_row[pos]
                pos += 1
                if t >= 2:
                    prev = weather[t - 2, loc, feat_idx]
                    if np.isfinite(lag1) and np.isfinite(prev):
                        delta = float(lag1 - prev)
                    else:
                        delta = np.nan
                elif t == 1:
                    delta = 0.0
                else:
                    delta = np.nan
                feature_row[pos] = delta
                pos += 1
                for window in active_weather_roll_max:
                    idx_start = t - window
                    if idx_start >= 0:
                        vals = weather[idx_start:t, loc, feat_idx]
                        finite_vals = np.where(np.isfinite(vals), vals, np.nan)
                        val = float(np.nanmax(finite_vals))
                    else:
                        val = np.nan
                    feature_row[pos] = val
                    pos += 1
                for window in active_weather_roll_sum:
                    idx_start = t - window
                    if idx_start >= 0:
                        vals = weather[idx_start:t, loc, feat_idx]
                        finite_vals = np.where(np.isfinite(vals), vals, 0.0)
                        val = float(np.nansum(finite_vals))
                    else:
                        val = np.nan
                    feature_row[pos] = val
                    pos += 1
                hazard_val = hazard_lag_vals.get(feat_name, np.nan)
                if np.isfinite(out_lag1_val) and np.isfinite(hazard_val):
                    feature_row[pos] = float(out_lag1_val * hazard_val)
                else:
                    feature_row[pos] = np.nan
                pos += 1
            feature_row[pos] = hour_sin[t]
            pos += 1
            feature_row[pos] = hour_cos[t]
            pos += 1
            feature_row[pos] = dow_sin[t]
            pos += 1
            feature_row[pos] = dow_cos[t]
            pos += 1
            feature_row[pos] = county_indices[loc]
            rows.append(feature_row)
            targets.append(float(target_val))
            occ.append(1 if target_val > 0 else 0)
            if meta is not None:
                meta.append((int(t), int(loc)))
    if not rows:
        empty = np.empty((0, total_dim), dtype=np.float32)
        empty_target = np.empty(0, dtype=np.float32)
        empty_occ = np.empty(0, dtype=np.int8)
        return empty, empty_target, empty_occ, meta
    X = np.vstack(rows)
    y = np.asarray(targets, dtype=np.float32)
    occ_arr = np.asarray(occ, dtype=np.int8)
    return X, y, occ_arr, meta


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_index: int,
    seed: int,
    device: str,
    calibrate: bool,
) -> Tuple[lgb.LGBMClassifier, Optional[IsotonicRegression]]:
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = float(neg_count / max(pos_count, 1))
    clf = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.005,
        num_leaves=1000,
        max_depth=12,
        n_estimators=10000,
        min_child_samples=100,
        min_sum_hessian_in_leaf=5.0,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        feature_fraction_bynode=0.8,
        reg_alpha=0.03,
        reg_lambda=0.03,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        device_type="gpu" if device == "gpu" else "cpu",
        verbosity=-1,
    )
    eval_set = [(X_val, y_val)] if len(X_val) else None
    clf.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="binary_logloss",
        categorical_feature=[cat_index],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    calibrator: Optional[IsotonicRegression] = None
    if calibrate and len(X_val) and len(np.unique(y_val)) > 1:
        val_probs = clf.predict_proba(X_val, raw_score=False)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs, y_val)
    return clf, calibrator


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_index: int,
    seed: int,
    device: str,
) -> lgb.LGBMRegressor:
    train_mask = y_train > 0
    val_mask = y_val > 0
    if train_mask.sum() == 0:
        raise RuntimeError("No positive outage samples in training data; cannot fit regressor.")
    reg = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.55,
        learning_rate=0.005,
        num_leaves=1000,
        max_depth=12,
        n_estimators=12000,
        min_child_samples=150,
        min_sum_hessian_in_leaf=10.0,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        feature_fraction_bynode=0.8,
        reg_alpha=0.03,
        reg_lambda=0.03,
        random_state=seed,
        n_jobs=-1,
        device_type="gpu" if device == "gpu" else "cpu",
        verbosity=-1,
    )
    eval_set = [(X_val[val_mask], y_val[val_mask])] if val_mask.sum() else None
    reg.fit(
        X_train[train_mask],
        y_train[train_mask],
        eval_set=eval_set,
        eval_metric="rmse",
        categorical_feature=[cat_index],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    return reg


def calibrate_probabilities(
    raw_probs: np.ndarray,
    calibrator: Optional[IsotonicRegression],
) -> np.ndarray:
    if raw_probs.size == 0:
        return raw_probs
    if calibrator is None:
        return np.clip(raw_probs, 0.0, 1.0)
    calibrated = calibrator.transform(raw_probs)
    return np.clip(calibrated, 0.0, 1.0)


def rolling_forecast_twostage(
    classifier: lgb.LGBMClassifier,
    calibrator: Optional[IsotonicRegression],
    regressor: lgb.LGBMRegressor,
    outages: np.ndarray,
    weather: np.ndarray,
    test_indices: np.ndarray,
    lookahead_steps: int,
    lookback_steps: int,
    lag_list: Sequence[int],
    hazard_idx: Sequence[int],
    hazard_names: Sequence[str],
    trig_cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    county_indices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    counties: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[int]]:
    records: List[Dict[str, Any]] = []
    raw_probs_all: List[float] = []
    occ_all: List[int] = []
    outages_work = outages.copy()
    block_lengths: List[int] = []
    total_steps = len(test_indices)
    for idx, block_start in enumerate(range(0, total_steps, lookahead_steps), start=1):
        block_indices = test_indices[block_start : block_start + lookahead_steps]
        if len(block_indices) == 0:
            continue
        block_lengths.append(len(block_indices))
        start_ts = timestamps[block_indices[0]]
        end_ts = timestamps[block_indices[-1]]
        print(f"[Forecast] Block {idx}: {len(block_indices)} steps ({start_ts} → {end_ts})")
        for t in block_indices:
            X_step, y_step, occ_step, meta = build_split_arrays(
                outages_work,
                weather,
                np.array([t]),
                lag_list,
                hazard_idx,
                hazard_names,
                trig_cache,
                county_indices,
                lookback_steps,
                return_meta=True,
            )
            if len(X_step) == 0 or not meta:
                continue
            raw_probs = classifier.predict_proba(X_step, raw_score=False)[:, 1]
            probs = calibrate_probabilities(raw_probs, calibrator)
            intensity = np.clip(regressor.predict(X_step), 0.0, None)
            preds = np.clip(probs * intensity, 0.0, None)
            for (time_idx, loc_idx), truth, pred, prob in zip(meta, y_step, preds, probs):
                records.append(
                    {
                        "timestamp": timestamps[time_idx],
                        "county_id": counties[loc_idx],
                        "outage_truth": float(truth),
                        "outage_pred": float(pred),
                        "p_nonzero": float(prob),
                    }
                )
                outages_work[time_idx, loc_idx] = pred
            raw_probs_all.extend(raw_probs.tolist())
            occ_all.extend(occ_step.tolist())
        outages_work[block_indices, :] = outages[block_indices, :]
    df = pd.DataFrame(records).sort_values(["county_id", "timestamp"]).reset_index(drop=True)
    return df, np.asarray(raw_probs_all, dtype=np.float32), np.asarray(occ_all, dtype=np.int8), block_lengths


def location_avg_rmse(df: pd.DataFrame) -> float:
    scores = []
    for _, group in df.groupby("county_id", sort=False):
        if group.empty:
            continue
        rms = math.sqrt(mean_squared_error(group["outage_truth"], group["outage_pred"]))
        scores.append(rms)
    return float(np.mean(scores)) if scores else float("nan")


def write_report(
    path: Path,
    dataset_name: str,
    lookback_steps: int,
    lookahead_steps: int,
    steps_per_day: int,
    block_lengths: List[int],
    metrics: Dict[str, float],
    df: pd.DataFrame,
) -> None:
    if df.empty:
        return
    abs_err = np.abs(df["outage_truth"] - df["outage_pred"])
    block_count = len(block_lengths)
    full_blocks = sum(1 for length in block_lengths if length == lookahead_steps)
    partial_blocks = block_count - full_blocks
    lines = [
        f"Dataset: {dataset_name.title()}",
        f"Lookback window: {lookback_steps} steps (~{lookback_steps / steps_per_day:.2f} days)",
        f"Lookahead horizon T: {lookahead_steps} steps (~{lookahead_steps / steps_per_day:.2f} days)",
        f"Forecast blocks: {block_count} (full: {full_blocks}, partial: {partial_blocks})",
        f"Predictions: {len(df)} samples across {df['county_id'].nunique()} counties and {df['timestamp'].nunique()} timesteps",
        (
            f"MSE {metrics['mse']:.2f} | MAE {metrics['mae']:.2f} | RMSE {metrics['rmse']:.2f} | "
            f"Loc-avg RMSE {metrics['loc_rmse']:.2f} | Zero-baseline RMSE {metrics['zero_rmse']:.2f}"
        ),
        f"Median | 90th pct absolute error: {np.median(abs_err):.2f} | {np.percentile(abs_err, 90):.2f}",
        f"Average outage truth/pred: {df['outage_truth'].mean():.2f}/{df['outage_pred'].mean():.2f}",
        f"Test window: {df['timestamp'].min()} → {df['timestamp'].max()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def save_calibration_plot(
    y_occ: np.ndarray,
    raw_probs: np.ndarray,
    calibrator: Optional[IsotonicRegression],
    output_path: Path,
    dataset_name: str,
) -> None:
    if raw_probs.size == 0:
        print("No samples for calibration plot; skipping.")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    if calibrator is not None:
        unique = np.unique(np.clip(raw_probs, 0.0, 1.0))
        grid = np.concatenate(([0.0], unique, [1.0]))
        fitted = calibrator.transform(grid)
        ax.step(grid, fitted, where="post", linewidth=2.0, label="Isotonic fit")
    else:
        frac_pos, mean_pred = calibration_curve(y_occ, raw_probs, n_bins=20, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label="Empirical")
    ax.scatter(
        raw_probs,
        y_occ,
        s=8,
        alpha=0.15,
        color="tab:orange",
        label="Samples",
    )
    ax.set_xlabel("Predicted non-zero probability (raw)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration Curve ({dataset_name.title()})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_test_series(
    df: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
) -> None:
    counties = sorted(df["county_id"].unique())
    if not counties:
        print("No test results to plot.")
        return
    ncols = 4
    nrows = math.ceil(len(counties) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.5), sharex=False)
    axes = axes.flatten()
    for ax in axes[len(counties) :]:
        ax.axis("off")
    for idx, county in enumerate(counties):
        ax = axes[idx]
        subset = df[df["county_id"] == county]
        ax.plot(subset["timestamp"], subset["outage_truth"], label="Actual", linewidth=1.0)
        ax.plot(subset["timestamp"], subset["outage_pred"], label="Forecast", linewidth=1.0)
        ax.set_title(county, fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Test Outage Forecasts ({dataset_name.title()})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- main ----------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage outage forecaster with new split rules.")
    parser.add_argument("--dataset", choices=DATASET_CONFIG.keys(), default="michigan")
    parser.add_argument("--nc-path", type=Path, default=None, help="Override path to the NetCDF file.")
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "two_stage")
    parser.add_argument("--lookback-days", type=float, default=2.0)
    parser.add_argument("--lookahead-days", type=float, default=1.0)
    parser.add_argument("--m-val", type=int, default=None, help="Override number of validation horizons.")
    parser.add_argument("--m-test", type=int, default=None, help="Override number of test horizons.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--use-weather", type=Path, default=None, help="CSV with weather forecasts; default uses ground truth.")
    parser.add_argument("--plot-results", action="store_true", help="Generate per-county test plots PDF.")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Number of model replicas to train and ensemble.")
    parser.add_argument("--disable-calibration", action="store_true", help="Skip isotonic probability calibration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    nc_path = args.nc_path or (Path(__file__).resolve().parent / cfg["nc_file"])
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    results_dir = args.results_dir if args.results_dir.is_absolute() else Path(__file__).resolve().parent / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(nc_path) as ds:
        outages = ds["out"].transpose("timestamp", "location").values.astype(np.float32)
        weather = ds["weather"].transpose("timestamp", "location", "feature").values.astype(np.float32)
        timestamps_raw = ds["timestamp"].values
        counties = decode_coord(ds["location"].values)
        feature_names = decode_coord(ds["feature"].values)

    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    common_present = [name for name in COMMON_WEATHER_FEATURES if name in feature_to_idx]
    if not common_present:
        raise ValueError("No common weather features were found in the NetCDF file.")
    indices = [feature_to_idx[name] for name in common_present]
    weather = weather[:, :, indices]
    feature_names = common_present

    timestamps = pd.to_datetime(timestamps_raw)
    steps_per_day = compute_steps_per_day(timestamps_raw)
    lookback_steps = min(int(round(args.lookback_days * steps_per_day)), MAX_LOOKBACK_POINTS)
    assert lookback_steps >= 1, "Lookback must be at least one timestep."
    lookahead_steps = int(round(args.lookahead_days * steps_per_day))
    if lookback_steps < 2 or lookahead_steps < 1:
        raise ValueError("Lookback must be >= 2 steps and lookahead >= 1 step.")

    m_val = args.m_val if args.m_val is not None else cfg["m_val"]
    m_test = args.m_test if args.m_test is not None else cfg["m_test"]
    plan = compute_split_plan(len(timestamps), lookback_steps, lookahead_steps, m_val, m_test)
    train_span = plan["train"][1] - max(plan["train"][0], lookback_steps)
    val_span = plan["val_eval"][1] - plan["val_eval"][0]
    test_span = plan["test_eval"][1] - plan["test_eval"][0]
    print(
        f"[Setup] Dataset={args.dataset} | Lookback={lookback_steps} steps | "
        f"Lookahead={lookahead_steps} steps | spans (train/val/test)={train_span}/{val_span}/{test_span}"
    )

    overrides = {}
    if args.use_weather:
        overrides = load_weather_override(args.use_weather, feature_names, timestamps, counties)
        if overrides:
            print(f"Loaded {len(overrides)} weather overrides from {args.use_weather}.")
    weather_used = apply_weather_overrides(weather, overrides)

    lag_list = build_lag_list(lookback_steps)
    hazard_idx = select_hazard_features(outages, weather_used, plan["train"], feature_names)
    hazard_names = [feature_names[idx] for idx in hazard_idx]
    print(f"[Setup] Selected {len(hazard_idx)} weather hazards from {len(feature_names)} total features.")
    trig_cache = trig_features(timestamps)
    county_indices = np.arange(len(counties), dtype=np.float32)

    train_mask = np.zeros(len(timestamps), dtype=bool)
    train_mask[max(lookback_steps, plan["train"][0]) : plan["train"][1]] = True
    val_mask = np.zeros(len(timestamps), dtype=bool)
    val_mask[plan["val_eval"][0] : plan["val_eval"][1]] = True
    test_mask = np.zeros(len(timestamps), dtype=bool)
    test_mask[plan["test_eval"][0] : plan["test_eval"][1]] = True

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train, y_train, occ_train, _ = build_split_arrays(
        outages,
        weather_used,
        train_idx,
        lag_list,
        hazard_idx,
        hazard_names,
        trig_cache,
        county_indices,
        lookback_steps,
        return_meta=False,
    )
    X_val, y_val, occ_val, _ = build_split_arrays(
        outages,
        weather_used,
        val_idx,
        lag_list,
        hazard_idx,
        hazard_names,
        trig_cache,
        county_indices,
        lookback_steps,
        return_meta=False,
    )

    if len(X_train) == 0 or len(X_val) == 0 or len(test_idx) == 0:
        raise RuntimeError("One of the splits produced zero samples. Check split parameters.")

    cat_index = X_train.shape[1] - 1  # county index is appended last
    y_train_occ = occ_train.astype(np.int8)
    y_val_occ = occ_val.astype(np.int8)

    print(
        f"[Training] Samples — train={len(X_train):,} | val={len(X_val):,} | features={X_train.shape[1]}"
    )

    ensemble_size = max(1, args.ensemble_size)
    use_calibration = not args.disable_calibration
    ensemble_preds: List[np.ndarray] = []
    ensemble_probs: List[np.ndarray] = []
    ensemble_raw_probs: List[np.ndarray] = []
    base_df: Optional[pd.DataFrame] = None
    test_occ_array: Optional[np.ndarray] = None
    block_lengths: Optional[List[int]] = None
    for member in range(ensemble_size):
        member_seed = args.seed + member
        print(f"[Training] Ensemble member {member + 1}/{ensemble_size} (seed={member_seed})")
        classifier, calibrator = train_classifier(
            X_train,
            y_train_occ,
            X_val,
            y_val_occ,
            cat_index,
            member_seed,
            args.device,
            use_calibration,
        )
        regressor = train_regressor(
            X_train, y_train, X_val, y_val, cat_index, member_seed, args.device
        )

        raw_val_probs = classifier.predict_proba(X_val, raw_score=False)[:, 1]
        val_probs = calibrate_probabilities(raw_val_probs, calibrator)
        val_intensity = regressor.predict(X_val)
        _ = np.clip(val_probs * val_intensity, 0.0, None)  # keep path for future diagnostics

        df_member, raw_test_probs_member, occ_member, block_lengths_member = rolling_forecast_twostage(
            classifier,
            calibrator,
            regressor,
            outages,
            weather_used,
            test_idx,
            lookahead_steps,
            lookback_steps,
            lag_list,
            hazard_idx,
            hazard_names,
            trig_cache,
            county_indices,
            timestamps,
            counties,
        )
        if df_member.empty:
            raise RuntimeError("Rolling forecast produced no samples.")
        if base_df is None:
            base_df = df_member[["timestamp", "county_id", "outage_truth"]].copy()
            base_df["outage_pred"] = 0.0
            base_df["p_nonzero"] = 0.0
            test_occ_array = occ_member
            block_lengths = block_lengths_member
        else:
            if not df_member[["timestamp", "county_id"]].equals(base_df[["timestamp", "county_id"]]):
                raise RuntimeError("Ensemble members produced misaligned forecasts.")
        ensemble_preds.append(df_member["outage_pred"].to_numpy())
        ensemble_probs.append(df_member["p_nonzero"].to_numpy())
        ensemble_raw_probs.append(raw_test_probs_member)

    assert base_df is not None and test_occ_array is not None and block_lengths is not None
    pred_mean = np.mean(ensemble_preds, axis=0)
    prob_mean = np.mean(ensemble_probs, axis=0)
    raw_test_probs = np.mean(ensemble_raw_probs, axis=0)
    base_df["outage_pred"] = pred_mean
    base_df["p_nonzero"] = prob_mean
    test_df = base_df

    y_true = test_df["outage_truth"].to_numpy()
    y_pred = test_df["outage_pred"].to_numpy()
    test_mse = mean_squared_error(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_rmse = math.sqrt(test_mse)
    zero_rmse = math.sqrt(mean_squared_error(y_true, np.zeros_like(y_true)))
    loc_rmse = location_avg_rmse(test_df)
    print(f"[{args.dataset}] Test MSE={test_mse:.3f} | MAE={test_mae:.3f} | RMSE={test_rmse:.3f}")
    print(f"[{args.dataset}] Zero baseline RMSE={zero_rmse:.3f}")
    print(f"[{args.dataset}] Location-averaged RMSE={loc_rmse:.3f}")

    forecast_path = results_dir / f"{args.dataset}_test_forecasts.csv"
    test_df[["timestamp", "county_id", "outage_truth", "outage_pred"]].to_csv(forecast_path, index=False)
    print(f"Saved test forecasts to {forecast_path}")

    calib_path = results_dir / f"{args.dataset}_calibration.pdf"
    save_calibration_plot(test_occ_array, raw_test_probs, calibrator, calib_path, args.dataset)
    print(f"Saved calibration curve to {calib_path}")

    if args.plot_results:
        plot_path = results_dir / f"{args.dataset}_test_plots.pdf"
        plot_test_series(test_df, plot_path, args.dataset)
        print(f"Saved per-county plots to {plot_path}")

    metrics_summary = {
        "mse": test_mse,
        "mae": test_mae,
        "rmse": test_rmse,
        "loc_rmse": loc_rmse,
        "zero_rmse": zero_rmse,
    }
    report_path = results_dir / f"{args.dataset}_report.txt"
    write_report(report_path, args.dataset, lookback_steps, lookahead_steps, steps_per_day, block_lengths, metrics_summary, test_df)
    print(f"Saved experiment report to {report_path}")


if __name__ == "__main__":
    main()

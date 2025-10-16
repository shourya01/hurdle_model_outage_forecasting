import argparse
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import warnings
import random
import xarray as xr
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm

# NOTES : GOOD SEEDS
# 48 hours: "21" [49.2538], 24 HOURS: "21" [22.2387] (LB 48)
# 48 hours: "11" [49.0495], 24 HOURS: "20" [21.8561] (LB 72)

warnings.filterwarnings(
    "ignore",
    message="No further splits with positive gain",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="LightGBM warning",
    category=UserWarning,
)
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", category=PerformanceWarning)

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError(
        "LightGBM is required for train_zif.py. Install it with `pip install lightgbm`."
    ) from exc


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_PATH = BASE_DIR / "train.nc"
DEFAULT_RESULTS_SUBDIR = Path("results") / "outage"
DEFAULT_WEATHER_RESULTS_SUBDIR = Path("results") / "weather"
DEFAULT_WEATHER_OUTPUT_NAME = "weather_forecast.csv"
EXP_NAME = "zeroinflated_outage"
CACHE_VERSION = 2
DEFAULT_LOOKBACK = 48
DEFAULT_LOOKAHEAD = 24
DEFAULT_DEVICE_TYPE = "cpu"
EXP_STRING = ""

POST_24_LAG_SPACING = 1  # dense variant: use every hour beyond 24 for lag features
LAG_LIST: List[int] = []  # initialized after parsing lookback
ROLL_SUM_WINDOWS: List[int] = []  # dense mode: drop outage rolling sums
ROLL_MAX_WINDOWS: List[int] = []  # dense mode: drop outage rolling maxima
WEATHER_ROLL_MAX_WINDOWS: List[int] = []  # dense mode: keep only raw hazard lags
WEATHER_ROLL_SUM_WINDOWS: List[int] = []
REQUIRED_HISTORY = 24
MAX_HAZARD_FEATURES = 24
CLIP_QUANTILES = (0.005, 0.995)
MIN_STD = 1e-6
BASELINE_DECAY = 0.85
WEATHER_MODEL_LAGS = [1, 2, 3, 6, 12, 24]
FORECAST_PLOT_PAST = 100

OutageHistory = Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]
WeatherHistory = Tuple[pd.DatetimeIndex, List[str], Dict[str, Dict[str, pd.Series]]]

HAZARD_NAME_PREFERENCE = [
    "sde",
    "sdwe",
    "gust",
    "fricv",
    "lai",
    "hail_1",
    "hail",
    "refc",
    "max_10si",
    "unknown_3",
    "unknown",
    "cnwat",
    "gh_3",
    "ustm",
    "snowc",
    "refd_1",
    "veril",
    "tcc_1",
    "refd",
    "lcc",
    "gh_5",
    "tcc",
    "sdlwrf",
    "pcdb",
    "sbt114",
    "veg",
    "unknown_5",
    "sbt124",
    "mcc",
    "gh",
    "pwat",
    "vstm",
    "pres",
    "lsm",
    "hcc",
    "gh_4",
    "vis",
    "orog",
    "cape",
]

HAZARD_KEYWORDS = (
    "gust",
    "hail",
    "ltng",
    "rain",
    "snow",
    "ice",
    "frzr",
    "prate",
    "tp",
    "pres",
    "temp",
    "t2",
    "d2",
    "sh",
    "wind",
    "u10",
    "v10",
    "wz",
    "cape",
    "ref",
    "vis",
    "tcoli",
    "tcolw",
)


def sanitize_feature_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    safe = safe.strip("_")
    return safe or "feature"


def make_unique(names: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for name in names:
        count = seen.get(name, 0)
        if count > 0:
            result.append(f"{name}_{count}")
        else:
            result.append(name)
        seen[name] = count + 1
    return result


def decode_char_array(values: np.ndarray) -> List[str]:
    decoded = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            decoded.append(item.decode("utf-8").strip())
        elif isinstance(item, str):
            decoded.append(item.strip())
        else:
            decoded.append("".join(ch.decode("utf-8") if isinstance(ch, (bytes, np.bytes_)) else str(ch) for ch in item).strip())
    return decoded


def to_datetime_index(hours_since_start: np.ndarray, start_ts: str) -> pd.DatetimeIndex:
    arr = np.asarray(hours_since_start)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr)
    base = pd.Timestamp(start_ts)
    arr = arr.astype(float)
    return base + pd.to_timedelta(arr, unit="h")


def load_long_dataframe(nc_path: str) -> Tuple[pd.DataFrame, pd.DatetimeIndex, List[str], List[str]]:
    with xr.open_dataset(nc_path) as ds:
        outages = ds["out"].values.astype(float)  # (location, timestamp)
        weather = ds["weather"].values.astype(float)  # (location, timestamp, feature)
        locations_raw = ds["location"].values
        timestamps_hours = ds["timestamp"].values
        feature_raw = ds["feature"].values
        start_ts = ds.attrs.get("time_start", "2022-01-01T00:00:00")

    outages = np.transpose(outages, (1, 0))  # (timestamp, location)
    weather = np.transpose(weather, (1, 0, 2))  # (timestamp, location, feature)

    timestamps = to_datetime_index(timestamps_hours, start_ts)
    county_ids = [str(x.decode("utf-8")) if isinstance(x, (bytes, np.bytes_)) else str(x) for x in np.array(locations_raw)]
    feature_names = decode_char_array(feature_raw)
    sanitized = [sanitize_feature_name(name) for name in feature_names]
    sanitized = make_unique(sanitized)
    weather_cols = [f"wx_{name}" for name in sanitized]

    multi_index = pd.MultiIndex.from_product([timestamps, county_ids], names=["ts", "county_id"])
    df = pd.DataFrame(index=multi_index).reset_index()
    df["outages"] = outages.reshape(-1)

    for idx, col in enumerate(weather_cols):
        df[col] = weather[:, :, idx].reshape(-1)

    return df, timestamps, county_ids, weather_cols


def assign_splits(
    df: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    lookback: int,
    lookahead: int,
    val_multiplier: int,
) -> Dict[str, List[pd.Timestamp]]:
    unique_ts = pd.Index(sorted(timestamps.unique()))
    holdout = max(48 - lookahead, 0)
    test_len = lookback + lookahead
    val_len = lookback + val_multiplier * lookahead
    train_len = len(unique_ts) - (val_len + test_len) - holdout
    if train_len <= REQUIRED_HISTORY:
        raise ValueError("Not enough history to satisfy look-back requirements.")

    split_map: Dict[pd.Timestamp, str] = {}
    train_ts = list(unique_ts[:train_len])
    val_ts = list(unique_ts[train_len:train_len + val_len])
    test_ts = list(unique_ts[train_len + val_len:train_len + val_len + test_len])
    holdout_ts = list(unique_ts[train_len + val_len + test_len:train_len + val_len + test_len + holdout])

    for ts in train_ts:
        split_map[ts] = "train"
    for ts in val_ts:
        split_map[ts] = "val"
    for ts in test_ts:
        split_map[ts] = "test"
    for ts in holdout_ts:
        split_map[ts] = "holdout"

    df["split"] = df["ts"].map(split_map)
    return {
        "train": train_ts,
        "val": val_ts,
        "test": test_ts,
        "holdout": holdout_ts,
    }


def select_hazard_columns(df: pd.DataFrame, weather_cols: List[str]) -> List[str]:
    available = set(weather_cols)
    preferred = [f"wx_{sanitize_feature_name(name)}" for name in HAZARD_NAME_PREFERENCE if f"wx_{sanitize_feature_name(name)}" in available]
    extra = [
        col
        for col in weather_cols
        if col not in preferred and any(keyword in col for keyword in HAZARD_KEYWORDS)
    ]
    if len(preferred) < MAX_HAZARD_FEATURES:
        preferred.extend(extra)
    if len(preferred) < MAX_HAZARD_FEATURES:
        preferred.extend([col for col in weather_cols if col not in preferred])

    train_mask = df["split"] == "train"
    if not np.any(train_mask):
        raise ValueError("Training split is empty; check data boundaries.")

    ranked = (
        df.loc[train_mask, preferred]
        .var(skipna=True)
        .sort_values(ascending=False)
        .index.tolist()
    )
    return ranked[:MAX_HAZARD_FEATURES]


def build_feature_matrix(df: pd.DataFrame, hazard_cols: List[str]) -> pd.DataFrame:
    feature_frames: List[pd.DataFrame] = []
    for county, group in df.sort_values(["county_id", "ts"]).groupby("county_id", sort=False):
        out_history: deque = deque(maxlen=REQUIRED_HISTORY)
        run_length: int | None = None
        weather_histories: Dict[str, deque] = {col: deque(maxlen=REQUIRED_HISTORY) for col in hazard_cols}
        records: Dict[int, Dict[str, float]] = {}

        for idx, row in group.iterrows():
            features: Dict[str, float] = {}
            hist = list(out_history)
            hist_len = len(hist)
            features["_history_len"] = float(hist_len)

            for lag in LAG_LIST:
                features[f"out_lag_{lag}"] = hist[-lag] if hist_len >= lag else np.nan
            for window in ROLL_SUM_WINDOWS:
                values = hist[-window:]
                features[f"out_roll_sum_{window}"] = float(np.sum(values)) if values else np.nan
            for window in ROLL_MAX_WINDOWS:
                values = hist[-window:]
                features[f"out_roll_max_{window}"] = float(np.max(values)) if values else np.nan

            features["out_run_length_since_nonzero"] = float(run_length) if run_length is not None else np.nan

            for col in hazard_cols:
                w_hist = weather_histories[col]
                w_list = list(w_hist)
                w_len = len(w_list)
                features[f"{col}_lag1"] = w_list[-1] if w_len >= 1 else np.nan
                if w_len >= 2:
                    features[f"{col}_delta_1"] = w_list[-1] - w_list[-2]
                elif w_len == 1:
                    features[f"{col}_delta_1"] = 0.0
                else:
                    features[f"{col}_delta_1"] = np.nan
                for window in WEATHER_ROLL_MAX_WINDOWS:
                    values = w_list[-window:]
                    features[f"{col}_roll_max_{window}"] = float(np.nanmax(values)) if values else np.nan
                for window in WEATHER_ROLL_SUM_WINDOWS:
                    values = w_list[-window:]
                    features[f"{col}_roll_sum_{window}"] = float(np.nansum(values)) if values else np.nan

            ts = row["ts"]
            hour = ts.hour
            dow = ts.dayofweek
            features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
            features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
            features["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
            features["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)

            records[idx] = features

            out_val = row["outages"]
            out_val = 0.0 if pd.isna(out_val) else float(out_val)
            out_history.append(out_val)
            if out_val > 0:
                run_length = 0
            else:
                run_length = 1 if run_length is None else run_length + 1

            for col in hazard_cols:
                w_val = row[col]
                w_val = float(w_val) if not pd.isna(w_val) else np.nan
                weather_histories[col].append(w_val)

        county_frame = pd.DataFrame.from_dict(records, orient="index")
        feature_frames.append(county_frame)

    features = pd.concat(feature_frames, axis=0).sort_index()
    return features


def last_nonzero_decay_baseline(df: pd.DataFrame, decay: float = BASELINE_DECAY) -> np.ndarray:
    preds: Dict[int, float] = {}
    carry: Dict[str, float] = {}
    for idx, row in df.sort_values(["county_id", "ts"]).iterrows():
        county = row["county_id"]
        prev = carry.get(county, 0.0)
        preds[idx] = prev
        actual = row["outages"]
        if actual > 0:
            carry[county] = float(actual)
        else:
            carry[county] = prev * decay
    ordered = df.index.to_list()
    return np.array([preds[idx] for idx in ordered])


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    metrics = {
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "roc_auc": float("nan"),
        "avg_precision": float("nan"),
        "n_pos": int(np.sum(y_true)),
        "n_total": int(len(y_true)),
    }
    if len(np.unique(y_true)) < 2:
        return metrics
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, probs))
    except ValueError:
        metrics["avg_precision"] = float("nan")
    return metrics


def build_dynamic_lag_list(lookback: int) -> List[int]:
    if lookback < 1:
        raise ValueError("Lookback must be at least 1 hour for dense lag generation.")
    return list(range(1, lookback + 1))


def update_required_history(lags: List[int]) -> None:
    global REQUIRED_HISTORY
    candidates: List[int] = []
    if lags:
        candidates.append(max(lags))
    if ROLL_SUM_WINDOWS:
        candidates.append(max(ROLL_SUM_WINDOWS))
    if ROLL_MAX_WINDOWS:
        candidates.append(max(ROLL_MAX_WINDOWS))
    if not candidates:
        candidates.append(REQUIRED_HISTORY)
    REQUIRED_HISTORY = max(candidates)


def mean_group_metric(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    group_col: str,
    metric_fn,
) -> float:
    values: List[float] = []
    for _, group in df.groupby(group_col):
        if group.empty:
            continue
        values.append(float(metric_fn(group[true_col], group[pred_col])))
    return float(np.mean(values)) if values else float("nan")


def rmse_metric(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def get_outage_history_length() -> int:
    lengths: List[int] = [REQUIRED_HISTORY]
    lengths.extend(LAG_LIST)
    lengths.extend(ROLL_SUM_WINDOWS)
    lengths.extend(ROLL_MAX_WINDOWS)
    if not lengths:
        return 1
    return max(max(lengths), 1)


def get_weather_history_length() -> int:
    lengths: List[int] = list(WEATHER_MODEL_LAGS)
    lengths.extend(WEATHER_ROLL_MAX_WINDOWS)
    lengths.extend(WEATHER_ROLL_SUM_WINDOWS)
    if not lengths:
        return 1
    return max(max(lengths), 1)


def prepare_weather_history(
    weather_df: pd.DataFrame,
    hazard_cols: List[str],
    medians: Dict[str, float],
    history_len: int,
) -> Dict[str, Dict[str, List[float]]]:
    history_seed: Dict[str, Dict[str, List[float]]] = {}
    for county, group in weather_df.groupby("county_id", sort=False):
        county_history: Dict[str, List[float]] = {}
        ordered = group.sort_values("ts")
        for hazard in hazard_cols:
            values = ordered[hazard].astype(float).tolist()
            tail = values[-history_len:]
            cleaned = [medians.get(hazard, 0.0) if pd.isna(val) else float(val) for val in tail]
            pad_value = medians.get(hazard, 0.0)
            while len(cleaned) < history_len:
                cleaned.insert(0, pad_value)
            county_history[hazard] = cleaned
        history_seed[county] = county_history
    return history_seed


def train_weather_forecasters(
    weather_df: pd.DataFrame,
    hazard_cols: List[str],
    random_state: int,
    device_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    history_len = get_weather_history_length()

    weather_medians: Dict[str, float] = {}
    for hazard in hazard_cols:
        if hazard in weather_df.columns:
            train_values = weather_df.loc[weather_df["split"] == "train", hazard]
            weather_medians[hazard] = float(train_values.median(skipna=True)) if not train_values.empty else 0.0
        else:
            weather_medians[hazard] = 0.0

    weather_models: Dict[str, Dict[str, Any]] = {}
    feature_maps: Dict[str, List[str]] = {}

    progress = tqdm(hazard_cols, desc="Training weather models", unit="feature")
    for hazard in progress:
        cols = ["county_id", "county_idx", "ts", "split", hazard]
        if any(col not in weather_df.columns for col in cols):
            weather_models[hazard] = {"model": None, "feature_cols": []}
            feature_maps[hazard] = []
            progress.set_postfix({"feature": hazard, "mse": "nan"})
            continue

        series_df = weather_df[cols].copy()
        series_df.sort_values(["county_id", "ts"], inplace=True)

        train_values = series_df.loc[series_df["split"] == "train", hazard].astype(float)
        finite_mask = np.isfinite(train_values)
        if finite_mask.any():
            non_nan = train_values[finite_mask]
            zero_ratio = float(np.mean(np.isclose(non_nan, 0.0))) if len(non_nan) else float("nan")
        else:
            zero_ratio = 1.0

        for lag in WEATHER_MODEL_LAGS:
            series_df[f"lag_{lag}"] = series_df.groupby("county_id")[hazard].shift(lag)

        ts_hours = series_df["ts"].dt.hour.astype(float)
        ts_dow = series_df["ts"].dt.dayofweek.astype(float)
        series_df["hour_sin"] = np.sin(2.0 * math.pi * ts_hours / 24.0)
        series_df["hour_cos"] = np.cos(2.0 * math.pi * ts_hours / 24.0)
        series_df["dow_sin"] = np.sin(2.0 * math.pi * ts_dow / 7.0)
        series_df["dow_cos"] = np.cos(2.0 * math.pi * ts_dow / 7.0)

        lag_features = [f"lag_{lag}" for lag in WEATHER_MODEL_LAGS]
        required_cols = [hazard, *lag_features]
        series_df.dropna(subset=required_cols, inplace=True)
        if series_df.empty:
            weather_models[hazard] = {"model": None, "feature_cols": []}
            feature_maps[hazard] = []
            progress.set_postfix({"feature": hazard, "mse": "nan"})
            continue

        feature_cols = lag_features + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "county_idx"]
        train_mask = series_df["split"].isin(["train", "val"])
        test_mask = series_df["split"] == "test"

        if not train_mask.any():
            weather_models[hazard] = {"model": None, "feature_cols": feature_cols}
            feature_maps[hazard] = feature_cols
            progress.set_postfix({"feature": hazard, "mse": "nan"})
            continue

        X_train = series_df.loc[train_mask, feature_cols]
        y_train = series_df.loc[train_mask, hazard].astype(float).to_numpy()

        if X_train.empty:
            weather_models[hazard] = {"model": None, "feature_cols": feature_cols}
            feature_maps[hazard] = feature_cols
            progress.set_postfix({"feature": hazard, "mse": "nan"})
            continue

        use_tweedie = not math.isnan(zero_ratio) and zero_ratio > 0.5

        model_params: Dict[str, Any] = {
            "learning_rate": 0.1,
            "num_leaves": 64,
            "max_depth": 8,
            "min_child_samples": 50,
            "min_sum_hessian_in_leaf": 5.0,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "feature_fraction_bynode": 0.8,
            "n_estimators": 800,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "random_state": random_state,
            "verbosity": -1,
        }
        if use_tweedie:
            model_params.update({
                "objective": "tweedie",
                "tweedie_variance_power": 1.55,
            })
        else:
            model_params["objective"] = "regression"

        if device_params and device_params.get("device_type") == "gpu":
            model_params["device_type"] = "gpu"
            if "gpu_device_id" in device_params:
                model_params["gpu_device_id"] = device_params["gpu_device_id"]

        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, y_train)

        mse_value = float("nan")
        if test_mask.any():
            X_test = series_df.loc[test_mask, feature_cols]
            if not X_test.empty:
                y_test = series_df.loc[test_mask, hazard].astype(float).to_numpy()
                preds = model.predict(X_test)
                mse_value = float(mean_squared_error(y_test, preds))

        train_median = series_df.loc[train_mask, hazard].median()
        if not np.isnan(train_median):
            weather_medians[hazard] = float(train_median)

        weather_models[hazard] = {"model": model, "feature_cols": feature_cols}
        feature_maps[hazard] = feature_cols
        progress.set_postfix({"feature": hazard, "mse": f"{mse_value:.4f}" if not math.isnan(mse_value) else "nan"})

    history_seed = prepare_weather_history(weather_df, hazard_cols, weather_medians, history_len)

    return {
        "models": weather_models,
        "medians": weather_medians,
        "feature_cols": feature_maps,
        "lags": WEATHER_MODEL_LAGS,
        "history_seed": history_seed,
        "history_len": history_len,
    }


def predict_counts(
    classifier: lgb.LGBMClassifier,
    calibrator: IsotonicRegression | None,
    regressor: lgb.LGBMRegressor,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = classifier.predict_proba(X)[:, 1]
    if calibrator is not None:
        probs = np.asarray(calibrator.transform(probs))
    intensity = regressor.predict(X)
    intensity = np.maximum(intensity, 0.0)
    expected = np.maximum(probs * intensity, 0.0)
    return probs, intensity, expected


def forecast_from_last_window(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_features: List[str],
    classifier: lgb.LGBMClassifier,
    calibrator: IsotonicRegression | None,
    regressor: lgb.LGBMRegressor,
    county_to_idx: Dict[str, int],
    medians: Dict[str, float],
    lookahead: int,
    hazard_cols: List[str],
    weather_artifacts: Dict[str, Any],
    external_weather_map: Optional[Dict[str, Dict[str, pd.Series]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    category_dtype = pd.api.types.CategoricalDtype(categories=list(county_to_idx.values()), ordered=False)
    last_timestamp = df["ts"].max()
    feature_set = feature_cols + cat_features

    hazard_models: Dict[str, Dict[str, Any]] = weather_artifacts.get("models", {})
    hazard_medians: Dict[str, float] = weather_artifacts.get("medians", {})
    hazard_feature_cols: Dict[str, List[str]] = weather_artifacts.get("feature_cols", {})
    weather_lags: List[int] = weather_artifacts.get("lags", WEATHER_MODEL_LAGS)
    history_len: int = weather_artifacts.get("history_len", get_weather_history_length())
    history_seed: Dict[str, Dict[str, List[float]]] = weather_artifacts.get("history_seed", {})

    county_frames: Dict[str, pd.DataFrame] = {}
    active_counties: List[str] = []
    for county in sorted(county_to_idx.keys()):
        county_frame = df[df["county_id"] == county].sort_values("ts")
        if county_frame.empty:
            continue
        county_frames[county] = county_frame
        active_counties.append(county)

    if not active_counties:
        return pd.DataFrame(), {hazard: pd.DataFrame(columns=["county_id", "timestamp", "prediction"]) for hazard in hazard_cols}

    weather_histories: Dict[str, Dict[str, deque]] = {}
    for county in active_counties:
        county_seed = history_seed.get(county, {})
        hazard_history: Dict[str, deque] = {}
        for hazard in hazard_cols:
            seed_values = county_seed.get(hazard, [])
            if not seed_values:
                default_val = float(hazard_medians.get(hazard, 0.0))
                seed_values = [default_val] * history_len
            while len(seed_values) < history_len:
                seed_values.insert(0, hazard_medians.get(hazard, 0.0))
            trimmed = [hazard_medians.get(hazard, 0.0) if pd.isna(val) else float(val) for val in seed_values[-history_len:]]
            hazard_history[hazard] = deque(trimmed, maxlen=history_len)
        weather_histories[county] = hazard_history

    outage_history_len = get_outage_history_length()
    outage_histories: Dict[str, deque] = {}
    run_lengths: Dict[str, float] = {}
    for county in active_counties:
        county_series = county_frames[county]["outages"].astype(float).tolist()
        history_vals = [0.0 if pd.isna(val) else float(val) for val in county_series[-outage_history_len:]]
        while len(history_vals) < outage_history_len:
            history_vals.insert(0, 0.0)
        outage_histories[county] = deque(history_vals, maxlen=outage_history_len)

        last_row = county_frames[county].iloc[-1]
        run_length_val = float(last_row.get("out_run_length_since_nonzero", float("nan")))
        if math.isnan(run_length_val):
            run_length_val = 0.0
            for val in reversed(outage_histories[county]):
                if val > 0:
                    break
                run_length_val += 1.0
        run_lengths[county] = run_length_val

    interaction_features = [col for col in feature_cols if col.startswith("int_out1_")]
    results: List[Dict[str, float]] = []
    weather_records: Dict[str, List[Dict[str, Any]]] = {hazard: [] for hazard in hazard_cols}

    for step in range(1, lookahead + 1):
        ts_future = last_timestamp + pd.Timedelta(hours=step)
        hour = ts_future.hour
        dow = ts_future.dayofweek
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        dow_sin = math.sin(2 * math.pi * dow / 7.0)
        dow_cos = math.cos(2 * math.pi * dow / 7.0)

        hazard_step_predictions: Dict[str, Dict[str, float]] = {}
        for hazard in hazard_cols:
            model_info = hazard_models.get(hazard, {})
            model = model_info.get("model") if isinstance(model_info, dict) else None
            feature_cols_hazard = model_info.get("feature_cols", []) if isinstance(model_info, dict) else []
            predictions_for_hazard: Dict[str, float] = {}

            if external_weather_map is not None and hazard in external_weather_map:
                hazard_forecast_map = external_weather_map[hazard]
                for county in active_counties:
                    county_series = hazard_forecast_map.get(county)
                    if county_series is not None and ts_future in county_series.index:
                        predictions_for_hazard[county] = float(county_series.loc[ts_future])
                    else:
                        history = weather_histories[county][hazard]
                        predictions_for_hazard[county] = float(history[-1]) if len(history) else float(hazard_medians.get(hazard, 0.0))
                hazard_step_predictions[hazard] = predictions_for_hazard
                continue

            if model is None:
                for county in active_counties:
                    history = weather_histories[county][hazard]
                    if len(history):
                        predictions_for_hazard[county] = float(history[-1])
                    else:
                        predictions_for_hazard[county] = float(hazard_medians.get(hazard, 0.0))
                hazard_step_predictions[hazard] = predictions_for_hazard
                continue

            feature_rows: List[Dict[str, float]] = []
            for county in active_counties:
                history = weather_histories[county][hazard]
                feature_row: Dict[str, float] = {}
                for lag in weather_lags:
                    if len(history) >= lag:
                        val = history[-lag]
                        if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
                            val = hazard_medians.get(hazard, 0.0)
                    else:
                        val = hazard_medians.get(hazard, 0.0)
                    feature_row[f"lag_{lag}"] = float(val)
                feature_row["hour_sin"] = hour_sin
                feature_row["hour_cos"] = hour_cos
                feature_row["dow_sin"] = dow_sin
                feature_row["dow_cos"] = dow_cos
                feature_row["county_idx"] = county_to_idx[county]
                feature_rows.append(feature_row)

            features_df = pd.DataFrame(feature_rows)
            for col in feature_cols_hazard:
                if col not in features_df.columns:
                    features_df[col] = hazard_medians.get(hazard, 0.0)
            if feature_cols_hazard:
                features_df = features_df[feature_cols_hazard]
            preds = model.predict(features_df)
            hazard_step_predictions[hazard] = {
                county: float(preds[idx]) for idx, county in enumerate(active_counties)
            }

        for county in active_counties:
            feature_values: Dict[str, float] = {}
            out_history = outage_histories[county]
            for lag in LAG_LIST:
                col_name = f"out_lag_{lag}"
                if len(out_history) >= lag:
                    feature_values[col_name] = float(out_history[-lag])
                else:
                    feature_values[col_name] = float(medians.get(col_name, 0.0))

            for window in ROLL_SUM_WINDOWS:
                col_name = f"out_roll_sum_{window}"
                if len(out_history) >= window:
                    recent = list(out_history)[-window:]
                    feature_values[col_name] = float(np.sum(recent))
                else:
                    feature_values[col_name] = float(medians.get(col_name, 0.0))

            for window in ROLL_MAX_WINDOWS:
                col_name = f"out_roll_max_{window}"
                if len(out_history) >= window:
                    recent = list(out_history)[-window:]
                    feature_values[col_name] = float(np.max(recent))
                else:
                    feature_values[col_name] = float(medians.get(col_name, 0.0))

            feature_values["out_run_length_since_nonzero"] = float(run_lengths[county])

            hazard_history = weather_histories[county]
            for hazard in hazard_cols:
                history = hazard_history[hazard]
                lag_key = f"{hazard}_lag1"
                if len(history):
                    feature_values[lag_key] = float(history[-1])
                else:
                    feature_values[lag_key] = float(hazard_medians.get(hazard, 0.0))

                delta_key = f"{hazard}_delta_1"
                if len(history) >= 2:
                    feature_values[delta_key] = float(history[-1] - history[-2])
                elif len(history) == 1:
                    feature_values[delta_key] = 0.0
                else:
                    feature_values[delta_key] = float(medians.get(delta_key, 0.0))

                for window in WEATHER_ROLL_MAX_WINDOWS:
                    roll_key = f"{hazard}_roll_max_{window}"
                    if len(history) >= window:
                        recent = list(history)[-window:]
                        feature_values[roll_key] = float(np.nanmax(recent))
                    else:
                        feature_values[roll_key] = float(hazard_medians.get(hazard, 0.0))

                for window in WEATHER_ROLL_SUM_WINDOWS:
                    roll_key = f"{hazard}_roll_sum_{window}"
                    if len(history) >= window:
                        recent = list(history)[-window:]
                        feature_values[roll_key] = float(np.nansum(recent))
                    else:
                        feature_values[roll_key] = float(medians.get(roll_key, 0.0))

            feature_values["hour_sin"] = hour_sin
            feature_values["hour_cos"] = hour_cos
            feature_values["dow_sin"] = dow_sin
            feature_values["dow_cos"] = dow_cos
            feature_values["county_te"] = float(medians.get("county_te", 0.0))

            out_lag1_val = feature_values.get("out_lag_1", float(medians.get("out_lag_1", 0.0)))
            for interaction_key in interaction_features:
                hazard_key = interaction_key.replace("int_out1_", "")
                hazard_lag_key = hazard_key
                hazard_lag_val = feature_values.get(hazard_lag_key)
                if hazard_lag_val is None or (
                    isinstance(hazard_lag_val, float) and math.isnan(hazard_lag_val)
                ):
                    hazard_lag_val = medians.get(hazard_lag_key, 0.0)
                feature_values[interaction_key] = float(out_lag1_val) * float(hazard_lag_val)

            row_feature_values: Dict[str, float] = {}
            for col in feature_cols:
                value = feature_values.get(col, medians.get(col, 0.0))
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    value = medians.get(col, 0.0)
                row_feature_values[col] = float(value)

            row_df = pd.DataFrame([row_feature_values])
            for cat_col in cat_features:
                if cat_col == "county_idx":
                    row_df[cat_col] = pd.Categorical([county_to_idx[county]], dtype=category_dtype)
                else:
                    cat_value = medians.get(cat_col, 0.0)
                    row_df[cat_col] = cat_value

            probs, intensity, expected = predict_counts(
                classifier,
                calibrator,
                regressor,
                row_df[feature_set],
            )

            expected_val = max(float(expected[0]), 0.0)
            results.append(
                {
                    "county_id": county,
                    "timestamp": ts_future,
                    "p_occ": float(probs[0]),
                    "intensity": float(intensity[0]),
                    "prediction": expected_val,
                }
            )

            outage_histories[county].append(expected_val)
            if float(probs[0]) >= 0.5:
                run_lengths[county] = 0.0
            else:
                run_lengths[county] = run_lengths[county] + 1.0

        for hazard in hazard_cols:
            county_predictions = hazard_step_predictions.get(hazard, {})
            for county in active_counties:
                history = weather_histories[county][hazard]
                predicted_val = county_predictions.get(county)
                if predicted_val is None or (
                    isinstance(predicted_val, float) and math.isnan(predicted_val)
                ):
                    fallback = history[-1] if len(history) else hazard_medians.get(hazard, 0.0)
                    predicted_val = fallback
                predicted_val = float(predicted_val)
                weather_records[hazard].append(
                    {
                        "county_id": county,
                        "timestamp": ts_future,
                        "prediction": predicted_val,
                    }
                )
                history.append(predicted_val)

    outage_forecast_df = pd.DataFrame(results)
    weather_forecasts = {
        hazard: pd.DataFrame(records) if records else pd.DataFrame(columns=["county_id", "timestamp", "prediction"])
        for hazard, records in weather_records.items()
    }
    return outage_forecast_df, weather_forecasts


def determine_subplot_layout(num_series: int) -> Tuple[int, int]:
    if num_series <= 0:
        return 1, 1
    cols = max(int(math.ceil(math.sqrt(num_series))), 1)
    rows = int(math.ceil(num_series / cols))
    return rows, cols


def load_recent_outage_history(train_path: Path, plot_past: int) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]:
    with xr.open_dataset(train_path) as ds:
        outages = ds["out"].values
        locations_raw = ds["location"].values
        timestamps_raw = ds["timestamp"].values
        start_ts = ds.attrs.get("time_start", "2022-01-01T00:00:00")

    timestamps = to_datetime_index(np.asarray(timestamps_raw), start_ts)
    if plot_past <= 0:
        raise ValueError("plot_past must be positive")
    history_ts = timestamps[-plot_past:]

    county_ids = [str(loc.decode("utf-8")) if isinstance(loc, (bytes, np.bytes_)) else str(loc) for loc in np.asarray(locations_raw)]
    history: Dict[str, pd.Series] = {}
    for idx, county in enumerate(county_ids):
        series = outages[idx, -plot_past:]
        history[county] = pd.Series(series, index=history_ts)
    return history_ts, history


def load_recent_weather_history(
    train_path: Path,
    plot_past: int,
    hazard_cols: List[str],
) -> WeatherHistory:
    with xr.open_dataset(train_path) as ds:
        timestamps_raw = ds["timestamp"].values
        start_ts = ds.attrs.get("time_start", "2022-01-01T00:00:00")
        locations_raw = ds["location"].values
        features_raw = ds["feature"].values

        weather_da = ds["weather"].transpose("timestamp", "location", "feature")
        total_steps = weather_da.sizes["timestamp"]
        history_steps = max(1, min(int(plot_past), total_steps))
        weather_window = weather_da.isel(timestamp=slice(total_steps - history_steps, total_steps)).values

    timestamps = to_datetime_index(np.asarray(timestamps_raw), start_ts)
    history_ts = timestamps[-history_steps:]

    county_ids = [
        str(loc.decode("utf-8")) if isinstance(loc, (bytes, np.bytes_)) else str(loc)
        for loc in np.asarray(locations_raw)
    ]

    feature_names = decode_char_array(features_raw)
    sanitized = make_unique([sanitize_feature_name(name) for name in feature_names])
    weather_map = {f"wx_{sanitized[idx]}": idx for idx in range(len(sanitized))}

    hazard_history: Dict[str, Dict[str, pd.Series]] = {}
    for hazard in hazard_cols:
        feature_idx = weather_map.get(hazard)
        if feature_idx is None:
            continue
        feature_values = weather_window[:, :, feature_idx]
        county_series: Dict[str, pd.Series] = {}
        for loc_idx, county in enumerate(county_ids):
            series = pd.Series(feature_values[:, loc_idx], index=history_ts)
            county_series[county] = series
        hazard_history[hazard] = county_series

    return history_ts, county_ids, hazard_history


def load_external_weather_forecasts(
    csv_path: Path,
    expected_counties: List[str],
    hazard_cols: List[str],
    last_timestamp: pd.Timestamp,
    lookahead: int,
) -> Tuple[Dict[str, Dict[str, pd.Series]], Dict[str, pd.DataFrame]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"External weather forecast file not found: {csv_path}")

    forecast_df = pd.read_csv(csv_path)
    required_cols = {"timestamp", "county"}
    if not required_cols.issubset(forecast_df.columns):
        raise ValueError(
            f"Weather forecast CSV must contain columns {sorted(required_cols)}; found {forecast_df.columns.tolist()}"
        )

    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], utc=False)
    forecast_df["county"] = forecast_df["county"].astype(str)

    available_counties = set(forecast_df["county"].unique())
    missing_counties = set(expected_counties) - available_counties
    if missing_counties:
        raise ValueError(f"Weather forecast CSV missing locations: {sorted(missing_counties)}")
    extra_counties = available_counties - set(expected_counties)
    if extra_counties:
        raise ValueError(f"Weather forecast CSV has unexpected locations: {sorted(extra_counties)}")

    expected_timestamps = [last_timestamp + pd.Timedelta(hours=i) for i in range(1, lookahead + 1)]
    expected_timestamp_set = set(expected_timestamps)

    truncated_forecast: List[pd.DataFrame] = []
    for county in expected_counties:
        county_df = forecast_df[forecast_df["county"] == county].sort_values("timestamp")
        if county_df.empty:
            raise ValueError(f"Location {county} has no forecast rows")
        if len(county_df) < lookahead:
            raise ValueError(
                f"Location {county} has {len(county_df)} forecast rows; expected at least {lookahead}"
            )
        county_df = county_df.iloc[:lookahead].copy()
        ts_values = set(pd.to_datetime(county_df["timestamp"]))
        if ts_values != expected_timestamp_set:
            raise ValueError(
                f"Location {county} forecast timestamps mismatch expected horizon; expected {expected_timestamps}"
            )
        truncated_forecast.append(county_df)

    if truncated_forecast:
        forecast_df = pd.concat(truncated_forecast, axis=0, ignore_index=True)

    column_map: Dict[str, str] = {}
    for col in forecast_df.columns:
        if col in required_cols:
            continue
        sanitized = f"wx_{sanitize_feature_name(col)}"
        column_map[sanitized] = col

    missing_features = [haz for haz in hazard_cols if haz not in column_map]
    if missing_features:
        raise ValueError(
            "Weather forecast CSV missing hazard columns: " + ", ".join(missing_features)
        )

    forecast_lookup: Dict[str, Dict[str, pd.Series]] = {}
    forecast_frames: Dict[str, pd.DataFrame] = {}
    for hazard in hazard_cols:
        original_col = column_map[hazard]
        hazard_df = forecast_df[["county", "timestamp", original_col]].copy()
        hazard_df.rename(columns={"county": "county_id", original_col: "prediction"}, inplace=True)
        hazard_df.sort_values(["county_id", "timestamp"], inplace=True)
        forecast_frames[hazard] = hazard_df

        county_map: Dict[str, pd.Series] = {}
        for county, group in hazard_df.groupby("county_id", sort=False):
            county_map[county] = pd.Series(group["prediction"].to_numpy(), index=pd.to_datetime(group["timestamp"]))
        forecast_lookup[hazard] = county_map

    return forecast_lookup, forecast_frames


def plot_forecast_history(
    exp_name: str,
    forecast_df: pd.DataFrame,
    save_dir: Path,
    plot_past: int,
    history_data: OutageHistory | None = None,
) -> Optional[Path]:
    if forecast_df.empty:
        return None

    forecast_copy = forecast_df.copy()
    forecast_copy["timestamp"] = pd.to_datetime(forecast_copy["timestamp"])

    if history_data is None:
        history_data = load_recent_outage_history(DEFAULT_TRAIN_DATA_PATH, plot_past)
    _, history = history_data
    counties = sorted(history.keys())
    if not counties:
        return None

    rows, cols = determine_subplot_layout(len(counties))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=False)
    axes_flat = axes.reshape(-1)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for ax, county in zip(axes_flat, counties):
        hist_series = history[county]
        county_forecast = forecast_copy[forecast_copy["county_id"] == county].sort_values("timestamp")

        ax.plot(hist_series.index, hist_series.values, label="History", color="tab:blue")
        if not county_forecast.empty:
            ax.plot(
                county_forecast["timestamp"],
                county_forecast["prediction"],
                label="Forecast",
                color="tab:orange",
            )
        ax.set_title(f"County {county}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Outages")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    for ax in axes_flat[len(counties):]:
        ax.axis("off")

    axes_flat[0].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    plot_path = save_dir / f"{exp_name}_forecast_plots.pdf"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def chunk_list(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx:idx + chunk_size]


def plot_forecast_history_book(
    exp_name: str,
    forecast_df: pd.DataFrame,
    save_dir: Path,
    history_data: OutageHistory,
    chunk_size: int = 12,
) -> Optional[Path]:
    if forecast_df.empty:
        return None

    history_ts, history = history_data
    counties = sorted(history.keys())
    if not counties:
        return None

    forecast_copy = forecast_df.copy()
    forecast_copy["timestamp"] = pd.to_datetime(forecast_copy["timestamp"])

    plot_dir = save_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    book_path = plot_dir / f"{exp_name}_forecast_pages.pdf"

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    with PdfPages(book_path) as pdf:
        county_chunks = list(chunk_list(counties, chunk_size))
        for county_chunk in tqdm(
            county_chunks,
            desc="Saving forecast pages",
            unit="page",
            total=len(county_chunks),
        ):
            rows, cols = determine_subplot_layout(len(county_chunk))
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=False)
            axes_flat = axes.reshape(-1)

            for ax, county in zip(axes_flat, county_chunk):
                hist_series = history[county]
                county_forecast = forecast_copy[forecast_copy["county_id"] == county].sort_values("timestamp")

                ax.plot(hist_series.index, hist_series.values, label="History", color="tab:blue", linewidth=1.0)
                if not county_forecast.empty:
                    ax.plot(
                        county_forecast["timestamp"],
                        county_forecast["prediction"],
                        label="Forecast",
                        color="tab:orange",
                        linewidth=1.0,
                    )
                ax.set_title(f"County {county}")
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Outages")
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                ax.tick_params(axis="x", labelrotation=45)

            for ax in axes_flat[len(county_chunk):]:
                ax.axis("off")

            axes_flat[0].legend(fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return book_path


def plot_weather_forecasts_pdf(
    exp_name: str,
    weather_forecasts: Dict[str, pd.DataFrame],
    save_dir: Path,
    weather_history: WeatherHistory,
) -> Optional[Path]:
    if not weather_forecasts:
        return None

    _history_ts, county_ids, hazard_history = weather_history
    hazard_order = [hazard for hazard in weather_forecasts if not weather_forecasts[hazard].empty]
    if not hazard_order:
        return None

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    hazard_forecast_groups: Dict[str, Dict[str, pd.DataFrame]] = {}
    for hazard in hazard_order:
        df_forecast = weather_forecasts[hazard].copy()
        if df_forecast.empty:
            continue
        df_forecast["timestamp"] = pd.to_datetime(df_forecast["timestamp"])
        hazard_forecast_groups[hazard] = {
            county: group.sort_values("timestamp")
            for county, group in df_forecast.groupby("county_id", sort=False)
        }

    weather_pdf_path = save_dir / f"{exp_name}_weather_forecasts.pdf"
    with PdfPages(weather_pdf_path) as pdf:
        for hazard in tqdm(
            hazard_order,
            desc="Saving weather forecast pages",
            unit="feature",
        ):
            history_map = hazard_history.get(hazard)
            forecast_map = hazard_forecast_groups.get(hazard)
            if not history_map or not forecast_map:
                continue
            counties_for_hazard = [county for county in county_ids if county in history_map]
            if not counties_for_hazard:
                continue

            rows, cols = determine_subplot_layout(len(counties_for_hazard))
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False, sharex=False)
            axes_flat = axes.reshape(-1)
            for ax, county in zip(axes_flat, counties_for_hazard):
                hist_series = history_map.get(county)
                forecast_series_df = forecast_map.get(county)

                if hist_series is not None:
                    ax.plot(hist_series.index, hist_series.values, label="History", color="tab:blue", linewidth=1.0)
                if forecast_series_df is not None and not forecast_series_df.empty:
                    ax.plot(
                        forecast_series_df["timestamp"],
                        forecast_series_df["prediction"],
                        label="Forecast",
                        color="tab:orange",
                        linewidth=1.0,
                    )
                ax.set_title(f"{county}")
                ax.set_xlabel("Timestamp")
                ax.set_ylabel(hazard)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                ax.tick_params(axis="x", labelrotation=45)

            for ax in axes_flat[len(counties_for_hazard):]:
                ax.axis("off")

            axes_flat[0].legend(fontsize=8)
            fig.suptitle(f"{hazard} forecast")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return weather_pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "weather_csv",
        nargs="?",
        type=Path,
        help="Path to the weather forecast CSV produced by train_weather.py. Defaults to the weather results directory if omitted.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_TRAIN_DATA_PATH,
        help="Path to the train.nc NetCDF file (relative paths resolved against the script directory).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_SUBDIR,
        help="Directory for saving artifacts (relative paths resolved against the script directory).",
    )
    parser.add_argument("--lookahead", type=int, choices=[24, 48], default=DEFAULT_LOOKAHEAD)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--exp-string", type=str, default=None, help="Optional experiment suffix for saved outputs")
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE_TYPE,
        choices=["cpu", "cuda"],
        help="Execution device for LightGBM (cpu or cuda)",
    )
    parser.add_argument(
        "--linear-leaves",
        action="store_true",
        help="Enable LightGBM linear leaves for both classifier and regressor",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility",
    )
    parser.add_argument(
        "--val-multiplier",
        type=int,
        default=3,
        help="Multiplier applied to lookahead when sizing the validation window",
    )
    args = parser.parse_args()

    train_data_path = args.data_path
    if not train_data_path.is_absolute():
        train_data_path = (BASE_DIR / train_data_path).resolve()
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training NetCDF file not found: {train_data_path}")

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (BASE_DIR / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.weather_csv is None:
        weather_csv_path = BASE_DIR / DEFAULT_WEATHER_RESULTS_SUBDIR / DEFAULT_WEATHER_OUTPUT_NAME
    else:
        weather_csv_path = args.weather_csv
        if not weather_csv_path.is_absolute():
            weather_csv_path = (BASE_DIR / weather_csv_path).resolve()
    if not weather_csv_path.exists():
        raise FileNotFoundError(f"Weather forecast CSV not found: {weather_csv_path}")

    lookahead = int(args.lookahead)
    lookback = int(args.lookback)
    val_multiplier = max(1, int(args.val_multiplier))
    device_type = args.device.lower()
    if device_type == "cuda":
        device_params = {"device_type": "gpu", "gpu_device_id": 0}
    else:
        device_type = "cpu"
        device_params = {"device_type": "cpu"}
    linear_leaves = bool(args.linear_leaves)
    master_seed = int(args.seed)
    if args.exp_string is not None:
        exp_suffix_value = args.exp_string.strip()
    else:
        exp_suffix_value = EXP_STRING.strip()
    exp_suffix = f"_{exp_suffix_value}" if exp_suffix_value else ""

    lag_list = build_dynamic_lag_list(lookback)
    globals()["LAG_LIST"] = lag_list
    update_required_history(lag_list)

    exp_name = f"{EXP_NAME}_LB_{lookback}_LA_{lookahead}{exp_suffix}"

    random.seed(master_seed)
    np.random.seed(master_seed)

    save_dir = results_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = BASE_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = exp_suffix_value or "default"
    cache_file = cache_dir / f"cache_{lookback}_{lookahead}_{cache_key}.joblib"

    cache_payload: Optional[Dict[str, Any]] = None
    if cache_file.exists():
        cache_payload = joblib.load(cache_file)

    if cache_payload and cache_payload.get("cache_version") == CACHE_VERSION:
        df = cache_payload["df"].copy()
        timestamps = cache_payload["timestamps"]
        county_ids = cache_payload["county_ids"]
        weather_cols = cache_payload["weather_cols"]
        hazard_cols = cache_payload["hazard_cols"]
        splits = cache_payload["splits"]
    else:
        df, timestamps, county_ids, weather_cols = load_long_dataframe(train_data_path)
        splits = assign_splits(df, timestamps, lookback, lookahead, val_multiplier)
        hazard_cols = select_hazard_columns(df, weather_cols)

        feature_df = build_feature_matrix(df, hazard_cols)
        df = df.join(feature_df, how="left")
        df = df[df["_history_len"] >= REQUIRED_HISTORY].copy()
        df.drop(columns=["_history_len"], inplace=True)

        df.sort_values(["county_id", "ts"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["persistence_baseline"] = df.groupby("county_id")["outages"].shift(lookahead)

        cache_payload = {
            "cache_version": CACHE_VERSION,
            "df": df,
            "timestamps": timestamps,
            "county_ids": county_ids,
            "weather_cols": weather_cols,
            "hazard_cols": hazard_cols,
            "splits": splits,
        }
        joblib.dump(cache_payload, cache_file)

    if "persistence_baseline" not in df.columns:
        df["persistence_baseline"] = df.groupby("county_id")["outages"].shift(lookahead)

    base_weather_cols = ["county_id", "ts", "split"]
    weather_training_df = df[base_weather_cols].copy()
    for hazard in hazard_cols:
        if hazard in df.columns:
            weather_training_df[hazard] = df[hazard]
        else:
            weather_training_df[hazard] = 0.0

    weather_medians: Dict[str, float] = {}
    for hazard in hazard_cols:
        if hazard in weather_training_df.columns:
            train_subset = weather_training_df.loc[weather_training_df["split"] == "train", hazard]
            weather_medians[hazard] = float(train_subset.median(skipna=True)) if not train_subset.empty else 0.0
        else:
            weather_medians[hazard] = 0.0

    history_len = get_weather_history_length()
    history_seed = prepare_weather_history(weather_training_df, hazard_cols, weather_medians, history_len)

    weather_artifacts = {
        "models": {hazard: {"model": None, "feature_cols": []} for hazard in hazard_cols},
        "medians": weather_medians,
        "feature_cols": {hazard: [] for hazard in hazard_cols},
        "lags": WEATHER_MODEL_LAGS,
        "history_seed": history_seed,
        "history_len": history_len,
    }

    external_weather_map, external_weather_frames = load_external_weather_forecasts(
        weather_csv_path,
        county_ids,
        hazard_cols,
        weather_training_df["ts"].max(),
        lookahead,
    )

    del weather_training_df

    county_to_idx = {cid: idx for idx, cid in enumerate(sorted(set(df["county_id"].tolist())))}

    # Drop raw weather columns now that rollups are computed
    df.drop(columns=weather_cols, inplace=True, errors="ignore")

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    if not train_mask.any():
        raise RuntimeError("Training split has no samples after feature filtering.")

    df["county_idx"] = df["county_id"].map(county_to_idx)
    category_dtype = pd.api.types.CategoricalDtype(categories=list(county_to_idx.values()), ordered=False)
    df["county_idx"] = df["county_idx"].astype(category_dtype)

    county_te_map = df.loc[train_mask].groupby("county_id")["outages"].mean()
    global_mean = float(df.loc[train_mask, "outages"].mean())
    df["county_te"] = df["county_id"].map(county_te_map).fillna(global_mean)

    feature_cols = [
        *(f"out_lag_{lag}" for lag in LAG_LIST),
        *(f"out_roll_sum_{window}" for window in ROLL_SUM_WINDOWS),
        *(f"out_roll_max_{window}" for window in ROLL_MAX_WINDOWS),
        "out_run_length_since_nonzero",
    ]
    for col in hazard_cols:
        feature_cols.extend(
            [
                f"{col}_lag1",
                f"{col}_delta_1",
                *(f"{col}_roll_max_{window}" for window in WEATHER_ROLL_MAX_WINDOWS),
                *(f"{col}_roll_sum_{window}" for window in WEATHER_ROLL_SUM_WINDOWS),
            ]
        )
    feature_cols.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos", "county_te"])
    feature_cols = [col for col in feature_cols if col in df.columns]

    interaction_cols: List[str] = []
    if "out_lag_1" in df.columns:
        top_hazards = hazard_cols[: min(5, len(hazard_cols))]
        for hz in top_hazards:
            lag_col = f"{hz}_lag1"
            if lag_col not in df.columns:
                continue
            new_col = f"int_out1_{lag_col}"
            df[new_col] = df["out_lag_1"] * df[lag_col]
            interaction_cols.append(new_col)

    if interaction_cols:
        feature_cols.extend(interaction_cols)

    df[feature_cols] = df.groupby("county_id")[feature_cols].ffill()
    medians_series = df.loc[train_mask, feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians_series)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    if device_type == "cuda":
        df[feature_cols] = df[feature_cols].astype(np.float32)

    train_stats = df.loc[train_mask, feature_cols]
    clip_bounds: Dict[str, Tuple[float | None, float | None]] = {}
    for col in feature_cols:
        col_data = train_stats[col].dropna()
        if col_data.empty:
            clip_bounds[col] = (None, None)
            continue
        lower = float(col_data.quantile(CLIP_QUANTILES[0]))
        upper = float(col_data.quantile(CLIP_QUANTILES[1]))
        df[col] = df[col].clip(lower=lower, upper=upper)
        clip_bounds[col] = (lower, upper)

    stds = train_stats.std().fillna(0.0)
    keep_cols = stds[stds > MIN_STD].index.tolist()
    dropped_cols = sorted(set(feature_cols) - set(keep_cols))
    if dropped_cols:
        df.drop(columns=dropped_cols, inplace=True)
    feature_cols = [col for col in feature_cols if col in keep_cols]

    medians = {col: float(medians_series.get(col, 0.0)) for col in feature_cols}

    cat_features = ["county_idx"]
    model_features = feature_cols + cat_features

    df["y_occ"] = (df["outages"] > 0).astype(int)
    df["y_cnt"] = df["outages"].astype(float)

    X_train = df.loc[train_mask, model_features]
    y_train_occ = df.loc[train_mask, "y_occ"].to_numpy()
    pos_count = y_train_occ.sum()
    neg_count = len(y_train_occ) - pos_count
    if pos_count == 0:
        raise RuntimeError("Training data contains no outage events; cannot fit occurrence model.")
    scale_pos_weight = float(neg_count / pos_count) if pos_count else 1.0

    classifier_params: Dict[str, Any] = {
        "objective": "binary",
        "learning_rate": 0.005,
        "num_leaves": 1000,
        "max_depth": 12,
        "min_child_samples": 100,
        "min_sum_hessian_in_leaf": 5.0,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "feature_fraction_bynode": 0.8,
        "reg_alpha": 0.03,
        "reg_lambda": 0.03,
        "n_estimators": 10000,
        "random_state": master_seed,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
    }
    if linear_leaves:
        classifier_params["linear_tree"] = True
    classifier_params.update(device_params)
    classifier = lgb.LGBMClassifier(**classifier_params)

    classifier.fit(
        X_train,
        y_train_occ,
        eval_set=[(df.loc[val_mask, model_features], df.loc[val_mask, "y_occ"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)],
        categorical_feature=cat_features,
    )

    calibrator: IsotonicRegression | None
    val_occ = df.loc[val_mask, "y_occ"].to_numpy()
    val_probs_raw = classifier.predict_proba(df.loc[val_mask, model_features])[:, 1]
    if len(np.unique(val_occ)) >= 2:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs_raw, val_occ)
    else:
        calibrator = None

    train_positive_mask = train_mask & (df["y_occ"] == 1)
    val_positive_mask = val_mask & (df["y_occ"] == 1)

    X_train_pos = df.loc[train_positive_mask, model_features]
    y_train_pos = df.loc[train_positive_mask, "y_cnt"].to_numpy()
    X_val_pos = df.loc[val_positive_mask, model_features]
    y_val_pos = df.loc[val_positive_mask, "y_cnt"].to_numpy()

    regressor_params: Dict[str, Any] = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.55,
        "learning_rate": 0.005,
        "num_leaves": 1000,
        "max_depth": 12,
        "min_child_samples": 150,
        "min_sum_hessian_in_leaf": 10.0,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "feature_fraction_bynode": 0.8,
        "reg_alpha": 0.03,
        "reg_lambda": 0.03,
        "n_estimators": 12000,
        "random_state": master_seed,
        "verbosity": -1,
    }
    if linear_leaves:
        regressor_params["linear_tree"] = True
    regressor_params.update(device_params)
    regressor = lgb.LGBMRegressor(**regressor_params)

    eval_set = [(X_val_pos, y_val_pos)] if len(X_val_pos) else [(X_train_pos, y_train_pos)]
    regressor.fit(
        X_train_pos,
        y_train_pos,
        eval_set=eval_set,
        eval_metric="mae",
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)],
        categorical_feature=cat_features,
    )

    metrics_records: List[Dict[str, float]] = []
    all_predictions: Dict[str, pd.DataFrame] = {}

    for split_name, mask in (("val", val_mask), ("test", test_mask)):
        if not mask.any():
            continue
        subset = df.loc[mask, ["county_id", "ts", "outages", "y_occ", "persistence_baseline"] + model_features].copy()
        subset.sort_values(["county_id", "ts"], inplace=True)
        subset.reset_index(drop=True, inplace=True)
        probs, intensity, expected = predict_counts(classifier, calibrator, regressor, subset[model_features])
        subset["p_occ"] = probs
        subset["intensity"] = intensity
        subset["prediction"] = expected
        subset["decay_pred"] = last_nonzero_decay_baseline(subset)
        subset["persistence_pred"] = subset["persistence_baseline"].fillna(0.0)

        if split_name == "test":
            subset = subset.groupby("county_id", sort=False).tail(lookahead).reset_index(drop=True)
        else:
            subset.reset_index(drop=True, inplace=True)

        mae = mean_group_metric(subset, "outages", "prediction", "county_id", mean_absolute_error)
        rmse = mean_group_metric(subset, "outages", "prediction", "county_id", rmse_metric)

        zero_df = subset.assign(zero_pred=0.0)
        zero_mae = mean_group_metric(zero_df, "outages", "zero_pred", "county_id", mean_absolute_error)
        zero_rmse = mean_group_metric(zero_df, "outages", "zero_pred", "county_id", rmse_metric)
        persistence_mae = mean_group_metric(subset, "outages", "persistence_pred", "county_id", mean_absolute_error)
        persistence_rmse = mean_group_metric(subset, "outages", "persistence_pred", "county_id", rmse_metric)
        decay_mae = mean_group_metric(subset, "outages", "decay_pred", "county_id", mean_absolute_error)
        decay_rmse = mean_group_metric(subset, "outages", "decay_pred", "county_id", rmse_metric)

        cls_metrics = classification_metrics(subset["y_occ"].to_numpy(), subset["p_occ"].to_numpy())
        metrics_records.append(
            {
                "split": split_name,
                "mae": mae,
                "rmse": rmse,
                "zero_mae": zero_mae,
                "zero_rmse": zero_rmse,
                "persistence_mae": persistence_mae,
                "persistence_rmse": persistence_rmse,
                "decay_mae": decay_mae,
                "decay_rmse": decay_rmse,
                **cls_metrics,
            }
        )

        all_predictions[split_name] = subset[["county_id", "ts", "outages", "p_occ", "intensity", "prediction", "persistence_pred"]].copy()

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = save_dir / f"{exp_name}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    for split_name, df_preds in all_predictions.items():
        out_path = save_dir / f"{exp_name}_{split_name}_predictions.csv"
        df_out = df_preds.rename(columns={"ts": "timestamp"})
        df_out.to_csv(out_path, index=False)

    test_metrics = next((row for row in metrics_records if row.get("split") == "test"), None)
    test_report_path = None
    if test_metrics:
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Zero baseline RMSE: {test_metrics['zero_rmse']:.4f}")
        if not math.isnan(test_metrics.get("persistence_rmse", float("nan"))):
            print(f"Persistence baseline RMSE: {test_metrics['persistence_rmse']:.4f}")
        test_report = pd.DataFrame([
            {
                "epoch": 0,
                "test_loss": test_metrics["rmse"],
                "zero_loss": test_metrics["zero_rmse"],
            }
        ])
        test_report_path = save_dir / f"{exp_name}_test.csv"
        test_report.to_csv(test_report_path, index=False)

    test_predictions = all_predictions.get("test")
    if test_predictions is not None and not test_predictions.empty:
        test_plot_df = test_predictions.rename(columns={"ts": "timestamp"}).copy()
        test_plot_df["timestamp"] = pd.to_datetime(test_plot_df["timestamp"])
        unique_counties = test_plot_df["county_id"].unique()
        num_series = len(unique_counties)
        if num_series > 0:
            grid_cols = max(int(math.ceil(math.sqrt(num_series))), 1)
            grid_rows = int(math.ceil(num_series / grid_cols))
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows), squeeze=False)
            axes_flat = axes.reshape(-1)

            for ax, county in zip(axes_flat, unique_counties[:num_series]):
                county_df = test_plot_df[test_plot_df["county_id"] == county].sort_values("timestamp")
                ax.plot(county_df["timestamp"], county_df["outages"], label="Actual")
                ax.plot(county_df["timestamp"], county_df["prediction"], linestyle="--", label="Prediction")
                ax.set_title(f"County {county}")
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Outages")

            for ax in axes_flat[num_series:]:
                ax.axis("off")

            axes_flat[0].legend(fontsize=8)
            fig.suptitle(f"{exp_name} test predictions")
            fig.autofmt_xdate()
            fig.tight_layout()
            plot_path = save_dir / f"{exp_name}_test_plots.pdf"
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)

    if test_mask.any():
        test_subset = df.loc[test_mask, model_features]
        test_probs, _, _ = predict_counts(classifier, calibrator, regressor, test_subset)
        y_test_occ = df.loc[test_mask, "y_occ"].to_numpy()
        if len(np.unique(y_test_occ)) >= 2:
            prob_true, prob_pred = calibration_curve(y_test_occ, test_probs, n_bins=10)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(prob_pred, prob_true, marker="o", label="Calibrated")
            ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Fraction observed")
            ax.set_title(f"Reliability ({lookahead}-Hour Model)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(save_dir / f"{exp_name}_reliability.pdf", bbox_inches="tight")
            plt.close(fig)

    medians["county_te"] = float(df.loc[train_mask, "county_te"].median())

    forecast_df, weather_forecasts = forecast_from_last_window(
        df,
        feature_cols,
        cat_features,
        classifier,
        calibrator,
        regressor,
        county_to_idx,
        medians,
        lookahead,
        hazard_cols,
        weather_artifacts,
        external_weather_map=external_weather_map,
    )
    weather_forecasts = {
        hazard: external_weather_frames.get(
            hazard,
            pd.DataFrame(columns=["county_id", "timestamp", "prediction"]),
        )
        for hazard in hazard_cols
    }
    forecast_path = save_dir / f"{exp_name}_forecast_{lookahead}h.csv"
    forecast_df.to_csv(forecast_path, index=False)

    outage_history = load_recent_outage_history(train_data_path, FORECAST_PLOT_PAST)
    forecast_plot_path = plot_forecast_history(
        exp_name,
        forecast_df,
        save_dir,
        FORECAST_PLOT_PAST,
        history_data=outage_history,
    )
    forecast_book_path = None
    weather_history = load_recent_weather_history(train_data_path, FORECAST_PLOT_PAST, hazard_cols)
    weather_forecast_path = plot_weather_forecasts_pdf(
        exp_name,
        weather_forecasts,
        save_dir,
        weather_history,
    )

    classifier.booster_.save_model(str(save_dir / f"{exp_name}_stage_a.txt"))
    regressor.booster_.save_model(str(save_dir / f"{exp_name}_stage_b.txt"))
    joblib.dump(calibrator, save_dir / f"{exp_name}_calibrator.joblib")

    metadata = {
        "lookback": lookback,
        "lookahead": lookahead,
        "hazard_columns": hazard_cols,
        "feature_columns": feature_cols,
        "categorical_features": cat_features,
        "clip_bounds": {col: [None if bound is None else float(bound) for bound in bounds] for col, bounds in clip_bounds.items()},
        "medians": medians,
        "train_timestamps": len(splits["train"]),
        "val_timestamps": len(splits["val"]),
        "test_timestamps": len(splits["test"]),
        "holdout_timestamps": len(splits["holdout"]),
        "val_multiplier": val_multiplier,
        "scale_pos_weight": scale_pos_weight,
        "persistence_shift_hours": lookahead,
        "device_type": device_params.get("device_type", "cpu"),
        "gpu_device_id": device_params.get("gpu_device_id"),
        "best_iteration_stage_a": getattr(classifier, "best_iteration_", None),
        "best_iteration_stage_b": getattr(regressor, "best_iteration_", None),
    }
    with open(save_dir / f"{exp_name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    if test_report_path is not None:
        print(f"Saved test metrics timeline to {test_report_path}")
    for split_name in all_predictions:
        print(f"Saved {split_name} predictions to {save_dir / f'{exp_name}_{split_name}_predictions.csv'}")
    print(f"Saved forecast to {forecast_path}")
    if forecast_plot_path is not None:
        print(f"Saved forecast plots to {forecast_plot_path}")
    if weather_forecast_path is not None:
        print(f"Saved weather forecast plots to {weather_forecast_path}")


if __name__ == "__main__":
    main()

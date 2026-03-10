"""
oracle_inference.py
═══════════════════════════════════════════════════════════════
Loads pre-trained XGBoost models from the GitHub repo and runs
inference. Drop this file in the same directory as app_oracle.py.

Called by app_oracle.py instead of train_and_predict().
The models are trained offline in Google Colab (see ngn_oracle_training.ipynb)
and committed to the GitHub repo under models/.

File layout expected in the repo root:
    models/
        reg_2h.pkl          ← XGBoost regressor for 2H horizon
        clf_2h.pkl          ← XGBoost classifier for 2H horizon
        reg_10h.pkl
        clf_10h.pkl
        reg_24h.pkl
        clf_24h.pkl
        reg_48h.pkl
        clf_48h.pkl
        reg_7d.pkl
        clf_7d.pkl
        scaler.pkl          ← RobustScaler fitted on training data
        feature_columns.json
        model_metadata.json
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── CONSTANTS ────────────────────────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
HORIZONS     = ["2h", "10h", "24h", "48h", "7d"]
HORIZON_LABELS = {
    "2h":  "2 Hours",
    "10h": "10 Hours",
    "24h": "24 Hours",
    "48h": "48 Hours",
    "7d":  "7 Days",
}
DIRECTION_MAP = {0: "DOWN ▼", 1: "STABLE ◆", 2: "UP ▲"}
DIRECTION_COLORS = {
    "DOWN ▼":   "var(--green)",   # NGN strengthening = good for NGN holders
    "STABLE ◆": "var(--amber)",
    "UP ▲":     "var(--red)",     # NGN weakening = bad for NGN holders
}


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """
    Load all trained models from the models/ directory.
    Cached by Streamlit so they are only loaded once per session.
    Returns a dict with all models, scaler, feature columns, and metadata.
    Returns None if models are not found (falls back to legacy mode).
    """
    if not os.path.exists(MODELS_DIR):
        return None

    required = ["scaler.pkl", "feature_columns.json"] + \
               [f"reg_{h}.pkl" for h in HORIZONS] + \
               [f"clf_{h}.pkl" for h in HORIZONS]

    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None

    try:
        bundle = {}

        # Load scaler
        bundle["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

        # Load feature columns
        with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
            bundle["feature_columns"] = json.load(f)

        # Load metadata
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                bundle["metadata"] = json.load(f)
        else:
            bundle["metadata"] = {}

        # Load each horizon's models
        bundle["regressors"]  = {}
        bundle["classifiers"] = {}
        for h in HORIZONS:
            bundle["regressors"][h]  = joblib.load(os.path.join(MODELS_DIR, f"reg_{h}.pkl"))
            bundle["classifiers"][h] = joblib.load(os.path.join(MODELS_DIR, f"clf_{h}.pkl"))

        bundle["loaded_at"] = datetime.datetime.now().isoformat()
        bundle["available"] = True
        return bundle

    except Exception as e:
        return {"available": False, "error": str(e)}


def models_available() -> bool:
    bundle = load_models()
    return bundle is not None and bundle.get("available", False)


# ── FEATURE ENGINEERING (must match the Colab notebook exactly) ───────────────
def build_inference_features(rate_history: list, feature_columns: list) -> np.ndarray | None:
    """
    Build the feature vector for inference from the rate history stored in
    st.session_state.rate_history.

    The feature engineering here must produce the SAME feature names and
    values as the Colab training notebook's build_features() function.
    Any mismatch = wrong predictions.

    Args:
        rate_history: list of dicts from st.session_state.rate_history
        feature_columns: list of feature names from feature_columns.json

    Returns:
        np.ndarray of shape (1, n_features) or None if insufficient history
    """
    # Need at least 90 days of history for all features (90d rolling windows)
    if len(rate_history) < 30:
        return None

    # Extract date + CBN rate series
    records = []
    for entry in rate_history:
        ts  = entry.get("timestamp", "")
        cbn = entry.get("cbn_rate") or entry.get("p2p_mid")
        if cbn and float(cbn) > 100 and ts:
            try:
                dt = datetime.datetime.fromisoformat(ts)
                records.append({"date": dt.date(), "cbn_rate": float(cbn)})
            except Exception:
                continue

    if len(records) < 10:
        return None

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    # Forward-fill to ensure continuous daily series
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(date_range).rename_axis("date").reset_index()
    df["cbn_rate"] = df["cbn_rate"].fillna(method="ffill").dropna()
    df = df.dropna(subset=["cbn_rate"]).reset_index(drop=True)

    r = df["cbn_rate"]

    # ── LAG FEATURES ──────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90]:
        df[f"lag_{lag}d"] = r.shift(lag)

    # ── RETURN / CHANGE FEATURES ──────────────────────────────────────────────
    for period in [1, 2, 3, 5, 7, 14, 30]:
        df[f"pct_change_{period}d"] = r.pct_change(period) * 100
        df[f"abs_change_{period}d"] = r.diff(period)

    # ── ROLLING STATISTICS ────────────────────────────────────────────────────
    for window in [3, 5, 7, 14, 21, 30, 60, 90]:
        df[f"ma_{window}d"]         = r.rolling(window).mean()
        df[f"std_{window}d"]        = r.rolling(window).std()
        df[f"ma_dev_{window}d"]     = r - r.rolling(window).mean()
        df[f"ma_dev_pct_{window}d"] = (r / r.rolling(window).mean() - 1) * 100

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    df["momentum_1d"]    = r.diff(1)
    df["momentum_3d"]    = r.diff(3)
    df["momentum_7d"]    = r.diff(7)
    df["momentum_accel"] = df["momentum_1d"].diff(1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = compute_rsi(r, 14)
    df["rsi_7"]  = compute_rsi(r, 7)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = r.ewm(span=12, adjust=False).mean()
    ema26 = r.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── BOLLINGER BANDS ───────────────────────────────────────────────────────
    for window in [20, 30]:
        ma  = r.rolling(window).mean()
        std = r.rolling(window).std()
        df[f"bb_upper_{window}"] = ma + 2 * std
        df[f"bb_lower_{window}"] = ma - 2 * std
        df[f"bb_pct_{window}"]   = (r - ma) / (2 * std + 1e-8)
        df[f"bb_width_{window}"] = (4 * std) / (ma + 1e-8) * 100

    # ── TREND SLOPE ───────────────────────────────────────────────────────────
    for window in [7, 14, 30]:
        slopes = []
        for i in range(len(df)):
            if i < window:
                slopes.append(np.nan)
            else:
                y    = r.values[i - window:i]
                x    = np.arange(window)
                slopes.append(float(np.polyfit(x, y, 1)[0]))
        df[f"trend_slope_{window}d"] = slopes

    # ── REGIME DETECTION ──────────────────────────────────────────────────────
    ma90 = r.rolling(90).mean()
    df["above_ma90"]     = (r > ma90).astype(int)
    df["regime_change"]  = df["above_ma90"].diff().abs()
    df["days_in_regime"] = df.groupby(
        (df["above_ma90"] != df["above_ma90"].shift()).cumsum()
    ).cumcount()

    df["vol_ratio"]       = df["std_30d"] / (df["std_90d"] + 1e-8)
    df["high_vol_regime"] = (df["vol_ratio"] > 1.2).astype(int)

    # ── CALENDAR FEATURES ─────────────────────────────────────────────────────
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["month"]          = df["date"].dt.month
    df["quarter"]        = df["date"].dt.quarter
    df["day_of_month"]   = df["date"].dt.day
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
    df["is_weekend"]     = (df["date"].dt.dayofweek >= 5).astype(int)
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── YOY ───────────────────────────────────────────────────────────────────
    df["yoy_change_pct"] = r.pct_change(252) * 100

    # Use the LAST row (most recent) for inference
    last_row = df.iloc[[-1]]

    # Build feature vector in the EXACT column order from training
    feat_vec = []
    for col in feature_columns:
        val = last_row[col].values[0] if col in last_row.columns else 0.0
        feat_vec.append(0.0 if pd.isna(val) or np.isinf(val) else float(val))

    return np.array(feat_vec).reshape(1, -1)


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(current_rate: float, rate_history: list) -> dict:
    """
    Run inference using the pre-trained XGBoost models.

    Returns a dict with predictions for all 5 horizons:
    {
        "available": True,
        "model_age_days": 3,
        "horizons": {
            "2h":  {"rate": 1389.0, "direction": "UP ▲", "direction_color": "var(--red)", ...},
            "10h": {...},
            ...
        }
    }
    """
    bundle = load_models()

    if not bundle or not bundle.get("available"):
        return {
            "available":  False,
            "error":      bundle.get("error", "Models not found. Train in Colab first.") if bundle else "Models not found in models/ directory.",
            "horizons":   {},
        }

    # Build feature vector
    feat_vec = build_inference_features(rate_history, bundle["feature_columns"])
    if feat_vec is None:
        return {
            "available": False,
            "error":     "Insufficient rate history for inference. Need at least 30 data points.",
            "horizons":  {},
        }

    try:
        X_scaled = bundle["scaler"].transform(feat_vec)
    except Exception as e:
        return {"available": False, "error": f"Scaler error: {e}", "horizons": {}}

    # Run predictions for each horizon
    results = {}
    meta    = bundle.get("metadata", {})
    perf    = meta.get("performance", {})

    for h in HORIZONS:
        try:
            reg  = bundle["regressors"][h]
            clf  = bundle["classifiers"][h]

            predicted_rate = float(reg.predict(X_scaled)[0])

            # Direction prediction + probabilities
            dir_probs  = clf.predict_proba(X_scaled)[0]   # [P(DOWN), P(STABLE), P(UP)]
            dir_idx    = int(clf.predict(X_scaled)[0])
            direction  = DIRECTION_MAP[dir_idx]
            dir_color  = DIRECTION_COLORS[direction]
            confidence = round(float(max(dir_probs)) * 100, 1)

            pct_change = round((predicted_rate - current_rate) / current_rate * 100, 2)

            # Uncertainty range: based on validation MAE from metadata
            val_mae = perf.get(h, {}).get("val_mae_ngn", current_rate * 0.005)
            pred_low  = round(predicted_rate - val_mae * 1.5, 0)
            pred_high = round(predicted_rate + val_mae * 1.5, 0)

            results[h] = {
                "label":        HORIZON_LABELS[h],
                "rate":         round(predicted_rate, 0),
                "pct_change":   pct_change,
                "direction":    direction,
                "dir_color":    dir_color,
                "confidence":   confidence,
                "prob_down":    round(float(dir_probs[0]) * 100, 1),
                "prob_stable":  round(float(dir_probs[1]) * 100, 1),
                "prob_up":      round(float(dir_probs[2]) * 100, 1),
                "pred_low":     pred_low,
                "pred_high":    pred_high,
                "val_mae":      val_mae,
                "dir_accuracy": perf.get(h, {}).get("dir_accuracy", None),
            }
        except Exception as e:
            results[h] = {"label": HORIZON_LABELS[h], "error": str(e)}

    # Model age
    trained_at = meta.get("trained_at", "")
    model_age_days = None
    if trained_at:
        try:
            trained_dt    = datetime.datetime.fromisoformat(trained_at)
            model_age_days = (datetime.datetime.now() - trained_dt).days
        except Exception:
            pass

    return {
        "available":     True,
        "model_age_days": model_age_days,
        "trained_at":    trained_at,
        "n_features":    len(bundle["feature_columns"]),
        "horizons":      results,
        "metadata":      meta,
    }


# ── STATUS BANNER ─────────────────────────────────────────────────────────────
def render_model_status_banner():
    """
    Renders a small status banner showing whether trained models are loaded.
    Call this near the top of app_oracle.py after the header.
    """
    bundle = load_models()

    if bundle is None or not bundle.get("available"):
        error_msg = bundle.get("error", "models/ folder not found") if bundle else "models/ folder not found"
        st.markdown(
            f'<div style="background:rgba(255,176,32,0.1);border:1px solid rgba(255,176,32,0.4);'
            f'border-left:4px solid var(--amber);border-radius:8px;padding:10px 16px;'
            f'margin-bottom:12px;font-size:12px;">'
            f'⚠️ <strong style="color:var(--amber);">Trained models not found.</strong> '
            f'Running in legacy mode (session-trained Ridge/RF/GB). '
            f'Train models in Google Colab and push <code>models/</code> to GitHub for full accuracy. '
            f'<span style="color:var(--muted);font-size:10px;">({error_msg})</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        return False

    meta = bundle.get("metadata", {})
    trained_at = meta.get("trained_at", "unknown")
    n_feat     = len(bundle.get("feature_columns", []))
    perf       = meta.get("performance", {})
    mae_24h    = perf.get("24h", {}).get("val_mae_ngn", "?")
    acc_24h    = perf.get("24h", {}).get("dir_accuracy", "?")

    # Age warning
    age_days = None
    try:
        age_days = (datetime.datetime.now() - datetime.datetime.fromisoformat(trained_at)).days
    except Exception:
        pass

    age_str   = f"{age_days}d old" if age_days is not None else ""
    age_color = "var(--red)" if (age_days or 0) > 14 else "var(--amber)" if (age_days or 0) > 7 else "var(--green)"
    age_warn  = f' <span style="color:{age_color};">⚠️ Stale — retrain recommended</span>' if (age_days or 0) > 14 else ""

    st.markdown(
        f'<div style="background:rgba(0,229,160,0.06);border:1px solid rgba(0,229,160,0.25);'
        f'border-radius:8px;padding:8px 16px;margin-bottom:12px;'
        f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        f'<span style="font-size:11px;color:var(--green);">✅ <strong>XGBoost models loaded</strong></span>'
        f'<span style="font-family:var(--font-mono);font-size:10px;color:var(--text2);">'
        f'Trained: {trained_at[:10]} &nbsp;·&nbsp; '
        f'<span style="color:{age_color};">{age_str}</span>{age_warn} &nbsp;·&nbsp; '
        f'{n_feat} features &nbsp;·&nbsp; '
        f'24H MAE: ₦{mae_24h} &nbsp;·&nbsp; 24H Dir Acc: {acc_24h}%'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    return True

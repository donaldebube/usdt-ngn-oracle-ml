"""
USDT/NGN Oracle — Unified Prediction Engine v3.0
═════════════════════════════════════════════════
Combines ML + AI prediction with full multi-timeframe forecasts:
  24h · 7d · 30d · 3m · 6m · 12m · 2yr

ML Engine:
  • Ridge Regression (trend baseline)
  • Random Forest (non-linear regime detection)
  • Gradient Boosting (sequential error correction)
  • Weighted Ensemble (Ridge×0.25 + RF×0.35 + GB×0.40)

Qualitative Engine:
  • 18 RSS feeds → 40+ live headlines
  • 10-dimension Gemini AI scoring
  • Oil / CBN / Geopolitics / Fed / EM Risk / Remittances

Multi-Timeframe Forecasting:
  • Decay-adjusted ML projections per timeframe
  • Qualitative scenario modifiers (bull / base / bear)
  • Confidence degrades correctly over time
  • Gemini AI narrative for each horizon

Run:
  pip install streamlit scikit-learn numpy pandas requests
  streamlit run app_oracle.py
"""

import streamlit as st
import requests
import json
import os
import datetime
import time
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="USDT/NGN Oracle · Unified",
    page_icon="₦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════
# DESIGN SYSTEM — Dark Terminal × Bloomberg Aesthetic
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');

:root {
  --bg:       #060912;
  --bg2:      #0a1020;
  --bg3:      #0d1628;
  --card:     #0f1d30;
  --card2:    #132236;
  --border:   #172440;
  --border2:  #1e3054;
  --border3:  #274070;

  --green:    #00e5a0;
  --green2:   rgba(0,229,160,0.10);
  --green3:   rgba(0,229,160,0.04);
  --red:      #ff4466;
  --red2:     rgba(255,68,102,0.10);
  --amber:    #ffb020;
  --amber2:   rgba(255,176,32,0.10);
  --blue:     #4488ff;
  --blue2:    rgba(68,136,255,0.10);
  --purple:   #b060ff;
  --purple2:  rgba(176,96,255,0.10);
  --cyan:     #00d4ff;
  --cyan2:    rgba(0,212,255,0.10);
  --gold:     #ffd700;
  --gold2:    rgba(255,215,0,0.08);

  --text:     #cfe0f5;
  --text2:    #9ab0cc;
  --muted:    #4a6080;
  --muted2:   #3a4e66;

  --font-mono: 'JetBrains Mono', monospace;
  --font-body: 'Space Grotesk', sans-serif;
  --font-display: 'Playfair Display', serif;

  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-xl: 20px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }
.block-container { padding: 1rem 1.8rem 2rem !important; max-width: 1500px !important; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
h1,h2,h3,h4 { font-family: var(--font-mono) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border3); border-radius: 4px; }

/* ── INPUT OVERRIDES ── */
.stTextInput>div>div>input,
.stTextArea textarea,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-body) !important;
}
.stButton>button {
  background: linear-gradient(135deg, #1a3060, #2a50a0) !important;
  color: #fff !important;
  border: 1px solid var(--border3) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  letter-spacing: 0.3px !important;
  transition: all 0.2s ease !important;
}
.stButton>button:hover {
  background: linear-gradient(135deg, #204080, #3060c0) !important;
  border-color: var(--blue) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(68,136,255,0.25) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2) !important;
  border-radius: var(--r-md) !important;
  border: 1px solid var(--border) !important;
  padding: 4px !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.5px !important;
  padding: 8px 14px !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: var(--card2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }
.stExpander { border: 1px solid var(--border) !important; border-radius: var(--r-md) !important; }
.stExpander details summary { font-family: var(--font-mono) !important; font-size: 12px !important; }

/* ── ANIMATIONS ── */
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
@keyframes ticker { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes glow { 0%,100%{box-shadow:0 0 8px rgba(0,229,160,0.2)} 50%{box-shadow:0 0 20px rgba(0,229,160,0.5)} }
@keyframes scanline { 0%{transform:translateY(-100%)} 100%{transform:translateY(100vh)} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }

/* ── LIVE DOT ── */
.live-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse 2s ease-in-out infinite;
  margin-right: 5px;
  vertical-align: middle;
}
.live-dot-amber { background: var(--amber); }
.live-dot-red   { background: var(--red); }

/* ── CARD BASE ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 20px 22px;
  position: relative;
  overflow: hidden;
  animation: fadeIn 0.3s ease;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border3), transparent);
}
.card-green  { border-top: 2px solid var(--green);  }
.card-red    { border-top: 2px solid var(--red);    }
.card-amber  { border-top: 2px solid var(--amber);  }
.card-blue   { border-top: 2px solid var(--blue);   }
.card-purple { border-top: 2px solid var(--purple); }
.card-cyan   { border-top: 2px solid var(--cyan);   }
.card-gold   { border-top: 2px solid var(--gold);   }

.card-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.card-value {
  font-family: var(--font-mono);
  font-size: 24px;
  font-weight: 700;
  line-height: 1.1;
  margin-bottom: 4px;
}
.card-sub { font-size: 11px; color: var(--text2); margin-top: 5px; }

/* ── SECTION HEADER ── */
.sec-header {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--muted);
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}

/* ── BADGE ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 100px;
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.badge-bull { background: var(--green2); color: var(--green); border: 1px solid rgba(0,229,160,0.3); }
.badge-bear { background: var(--red2);   color: var(--red);   border: 1px solid rgba(255,68,102,0.3); }
.badge-neu  { background: var(--amber2); color: var(--amber); border: 1px solid rgba(255,176,32,0.3); }

/* ── TIMEFRAME PREDICTION CARD ── */
.tf-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 16px 18px;
  position: relative;
  overflow: hidden;
  transition: all 0.2s ease;
  cursor: default;
}
.tf-card:hover {
  border-color: var(--border3);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.tf-card-accent {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
}
.tf-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.tf-value {
  font-family: var(--font-mono);
  font-size: 20px;
  font-weight: 700;
  line-height: 1.2;
}
.tf-range {
  font-size: 10px;
  color: var(--text2);
  margin-top: 4px;
  font-family: var(--font-mono);
}
.tf-change {
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
  margin-top: 6px;
}
.tf-conf {
  font-size: 9px;
  color: var(--muted);
  margin-top: 4px;
  font-family: var(--font-mono);
}

/* ── PROGRESS BAR ── */
.prog-wrap { margin-bottom: 12px; }
.prog-label { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px; }
.prog-track { background: var(--border); border-radius: 3px; height: 5px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 3px; transition: width 0.5s ease; }

/* ── ALERT BOX ── */
.alert-box {
  border-radius: var(--r-md);
  padding: 12px 16px;
  font-size: 13px;
  margin-bottom: 10px;
  border-left: 3px solid;
  line-height: 1.5;
}
.alert-bull { background: var(--green2); border-color: var(--green); color: #a0ead4; }
.alert-bear { background: var(--red2);   border-color: var(--red);   color: #ffaabb; }
.alert-info { background: var(--blue2);  border-color: var(--blue);  color: #99bbff; }
.alert-warn { background: var(--amber2); border-color: var(--amber); color: #ffd580; }

/* ── MODEL BADGES ── */
.model-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 100px;
  font-family: var(--font-mono); font-size: 10px; font-weight: 700;
}
.badge-ridge  { background: rgba(68,136,255,0.15);  color: #4488ff; border: 1px solid rgba(68,136,255,0.3); }
.badge-rf     { background: rgba(0,229,160,0.10);   color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }
.badge-gb     { background: rgba(176,96,255,0.12);  color: #b060ff; border: 1px solid rgba(176,96,255,0.3); }
.badge-ens    { background: rgba(255,176,32,0.15);  color: #ffb020; border: 1px solid rgba(255,176,32,0.3); }

/* ── TICKER ── */
.ticker-wrap {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  overflow: hidden;
  padding: 8px 0;
  margin-bottom: 16px;
  white-space: nowrap;
}
.ticker-inner {
  display: inline-flex;
  gap: 40px;
  animation: ticker 35s linear infinite;
  padding: 0 20px;
}
.ticker-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted);
}
.ticker-item .val { color: var(--text); font-weight: 600; }
.ticker-item .up  { color: var(--green); }
.ticker-item .dn  { color: var(--red); }
.ticker-sep { color: var(--border3); }

/* ── SPREAD TABLE ── */
.spread-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.spread-table th {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.spread-table td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}
.spread-table tr:last-child td { border-bottom: none; }
.spread-table tr:hover td { background: rgba(255,255,255,0.02); }

/* ── CHAT ── */
.chat-u {
  background: var(--blue2);
  border: 1px solid rgba(68,136,255,0.2);
  border-radius: 14px 14px 3px 14px;
  padding: 12px 16px;
  margin: 8px 0;
  margin-left: 15%;
  font-size: 13px;
  line-height: 1.6;
}
.chat-a {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px 14px 14px 3px;
  padding: 12px 16px;
  margin: 8px 0;
  margin-right: 15%;
  font-size: 13px;
  line-height: 1.6;
}
.chat-badge {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--green);
  margin-bottom: 5px;
}

/* ── HEADLINE ROW ── */
.hl-row {
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  line-height: 1.5;
}
.hl-row:last-child { border-bottom: none; }

/* ── SCENARIO CARD ── */
.scenario-card {
  border-radius: var(--r-md);
  padding: 14px 16px;
  border: 1px solid;
  margin-bottom: 10px;
}
.scenario-bull { background: var(--green3); border-color: rgba(0,229,160,0.3); }
.scenario-base { background: rgba(68,136,255,0.05); border-color: rgba(68,136,255,0.2); }
.scenario-bear { background: rgba(255,68,102,0.05); border-color: rgba(255,68,102,0.25); }

/* ── DIVIDER ── */
.hdivider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border3), transparent);
  margin: 18px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SESSION STATE + PERSISTENCE
# ══════════════════════════════════════════════════════
HISTORY_FILE = "oracle_rate_history.json"
MAX_HISTORY  = 2000

def _save_history():
    try:
        data = st.session_state.rate_history[-MAX_HISTORY:]
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, default=str)
    except Exception:
        pass

def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        valid = [d for d in data if isinstance(d, dict) and d.get("p2p_mid") and d.get("timestamp")]
        return valid[-MAX_HISTORY:]
    except Exception:
        return []

def init():
    defaults = {
        "chat": [],
        "result": None,
        "last_time": None,
        "rate_history": [],
        "ml_metrics": {},
        "alerts": [],
        "alert_triggered": [],
        "auto_refresh": False,
        "refresh_interval": 60,
        "prev_rate": None,
        "user_email": "",
        "global_signals": None,
        "global_signals_time": None,
        "history_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not st.session_state.history_loaded:
        saved = _load_history()
        if saved:
            existing_times = {d.get("timestamp") for d in st.session_state.rate_history}
            new_pts = [d for d in saved if d.get("timestamp") not in existing_times]
            st.session_state.rate_history = new_pts + st.session_state.rate_history
        st.session_state.history_loaded = True

init()


# ══════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════
try:
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
    NEWS_KEY   = st.secrets.get("NEWS_KEY", "")
    RESEND_KEY = st.secrets.get("RESEND_API_KEY", "")
except Exception:
    GEMINI_KEY = ""
    NEWS_KEY   = ""
    RESEND_KEY = ""

if not GEMINI_KEY:
    st.error("⚠️ **GEMINI_KEY not configured.** Add it to `.streamlit/secrets.toml` to run the Oracle.")
    st.stop()


# ══════════════════════════════════════════════════════
# GEMINI ENGINE
# ══════════════════════════════════════════════════════
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]

@st.cache_data(ttl=300)
def check_gemini_key(key: str) -> tuple:
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        try:
            r = requests.post(url, json={
                "contents": [{"parts": [{"text": "Reply: OK"}]}],
                "generationConfig": {"maxOutputTokens": 5}
            }, timeout=12)
            if r.status_code == 200: return True, model, ""
            if r.status_code == 403: return False, "", "API key invalid or not authorised"
            if r.status_code == 429: return True, model, "Rate limited"
        except: continue
    return False, "", "No working Gemini model found"

_key_ok, _working_model, _key_err = check_gemini_key(GEMINI_KEY)
if not _key_ok:
    st.error(f"❌ Gemini key error: {_key_err}. Check aistudio.google.com")
    st.stop()

def gemini(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 4096) -> str:
    parts = []
    if system:
        parts.append({"text": f"SYSTEM:\n{system}\n\n---\n\n"})
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }
    errors = []
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        try:
            r = requests.post(url, json=payload, timeout=45)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 429:
                return "❌ Rate limit hit. Wait 60s and try again."
            elif r.status_code == 403:
                return "❌ API key invalid."
            else:
                errors.append(f"{model}: HTTP {r.status_code}")
                continue
        except Exception as e:
            errors.append(f"{model}: {str(e)[:50]}")
            continue
    return f"❌ All Gemini models failed: {' | '.join(errors)}"

def _parse_json(raw: str) -> dict:
    """Robust JSON parser that handles Gemini's markdown wrapping."""
    clean = raw.strip()
    if "```" in clean:
        for p in clean.split("```"):
            p = p.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("{"): clean = p; break
    if not clean.startswith("{"):
        idx = clean.find("{")
        if idx >= 0: clean = clean[idx:]
    last = clean.rfind("}")
    if last >= 0: clean = clean[:last+1]
    return json.loads(clean)


# ══════════════════════════════════════════════════════
# GLOBAL SIGNALS ENGINE
# ══════════════════════════════════════════════════════
SIGNALS_TTL = 300

def _signals_stale() -> bool:
    t = st.session_state.global_signals_time
    if t is None: return True
    return (datetime.datetime.now() - t).total_seconds() > SIGNALS_TTL

def fetch_global_signals() -> dict:
    sig = {"fetched_at": datetime.datetime.now().isoformat(), "sources": [], "errors": []}

    # 1. CRYPTO PRICES
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,tether,binancecoin,solana,ripple"
            "&vs_currencies=usd,ngn&include_24hr_change=true&include_market_cap=true",
            timeout=12, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            sig["btc_usd"]     = d.get("bitcoin",{}).get("usd")
            sig["btc_24h"]     = d.get("bitcoin",{}).get("usd_24h_change")
            sig["btc_mcap"]    = d.get("bitcoin",{}).get("usd_market_cap")
            sig["eth_usd"]     = d.get("ethereum",{}).get("usd")
            sig["eth_24h"]     = d.get("ethereum",{}).get("usd_24h_change")
            sig["bnb_usd"]     = d.get("binancecoin",{}).get("usd")
            sig["sol_usd"]     = d.get("solana",{}).get("usd")
            sig["xrp_usd"]     = d.get("ripple",{}).get("usd")
            sig["usdt_ngn_cg"] = d.get("tether",{}).get("ngn")
            sig["sources"].append("CoinGecko")
    except Exception as e:
        sig["errors"].append(f"CoinGecko: {str(e)[:60]}")

    # 2. FX RATES
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
        if r.status_code == 200:
            rates = r.json().get("rates", {})
            sig["usd_ngn_official"] = rates.get("NGN")
            sig["usd_eur"]  = rates.get("EUR")
            sig["usd_gbp"]  = rates.get("GBP")
            sig["usd_zar"]  = rates.get("ZAR")
            sig["usd_kes"]  = rates.get("KES")
            sig["usd_ghs"]  = rates.get("GHS")
            sig["usd_jpy"]  = rates.get("JPY")
            sig["usd_cny"]  = rates.get("CNY")
            sig["eurusd"]   = round(1 / rates["EUR"], 5) if rates.get("EUR") else None
            sig["dxy_proxy"]= round(rates["EUR"] * 100, 3) if rates.get("EUR") else None
            sig["sources"].append("ExchangeRate-API")
    except Exception as e:
        sig["errors"].append(f"FX API: {str(e)[:60]}")

    # 3. FEAR & GREED
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r.status_code == 200:
            fng = r.json().get("data", [{}])[0]
            sig["fear_greed_value"] = int(fng.get("value", 50))
            sig["fear_greed_label"] = fng.get("value_classification", "N/A")
            sig["sources"].append("Fear&Greed Index")
    except Exception as e:
        sig["errors"].append(f"FnG: {str(e)[:40]}")

    # 4. OIL HEADLINE
    try:
        r = requests.get(
            "https://news.google.com/rss/search?q=Brent+crude+oil+price+today&hl=en-US&gl=US&ceid=US:en",
            timeout=7, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
            sig["oil_headline"] = titles[1] if len(titles) > 1 else None
    except: pass

    # 5. GLOBAL NEWS (18 RSS feeds)
    rss_topics = [
        ("Nigeria naira exchange rate",        "🇳🇬 NGN"),
        ("CBN central bank Nigeria forex",     "🏦 CBN"),
        ("crude oil price Brent OPEC today",   "🛢️ Oil"),
        ("Iran oil sanctions",                 "⚠️ Iran"),
        ("US Federal Reserve interest rates",  "🇺🇸 Fed"),
        ("Bitcoin crypto market today",        "₿ BTC"),
        ("Nigeria economy inflation 2025",     "📉 NG Macro"),
        ("Middle East conflict oil supply",    "🌍 MidEast"),
        ("dollar index DXY strength",          "💵 DXY"),
        ("OPEC production output cut",         "🛢️ OPEC"),
        ("Russia Ukraine war commodity",       "⚡ Russia"),
        ("Nigeria crypto P2P USDT",            "💱 NG Crypto"),
        ("emerging markets currency selloff",  "📊 EM FX"),
        ("US inflation CPI report",            "📈 US CPI"),
        ("IMF World Bank Nigeria",             "🏛️ IMF/WB"),
        ("China economy trade slowdown",       "🇨🇳 China"),
        ("Nigeria remittance diaspora",        "💸 Remittance"),
        ("gold price safe haven",              "🥇 Gold"),
    ]
    headlines_raw = []
    for query, tag in rss_topics:
        try:
            encoded = requests.utils.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
                descs  = re.findall(r"<description><!\[CDATA\[(.*?)\]\]></description>", r.text)
                for i, title in enumerate(titles[1:3]):
                    desc = re.sub(r"<[^>]+>", "", descs[i] if i < len(descs) else "")[:120]
                    headlines_raw.append({"tag": tag, "title": title.strip(), "desc": desc.strip(),
                                          "full": f"{tag} | {title.strip()}"})
        except: pass

    if NEWS_KEY:
        for q in ["Nigeria naira USDT", "oil price Iran", "CBN forex", "Nigeria economy"]:
            try:
                r = requests.get(
                    f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}"
                    f"&sortBy=publishedAt&pageSize=3&language=en&apiKey={NEWS_KEY}",
                    timeout=7
                )
                if r.status_code == 200:
                    for a in r.json().get("articles", [])[:2]:
                        t = a.get("title", "")
                        d = (a.get("description") or "")[:120]
                        if t:
                            headlines_raw.append({"tag": "📰 NewsAPI", "title": t, "desc": d,
                                                  "full": f"📰 NewsAPI | {t}"})
            except: pass

    sig["headlines"]      = headlines_raw
    sig["headline_count"] = len(headlines_raw)
    if headlines_raw:
        sig["sources"].append(f"Google News RSS ({len(headlines_raw)} headlines)")

    # 6. GEMINI DEEP ANALYSIS
    if headlines_raw:
        now_str  = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M WAT")
        btc_str  = f"${sig.get('btc_usd',0):,.0f} ({sig.get('btc_24h',0):+.1f}%)" if sig.get("btc_usd") else "N/A"
        fng_str  = f"{sig.get('fear_greed_value','?')} — {sig.get('fear_greed_label','N/A')}"
        headlines_block = "\n".join(f"  {i+1}. {h['full']}" for i, h in enumerate(headlines_raw[:40]))

        q_prompt = f"""Senior FX strategist — Today: {now_str}

LIVE SNAPSHOT:
- BTC: {btc_str} | ETH: ${sig.get('eth_usd',0):,.0f} ({sig.get('eth_24h',0):+.1f}%)
- Fear & Greed: {fng_str}
- EUR/USD: {sig.get('eurusd','N/A')} | DXY proxy: {sig.get('dxy_proxy','N/A')}
- USD/ZAR: {sig.get('usd_zar','N/A')} | USD/KES: {sig.get('usd_kes','N/A')} | USD/GHS: {sig.get('usd_ghs','N/A')}
- Oil headline: {sig.get('oil_headline','N/A')}

LIVE HEADLINES:
{headlines_block}

Return ONLY valid JSON (no markdown):
{{
  "overall_score": <-100 to +100, positive=USDT rises/NGN weakens>,
  "nigeria_macro": <-100 to 100>,
  "cbn_policy": <-100 to 100>,
  "oil_impact": <-100 to 100>,
  "usd_fed_impact": <-100 to 100>,
  "crypto_sentiment": <-100 to 100>,
  "geopolitical_risk": <-100 to 100>,
  "political_risk_nigeria": <-100 to 100>,
  "remittance_flow": <-100 to 100>,
  "global_em_risk": <-100 to 100>,
  "market_mood": "RISK_ON|RISK_OFF|NEUTRAL",
  "top_mover_today": "<specific event moving market right now>",
  "breaking_event": "<breaking news or null>",
  "oil_analysis": "<2 sentences: oil situation and NGN impact>",
  "geopolitical_analysis": "<2 sentences: active geopolitical risks affecting rate>",
  "cbn_analysis": "<2 sentences: CBN stance and likely near-term action>",
  "crypto_analysis": "<1 sentence: crypto sentiment and P2P Nigeria impact>",
  "em_analysis": "<1 sentence: broader EM currency pressure>",
  "top_bullish_catalyst": "<most important reason USDT/NGN could RISE — cite actual news>",
  "top_bearish_catalyst": "<most important reason USDT/NGN could FALL — cite actual news>",
  "overall_qualitative_direction": "BULLISH_USDT|BEARISH_USDT|NEUTRAL",
  "qualitative_confidence": <0-100>,
  "30min_bias": "BUY|SELL|HOLD",
  "key_watch_items": ["<watch item 1>", "<watch item 2>", "<watch item 3>"],
  "medium_term_outlook": "<3-sentence outlook for next 30-90 days based on fundamentals>",
  "long_term_outlook": "<2-sentence 6-12 month outlook for USDT/NGN>",
  "structural_ngn_risks": ["<structural risk 1>", "<structural risk 2>", "<structural risk 3>"]
}}"""

        try:
            raw_q = gemini(q_prompt, "You are a quantitative FX strategist. Return only valid JSON. No markdown.")
            sig["analysis"] = _parse_json(raw_q)
            sig["sources"].append("Gemini deep analysis")
        except Exception as e:
            sig["analysis"] = {}
            sig["errors"].append(f"Gemini analysis: {str(e)[:80]}")

    return sig

def maybe_refresh_signals(force: bool = False):
    if force or _signals_stale():
        try:
            sig = fetch_global_signals()
            st.session_state.global_signals      = sig
            st.session_state.global_signals_time = datetime.datetime.now()
        except Exception: pass


# ══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
FEATURE_COLS = [
    "p2p_spread_abs", "p2p_spread_pct", "p2p_buy_std", "p2p_sell_std",
    "premium_pct", "premium_abs", "official_rate",
    "btc_24h_change", "eth_24h_change", "usdt_ngn_cg",
    "chainlink_change", "uniswap_change",
    "eurusd", "dxy_proxy", "usd_zar", "usd_kes", "usd_ghs",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_business",
    "news_overall", "news_nigeria", "news_cbn", "news_oil", "news_usd",
    "news_crypto", "news_geopolitics", "news_political_risk",
    "news_remittance", "news_em_risk",
    "momentum_1", "momentum_avg", "volatility", "trend_slope", "trend_accel",
    "rate_ma5_dev",
]

def features_to_vector(feat: dict) -> np.ndarray:
    return np.array([float(feat.get(c, 0.0)) for c in FEATURE_COLS], dtype=float)

def collect_features() -> tuple:
    feat = {}
    raw  = {}
    raw["rate_source"] = "unknown"
    raw["rate_status"] = "fetching"

    # P2P RATES
    try:
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        for side, key in [("BUY", "p2p_buy"), ("SELL", "p2p_sell")]:
            payload = {"asset": "USDT", "fiat": "NGN", "merchantCheck": False,
                       "page": 1, "payTypes": [], "publisherType": None, "rows": 10, "tradeType": side}
            r = requests.post("https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search",
                              json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                prices = [float(item["adv"]["price"]) for item in r.json().get("data", [])
                          if float(item.get("adv", {}).get("price", 0)) > 100]
                if prices:
                    raw[key] = round(sum(prices) / len(prices), 2)
                    if side == "BUY":
                        feat["p2p_buy_min"] = min(prices)
                        feat["p2p_buy_max"] = max(prices)
                        feat["p2p_buy_std"] = float(np.std(prices))
                    else:
                        feat["p2p_sell_min"] = min(prices)
                        feat["p2p_sell_max"] = max(prices)
                        feat["p2p_sell_std"] = float(np.std(prices))
        if raw.get("p2p_buy") or raw.get("p2p_sell"):
            raw["rate_source"] = "Binance P2P"
            raw["rate_status"] = "live"
    except Exception as e:
        raw["binance_error"] = str(e)[:100]

    if not (raw.get("p2p_buy") and raw.get("p2p_sell")):
        try:
            r = requests.get(
                "https://api2.bybit.com/fiat/otc/item/list?userId=&tokenId=USDT&currencyId=NGN"
                "&payment=&side=1&size=10&page=1&amount=&authMaker=false&canTrade=false",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=12
            )
            if r.status_code == 200:
                prices = [float(i.get("price", 0)) for i in r.json().get("result", {}).get("items", [])
                          if float(i.get("price", 0)) > 100]
                if prices:
                    avg = round(sum(prices) / len(prices), 2)
                    if not raw.get("p2p_buy"):  raw["p2p_buy"]  = avg
                    if not raw.get("p2p_sell"): raw["p2p_sell"] = round(avg * 0.998, 2)
                    feat["p2p_buy_std"]  = float(np.std(prices)) if len(prices) > 1 else 0.0
                    feat["p2p_sell_std"] = 0.0
                    raw["rate_source"] = "Bybit P2P"
                    raw["rate_status"] = "live"
        except: pass

    if not raw.get("p2p_buy"):
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=ngn",
                             timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                ngn_val = r.json().get("tether", {}).get("ngn")
                if ngn_val and float(ngn_val) > 100:
                    raw["p2p_buy"]  = float(ngn_val)
                    raw["p2p_sell"] = round(float(ngn_val) * 0.997, 2)
                    feat["p2p_buy_std"] = feat["p2p_sell_std"] = 0.0
                    raw["rate_source"] = "CoinGecko"
                    raw["rate_status"] = "live"
        except: pass

    if not raw.get("p2p_buy"):
        raw["p2p_buy"]   = 1620.0
        raw["p2p_sell"]  = 1615.0
        raw["rate_source"] = "⚠️ Fallback estimate"
        raw["rate_status"] = "estimated"
        feat["p2p_buy_std"] = feat["p2p_sell_std"] = 0.0

    if raw.get("p2p_buy") and raw.get("p2p_sell"):
        raw["p2p_mid"]    = round((raw["p2p_buy"] + raw["p2p_sell"]) / 2, 2)
        raw["p2p_spread"] = round(raw["p2p_buy"] - raw["p2p_sell"], 2)
        feat["p2p_spread_abs"] = raw["p2p_spread"]
        feat["p2p_spread_pct"] = round(raw["p2p_spread"] / max(raw["p2p_sell"], 1) * 100, 4)
    else:
        raw["p2p_mid"] = raw.get("p2p_buy", 1620.0)
        raw["p2p_spread"] = 0.0
        feat["p2p_spread_abs"] = feat["p2p_spread_pct"] = 0.0

    # OFFICIAL RATE
    official = None
    for url in ["https://open.er-api.com/v6/latest/USD", "https://api.exchangerate-api.com/v4/latest/USD"]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                rates_obj = r.json().get("rates", {})
                ngn = rates_obj.get("NGN")
                if ngn:
                    official = float(ngn)
                    for ccy in ["EUR", "GBP", "ZAR", "KES", "GHS", "EGP", "XOF"]:
                        v = rates_obj.get(ccy)
                        if v: feat[f"usd_{ccy.lower()}"] = float(v)
                    break
        except: pass
    raw["official"] = official
    if official and raw.get("p2p_mid"):
        raw["premium_pct"] = round((raw["p2p_mid"] - official) / official * 100, 4)
        feat["official_rate"] = official
        feat["premium_pct"]   = raw["premium_pct"]
        feat["premium_abs"]   = round(raw["p2p_mid"] - official, 2)

    # CRYPTO FEATURES
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,tether&vs_currencies=usd,ngn&include_24hr_change=true",
            timeout=12, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            feat["btc_24h_change"] = d.get("bitcoin", {}).get("usd_24h_change", np.nan)
            feat["eth_24h_change"] = d.get("ethereum", {}).get("usd_24h_change", np.nan)
            feat["usdt_ngn_cg"]    = d.get("tether", {}).get("ngn", np.nan)
            raw["btc_change"]      = feat["btc_24h_change"]
    except: pass

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=chainlink,uniswap&vs_currencies=usd&include_24hr_change=true",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            feat["chainlink_change"] = d.get("chainlink", {}).get("usd_24h_change", np.nan)
            feat["uniswap_change"]   = d.get("uniswap", {}).get("usd_24h_change", np.nan)
    except: pass

    # USD STRENGTH
    eur = feat.get("usd_eur")
    if eur:
        feat["eurusd"]     = round(1 / eur, 6)
        feat["dxy_proxy"]  = round(eur * 100, 4)
        feat["usd_strong"] = 1 if feat["eurusd"] < 1.05 else 0

    # TEMPORAL FEATURES
    now = datetime.datetime.now()
    feat["hour"]        = now.hour
    feat["dow"]         = now.weekday()
    feat["is_weekend"]  = int(now.weekday() >= 5)
    feat["is_business"] = int(8 <= now.hour <= 17 and now.weekday() < 5)
    feat["month"]       = now.month
    feat["hour_sin"]    = float(np.sin(2 * np.pi * now.hour / 24))
    feat["hour_cos"]    = float(np.cos(2 * np.pi * now.hour / 24))
    feat["dow_sin"]     = float(np.sin(2 * np.pi * now.weekday() / 7))
    feat["dow_cos"]     = float(np.cos(2 * np.pi * now.weekday() / 7))
    feat["month_sin"]   = float(np.sin(2 * np.pi * now.month / 12))
    feat["month_cos"]   = float(np.cos(2 * np.pi * now.month / 12))

    # QUALITATIVE INTELLIGENCE
    cached_sig  = st.session_state.global_signals or {}
    cached_anal = cached_sig.get("analysis", {})
    headlines_all = [h.get("full", "") for h in cached_sig.get("headlines", [])]

    if cached_anal and not _signals_stale():
        feat["news_overall"]        = float(cached_anal.get("overall_score", 0))
        feat["news_nigeria"]        = float(cached_anal.get("nigeria_macro", 0))
        feat["news_cbn"]            = float(cached_anal.get("cbn_policy", 0))
        feat["news_oil"]            = float(cached_anal.get("oil_impact", 0))
        feat["news_usd"]            = float(cached_anal.get("usd_fed_impact", 0))
        feat["news_crypto"]         = float(cached_anal.get("crypto_sentiment", 0))
        feat["news_geopolitics"]    = float(cached_anal.get("geopolitical_risk", 0))
        feat["news_political_risk"] = float(cached_anal.get("political_risk_nigeria", 0))
        feat["news_remittance"]     = float(cached_anal.get("remittance_flow", 0))
        feat["news_em_risk"]        = float(cached_anal.get("global_em_risk", 0))
        raw["news_intel"]           = cached_anal
        raw["news_headlines"]       = headlines_all[:40]
        raw["news_headlines_count"] = len(headlines_all)
    else:
        maybe_refresh_signals(force=True)
        fresh_sig  = st.session_state.global_signals or {}
        fresh_anal = fresh_sig.get("analysis", {})
        fresh_hl   = [h.get("full", "") for h in fresh_sig.get("headlines", [])]
        if fresh_anal:
            feat["news_overall"]        = float(fresh_anal.get("overall_score", 0))
            feat["news_nigeria"]        = float(fresh_anal.get("nigeria_macro", 0))
            feat["news_cbn"]            = float(fresh_anal.get("cbn_policy", 0))
            feat["news_oil"]            = float(fresh_anal.get("oil_impact", 0))
            feat["news_usd"]            = float(fresh_anal.get("usd_fed_impact", 0))
            feat["news_crypto"]         = float(fresh_anal.get("crypto_sentiment", 0))
            feat["news_geopolitics"]    = float(fresh_anal.get("geopolitical_risk", 0))
            feat["news_political_risk"] = float(fresh_anal.get("political_risk_nigeria", 0))
            feat["news_remittance"]     = float(fresh_anal.get("remittance_flow", 0))
            feat["news_em_risk"]        = float(fresh_anal.get("global_em_risk", 0))
            raw["news_intel"]           = fresh_anal
            raw["news_headlines"]       = fresh_hl[:40]
            raw["news_headlines_count"] = len(fresh_hl)
        else:
            for k in ["news_overall","news_nigeria","news_cbn","news_oil","news_usd",
                      "news_crypto","news_geopolitics","news_political_risk","news_remittance","news_em_risk"]:
                feat[k] = 0.0
            raw["news_intel"] = {}
            raw["news_headlines"] = []
            raw["news_headlines_count"] = 0

    # HISTORICAL MOMENTUM
    hist = st.session_state.rate_history
    if len(hist) >= 2:
        recent_rates = [h["p2p_mid"] for h in hist[-10:] if h.get("p2p_mid")]
        if len(recent_rates) >= 2:
            feat["momentum_1"]   = recent_rates[-1] - recent_rates[-2]
            feat["momentum_avg"] = recent_rates[-1] - np.mean(recent_rates[:-1])
            feat["volatility"]   = float(np.std(recent_rates))
            x = np.arange(len(recent_rates))
            slope = float(np.polyfit(x, recent_rates, 1)[0])
            feat["trend_slope"]  = slope
            feat["trend_accel"]  = feat["momentum_1"] - (recent_rates[-2] - recent_rates[-3]) if len(recent_rates) >= 3 else 0.0
        if len(recent_rates) >= 5:
            feat["rate_ma5"]     = float(np.mean(recent_rates[-5:]))
            feat["rate_ma5_dev"] = recent_rates[-1] - feat["rate_ma5"]
    else:
        for k in ["momentum_1","momentum_avg","volatility","trend_slope","trend_accel"]:
            feat[k] = 0.0

    for k, v in feat.items():
        if isinstance(v, float) and np.isnan(v):
            feat[k] = 0.0

    raw["timestamp"] = datetime.datetime.now().isoformat()
    raw["features"]  = feat
    return raw, feat


# ══════════════════════════════════════════════════════
# ML TRAINING ENGINE
# ══════════════════════════════════════════════════════
def build_training_data() -> tuple:
    hist = st.session_state.rate_history
    if len(hist) < 5:
        return None, None, None
    X_rows, y_vals, times = [], [], []
    for i in range(len(hist) - 1):
        feat = hist[i].get("features", {})
        next_rate = hist[i + 1].get("p2p_mid")
        if next_rate and feat:
            X_rows.append(features_to_vector(feat))
            y_vals.append(float(next_rate))
            times.append(hist[i].get("timestamp", ""))
    if len(X_rows) < 4:
        return None, None, None
    return np.array(X_rows), np.array(y_vals), times

def train_and_predict(current_feat: dict, current_rate: float) -> dict:
    X, y, times = build_training_data()
    cold_start = (X is None)

    if cold_start:
        score = 0.0
        score += current_feat.get("btc_24h_change", 0) * 0.3
        score += -current_feat.get("dxy_proxy", 0) * 0.01
        score += current_feat.get("news_overall", 0) * 0.5
        score += -current_feat.get("premium_pct", 0) * 0.2
        score += current_feat.get("momentum_1", 0) * 0.8
        direction_factor = 1 + score / 5000.0
        est = round(current_rate * direction_factor, 2)
        return {
            "cold_start": True,
            "n_training_points": 0,
            "ridge_pred": est, "rf_pred": est, "gb_pred": est, "ensemble": est,
            "pred_low": round(current_rate * 0.992, 2),
            "pred_high": round(current_rate * 1.008, 2),
            "confidence": 35, "direction": "BULLISH" if est > current_rate else "BEARISH" if est < current_rate else "NEUTRAL",
            "model_agreement": 100.0,
            "ridge_cv_mae": None, "rf_cv_mae": None, "gb_cv_mae": None,
            "rf_feature_importance": {},
            "note": "COLD START — fewer than 5 data points. Confidence capped at 35%.",
        }

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_pred   = scaler.transform(features_to_vector(current_feat).reshape(1, -1))

    # Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_scaled, y)
    ridge_pred = float(ridge.predict(x_pred)[0])
    ridge_cv_mae = None
    if len(X_scaled) >= 5:
        cv = cross_val_score(ridge, X_scaled, y, cv=min(5, len(X_scaled)), scoring="neg_mean_absolute_error")
        ridge_cv_mae = float(-cv.mean())

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_pred = float(rf.predict(x_pred)[0])
    rf_cv_mae = None
    if len(X_scaled) >= 5:
        cv = cross_val_score(rf, X_scaled, y, cv=min(5, len(X_scaled)), scoring="neg_mean_absolute_error")
        rf_cv_mae = float(-cv.mean())
    imp = dict(zip(FEATURE_COLS, rf.feature_importances_))
    top_features = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10])

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.8, random_state=42)
    gb.fit(X_scaled, y)
    gb_pred = float(gb.predict(x_pred)[0])
    gb_cv_mae = None
    if len(X_scaled) >= 5:
        cv = cross_val_score(gb, X_scaled, y, cv=min(5, len(X_scaled)), scoring="neg_mean_absolute_error")
        gb_cv_mae = float(-cv.mean())

    # Ensemble
    weights  = np.array([0.25, 0.35, 0.40])
    preds    = np.array([ridge_pred, rf_pred, gb_pred])
    ensemble = float(np.dot(weights, preds))

    # Confidence
    pred_std      = float(np.std(preds))
    pred_mean     = float(np.mean(preds))
    agreement_score = max(0.0, 1.0 - (pred_std / max(pred_mean, 1.0))) * 100
    maes = [m for m in [ridge_cv_mae, rf_cv_mae, gb_cv_mae] if m is not None]
    mae_conf = max(0.0, min(100.0, 100.0 - (np.mean(maes) / max(current_rate, 1.0) * 100 * 5))) if maes else 50.0
    n = len(X_scaled)
    size_bonus = min(20.0, n * 0.4)
    raw_conf   = agreement_score * 0.45 + mae_conf * 0.40 + size_bonus * 0.15
    confidence = int(min(92, max(30, round(raw_conf))))

    # Prediction interval
    vol = current_feat.get("volatility", current_rate * 0.003)
    half_range = max(pred_std * 2, vol * 1.5, current_rate * 0.003)
    pred_low   = round(ensemble - half_range, 2)
    pred_high  = round(ensemble + half_range, 2)

    direction = ("BULLISH" if ensemble > current_rate * 1.0005
                 else "BEARISH" if ensemble < current_rate * 0.9995
                 else "NEUTRAL")

    y_in_sample = [float(gb.predict(X_scaled[i:i+1])[0]) for i in range(len(X_scaled))]
    r2 = float(r2_score(y, y_in_sample)) if len(y) > 1 else None

    metrics = {
        "n_training_points": n,
        "ridge_cv_mae": ridge_cv_mae, "rf_cv_mae": rf_cv_mae, "gb_cv_mae": gb_cv_mae,
        "r2_in_sample": r2, "pred_std": pred_std,
        "agreement_score": agreement_score, "mae_conf": mae_conf, "size_bonus": size_bonus,
        "rf_feature_importance": top_features,
    }
    st.session_state.ml_metrics = metrics

    return {
        "cold_start": False,
        "n_training_points": n,
        "ridge_pred": round(ridge_pred, 2), "rf_pred": round(rf_pred, 2),
        "gb_pred": round(gb_pred, 2), "ensemble": round(ensemble, 2),
        "pred_low": pred_low, "pred_high": pred_high,
        "confidence": confidence, "direction": direction,
        "model_agreement": round(agreement_score, 1),
        "ridge_cv_mae": ridge_cv_mae, "rf_cv_mae": rf_cv_mae, "gb_cv_mae": gb_cv_mae,
        "r2_in_sample": r2, "rf_feature_importance": top_features,
        "note": f"Ensemble of Ridge + RF + GradientBoosting trained on {n} observations.",
    }


# ══════════════════════════════════════════════════════
# MULTI-TIMEFRAME FORECASTING ENGINE
# ══════════════════════════════════════════════════════
TIMEFRAMES = [
    {"label": "24H",   "hours": 24,    "key": "24h"},
    {"label": "7 DAY", "hours": 168,   "key": "7d"},
    {"label": "30 DAY","hours": 720,   "key": "30d"},
    {"label": "3 MO",  "hours": 2160,  "key": "3m"},
    {"label": "6 MO",  "hours": 4320,  "key": "6m"},
    {"label": "12 MO", "hours": 8760,  "key": "12m"},
    {"label": "2 YR",  "hours": 17520, "key": "2yr"},
]

def build_multi_timeframe_forecast(current_rate: float, ml: dict, feat: dict, raw: dict) -> dict:
    """
    Generate statistically-grounded multi-timeframe predictions.
    Uses:
    1. ML ensemble directional bias as the near-term anchor
    2. Qualitative macro score for medium-term direction
    3. Historical volatility scaling for uncertainty bounds
    4. Confidence decays with √time (statistical law)
    5. Structural NGN bias toward depreciation over long-term
    """
    ensemble      = ml.get("ensemble", current_rate)
    base_conf     = ml.get("confidence", 35)
    direction     = ml.get("direction", "NEUTRAL")
    vol           = feat.get("volatility", current_rate * 0.005) or current_rate * 0.005
    news_score    = feat.get("news_overall", 0)
    premium_pct   = feat.get("premium_pct", 5)
    trend_slope   = feat.get("trend_slope", 0)
    btc_change    = feat.get("btc_24h_change", 0) or 0
    oil_impact    = feat.get("news_oil", 0) or 0
    cbn_score     = feat.get("news_cbn", 0) or 0
    em_risk       = feat.get("news_em_risk", 0) or 0

    # 24h: tight ML-anchored prediction
    step_24h = (ensemble - current_rate)

    # Structural factors that compound over time
    annual_depreciation_rate = 0.08   # Historical NGN long-term depreciation ~8-15%/yr
    if abs(news_score) > 30:          # Strong qualitative signal
        annual_depreciation_rate += news_score * 0.001
    if premium_pct > 8:               # High black-market premium = structural pressure
        annual_depreciation_rate += 0.02
    if cbn_score > 20:                # CBN tightening bearish for parallel market
        annual_depreciation_rate -= 0.02
    annual_depreciation_rate = max(0.02, min(0.25, annual_depreciation_rate))

    forecasts = {}
    for tf in TIMEFRAMES:
        hours   = tf["hours"]
        years   = hours / 8760.0
        days    = hours / 24.0

        # Central estimate
        if hours <= 24:
            # Near-term: ML-anchored
            central = ensemble
            trend_contrib = step_24h
        elif hours <= 168:
            # 7 days: blend ML direction with weekly trend
            weekly_trend = trend_slope * days * 4   # extrapolate trend
            central = current_rate + step_24h + weekly_trend
        elif hours <= 720:
            # 30 days: qualitative + trend
            monthly_factor = (1 + annual_depreciation_rate) ** years
            central = current_rate * monthly_factor
            # Adjust for current news score
            news_adj = (news_score / 100) * current_rate * 0.015
            central += news_adj
        else:
            # 3m, 6m, 12m, 2yr: fundamental macro model
            # Compound the depreciation rate
            factor = (1 + annual_depreciation_rate) ** years
            central = current_rate * factor
            # Oil impact diminishes over time
            oil_adj = (oil_impact / 100) * current_rate * 0.02 * (1 / (1 + years))
            # EM risk premium
            em_adj  = (em_risk / 100) * current_rate * 0.03 * min(years, 1)
            central += oil_adj + em_adj

        # Uncertainty range: volatility scales with √time
        base_vol = max(vol, current_rate * 0.003)
        vol_scale = base_vol * np.sqrt(days) * 0.4
        # Add structural uncertainty for longer horizons
        structural_uncertainty = current_rate * years * 0.04
        half_range = vol_scale + structural_uncertainty
        # Ensure minimum meaningful range
        min_range = current_rate * 0.005 * np.sqrt(days)
        half_range = max(half_range, min_range)

        low    = round(central - half_range, 0)
        high   = round(central + half_range, 0)
        central = round(central, 0)

        # Confidence: decays with √time, anchored to base confidence
        conf_decay = base_conf * (1 / (1 + np.sqrt(years) * 0.8))
        # Floor: at 2yr, min confidence is 20 (non-trivial direction)
        conf = int(max(20, min(base_conf, round(conf_decay))))

        pct_change = round((central - current_rate) / current_rate * 100, 1)
        direction_label = "▲ HIGHER" if central > current_rate * 1.002 else "▼ LOWER" if central < current_rate * 0.998 else "◆ STABLE"

        # Bull/base/bear scenarios
        bull_case = round(low * 0.95, 0)    # NGN strengthens (USDT falls)
        bear_case = round(high * 1.12, 0)   # NGN weakens more (USDT rises)

        forecasts[tf["key"]] = {
            "label":         tf["label"],
            "hours":         hours,
            "central":       central,
            "low":           low,
            "high":          high,
            "bull_case":     bull_case,
            "bear_case":     bear_case,
            "pct_change":    pct_change,
            "direction":     direction_label,
            "confidence":    conf,
        }

    return forecasts


def build_forecast_narratives(current_rate: float, forecasts: dict, ml: dict, feat: dict, raw: dict) -> dict:
    """Ask Gemini to generate qualitative narratives for each timeframe."""
    q_intel = raw.get("news_intel", {})
    sig = st.session_state.global_signals or {}
    anal = sig.get("analysis", {})

    f24  = forecasts.get("24h", {})
    f7d  = forecasts.get("7d", {})
    f30d = forecasts.get("30d", {})
    f3m  = forecasts.get("3m", {})
    f6m  = forecasts.get("6m", {})
    f12m = forecasts.get("12m", {})
    f2yr = forecasts.get("2yr", {})

    prompt = f"""You are Nigeria's premier FX strategist at a tier-1 investment bank.
Provide sharp, specific multi-timeframe narratives for USDT/NGN predictions.

CURRENT LIVE DATA:
- P2P Mid Rate: ₦{current_rate:,.0f}
- ML Prediction (24h): ₦{f24.get('central',0):,.0f} ({f24.get('pct_change',0):+.1f}%) — Range: ₦{f24.get('low',0):,.0f}–₦{f24.get('high',0):,.0f}
- ML Confidence: {ml.get('confidence',0)}% | Model Agreement: {ml.get('model_agreement',0):.1f}%
- Training Points: {ml.get('n_training_points',0)}

ML FEATURE IMPORTANCES (top):
{json.dumps(dict(list(ml.get('rf_feature_importance',{}).items())[:5]), indent=2)}

QUALITATIVE MACRO SCORES (-100=NGN strengthens, +100=NGN weakens):
- Overall: {feat.get('news_overall',0):+.0f}
- Nigeria Macro: {feat.get('news_nigeria',0):+.0f}
- CBN Policy: {feat.get('news_cbn',0):+.0f}
- Oil Markets: {feat.get('news_oil',0):+.0f}
- USD/Fed: {feat.get('news_usd',0):+.0f}
- Geopolitics: {feat.get('news_geopolitics',0):+.0f}
- EM Risk: {feat.get('news_em_risk',0):+.0f}

KEY INTEL:
- Oil: {q_intel.get('oil_analysis','N/A')}
- CBN: {q_intel.get('cbn_analysis','N/A')}
- Top Bull Catalyst: {q_intel.get('top_bullish_catalyst','N/A')}
- Top Bear Catalyst: {q_intel.get('top_bearish_catalyst','N/A')}
- Medium-term outlook: {anal.get('medium_term_outlook','N/A')}
- Long-term outlook: {anal.get('long_term_outlook','N/A')}
- Structural NGN risks: {anal.get('structural_ngn_risks',[])}

STATISTICAL FORECASTS:
- 24H:  ₦{f24.get('central',0):,.0f} ({f24.get('pct_change',0):+.1f}%) | Conf: {f24.get('confidence',0)}%
- 7D:   ₦{f7d.get('central',0):,.0f} ({f7d.get('pct_change',0):+.1f}%) | Conf: {f7d.get('confidence',0)}%
- 30D:  ₦{f30d.get('central',0):,.0f} ({f30d.get('pct_change',0):+.1f}%) | Conf: {f30d.get('confidence',0)}%
- 3M:   ₦{f3m.get('central',0):,.0f} ({f3m.get('pct_change',0):+.1f}%) | Conf: {f3m.get('confidence',0)}%
- 6M:   ₦{f6m.get('central',0):,.0f} ({f6m.get('pct_change',0):+.1f}%) | Conf: {f6m.get('confidence',0)}%
- 12M:  ₦{f12m.get('central',0):,.0f} ({f12m.get('pct_change',0):+.1f}%) | Conf: {f12m.get('confidence',0)}%
- 2YR:  ₦{f2yr.get('central',0):,.0f} ({f2yr.get('pct_change',0):+.1f}%) | Conf: {f2yr.get('confidence',0)}%

Return ONLY valid JSON (no markdown, no backticks):
{{
  "exec_summary": "<4-5 sentence executive summary combining ML + qualitative signals. What does everything say about USDT/NGN right now?>",
  "trade_recommendation": "<Specific, actionable trading recommendation with timing>",
  "best_convert_time": "<Best time window for NGN→USDT or USDT→NGN conversion based on signals>",

  "n24h_narrative": "<2 sentences: 24h outlook driven by specific signals>",
  "n24h_drivers": ["<driver 1>", "<driver 2>"],
  "n24h_risk": "<Main risk to this 24h forecast>",

  "n7d_narrative": "<2 sentences: 7-day outlook — what events this week drive it?>",
  "n7d_drivers": ["<driver 1>", "<driver 2>"],
  "n7d_risk": "<Main risk to 7d forecast>",

  "n30d_narrative": "<2 sentences: 30-day outlook — macro factors>",
  "n30d_bull_scenario": "<NGN strengthens scenario — what would cause it?>",
  "n30d_bear_scenario": "<NGN weakens scenario — what would cause it?>",

  "n3m_narrative": "<2 sentences: 3-month outlook — policy, oil, elections, IMF>",
  "n6m_narrative": "<2 sentences: 6-month view — Fed rates, Nigeria growth, OPEC>",
  "n12m_narrative": "<2 sentences: 12-month view — structural CBN policy trajectory>",
  "n2yr_narrative": "<2 sentences: 2-year view — Nigeria economic reform trajectory>",

  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>", "<risk 4>"],
  "key_upside_catalysts": ["<catalyst 1>", "<catalyst 2>", "<catalyst 3>"],
  "cbn_watch": "<What to watch from CBN in coming weeks>",
  "oil_impact_summary": "<How oil movements feed through to NGN over each timeframe>",

  "data_quality_note": "<Brief honest assessment: cold start or trained? How many data points? What would improve predictions?>",
  "disclaimer_note": "<Brief risk disclaimer for each timeframe horizon>"
}}"""

    try:
        raw_out = gemini(prompt,
            "You are Nigeria's most rigorous FX strategist. Be specific, cite real events, give real numbers. Return only valid JSON.")
        return _parse_json(raw_out)
    except Exception as e:
        return {
            "exec_summary": "Analysis complete. Statistical models have generated forecasts across all timeframes. See individual timeframe cards for details.",
            "trade_recommendation": "Based on current ML signals and qualitative factors — exercise caution and DYOR.",
            "best_convert_time": "Monitor the next 24-48 hours for confirmation of direction.",
            "n24h_narrative": f"ML models project ₦{forecasts.get('24h',{}).get('central',0):,.0f} within 24 hours.",
            "n7d_narrative": f"7-day projection of ₦{forecasts.get('7d',{}).get('central',0):,.0f} based on current trend.",
            "n30d_narrative": f"30-day target ₦{forecasts.get('30d',{}).get('central',0):,.0f} driven by macro fundamentals.",
            "n3m_narrative": "3-month outlook subject to CBN policy, oil, and global EM conditions.",
            "n6m_narrative": "6-month view reflects structural NGN depreciation pressures.",
            "n12m_narrative": "12-month forecast reflects long-term trajectory of Nigeria FX reform.",
            "n2yr_narrative": "2-year projection reflects fundamental macroeconomic trajectory.",
            "key_risks": ["Policy surprise from CBN", "Global oil price shock", "USD strength surge", "Nigeria political risk"],
            "key_upside_catalysts": ["Oil price rise", "CBN intervention", "Strong remittance inflows"],
            "cbn_watch": "Monitor CBN FX intervention frequency and reserve levels.",
            "oil_impact_summary": "Oil prices directly impact Nigeria FX earnings and NGN support.",
            "data_quality_note": ("ML engine in cold start mode." if ml.get("cold_start") else f"ML engine trained on {ml.get('n_training_points',0)} observations."),
            "disclaimer_note": "All forecasts carry uncertainty. Confidence degrades with horizon. Not financial advice."
        }


def run_full_analysis() -> dict:
    raw, feat = collect_features()
    p2p_mid = raw.get("p2p_mid") or raw.get("p2p_buy") or 1620.0
    raw["p2p_mid"] = p2p_mid

    st.session_state.rate_history.append({
        "timestamp": raw.get("timestamp"),
        "p2p_mid":   p2p_mid,
        "p2p_buy":   raw.get("p2p_buy"),
        "p2p_sell":  raw.get("p2p_sell"),
        "official":  raw.get("official"),
        "features":  feat,
    })
    _save_history()

    ml = train_and_predict(feat, p2p_mid)
    forecasts = build_multi_timeframe_forecast(p2p_mid, ml, feat, raw)
    narratives = build_forecast_narratives(p2p_mid, forecasts, ml, feat, raw)

    return {
        "success": True, "raw": raw, "features": feat,
        "ml": ml, "forecasts": forecasts, "narratives": narratives,
        "timestamp": raw.get("timestamp"),
        "n_history": len(st.session_state.rate_history),
    }


# ══════════════════════════════════════════════════════
# CHAT ENGINE
# ══════════════════════════════════════════════════════
def chat_response(msg: str, result: dict) -> str:
    ml = result.get("ml", {})
    raw = result.get("raw", {})
    feat = result.get("features", {})
    forecasts = result.get("forecasts", {})
    narratives = result.get("narratives", {})

    ctx = f"""USDT/NGN ORACLE — FULL CONTEXT:
Current P2P Rate: ₦{raw.get("p2p_mid", 0):,.2f}
ML Ensemble: ₦{ml.get("ensemble", 0):,.2f} | Direction: {ml.get("direction","N/A")} | Confidence: {ml.get("confidence",0)}%
Training Points: {ml.get("n_training_points",0)} | Cold Start: {ml.get("cold_start",True)}

MULTI-TIMEFRAME FORECASTS:
{json.dumps({k: {
    "central": v.get("central"), "low": v.get("low"), "high": v.get("high"),
    "pct_change": v.get("pct_change"), "confidence": v.get("confidence")
} for k, v in forecasts.items()}, indent=2)}

KEY NARRATIVES:
- Executive Summary: {narratives.get("exec_summary","N/A")}
- 24H: {narratives.get("n24h_narrative","N/A")}
- 7D: {narratives.get("n7d_narrative","N/A")}
- 30D: {narratives.get("n30d_narrative","N/A")}
- Trade Rec: {narratives.get("trade_recommendation","N/A")}
- Key Risks: {narratives.get("key_risks",[])}

QUALITATIVE SCORES:
- Overall: {feat.get("news_overall",0):+.0f} | Nigeria: {feat.get("news_nigeria",0):+.0f} | CBN: {feat.get("news_cbn",0):+.0f}
- Oil: {feat.get("news_oil",0):+.0f} | USD/Fed: {feat.get("news_usd",0):+.0f} | Geopolitics: {feat.get("news_geopolitics",0):+.0f}
- Breaking: {raw.get("news_intel",{}).get("breaking_event","None")}

TOP ML FEATURES: {json.dumps(dict(list(ml.get("rf_feature_importance",{}).items())[:5]))}
"""
    hist = "".join(f"\n{'User' if m['r']=='u' else 'Oracle'}: {m['c']}" for m in st.session_state.chat[-6:])

    system = """You are the USDT/NGN Oracle — Nigeria's most authoritative AI FX analyst.
You have access to live ML models, 40+ news headlines, and multi-timeframe forecasts.
Be precise, cite specific data points, reference actual events. Never be vague.
If a user asks about a specific timeframe, give them the specific numbers.
Speak like a Bloomberg terminal analyst — direct, data-driven, no fluff."""

    return gemini(f"{ctx}\n\nConversation:{hist}\n\nUser: {msg}\n\nOracle:", system, max_tokens=1500)


# ══════════════════════════════════════════════════════
# EMAIL ALERTS
# ══════════════════════════════════════════════════════
def send_email_alert(to_email, subject, html_body, key) -> bool:
    if not to_email or not key: return False
    try:
        r = requests.post("https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"from": "USDT/NGN Oracle <onboarding@resend.dev>",
                  "to": [to_email], "subject": subject, "html": html_body}, timeout=15)
        return r.status_code == 200
    except: return False

def build_email_html(msg, rate, direction, confidence, pred_low, pred_high, recommendation):
    dc = "#00e5a0" if direction=="BULLISH" else "#ff4466" if direction=="BEARISH" else "#ffb020"
    da = "▲" if direction=="BULLISH" else "▼"
    return f"""<div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;
    background:#060912;color:#cfe0f5;border-radius:16px;overflow:hidden;border:1px solid #172440;">
    <div style="background:linear-gradient(135deg,#0f1d30,#060912);padding:28px 32px;border-bottom:2px solid #b060ff;">
    <div style="font-size:10px;letter-spacing:3px;color:#b060ff;text-transform:uppercase;margin-bottom:8px;">
    🇳🇬 USDT/NGN Oracle · Unified Intelligence</div>
    <div style="font-size:22px;font-weight:700;">Price Alert Triggered</div></div>
    <div style="padding:28px 32px;">
    <div style="background:#0f1d30;border:1px solid #172440;border-left:4px solid #ffb020;
    border-radius:10px;padding:14px 18px;margin-bottom:20px;font-size:15px;font-weight:600;color:#ffb020;">
    🔔 {msg}</div>
    <table style="width:100%;border-collapse:collapse;margin-bottom:18px;">
    <tr><td style="padding:9px 0;border-bottom:1px solid #172440;color:#6b84a0;font-size:12px;">Direction</td>
    <td style="text-align:right;font-weight:700;color:{dc};">{da} {direction}</td></tr>
    <tr><td style="padding:9px 0;border-bottom:1px solid #172440;color:#6b84a0;font-size:12px;">Current Rate</td>
    <td style="text-align:right;font-weight:700;color:#00e5a0;">₦{rate:,.0f}</td></tr>
    <tr><td style="padding:9px 0;border-bottom:1px solid #172440;color:#6b84a0;font-size:12px;">24H ML Range</td>
    <td style="text-align:right;font-weight:600;">₦{pred_low:,.0f} – ₦{pred_high:,.0f}</td></tr>
    <tr><td style="padding:9px 0;color:#6b84a0;font-size:12px;">Confidence</td>
    <td style="text-align:right;font-weight:700;color:#ffb020;">{confidence}%</td></tr>
    </table>
    <div style="background:#0f1d30;border:1px solid #172440;border-radius:10px;padding:14px 18px;margin-bottom:20px;">
    <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#b060ff;margin-bottom:7px;">Recommendation</div>
    <div style="font-size:12px;line-height:1.7;color:#9ab0cc;">{recommendation}</div></div>
    <div style="font-size:10px;color:#4a6080;text-align:center;border-top:1px solid #172440;padding-top:14px;line-height:1.6;">
    ⚠️ Not financial advice. ML + AI predictions carry uncertainty. Always DYOR.</div>
    </div></div>"""

def check_and_trigger_alerts(rate, ml, narratives):
    triggered = []
    user_email = st.session_state.get("user_email", "")
    try: rk = st.secrets.get("RESEND_API_KEY", "")
    except: rk = ""
    for i, a in enumerate(st.session_state.alerts):
        msg = ""
        if a["type"] == "above" and rate >= a["level"] and i not in st.session_state.alert_triggered:
            msg = f"Rate crossed ABOVE ₦{a['level']:,} — now at ₦{rate:,.0f}"
            triggered.append((i, f"🔔 {msg}"))
            st.session_state.alert_triggered.append(i)
        elif a["type"] == "below" and rate <= a["level"] and i not in st.session_state.alert_triggered:
            msg = f"Rate dropped BELOW ₦{a['level']:,} — now at ₦{rate:,.0f}"
            triggered.append((i, f"🔔 {msg}"))
            st.session_state.alert_triggered.append(i)
        if msg and user_email and rk:
            d = ml.get("direction","N/A"); c = ml.get("confidence",0)
            html = build_email_html(msg, rate, d, c, ml.get("pred_low",0), ml.get("pred_high",0),
                                    narratives.get("trade_recommendation","See Oracle for details."))
            send_email_alert(user_email, f"🔔 USDT/NGN Oracle Alert: {msg}", html, rk)
    return triggered


# ══════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════
def prog_bar(label, val, color, min_val=-100, max_val=100):
    norm = max(0, min(100, (val - min_val) / (max_val - min_val) * 100))
    st.markdown(f"""<div class="prog-wrap">
    <div class="prog-label">
      <span style="color:var(--text2);font-size:11px;">{label}</span>
      <span style="font-family:var(--font-mono);font-size:11px;color:{color};">{val:+.1f}</span>
    </div>
    <div class="prog-track"><div class="prog-fill" style="width:{norm}%;background:{color};"></div></div>
    </div>""", unsafe_allow_html=True)

def signal_color(score):
    if score > 20:  return "var(--red)"
    if score < -20: return "var(--green)"
    return "var(--amber)"

def headline_color(tag: str) -> str:
    t = tag.upper()
    if any(x in t for x in ["IRAN","MIDEAST","RUSSIA","WAR","CONFLICT","SANCTION"]): return "var(--red)"
    if any(x in t for x in ["OIL","OPEC","BRENT"]): return "var(--amber)"
    if any(x in t for x in ["NGN","NAIRA","CBN"]): return "var(--green)"
    if any(x in t for x in ["BTC","CRYPTO","BITCOIN"]): return "var(--purple)"
    if any(x in t for x in ["FED","USD","DXY","CPI"]): return "var(--blue)"
    return "var(--text2)"

def tf_accent_color(key: str) -> str:
    colors = {"24h":"#00e5a0","7d":"#4488ff","30d":"#b060ff","3m":"#ffb020","6m":"#ff4466","12m":"#00d4ff","2yr":"#ffd700"}
    return colors.get(key, "#4488ff")


# ══════════════════════════════════════════════════════
# ── UI ──
# ══════════════════════════════════════════════════════

# ── ACTION BAR ──
ab1, ab2, ab3, ab4, ab5 = st.columns([2.5, 1.2, 1.2, 1.2, 5])
with ab1:
    run_btn = st.button("⚡ Run Full Analysis", use_container_width=True, type="primary")
with ab2:
    auto_ref = st.toggle("Auto-refresh", value=st.session_state.auto_refresh, key="ar_tog")
    st.session_state.auto_refresh = auto_ref
with ab3:
    if auto_ref:
        iv = st.selectbox("Interval", [15, 30, 60, 120], index=2,
                          format_func=lambda x: f"{x}m", label_visibility="collapsed", key="ar_iv")
        st.session_state.refresh_interval = iv
with ab5:
    if st.session_state.last_time:
        el = int((datetime.datetime.now() - st.session_state.last_time).total_seconds() // 60)
        pts = len(st.session_state.rate_history)
        st.markdown(
            f'<p style="font-family:var(--font-mono);font-size:10px;color:var(--text2);'
            f'margin:10px 0 0 0;text-align:right;">'
            f'<span class="live-dot"></span>Last run {el}m ago &nbsp;·&nbsp; {pts} training pts</p>',
            unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div style="padding:10px 0 20px;">
  <div style="font-family:var(--font-mono);font-size:9px;color:var(--cyan);
  letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
    <span class="live-dot"></span>UNIFIED ML + AI INTELLIGENCE PLATFORM
  </div>
  <h1 style="font-family:var(--font-mono);font-size:28px;font-weight:700;
  margin:0 0 6px;background:linear-gradient(135deg,#cfe0f5 0%,#4488ff 50%,#b060ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;">
    USDT / NGN Oracle
  </h1>
  <p style="color:var(--text2);font-size:12px;margin:0;font-family:var(--font-mono);">
    Ridge · Random Forest · Gradient Boosting · Ensemble · Gemini AI · 24H–2YR Multi-Timeframe Forecasts
  </p>
</div>""", unsafe_allow_html=True)

# ── METHODOLOGY NOTE ──
st.markdown("""
<div class="alert-box alert-info" style="margin-bottom:18px;font-size:12px;">
  <strong style="color:var(--blue);">🧪 How This Works:</strong>
  Three statistical ML models (Ridge, Random Forest, Gradient Boosting) train on your live session history.
  Gemini AI scores 40+ live news headlines across 10 macro dimensions. The combined signals power
  <strong>7 timeframe forecasts</strong> from 24 hours to 2 years.
  <em>Confidence degrades mathematically with horizon — this is a feature, not a bug.</em>
  Reach 5+ runs for full ML accuracy. Run every 15–60 minutes for best training data.
</div>""", unsafe_allow_html=True)


# ── RUN + AUTO-REFRESH ──
if run_btn:
    with st.spinner("Collecting live features · Training ML models · Generating 7-timeframe forecasts · Running Gemini interpretation..."):
        result = run_full_analysis()
        st.session_state.result    = result
        st.session_state.last_time = datetime.datetime.now()
    st.rerun()

if auto_ref and st.session_state.last_time and GEMINI_KEY:
    elapsed_sec  = (datetime.datetime.now() - st.session_state.last_time).total_seconds()
    interval_sec = st.session_state.refresh_interval * 60
    if elapsed_sec >= interval_sec:
        with st.spinner("Auto-refreshing Oracle..."):
            result = run_full_analysis()
            st.session_state.result    = result
            st.session_state.last_time = datetime.datetime.now()
        st.rerun()
    else:
        rem = int((interval_sec - elapsed_sec) // 60)
        st.markdown(f'<p style="font-size:10px;color:var(--green);text-align:right;margin-bottom:0;">🔄 Auto-refresh in {rem}m</p>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ── EMPTY STATE ──
# ══════════════════════════════════════════════════════
if not st.session_state.result:
    pts    = len(st.session_state.rate_history)
    needed = max(0, 5 - pts)
    persisted = (
        f'<div class="alert-box alert-bull" style="max-width:440px;margin:0 auto 12px;">'
        f'📂 Loaded <strong>{pts}</strong> data points from previous sessions. '
        f'{"✅ ML engine is ready!" if pts >= 5 else f"Need {needed} more run(s) for full ML."}'
        f'</div>'
    ) if pts > 0 else ""

    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px 40px;">
      <div style="font-size:50px;margin-bottom:20px;">🔮</div>
      <h2 style="font-family:var(--font-mono);font-size:26px;font-weight:700;
      background:linear-gradient(135deg,#cfe0f5,#4488ff,#b060ff);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;margin-bottom:14px;">Oracle Ready</h2>
      <p style="color:var(--text2);max-width:540px;margin:0 auto 20px;line-height:1.8;font-size:14px;">
        The Unified Oracle combines <strong style="color:var(--amber);">Ridge + Random Forest + Gradient Boosting</strong>
        with <strong style="color:var(--cyan);">live Gemini AI intelligence</strong> to generate
        multi-timeframe forecasts from <strong style="color:var(--green);">24 hours to 2 years</strong>.
      </p>
      {persisted}
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
      max-width:760px;margin:24px auto;">
        {" ".join(f'''<div class="card card-{'green' if i==0 else 'blue' if i==1 else 'purple' if i==2 else 'amber'}" style="padding:18px 14px;">
          <div style="font-size:26px;margin-bottom:8px;">{icon}</div>
          <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;">{title}</div>
          <div style="font-size:10px;color:var(--muted);">{sub}</div>
        </div>''' for i, (icon, title, sub) in enumerate([
            ("📡","Live P2P Rates","Binance · Bybit · CoinGecko"),
            ("🌍","40+ Headlines","18 RSS feeds · Gemini scoring"),
            ("🤖","ML Ensemble","Ridge · RF · GradBoost"),
            ("📈","7 Timeframes","24H → 2YR forecasts"),
        ]))}
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ── MAIN DISPLAY ──
# ══════════════════════════════════════════════════════
else:
    r          = st.session_state.result
    ml         = r.get("ml", {})
    raw        = r.get("raw", {})
    feat       = r.get("features", {})
    forecasts  = r.get("forecasts", {})
    narratives = r.get("narratives", {})
    metrics    = st.session_state.ml_metrics

    p2p_mid  = raw.get("p2p_mid", 0)
    official = raw.get("official", 0) or 0
    ensemble = ml.get("ensemble", 0)
    direction= ml.get("direction", "NEUTRAL")
    conf     = ml.get("confidence", 0)
    pred_low = ml.get("pred_low", 0)
    pred_high= ml.get("pred_high", 0)
    n_pts    = ml.get("n_training_points", 0)
    cold     = ml.get("cold_start", True)
    prem     = feat.get("premium_pct", 0) or 0
    p2p_buy  = raw.get("p2p_buy", 0) or 0
    p2p_sell = raw.get("p2p_sell", 0) or 0
    spread   = (p2p_buy - p2p_sell) if p2p_buy and p2p_sell else 0

    _sig  = st.session_state.global_signals or {}
    _anal = _sig.get("analysis", {})
    _bias = _anal.get("30min_bias", "—")
    _fng  = _sig.get("fear_greed_value", "—")
    _fng_l= _sig.get("fear_greed_label", "N/A")

    dc    = "var(--green)" if direction=="BULLISH" else "var(--red)" if direction=="BEARISH" else "var(--amber)"
    da    = "▲" if direction=="BULLISH" else "▼" if direction=="BEARISH" else "◆"
    cc    = "var(--green)" if conf>=65 else "var(--amber)" if conf>=45 else "var(--red)"
    prem_col = "var(--red)" if prem>8 else "var(--amber)" if prem>4 else "var(--green)"
    bias_col = "var(--green)" if _bias=="BUY" else "var(--red)" if _bias=="SELL" else "var(--amber)"

    # ── LIVE TICKER ──
    btc_u = _sig.get("btc_usd"); btc_c = _sig.get("btc_24h") or 0
    eth_u = _sig.get("eth_usd"); eth_c = _sig.get("eth_24h") or 0
    eur   = _sig.get("eurusd"); dxy = _sig.get("dxy_proxy")
    zar   = _sig.get("usd_zar"); ghs = _sig.get("usd_ghs")

    def ticker_item(label, val, change=None, fmt=""):
        if val is None: return ""
        v_str = f"{fmt}{val:,.2f}" if isinstance(val, float) else f"{val}"
        ch_str = ""
        if change is not None:
            cls = "up" if float(change) >= 0 else "dn"
            ch_str = f' <span class="{cls}">{float(change):+.2f}%</span>'
        return f'<span class="ticker-item">{label} <span class="val">{v_str}</span>{ch_str}</span><span class="ticker-sep">·</span>'

    ticker_items = "".join([
        ticker_item("USDT/NGN P2P", p2p_mid, None, "₦"),
        ticker_item("Official", official, None, "₦"),
        ticker_item("BTC", btc_u, btc_c, "$"),
        ticker_item("ETH", eth_u, eth_c, "$"),
        ticker_item("EUR/USD", eur, None),
        ticker_item("DXY Proxy", dxy, None),
        ticker_item("USD/ZAR", zar, None),
        ticker_item("USD/GHS", ghs, None),
        f'<span class="ticker-item">F&G <span class="val">{_fng} — {_fng_l}</span></span><span class="ticker-sep">·</span>',
        f'<span class="ticker-item">30m Bias <span style="color:{bias_col};font-weight:700;">⚡{_bias}</span></span>',
    ])
    st.markdown(f"""
    <div class="ticker-wrap">
      <div class="ticker-inner">{ticker_items}{ticker_items}</div>
    </div>""", unsafe_allow_html=True)

    # ── TOP METRIC CARDS ──
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    def metric_card(col, accent, label, value, sub, val_color=None):
        vc = val_color or "var(--text)"
        col.markdown(f"""<div class="card card-{accent}">
        <div class="card-label">{label}</div>
        <div class="card-value" style="color:{vc};font-size:22px;">{value}</div>
        <div class="card-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    with c1:
        metric_card(c1, "green", "P2P Live Rate", f"₦{p2p_mid:,.0f}",
                    f"Buy ₦{p2p_buy:,.0f} · Sell ₦{p2p_sell:,.0f}", "var(--green)")
    with c2:
        metric_card(c2, "amber", "B.M. Premium",
                    f"{prem:+.2f}%", f"vs CBN ₦{official:,.0f}", prem_col)
    with c3:
        metric_card(c3, "green" if direction=="BULLISH" else "red" if direction=="BEARISH" else "amber",
                    "ML Prediction 24H", f"{da} ₦{ensemble:,.0f}",
                    f"₦{pred_low:,.0f} – ₦{pred_high:,.0f}", dc)
    with c4:
        metric_card(c4, "purple", "ML Confidence",
                    f"{conf}%", f"{'⚠️ Cold start' if cold else f'✅ {n_pts} training pts'}", cc)
    with c5:
        metric_card(c5, "blue", "News Signal",
                    f"{feat.get('news_overall',0):+.0f}",
                    f"Score: -100=NGN↑ · +100=USDT↑", signal_color(feat.get("news_overall",0)))
    with c6:
        metric_card(c6, "green" if _bias=="BUY" else "red" if _bias=="SELL" else "amber",
                    "30-Min Bias", f"⚡ {_bias}", "Qualitative signal", bias_col)
    with c7:
        f7d_val = forecasts.get("7d", {}).get("central", 0)
        f7d_pct = forecasts.get("7d", {}).get("pct_change", 0)
        c7_color = "var(--green)" if f7d_val > p2p_mid else "var(--red)" if f7d_val < p2p_mid else "var(--amber)"
        metric_card(c7, "cyan", "7-Day Forecast",
                    f"₦{f7d_val:,.0f}", f"{f7d_pct:+.1f}% from now", c7_color)

    if cold:
        st.markdown(f'<div class="alert-box alert-warn" style="margin-top:12px;">⚠️ <strong>Cold Start Mode</strong> — {ml.get("note","")} Run analysis 5+ times to unlock full ML accuracy.</div>',
                    unsafe_allow_html=True)

    # ── TRIGGERED ALERTS ──
    triggered_alerts = check_and_trigger_alerts(p2p_mid, ml, narratives)
    for _, msg in triggered_alerts:
        st.markdown(f'<div class="alert-box alert-warn">{msg}</div>', unsafe_allow_html=True)

    # ── RATE SOURCE BADGE ──
    src = raw.get("rate_source", "unknown")
    status = raw.get("rate_status", "unknown")
    src_col = "var(--green)" if status=="live" else "var(--amber)" if status=="estimated" else "var(--red)"
    ts = raw.get("timestamp","")[:19].replace("T"," ")
    st.markdown(
        f'<p style="font-size:10px;font-family:var(--font-mono);color:{src_col};margin-bottom:0;">'
        f'<span class="live-dot" style="background:{src_col};"></span>'
        f'Source: {src} &nbsp;·&nbsp; Buy ₦{p2p_buy:,.0f} &nbsp;|&nbsp; Sell ₦{p2p_sell:,.0f}'
        f' &nbsp;·&nbsp; <span style="color:var(--muted);">Updated {ts}</span></p>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════ TABS ══════
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Analysis",
        "📈 Forecasts",
        "🌍 Global Signals",
        "💱 Converter",
        "📉 History",
        "💬 Chat",
        "🔔 Alerts",
        "🔬 ML Metrics",
    ])


    # ══════ TAB 1: ANALYSIS ══════
    with tab1:
        left, right = st.columns([3, 2])

        with left:
            # Executive Summary
            exec_sum = narratives.get("exec_summary", "")
            if exec_sum:
                st.markdown(f"""<div class="card card-purple" style="margin-bottom:16px;">
                <div class="sec-header">🧠 EXECUTIVE SUMMARY</div>
                <p style="font-size:13px;color:var(--text);line-height:1.75;margin:0;">{exec_sum}</p>
                </div>""", unsafe_allow_html=True)

            # Trade Recommendation
            trade_rec = narratives.get("trade_recommendation", "")
            best_time = narratives.get("best_convert_time", "")
            if trade_rec:
                dir_badge = f'<span class="badge badge-{"bull" if direction=="BULLISH" else "bear" if direction=="BEARISH" else "neu"}">{da} {direction}</span>'
                st.markdown(f"""<div class="card card-{'green' if direction=='BULLISH' else 'red' if direction=='BEARISH' else 'amber'}" style="margin-bottom:16px;">
                <div class="sec-header">⚡ TRADE RECOMMENDATION &nbsp;&nbsp; {dir_badge}</div>
                <p style="font-size:13px;color:var(--text);line-height:1.7;margin:0 0 10px;">{trade_rec}</p>
                {f'<div style="font-size:11px;color:var(--text2);border-top:1px solid var(--border);padding-top:8px;margin-top:8px;">⏰ Best time: {best_time}</div>' if best_time else ""}
                </div>""", unsafe_allow_html=True)

            # Model Predictions Breakdown
            st.markdown("""<div class="card" style="margin-bottom:16px;">
            <div class="sec-header">🤖 INDIVIDUAL MODEL PREDICTIONS</div>""", unsafe_allow_html=True)
            for badge_cls, label, pred, desc in [
                ("badge-ridge", "Ridge Regression",    ml.get("ridge_pred", 0), "Regularised linear — captures long-term trend direction."),
                ("badge-rf",    "Random Forest",        ml.get("rf_pred", 0),    "Decision tree ensemble — detects non-linear patterns."),
                ("badge-gb",    "Gradient Boosting",    ml.get("gb_pred", 0),    "Sequential error-correction — best for time-series."),
                ("badge-ens",   "Weighted Ensemble",    ml.get("ensemble", 0),   "Ridge×0.25 + RF×0.35 + GB×0.40 — final prediction."),
            ]:
                diff  = pred - p2p_mid
                color = "var(--green)" if diff >= 0 else "var(--red)"
                arr   = "▲" if diff >= 0 else "▼"
                pct   = diff / max(p2p_mid, 1) * 100
                st.markdown(f"""
                <div style="padding:12px 0;border-bottom:1px solid var(--border);">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
                    <span class="model-badge {badge_cls}">{label}</span>
                    <div style="display:flex;align-items:center;gap:12px;">
                      <span style="font-family:var(--font-mono);font-size:17px;font-weight:700;color:{color};">₦{pred:,.0f}</span>
                      <span style="font-family:var(--font-mono);font-size:11px;color:{color};">{arr} {diff:+.0f} ({pct:+.2f}%)</span>
                    </div>
                  </div>
                  <p style="font-size:11px;color:var(--muted);margin:0;">{desc}</p>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Key Headlines Driving This Prediction
            headlines_driving = narratives.get("key_headlines_driving_prediction") or []
            if not headlines_driving:
                headlines_driving = [raw.get("news_intel",{}).get("top_bullish_catalyst",""),
                                     raw.get("news_intel",{}).get("top_bearish_catalyst","")]
            if any(headlines_driving):
                st.markdown("""<div class="card" style="margin-bottom:16px;">
                <div class="sec-header">📰 KEY HEADLINES DRIVING PREDICTION</div>""", unsafe_allow_html=True)
                for h in [x for x in headlines_driving if x]:
                    st.markdown(f'<div class="hl-row">📌 <span style="font-size:12px;color:var(--text);">{h}</span></div>',
                                unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            # Confidence Ring visualization
            conf_color = "var(--green)" if conf>=65 else "var(--amber)" if conf>=45 else "var(--red)"
            st.markdown(f"""<div class="card" style="margin-bottom:16px;text-align:center;padding:24px 22px;">
            <div class="sec-header" style="text-align:left;">📊 CONFIDENCE BREAKDOWN</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:24px;margin:16px 0;">
              <div style="border: 6px solid {conf_color};border-radius:50%;width:100px;height:100px;
              display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <div style="font-family:var(--font-mono);font-size:24px;font-weight:700;color:{conf_color};">{conf}%</div>
                <div style="font-size:9px;color:var(--muted);letter-spacing:1px;">CONF</div>
              </div>
              <div style="text-align:left;">
                <div style="font-size:11px;color:var(--text2);margin-bottom:6px;">Model Agreement</div>
                <div style="font-family:var(--font-mono);font-size:18px;color:var(--blue);">{ml.get("model_agreement",0):.1f}%</div>
                <div style="font-size:10px;color:var(--muted);margin-top:8px;">Training Points</div>
                <div style="font-family:var(--font-mono);font-size:18px;color:var(--purple);">{n_pts}</div>
              </div>
            </div></div>""", unsafe_allow_html=True)

            # Rate Summary Table
            st.markdown("""<div class="card" style="margin-bottom:16px;">
            <div class="sec-header">💹 RATE SUMMARY</div>
            <table class="spread-table">
            <tr><th>Metric</th><th>Value</th></tr>""", unsafe_allow_html=True)
            for lbl, val, clr in [
                ("P2P Live Rate",   f"₦{p2p_mid:,.2f}",              "var(--green)"),
                ("Official (CBN)",  f"₦{official:,.2f}",             "var(--blue)"),
                ("B.M. Premium",    f"{prem:+.2f}%",                  "var(--amber)"),
                ("P2P Spread",      f"₦{spread:.0f}",                 "var(--text2)"),
                ("Ridge Target",    f"₦{ml.get('ridge_pred',0):,.0f}","var(--blue)"),
                ("RF Target",       f"₦{ml.get('rf_pred',0):,.0f}",   "var(--green)"),
                ("GB Target",       f"₦{ml.get('gb_pred',0):,.0f}",   "var(--purple)"),
                ("Ensemble (24H)",  f"₦{ensemble:,.0f}",              "var(--amber)"),
                ("Range Low",       f"₦{pred_low:,.0f}",              "var(--text2)"),
                ("Range High",      f"₦{pred_high:,.0f}",             "var(--text2)"),
            ]:
                st.markdown(f'<tr><td style="font-size:11px;color:var(--text2);">{lbl}</td>'
                            f'<td style="font-family:var(--font-mono);color:{clr};font-size:12px;">{val}</td></tr>',
                            unsafe_allow_html=True)
            st.markdown('</table></div>', unsafe_allow_html=True)

            # Key Risks
            risks = narratives.get("key_risks", [])
            catalysts = narratives.get("key_upside_catalysts", [])
            if risks or catalysts:
                st.markdown("""<div class="card" style="margin-bottom:16px;">
                <div class="sec-header">⚠️ KEY RISKS & CATALYSTS</div>""", unsafe_allow_html=True)
                for r_item in risks[:4]:
                    if r_item:
                        st.markdown(f'<div style="font-size:11px;color:#ffaabb;padding:5px 0;border-bottom:1px solid var(--border);">🔻 {r_item}</div>',
                                    unsafe_allow_html=True)
                for c_item in catalysts[:3]:
                    if c_item:
                        st.markdown(f'<div style="font-size:11px;color:#a0ead4;padding:5px 0;border-bottom:1px solid var(--border);">🔺 {c_item}</div>',
                                    unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Full-width qualitative section
        q_intel     = raw.get("news_intel", {})
        n_headlines = raw.get("news_headlines_count", 0)

        if q_intel:
            st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;
            text-transform:uppercase;color:var(--purple);margin-bottom:16px;">
            🌐 LIVE WORLD INTELLIGENCE — {n_headlines} headlines analysed</div>""", unsafe_allow_html=True)

            breaking = q_intel.get("breaking_event")
            if breaking and str(breaking).lower() not in ("null","none","n/a",""):
                st.markdown(f"""<div style="background:rgba(255,68,102,0.1);border:1px solid var(--red);
                border-left:4px solid var(--red);border-radius:10px;padding:12px 16px;margin-bottom:14px;">
                <span style="font-size:9px;color:var(--red);font-family:var(--font-mono);
                letter-spacing:2px;text-transform:uppercase;">⚡ BREAKING EVENT</span>
                <div style="font-size:13px;color:var(--text);margin-top:5px;line-height:1.6;">{breaking}</div>
                </div>""", unsafe_allow_html=True)

            # 3 analysis cards
            qa1, qa2, qa3 = st.columns(3)
            with qa1:
                oil_s = feat.get("news_oil", 0)
                oil_c = signal_color(oil_s)
                st.markdown(f"""<div class="card" style="height:100%;margin-bottom:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                  <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;
                  text-transform:uppercase;color:var(--muted);">🛢️ Oil Markets</div>
                  <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{oil_c};">{oil_s:+.0f}</div>
                </div>
                <p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0;">
                {q_intel.get("oil_analysis","No oil analysis available.")}</p></div>""", unsafe_allow_html=True)
            with qa2:
                geo_s = feat.get("news_geopolitics", 0)
                geo_c = signal_color(geo_s)
                bull  = q_intel.get("top_bullish_catalyst","")[:160]
                st.markdown(f"""<div class="card" style="height:100%;margin-bottom:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                  <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;
                  text-transform:uppercase;color:var(--muted);">🌍 Geopolitics</div>
                  <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{geo_c};">{geo_s:+.0f}</div>
                </div>
                <p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0 0 10px;">
                {q_intel.get("geopolitical_analysis","N/A")}</p>
                <div style="font-size:10px;color:var(--muted);">📈 Bullish: <span style="color:var(--text2);">{bull}</span></div>
                </div>""", unsafe_allow_html=True)
            with qa3:
                cbn_s = feat.get("news_cbn", 0)
                cbn_c = signal_color(cbn_s)
                bear  = q_intel.get("top_bearish_catalyst","")[:160]
                st.markdown(f"""<div class="card" style="height:100%;margin-bottom:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                  <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;
                  text-transform:uppercase;color:var(--muted);">🏦 CBN Watch</div>
                  <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{cbn_c};">{cbn_s:+.0f}</div>
                </div>
                <p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0 0 10px;">
                {q_intel.get("cbn_analysis","N/A")}</p>
                <div style="font-size:10px;color:var(--muted);">📉 Bearish: <span style="color:var(--text2);">{bear}</span></div>
                </div>""", unsafe_allow_html=True)

            # 10-dimension score bars
            st.markdown('<div class="card" style="margin-top:14px;"><div class="sec-header">📡 10-DIMENSION QUALITATIVE SIGNAL SCORES</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;color:var(--muted);margin-bottom:14px;">-100 = very bullish NGN (USDT falls) &nbsp; · &nbsp; +100 = very bullish USDT (NGN weakens)</div>',
                        unsafe_allow_html=True)
            dims = [
                ("Overall Signal",    feat.get("news_overall",0)),
                ("Nigeria Macro",     feat.get("news_nigeria",0)),
                ("CBN Policy",        feat.get("news_cbn",0)),
                ("Oil Markets",       feat.get("news_oil",0)),
                ("USD / Fed",         feat.get("news_usd",0)),
                ("Crypto Sentiment",  feat.get("news_crypto",0)),
                ("Geopolitics",       feat.get("news_geopolitics",0)),
                ("Nigeria Politics",  feat.get("news_political_risk",0)),
                ("Remittances",       feat.get("news_remittance",0)),
                ("Global EM Risk",    feat.get("news_em_risk",0)),
            ]
            cols_dim = st.columns(2)
            for i, (lbl, score) in enumerate(dims):
                with cols_dim[i % 2]:
                    prog_bar(lbl, score, signal_color(score))
            st.markdown('</div>', unsafe_allow_html=True)


    # ══════ TAB 2: MULTI-TIMEFRAME FORECASTS ══════
    with tab2:
        st.markdown(f"""<div class="card card-purple" style="margin-bottom:20px;">
        <div class="sec-header">📈 MULTI-TIMEFRAME FORECAST ENGINE</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-size:12px;color:var(--text2);">
          <div>📌 <strong style="color:var(--text);">Current Rate:</strong> <span style="font-family:var(--font-mono);color:var(--green);">₦{p2p_mid:,.0f}</span></div>
          <div>🤖 <strong style="color:var(--text);">ML Training Points:</strong> <span style="font-family:var(--font-mono);color:var(--amber);">{n_pts}</span></div>
          <div>📰 <strong style="color:var(--text);">Headlines Analysed:</strong> <span style="font-family:var(--font-mono);color:var(--blue);">{n_headlines}</span></div>
        </div>
        <div style="margin-top:10px;font-size:11px;color:var(--muted);">
          ℹ️ Confidence degrades with time horizon (statistical law: uncertainty ∝ √time).
          Near-term forecasts are ML-anchored; medium/long-term use macro depreciation models + qualitative scores.
        </div>
        </div>""", unsafe_allow_html=True)

        # ── 7 TIMEFRAME CARDS ──
        # Row 1: 24H + 7D + 30D
        tf_row1 = st.columns(3)
        tf_keys_row1 = ["24h","7d","30d"]
        # Row 2: 3M + 6M + 12M + 2YR
        tf_row2 = st.columns(4)
        tf_keys_row2 = ["3m","6m","12m","2yr"]

        for cols, keys in [(tf_row1, tf_keys_row1), (tf_row2, tf_keys_row2)]:
            for col, key in zip(cols, keys):
                f  = forecasts.get(key, {})
                ac = tf_accent_color(key)
                central = f.get("central", 0)
                pct = f.get("pct_change", 0)
                direction_lbl = f.get("direction","")
                fconf = f.get("confidence", 0)
                flow = f.get("low", 0)
                fhigh = f.get("high", 0)
                bull_c = f.get("bull_case", 0)
                bear_c = f.get("bear_case", 0)
                fcolor = "var(--green)" if "HIGHER" in direction_lbl else "var(--red)" if "LOWER" in direction_lbl else "var(--amber)"
                col.markdown(f"""
                <div class="tf-card">
                  <div class="tf-card-accent" style="background:{ac};"></div>
                  <div class="tf-label">{f.get("label","")}</div>
                  <div class="tf-value" style="color:{ac};">₦{central:,.0f}</div>
                  <div class="tf-change" style="color:{fcolor};">{direction_lbl}</div>
                  <div class="tf-range">Range: ₦{flow:,.0f} – ₦{fhigh:,.0f}</div>
                  <div style="font-family:var(--font-mono);font-size:12px;color:{fcolor};margin-top:4px;">{pct:+.1f}% from now</div>
                  <div class="tf-conf">Confidence: {fconf}%</div>
                  <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:10px;color:var(--muted);">
                    Bull ₦{bull_c:,.0f} &nbsp;|&nbsp; Bear ₦{bear_c:,.0f}
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── NARRATIVE SECTION ──
        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:16px;">💬 TIMEFRAME NARRATIVES</div>',
                    unsafe_allow_html=True)

        nar_data = [
            ("24H",   "24h", "n24h_narrative",  "n24h_risk",   "n24h_drivers"),
            ("7 DAY", "7d",  "n7d_narrative",   "n7d_risk",    "n7d_drivers"),
            ("30 DAY","30d", "n30d_narrative",  None,          None),
            ("3 MO",  "3m",  "n3m_narrative",   None,          None),
            ("6 MO",  "6m",  "n6m_narrative",   None,          None),
            ("12 MO", "12m", "n12m_narrative",  None,          None),
            ("2 YR",  "2yr", "n2yr_narrative",  None,          None),
        ]

        for lbl, key, nar_key, risk_key, drivers_key in nar_data:
            nar   = narratives.get(nar_key, "")
            risk  = narratives.get(risk_key, "") if risk_key else ""
            drvrs = narratives.get(drivers_key, []) if drivers_key else []
            ac    = tf_accent_color(key)
            f_data= forecasts.get(key, {})
            f_pct = f_data.get("pct_change", 0)
            f_conf= f_data.get("confidence", 0)
            fclr  = "var(--green)" if f_pct > 0.5 else "var(--red)" if f_pct < -0.5 else "var(--amber)"
            central_val = f_data.get("central", 0)
            fallback_nar = f"Forecast: \u20a6{central_val:,.0f} ({f_pct:+.1f}% from current rate)"

            drivers_html = "".join([
                f'<span style="background:rgba(68,136,255,0.12);color:var(--blue);border:1px solid rgba(68,136,255,0.25);'
                f'border-radius:4px;padding:2px 8px;font-size:10px;font-family:var(--font-mono);margin-right:6px;">{d}</span>'
                for d in drvrs if d
            ])
            drivers_row = f'<div style="margin-bottom:6px;">{drivers_html}</div>' if drivers_html else ""
            risk_row    = f'<div style="margin-top:8px;font-size:11px;color:#ffaabb;"><strong>\u26a0\ufe0f Key risk:</strong> {risk}</div>' if risk else ""
            nar_text    = nar if nar else fallback_nar

            # Render the main card (no scenarios_html inside — avoids CSS {} conflict)
            st.markdown(
                f'<div class="card" style="margin-bottom:10px;border-left:3px solid {ac};">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<div style="font-family:var(--font-mono);font-size:11px;font-weight:700;color:{ac};">{lbl} HORIZON</div>'
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f'<span style="font-family:var(--font-mono);font-size:13px;color:{fclr};font-weight:700;">'
                f'\u20a6{central_val:,.0f} ({f_pct:+.1f}%)</span>'
                f'<span style="font-family:var(--font-mono);font-size:9px;color:var(--muted);">CONF {f_conf}%</span>'
                f'</div></div>'
                f'{drivers_row}'
                f'<p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0;">{nar_text}</p>'
                f'{risk_row}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Render 30d bull/bear scenarios as a SEPARATE st.markdown call (avoids f-string CSS {} clash)
            if key == "30d":
                bull_s = narratives.get("n30d_bull_scenario", "")
                bear_s = narratives.get("n30d_bear_scenario", "")
                if bull_s or bear_s:
                    st.markdown(
                        '<div style="display:flex;gap:10px;margin-top:-6px;margin-bottom:10px;">'
                        '<div class="scenario-card scenario-bull" style="flex:1;">'
                        '<div style="font-size:9px;color:var(--green);font-family:var(--font-mono);'
                        'letter-spacing:1px;margin-bottom:5px;">&#x1F4C8; BULL SCENARIO</div>'
                        f'<div style="font-size:11px;color:var(--text2);line-height:1.5;">{bull_s}</div>'
                        '</div>'
                        '<div class="scenario-card scenario-bear" style="flex:1;">'
                        '<div style="font-size:9px;color:var(--red);font-family:var(--font-mono);'
                        'letter-spacing:1px;margin-bottom:5px;">&#x1F4C9; BEAR SCENARIO</div>'
                        f'<div style="font-size:11px;color:var(--text2);line-height:1.5;">{bear_s}</div>'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

        # Oil impact + CBN watch
        oil_impact_sum = narratives.get("oil_impact_summary", "")
        cbn_watch_str  = narratives.get("cbn_watch", "")
        if oil_impact_sum or cbn_watch_str:
            oi1, oi2 = st.columns(2)
            with oi1:
                if oil_impact_sum:
                    st.markdown(f"""<div class="card">
                    <div class="sec-header">🛢️ OIL → NGN TRANSMISSION MECHANISM</div>
                    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{oil_impact_sum}</p>
                    </div>""", unsafe_allow_html=True)
            with oi2:
                if cbn_watch_str:
                    st.markdown(f"""<div class="card">
                    <div class="sec-header">🏦 CBN WATCH LIST</div>
                    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{cbn_watch_str}</p>
                    </div>""", unsafe_allow_html=True)

        # Data quality note
        dq_note = narratives.get("data_quality_note", "")
        disc_note = narratives.get("disclaimer_note", "")
        if dq_note:
            st.markdown(f'<div class="alert-box alert-info" style="margin-top:14px;font-size:11px;"><strong>🔬 Data Quality:</strong> {dq_note}</div>',
                        unsafe_allow_html=True)
        if disc_note:
            st.markdown(f'<div style="font-size:10px;color:var(--muted);margin-top:8px;line-height:1.6;">⚠️ {disc_note}</div>',
                        unsafe_allow_html=True)


    # ══════ TAB 3: GLOBAL SIGNALS ══════
    with tab3:
        sig  = st.session_state.global_signals or {}
        anal = sig.get("analysis", {})
        now_gs = st.session_state.global_signals_time

        age_str, next_str = "never", "5m 0s"
        if now_gs:
            age_s   = int((datetime.datetime.now() - now_gs).total_seconds())
            age_str = f"{age_s}s ago" if age_s < 60 else f"{age_s//60}m {age_s%60}s ago"
            rem     = max(0, SIGNALS_TTL - age_s)
            next_str= f"{rem//60}m {rem%60}s"

        gs_col1, gs_col2 = st.columns([6,1])
        with gs_col1:
            n_hl = sig.get("headline_count", 0)
            n_src= len(sig.get("sources", []))
            st.markdown(f"""<div style="margin-bottom:14px;">
            <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;
            text-transform:uppercase;color:var(--purple);margin-bottom:4px;">🌐 LIVE GLOBAL SIGNALS</div>
            <div style="font-size:11px;color:var(--text2);">
              {n_hl} headlines &nbsp;·&nbsp; {n_src} sources &nbsp;·&nbsp;
              <span style="color:var(--green);">Updated {age_str}</span> &nbsp;·&nbsp;
              Next refresh: {next_str}
            </div></div>""", unsafe_allow_html=True)
        with gs_col2:
            if st.button("↻ Refresh", key="gs_refresh", use_container_width=True):
                maybe_refresh_signals(force=True)
                st.rerun()

        if not sig:
            st.markdown('<div class="card" style="text-align:center;padding:40px;"><div style="font-size:32px;margin-bottom:12px;">📡</div><p style="color:var(--text2);">Press ↻ Refresh to load live global signals.</p></div>',
                        unsafe_allow_html=True)
        else:
            # Breaking event
            breaking = anal.get("breaking_event","")
            if breaking and str(breaking).lower() not in ("null","none","n/a",""):
                st.markdown(f"""<div style="background:rgba(255,68,102,0.1);border:1px solid var(--red);
                border-left:4px solid var(--red);border-radius:10px;padding:12px 16px;margin-bottom:12px;">
                <span style="font-size:9px;color:var(--red);font-family:var(--font-mono);
                letter-spacing:2px;text-transform:uppercase;">⚡ BREAKING EVENT</span>
                <div style="font-size:13px;color:var(--text);margin-top:5px;line-height:1.6;">{breaking}</div>
                </div>""", unsafe_allow_html=True)

            # Top mover
            top_mover = anal.get("top_mover_today","")
            if top_mover:
                st.markdown(f"""<div style="background:var(--purple2);border:1px solid rgba(176,96,255,0.3);
                border-radius:10px;padding:10px 16px;margin-bottom:14px;">
                <span style="font-size:9px;color:var(--purple);font-family:var(--font-mono);
                letter-spacing:2px;text-transform:uppercase;">🎯 TOP MOVER TODAY</span>
                <div style="font-size:12px;color:var(--text);margin-top:5px;">{top_mover}</div>
                </div>""", unsafe_allow_html=True)

            # Market metrics row
            btc_u = sig.get("btc_usd"); btc_c = sig.get("btc_24h") or 0
            eth_u = sig.get("eth_usd"); eth_c = sig.get("eth_24h") or 0
            eur   = sig.get("eurusd"); dxy = sig.get("dxy_proxy")
            zar   = sig.get("usd_zar"); fng_v = sig.get("fear_greed_value","—")
            fng_l = sig.get("fear_greed_label","N/A")

            gm = st.columns(4)
            for col, lbl, val, sub, clr in [
                (gm[0], "BTC/USD",        f"${btc_u:,.0f}" if btc_u else "N/A", f"{btc_c:+.2f}% 24h", "var(--green)" if btc_c>=0 else "var(--red)"),
                (gm[1], "ETH/USD",        f"${eth_u:,.0f}" if eth_u else "N/A", f"{eth_c:+.2f}% 24h", "var(--green)" if eth_c>=0 else "var(--red)"),
                (gm[2], "EUR/USD",        f"{eur:.4f}" if eur else "N/A",       f"DXY proxy: {dxy}", "var(--blue)"),
                (gm[3], "Fear & Greed",   f"{fng_v}",                           fng_l, "var(--green)" if isinstance(fng_v,int) and fng_v>60 else "var(--red)" if isinstance(fng_v,int) and fng_v<30 else "var(--amber)"),
            ]:
                col.markdown(f"""<div class="card" style="margin-bottom:12px;">
                <div class="card-label">{lbl}</div>
                <div style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:{clr};">{val}</div>
                <div class="card-sub">{sub}</div></div>""", unsafe_allow_html=True)

            # Signal cards: Oil, Geo, CBN, Crypto, EM
            signal_cards = [
                ("🛢️ Oil Markets",      "oil_impact",           "oil_analysis",           "CoinGecko+RSS",   "OIL-ORACLE"),
                ("🌍 Geopolitics",       "geopolitical_risk",    "geopolitical_analysis",  "Google News",     "GEO-SIGNAL"),
                ("🏦 CBN Policy",        "cbn_policy",           "cbn_analysis",           "CBN Watch",       "CBN-POLICY"),
                ("₿ Crypto Sentiment",   "crypto_sentiment",     "crypto_analysis",        "CoinGecko",       "CRYPTO-SIG"),
                ("📊 EM FX Risk",        "global_em_risk",       "em_analysis",            "FX Data",         "EM-RISK"),
                ("💸 Remittance Flow",   "remittance_flow",      None,                     "NG News",         "REMIT-SIG"),
            ]
            for icon_lbl, score_key, analysis_key, src_lbl, src_tag in signal_cards:
                score = anal.get(score_key, 0) or 0
                body  = anal.get(analysis_key, "") if analysis_key else f"Score: {score:+.0f}"
                score_c = signal_color(score)
                score_lbl = "BEARISH NGN" if score > 15 else "BULLISH NGN" if score < -15 else "NEUTRAL"
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--r-lg);
                padding:14px 18px;margin-bottom:10px;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
                    <span style="font-family:var(--font-mono);font-size:11px;font-weight:700;color:var(--text);">{icon_lbl}</span>
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{score_c};">{score:+.0f}</span>
                      <span style="background:rgba({score_c.replace('var(--','').replace(')','')},0.1);color:{score_c};
                      border:1px solid {score_c}44;border-radius:4px;padding:2px 8px;font-size:9px;
                      font-family:var(--font-mono);font-weight:700;">{score_lbl}</span>
                    </div>
                  </div>
                  <div style="font-size:12px;color:var(--text2);line-height:1.65;">
                    {body if body else "No analysis available."}
                  </div>
                </div>""", unsafe_allow_html=True)

            # Bull/Bear catalysts
            bull = anal.get("top_bullish_catalyst","")
            bear = anal.get("top_bearish_catalyst","")
            bb1, bb2 = st.columns(2)
            if bull:
                bb1.markdown(f"""<div style="background:var(--green3);border:1px solid rgba(0,229,160,0.3);
                border-radius:var(--r-md);padding:14px 16px;">
                <div style="font-size:9px;color:var(--green);font-family:var(--font-mono);
                letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">📈 Top Bullish Catalyst (USDT)</div>
                <div style="font-size:12px;color:var(--text2);line-height:1.65;">{bull}</div>
                </div>""", unsafe_allow_html=True)
            if bear:
                bb2.markdown(f"""<div style="background:rgba(255,68,102,0.05);border:1px solid rgba(255,68,102,0.25);
                border-radius:var(--r-md);padding:14px 16px;">
                <div style="font-size:9px;color:var(--red);font-family:var(--font-mono);
                letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">📉 Top Bearish Catalyst (USDT)</div>
                <div style="font-size:12px;color:var(--text2);line-height:1.65;">{bear}</div>
                </div>""", unsafe_allow_html=True)

            # Watch items
            watch = anal.get("key_watch_items", [])
            if watch:
                st.markdown(f"""<div style="margin-top:14px;font-size:11px;border-top:1px solid var(--border);padding-top:12px;color:var(--muted);">
                <strong style="font-family:var(--font-mono);font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);">👁 WATCH THIS WEEK: </strong>
                {"  &nbsp;·&nbsp;  ".join(f'<span style="color:var(--amber);">{w}</span>' for w in watch)}
                </div>""", unsafe_allow_html=True)

            # Medium/Long-term outlooks from qualitative
            mt_outlook = anal.get("medium_term_outlook","")
            lt_outlook = anal.get("long_term_outlook","")
            if mt_outlook or lt_outlook:
                mo1, mo2 = st.columns(2)
                if mt_outlook:
                    mo1.markdown(f"""<div class="card" style="margin-top:14px;">
                    <div class="sec-header">📅 30–90 DAY QUALITATIVE OUTLOOK</div>
                    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{mt_outlook}</p>
                    </div>""", unsafe_allow_html=True)
                if lt_outlook:
                    mo2.markdown(f"""<div class="card" style="margin-top:14px;">
                    <div class="sec-header">📆 6–12 MONTH QUALITATIVE OUTLOOK</div>
                    <p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{lt_outlook}</p>
                    </div>""", unsafe_allow_html=True)

            # All Headlines
            headlines = sig.get("headlines", [])
            if headlines:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander(f"📋 All {len(headlines)} live scraped headlines"):
                    for h in headlines:
                        clr  = headline_color(h.get("tag",""))
                        desc = f'<br><span style="color:var(--muted);font-size:10px;">{h["desc"]}</span>' if h.get("desc") else ""
                        st.markdown(
                            f'<div class="hl-row"><span style="color:{clr};font-family:var(--font-mono);">{h.get("tag","")}</span> '
                            f'<span style="color:var(--text);">{h.get("title","")}</span>{desc}</div>',
                            unsafe_allow_html=True)


    # ══════ TAB 4: CONVERTER ══════
    with tab4:
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:16px;">💱 SMART CURRENCY CONVERTER</div>',
                    unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            usdt_in = st.number_input("USDT Amount", min_value=0.0, value=100.0, step=10.0)
            rates_for_conv = {
                "P2P Sell (you receive)": p2p_sell or p2p_mid,
                "P2P Mid": p2p_mid,
                "P2P Buy (you pay)": p2p_buy or p2p_mid,
                "Official (CBN)": official or p2p_mid,
                "ML Prediction (24H)": ensemble,
                "7D Forecast": forecasts.get("7d",{}).get("central", p2p_mid),
                "30D Forecast": forecasts.get("30d",{}).get("central", p2p_mid),
            }
            st.markdown('<div class="card"><div class="sec-header">USDT → NGN</div>', unsafe_allow_html=True)
            for lbl, rate_v in rates_for_conv.items():
                if rate_v:
                    ngn_out = usdt_in * rate_v
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);">'
                                f'<span style="font-size:12px;color:var(--text2);">{lbl}</span>'
                                f'<span style="font-family:var(--font-mono);font-size:13px;font-weight:700;color:var(--green);">₦{ngn_out:,.2f}</span>'
                                f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            ngn_in = st.number_input("NGN Amount", min_value=0.0, value=100000.0, step=1000.0)
            st.markdown('<div class="card"><div class="sec-header">NGN → USDT</div>', unsafe_allow_html=True)
            for lbl, rate_v in rates_for_conv.items():
                if rate_v:
                    usdt_out = ngn_in / rate_v
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);">'
                                f'<span style="font-size:12px;color:var(--text2);">{lbl}</span>'
                                f'<span style="font-family:var(--font-mono);font-size:13px;font-weight:700;color:var(--cyan);">${usdt_out:,.4f}</span>'
                                f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


    # ══════ TAB 5: HISTORY ══════
    with tab5:
        hist_data = st.session_state.rate_history
        if len(hist_data) < 2:
            st.markdown('<div class="card" style="text-align:center;padding:40px;"><p style="color:var(--text2);">Run the analysis at least twice to see history.</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:10px;font-family:var(--font-mono);color:var(--text2);margin-bottom:16px;"><span class="live-dot"></span>{len(hist_data)} data points in history</div>',
                        unsafe_allow_html=True)
            df = pd.DataFrame([{
                "Time":     d.get("timestamp","")[:16].replace("T"," "),
                "P2P Mid":  d.get("p2p_mid"),
                "P2P Buy":  d.get("p2p_buy"),
                "P2P Sell": d.get("p2p_sell"),
                "Official": d.get("official"),
            } for d in hist_data[-50:]])
            df = df.dropna(subset=["P2P Mid"])

            import json as _json
            # Chart data
            chart_vals = [d.get("p2p_mid") for d in hist_data[-50:] if d.get("p2p_mid")]
            if len(chart_vals) >= 2:
                chart_min = min(chart_vals) * 0.998
                chart_max = max(chart_vals) * 1.002
                n_pts_chart = len(chart_vals)
                path_d = " ".join(
                    f"{'M' if i==0 else 'L'} {i*(560/(n_pts_chart-1)):.1f} {((chart_max - v)/(chart_max-chart_min)*80):.1f}"
                    for i, v in enumerate(chart_vals)
                )
                latest_v = chart_vals[-1]
                trend_up = chart_vals[-1] >= chart_vals[0]
                line_color = "#00e5a0" if trend_up else "#ff4466"

                st.markdown(f"""<div class="card" style="margin-bottom:16px;">
                <div class="sec-header">📉 P2P RATE CHART (last {n_pts_chart} observations)</div>
                <svg width="100%" viewBox="0 0 600 100" style="overflow:visible;">
                  <defs>
                    <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stop-color="{line_color}" stop-opacity="0.2"/>
                      <stop offset="100%" stop-color="{line_color}" stop-opacity="0"/>
                    </linearGradient>
                  </defs>
                  <path d="{path_d} L {560:.1f} 80 L 0 80 Z" fill="url(#cg)"/>
                  <path d="{path_d}" fill="none" stroke="{line_color}" stroke-width="2"/>
                  <text x="0" y="90" fill="var(--muted)" font-size="9" font-family="JetBrains Mono">₦{chart_min:,.0f}</text>
                  <text x="0" y="8" fill="var(--muted)" font-size="9" font-family="JetBrains Mono">₦{chart_max:,.0f}</text>
                  <text x="480" y="90" fill="{line_color}" font-size="9" font-family="JetBrains Mono">₦{latest_v:,.0f}</text>
                </svg></div>""", unsafe_allow_html=True)

            # Table
            st.markdown('<div class="card"><div class="sec-header">DATA TABLE</div>', unsafe_allow_html=True)
            st.dataframe(
                df.tail(30).sort_values("Time", ascending=False),
                use_container_width=True,
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Stats
            if len(chart_vals) >= 3:
                s1, s2, s3, s4 = st.columns(4)
                for col, lbl, val, clr in [
                    (s1, "Avg Rate", f"₦{np.mean(chart_vals):,.0f}", "var(--text)"),
                    (s2, "Volatility (σ)", f"₦{np.std(chart_vals):,.0f}", "var(--amber)"),
                    (s3, "Session High", f"₦{max(chart_vals):,.0f}", "var(--green)"),
                    (s4, "Session Low", f"₦{min(chart_vals):,.0f}", "var(--red)"),
                ]:
                    col.markdown(f"""<div class="card" style="margin-top:12px;padding:14px 16px;">
                    <div class="card-label">{lbl}</div>
                    <div style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:{clr};">{val}</div>
                    </div>""", unsafe_allow_html=True)


    # ══════ TAB 6: CHAT ══════
    with tab6:
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:12px;">💬 ASK THE ORACLE</div>',
                    unsafe_allow_html=True)

        # Quick prompts
        quick_prompts = [
            "What's your 7-day forecast and why?",
            "Which timeframe has the highest confidence?",
            "What are the biggest risks to the 30-day forecast?",
            "Should I convert USDT to NGN today?",
            "How does oil price affect NGN right now?",
            "What's the 2-year structural outlook?",
        ]
        clicked = None
        qcols = st.columns(3)
        for i, qp in enumerate(quick_prompts):
            if qcols[i%3].button(qp, key=f"qp_{i}", use_container_width=True):
                clicked = qp

        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

        # Chat messages
        for msg_data in st.session_state.chat:
            if msg_data["r"] == "u":
                st.markdown(f'<div class="chat-u">👤 {msg_data["c"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-a"><div class="chat-badge">🔮 ORACLE</div>{msg_data["c"]}</div>',
                            unsafe_allow_html=True)

        inp_col, btn_col = st.columns([5,1])
        with inp_col:
            user_msg = st.text_input("Ask Oracle anything about USDT/NGN...", key="chat_input", label_visibility="collapsed",
                                     placeholder="e.g. What's the 30-day forecast and what's driving it?")
        with btn_col:
            send = st.button("Send →", use_container_width=True)

        question = user_msg if (send and user_msg) else clicked
        if question:
            st.session_state.chat.append({"r":"u","c":question})
            with st.spinner("Oracle analyzing..."):
                reply = chat_response(question, r)
            st.session_state.chat.append({"r":"a","c":reply})
            st.rerun()

        if st.session_state.chat:
            if st.button("🗑 Clear Chat", key="clear_chat"):
                st.session_state.chat = []
                st.rerun()


    # ══════ TAB 7: ALERTS ══════
    with tab7:
        al1, al2 = st.columns(2)
        with al1:
            st.markdown("""<div class="card"><div class="sec-header">📧 EMAIL ALERTS</div>
            <p style="font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:12px;">
            Get notified when the rate crosses your target — includes live ML prediction and recommendation.</p>
            </div>""", unsafe_allow_html=True)

            user_email = st.text_input("Your Email Address", value=st.session_state.user_email,
                                       placeholder="yourname@gmail.com", key="alert_email_input")
            st.session_state.user_email = user_email

            if user_email:
                if st.button("🧪 Send Test Email", use_container_width=True):
                    rk = RESEND_KEY
                    if rk:
                        test_html = build_email_html("Test — Oracle email is connected!",
                                                     p2p_mid, direction, conf, pred_low, pred_high,
                                                     narratives.get("trade_recommendation","See Oracle for full analysis."))
                        ok = send_email_alert(user_email, "✅ USDT/NGN Oracle — Test Alert", test_html, rk)
                        if ok: st.success("✅ Test email sent!")
                        else:  st.error("❌ Failed. Check RESEND_API_KEY in secrets.")
                    else:
                        st.error("❌ RESEND_API_KEY not configured in secrets.toml.")
                if RESEND_KEY:
                    st.markdown('<div class="alert-box alert-bull" style="margin-top:8px;font-size:11px;">✅ Email service connected</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-box alert-warn" style="margin-top:8px;font-size:11px;">⚠️ Add RESEND_API_KEY to secrets.toml. Free at resend.com</div>',
                                unsafe_allow_html=True)

        with al2:
            st.markdown(f"""<div class="card"><div class="sec-header">🔔 SET PRICE ALERT</div>
            <p style="font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:12px;">
            Alert fires on next analysis run when the live P2P rate crosses your target.</p>
            </div>""", unsafe_allow_html=True)

            a_level = st.number_input("Alert price (₦)", min_value=100.0, max_value=9999.0,
                                       value=float(round(p2p_mid * 1.01)), step=10.0, key="alert_price_input")
            a_type  = st.selectbox("Alert when rate goes:", ["above", "below"], key="alert_type_select")
            if st.button("+ Add Alert", use_container_width=True, key="add_alert_btn"):
                st.session_state.alerts.append({"level": a_level, "type": a_type})
                em = "📧" if user_email else "🔕"
                st.success(f"Alert set: {em} notify when rate goes {a_type} ₦{a_level:,.0f}")

            st.markdown(f"""<div style="background:var(--bg2);border:1px solid var(--border);
            border-radius:var(--r-sm);padding:12px 14px;margin-top:8px;text-align:center;">
            <div style="font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;font-family:var(--font-mono);">Current P2P Rate</div>
            <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:var(--green);margin-top:4px;">₦{p2p_mid:,.2f}</div>
            <div style="font-size:10px;color:var(--muted2);">{raw.get("rate_source","—")}</div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.alerts:
            st.markdown('<div class="card" style="margin-top:16px;"><div class="sec-header">ACTIVE ALERTS</div>',
                        unsafe_allow_html=True)
            for i, a in enumerate(st.session_state.alerts):
                triggered_flag = i in st.session_state.alert_triggered
                acol1, acol2, acol3 = st.columns([4,2,1])
                with acol1:
                    arrow = "▲" if a["type"]=="above" else "▼"
                    em = "📧" if st.session_state.user_email else "🔕"
                    status_badge = ('<span style="color:var(--green);font-size:10px;">✅ Triggered</span>'
                                   if triggered_flag else '<span style="color:var(--amber);font-size:10px;">⏳ Watching</span>')
                    st.markdown(f'<span style="font-size:13px;">{em} {arrow} ₦{a["level"]:,} ({a["type"]})</span> {status_badge}',
                                unsafe_allow_html=True)
                with acol2:
                    dist = a["level"] - p2p_mid
                    dc2 = "var(--green)" if ((a["type"]=="above" and dist>0) or (a["type"]=="below" and dist<0)) else "var(--red)"
                    st.markdown(f'<span style="font-family:var(--font-mono);font-size:11px;color:{dc2};">{dist:+,.0f} from now</span>',
                                unsafe_allow_html=True)
                with acol3:
                    if st.button("✕", key=f"del_alert_{i}"):
                        st.session_state.alerts.pop(i)
                        if i in st.session_state.alert_triggered:
                            st.session_state.alert_triggered.remove(i)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="card" style="text-align:center;padding:24px;margin-top:16px;">
            <div style="font-size:24px;opacity:0.3;margin-bottom:8px;">🔔</div>
            <p style="color:var(--text2);font-size:12px;">No active alerts. Set one above.</p>
            </div>""", unsafe_allow_html=True)


    # ══════ TAB 8: ML METRICS ══════
    with tab8:
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--purple);margin-bottom:16px;">🔬 ML MODEL DIAGNOSTICS</div>',
                    unsafe_allow_html=True)

        if cold:
            st.markdown(f'<div class="alert-box alert-warn">⚠️ <strong>Cold Start Mode</strong> — {ml.get("note","")} Run analysis 5+ times to unlock full ML metrics.</div>',
                        unsafe_allow_html=True)
        else:
            mm1, mm2, mm3, mm4 = st.columns(4)
            for col, lbl, val, clr in [
                (mm1, "Training Points", str(metrics.get("n_training_points",0)), "var(--amber)"),
                (mm2, "R² In-Sample",    f"{metrics.get('r2_in_sample',0):.4f}" if metrics.get("r2_in_sample") else "N/A", "var(--blue)"),
                (mm3, "Model Agreement", f"{metrics.get('agreement_score',0):.1f}%", "var(--green)"),
                (mm4, "MAE Confidence",  f"{metrics.get('mae_conf',0):.1f}%", "var(--purple)"),
            ]:
                col.markdown(f"""<div class="card">
                <div class="card-label">{lbl}</div>
                <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{clr};">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Cross-val MAEs
            ml_met1, ml_met2 = st.columns(2)
            with ml_met1:
                st.markdown('<div class="card"><div class="sec-header">CROSS-VALIDATED MAE (5-FOLD)</div>',
                            unsafe_allow_html=True)
                for m_lbl, m_badge, mae_val in [
                    ("Ridge Regression",   "badge-ridge", metrics.get("ridge_cv_mae")),
                    ("Random Forest",      "badge-rf",    metrics.get("rf_cv_mae")),
                    ("Gradient Boosting",  "badge-gb",    metrics.get("gb_cv_mae")),
                ]:
                    val_s = f"₦{mae_val:.2f}" if mae_val else "N/A"
                    pct_s = f"({mae_val/max(p2p_mid,1)*100:.3f}% of rate)" if mae_val else ""
                    st.markdown(f"""<div style="padding:10px 0;border-bottom:1px solid var(--border);
                    display:flex;align-items:center;justify-content:space-between;">
                    <span class="model-badge {m_badge}">{m_lbl}</span>
                    <div style="font-family:var(--font-mono);font-size:13px;">
                    <span style="color:var(--amber);">{val_s}</span>
                    <span style="color:var(--muted);font-size:10px;"> {pct_s}</span></div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with ml_met2:
                # Feature Importances
                top_f = ml.get("rf_feature_importance", {})
                if top_f:
                    st.markdown('<div class="card"><div class="sec-header">RF FEATURE IMPORTANCES (TOP 10)</div>',
                                unsafe_allow_html=True)
                    max_imp = max(top_f.values()) if top_f else 1
                    for fname, fval in list(top_f.items())[:10]:
                        pct = fval / max_imp * 100
                        clean_name = fname.replace("_"," ").upper()
                        st.markdown(f"""<div class="prog-wrap">
                        <div class="prog-label">
                          <span style="font-size:10px;color:var(--text2);">{clean_name}</span>
                          <span style="font-family:var(--font-mono);font-size:10px;color:var(--amber);">{fval:.4f}</span>
                        </div>
                        <div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:var(--blue);"></div></div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # Methodology explainer
            st.markdown(f"""<div class="card" style="margin-top:14px;">
            <div class="sec-header">📚 MODEL METHODOLOGY</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;font-size:12px;color:var(--text2);line-height:1.65;">
              <div><strong style="color:var(--blue);">Ridge Regression (weight: 25%)</strong><br>
              Regularised linear model. Best for capturing stable directional trends. Prevents overfitting via L2 penalty.</div>
              <div><strong style="color:var(--green);">Random Forest (weight: 35%)</strong><br>
              Ensemble of 200 decision trees. Detects non-linear patterns, regime shifts. Feature importance from node impurity.</div>
              <div><strong style="color:var(--purple);">Gradient Boosting (weight: 40%)</strong><br>
              Sequential error-correction (150 estimators). Most powerful for time-series patterns. Highest ensemble weight.</div>
              <div><strong style="color:var(--amber);">Confidence Score</strong><br>
              = 45% × model agreement + 40% × MAE-based accuracy + 15% × sample size bonus. Capped at 92%.</div>
            </div>
            </div>""", unsafe_allow_html=True)

    # ── DISCLAIMER ──
    st.markdown("""
    <div style="font-size:10px;color:var(--muted);margin-top:24px;line-height:1.7;
    padding:14px 16px;border-top:1px solid var(--border);text-align:center;">
      ⚠️ Statistical models and AI predictions carry inherent uncertainty.
      Confidence degrades with time horizon — this is by design.
      Not financial advice. Always do your own research (DYOR) before converting funds.
      Short-term ML accuracy improves with more training data — run analysis frequently.
    </div>""", unsafe_allow_html=True)
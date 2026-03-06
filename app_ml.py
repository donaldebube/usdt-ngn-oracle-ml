"""
USDT/NGN Oracle — ML Edition (app_ml.py)
─────────────────────────────────────────
Statistical ML prediction engine using:
  • Linear Regression (trend baseline)
  • Random Forest Regressor (non-linear patterns)
  • Gradient Boosting Regressor (sequential error correction)
  • Ensemble voting for final price prediction
  • Real confidence score = 1 − (std_dev of model outputs / mean)
  • Gemini AI for natural language interpretation of ML output only

Run locally:
  pip install streamlit scikit-learn numpy pandas requests
  streamlit run app_ml.py

Streamlit Cloud: set GEMINI_KEY (and optionally NEWS_KEY) in Secrets.
"""

import streamlit as st
import requests
import json
import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="USDT/NGN Oracle — ML",
    page_icon="₦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS (same dark theme as main app)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080c14; --bg2: #0c1220; --bg3: #101828; --card: #111d2e;
    --border: #1a2942; --border2: #243550;
    --green: #05d68a; --green2: rgba(5,214,138,0.12);
    --red: #f0455a;   --red2: rgba(240,69,90,0.12);
    --amber: #f5a623; --amber2: rgba(245,166,35,0.12);
    --blue: #4f8ef7;  --blue2: rgba(79,142,247,0.12);
    --purple: #a78bfa;
    --text: #dce8f8;  --muted: #4a6080; --muted2: #6b84a0;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 1.2rem 2rem 2rem 2rem !important; max-width: 1400px !important; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace !important; }
.mcard {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 22px; position: relative; overflow: hidden;
}
.mcard-green  { border-top: 2px solid var(--green); }
.mcard-red    { border-top: 2px solid var(--red); }
.mcard-amber  { border-top: 2px solid var(--amber); }
.mcard-blue   { border-top: 2px solid var(--blue); }
.mcard-purple { border-top: 2px solid var(--purple); }
.mcard-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; font-family: 'IBM Plex Mono', monospace; }
.mcard-value { font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 700; line-height: 1.1; margin-bottom: 4px; }
.mcard-sub   { font-size: 12px; color: var(--muted2); margin-top: 4px; }
.ocard { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px 24px; margin-bottom: 16px; }
.ocard-title { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.live-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: blink 2s ease-in-out infinite; margin-right: 6px; vertical-align: middle; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.chat-u { background: var(--blue2); border: 1px solid rgba(79,142,247,0.2); border-radius: 14px 14px 3px 14px; padding: 12px 16px; margin: 10px 0; margin-left: 12%; font-size: 14px; line-height: 1.6; }
.chat-a { background: var(--card); border: 1px solid var(--border); border-radius: 14px 14px 14px 3px; padding: 12px 16px; margin: 10px 0; margin-right: 12%; font-size: 14px; line-height: 1.6; }
.chat-badge { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--green); margin-bottom: 5px; }
.prog-wrap { margin-bottom: 14px; }
.prog-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; }
.prog-track { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 4px; }
.alert-box  { border-radius: 10px; padding: 12px 16px; font-size: 13px; margin-bottom: 10px; border-left: 3px solid; line-height: 1.5; }
.alert-bull { background: var(--green2); border-color: var(--green); color: #a7f3d0; }
.alert-bear { background: var(--red2);   border-color: var(--red);   color: #fca5a5; }
.alert-info { background: var(--blue2);  border-color: var(--blue);  color: #bfdbfe; }
.alert-warn { background: var(--amber2); border-color: var(--amber); color: #fde68a; }
.spread-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.spread-table th { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
.spread-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.spread-table tr:last-child td { border-bottom: none; }
.stTextInput>div>div>input, .stTextArea textarea, .stNumberInput>div>div>input {
    background: var(--card) !important; border: 1px solid var(--border2) !important;
    color: var(--text) !important; border-radius: 8px !important;
}
.stButton>button {
    background: linear-gradient(135deg, #1a3a6e, #2d5fb8) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
}
.stButton>button:hover { filter: brightness(1.15) !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ML-specific badges */
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 100px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-ridge { background: rgba(79,142,247,0.15); color: #4f8ef7; border: 1px solid rgba(79,142,247,0.3); }
.badge-rf    { background: rgba(5,214,138,0.12); color: #05d68a; border: 1px solid rgba(5,214,138,0.3); }
.badge-gb    { background: rgba(167,139,250,0.12); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-ens   { background: rgba(245,166,35,0.15); color: #f5a623; border: 1px solid rgba(245,166,35,0.3); }

.confidence-ring {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; width: 120px; height: 120px;
    border-radius: 50%; border: 6px solid;
    font-family: 'IBM Plex Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init():
    defaults = {
        "chat": [], "result": None, "last_time": None,
        "rate_history": [],      # list of dicts: {time, features, actual_rate}
        "model_history": [],     # list of dicts for audit trail
        "alerts": [], "alert_triggered": [],
        "auto_refresh": False, "refresh_interval": 60,
        "prev_rate": None,
        "ml_metrics": {},        # stored cross-val metrics
        "user_email": "",        # email for price alerts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()


# ─────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────
try:
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
    NEWS_KEY   = st.secrets.get("NEWS_KEY", "")
    RESEND_KEY = st.secrets.get("RESEND_API_KEY", "")
except Exception:
    GEMINI_KEY = ""
    NEWS_KEY   = ""
    RESEND_KEY = ""

if not GEMINI_KEY:
    st.error("⚠️ Gemini API key not configured. Add GEMINI_KEY to Streamlit Secrets.")
    st.stop()

# ── VERIFY GEMINI KEY IS WORKING ──
@st.cache_data(ttl=300)  # cache for 5 mins so it doesn't re-check every rerun
def check_gemini_key(key: str) -> tuple:
    """Returns (working: bool, working_model: str, error: str)"""
    test_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
    ]
    for model in test_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        try:
            r = requests.post(url, json={
                "contents": [{"parts": [{"text": "Reply with just the word OK"}]}],
                "generationConfig": {"maxOutputTokens": 5}
            }, timeout=15)
            if r.status_code == 200:
                return True, model, ""
            elif r.status_code == 403:
                return False, "", "API key is invalid or not authorised"
            elif r.status_code == 429:
                return True, model, "Rate limited (but key works)"
        except:
            continue
    return False, "", "No working Gemini model found for this key"

_key_ok, _working_model, _key_err = check_gemini_key(GEMINI_KEY)
if not _key_ok:
    st.error(f"❌ Gemini API key issue: {_key_err}. Please check your key at aistudio.google.com")
    st.stop()


# ─────────────────────────────────────────────
# GEMINI HELPER
# ─────────────────────────────────────────────
def gemini(prompt: str, system: str = "") -> str:
    models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
    ]
    parts = []
    if system:
        parts.append({"text": f"SYSTEM:\n{system}\n\n---\n\n"})
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
    }
    errors = []
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        try:
            r = requests.post(url, json=payload, timeout=40)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 429:
                return "❌ Rate limit hit. Wait 60 seconds and try again."
            elif r.status_code == 400:
                detail = r.json().get("error", {}).get("message", "Bad request")
                errors.append(f"{model}: 400 – {detail[:80]}")
                continue
            elif r.status_code == 403:
                return "❌ API key invalid or not authorised. Check your GEMINI_KEY in secrets.toml."
            elif r.status_code == 404:
                errors.append(f"{model}: not found")
                continue
            else:
                detail = ""
                try: detail = r.json().get("error", {}).get("message", "")[:80]
                except: pass
                errors.append(f"{model}: HTTP {r.status_code} {detail}")
                continue
        except Exception as e:
            errors.append(f"{model}: {str(e)[:60]}")
            continue
    error_summary = " | ".join(errors)
    return f"❌ All Gemini models failed. Errors: {error_summary}"


# ─────────────────────────────────────────────
# ── DATA COLLECTION ──
# Returns a flat dict of ALL numeric features
# ─────────────────────────────────────────────
def collect_features() -> dict:
    """
    Collect every numeric signal that can statistically affect USDT/NGN.
    All values normalised to floats. Missing = np.nan (handled by imputation later).
    Returns both raw rates and a features dict.
    """
    feat = {}
    raw  = {}   # store raw rate data separately

    # ── 1. RATE FETCHING: Binance P2P → Bybit P2P → CoinGecko → Fallback ──
    raw["rate_source"] = "unknown"
    raw["rate_status"] = "fetching"

    # SOURCE A: Binance P2P
    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        for side, key in [("BUY", "p2p_buy"), ("SELL", "p2p_sell")]:
            payload = {
                "asset": "USDT", "fiat": "NGN", "merchantCheck": False,
                "page": 1, "payTypes": [], "publisherType": None,
                "rows": 10, "tradeType": side
            }
            r = requests.post(
                "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search",
                json=payload, headers=headers, timeout=15
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                prices = []
                for item in data:
                    try:
                        price = float(item["adv"]["price"])
                        if price > 100:  # sanity check — NGN rate should be > 100
                            prices.append(price)
                    except:
                        pass
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

    # SOURCE B: Bybit P2P (fallback if Binance didn't return both sides)
    if not (raw.get("p2p_buy") and raw.get("p2p_sell")):
        try:
            r = requests.get(
                "https://api2.bybit.com/fiat/otc/item/list"
                "?userId=&tokenId=USDT&currencyId=NGN&payment=&side=1"
                "&size=10&page=1&amount=&authMaker=false&canTrade=false",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12
            )
            if r.status_code == 200:
                items = r.json().get("result", {}).get("items", [])
                prices = [float(i.get("price", 0)) for i in items if float(i.get("price", 0)) > 100]
                if prices:
                    bybit_avg = round(sum(prices) / len(prices), 2)
                    if not raw.get("p2p_buy"):
                        raw["p2p_buy"] = bybit_avg
                        feat["p2p_buy_std"] = float(np.std(prices)) if len(prices) > 1 else 0.0
                    if not raw.get("p2p_sell"):
                        raw["p2p_sell"] = round(bybit_avg * 0.998, 2)  # estimate sell side
                        feat["p2p_sell_std"] = 0.0
                    raw["rate_source"] = "Bybit P2P"
                    raw["rate_status"] = "live"
        except Exception as e:
            raw["bybit_error"] = str(e)[:100]

    # SOURCE C: CoinGecko USDT/NGN (second fallback)
    if not raw.get("p2p_buy"):
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=tether&vs_currencies=ngn",
                timeout=10, headers={"User-Agent": "Mozilla/5.0"}
            )
            if r.status_code == 200:
                ngn_val = r.json().get("tether", {}).get("ngn")
                if ngn_val and float(ngn_val) > 100:
                    raw["p2p_buy"]  = float(ngn_val)
                    raw["p2p_sell"] = round(float(ngn_val) * 0.997, 2)
                    feat["p2p_buy_std"]  = 0.0
                    feat["p2p_sell_std"] = 0.0
                    raw["rate_source"] = "CoinGecko (aggregated)"
                    raw["rate_status"] = "live"
        except Exception as e:
            raw["coingecko_error"] = str(e)[:100]

    # SOURCE D: Hard fallback (last resort — clearly labelled)
    if not raw.get("p2p_buy"):
        raw["p2p_buy"]    = 1620.0
        raw["p2p_sell"]   = 1615.0
        raw["rate_source"] = "⚠️ Fallback estimate (all sources failed)"
        raw["rate_status"] = "estimated"
        feat["p2p_buy_std"]  = 0.0
        feat["p2p_sell_std"] = 0.0

    # Derived P2P features
    if raw.get("p2p_buy") and raw.get("p2p_sell"):
        raw["p2p_mid"]    = round((raw["p2p_buy"] + raw["p2p_sell"]) / 2, 2)
        raw["p2p_spread"] = round(raw["p2p_buy"] - raw["p2p_sell"], 2)
        feat["p2p_spread_abs"]  = raw["p2p_spread"]
        feat["p2p_spread_pct"]  = round(raw["p2p_spread"] / max(raw["p2p_sell"], 1) * 100, 4)
    elif raw.get("p2p_buy"):
        raw["p2p_mid"] = raw["p2p_buy"]
        raw["p2p_spread"] = 0.0
        feat["p2p_spread_abs"] = 0.0
        feat["p2p_spread_pct"] = 0.0

    # ── 2. OFFICIAL / INTERBANK RATE ──
    official = None
    for url in [
        "https://open.er-api.com/v6/latest/USD",
        "https://api.exchangerate-api.com/v4/latest/USD"
    ]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ngn = r.json().get("rates", {}).get("NGN")
                if ngn:
                    official = float(ngn)
                    # also grab other crosses for EM context
                    rates_obj = r.json().get("rates", {})
                    for ccy in ["EUR", "GBP", "ZAR", "KES", "GHS", "EGP", "XOF"]:
                        v = rates_obj.get(ccy)
                        if v:
                            feat[f"usd_{ccy.lower()}"] = float(v)
                    break
        except:
            pass

    raw["official"] = official
    if official and raw.get("p2p_mid"):
        raw["premium_pct"] = round((raw["p2p_mid"] - official) / official * 100, 4)
        feat["official_rate"]  = official
        feat["premium_pct"]    = raw["premium_pct"]
        feat["premium_abs"]    = round(raw["p2p_mid"] - official, 2)

    # ── 3. CRYPTO MARKET ──
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,tether&vs_currencies=usd,ngn"
            "&include_24hr_change=true&include_market_cap=true",
            timeout=12, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            btc = d.get("bitcoin", {})
            eth = d.get("ethereum", {})
            usdt = d.get("tether", {})

            feat["btc_usd"]        = btc.get("usd", np.nan)
            feat["btc_24h_change"] = btc.get("usd_24h_change", np.nan)
            feat["eth_usd"]        = eth.get("usd", np.nan)
            feat["eth_24h_change"] = eth.get("usd_24h_change", np.nan)
            feat["usdt_ngn_cg"]    = usdt.get("ngn", np.nan)   # CoinGecko aggregated
            feat["btc_mcap"]       = btc.get("usd_market_cap", np.nan)

            raw["btc_change"] = feat["btc_24h_change"]
    except:
        pass

    # ── 4. MACRO: OIL PROXY (Brent) via commodity index ──
    # We use BTC as a risk-on/risk-off proxy for oil correlation
    # and supplement with a free commodity endpoint
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=wrapped-bitcoin,chainlink,uniswap&vs_currencies=usd"
            "&include_24hr_change=true",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            cl = d.get("chainlink", {})
            uni = d.get("uniswap", {})
            feat["chainlink_change"] = cl.get("usd_24h_change", np.nan)
            feat["uniswap_change"]   = uni.get("usd_24h_change", np.nan)
    except:
        pass

    # ── 5. USD STRENGTH (DXY proxy via EUR/USD) ──
    eur = feat.get("usd_eur")
    if eur:
        eurusd = round(1 / eur, 6) if eur else np.nan
        feat["eurusd"]      = eurusd
        feat["dxy_proxy"]   = round(eur * 100, 4)    # higher = stronger USD
        feat["usd_strong"]  = 1 if eurusd < 1.05 else 0

    # ── 6. TEMPORAL FEATURES ──
    now = datetime.datetime.now()
    feat["hour"]         = now.hour
    feat["dow"]          = now.weekday()      # 0=Mon
    feat["is_weekend"]   = int(now.weekday() >= 5)
    feat["is_business"]  = int(8 <= now.hour <= 17 and now.weekday() < 5)
    feat["month"]        = now.month
    feat["day_of_month"] = now.day
    # cyclical encoding (sin/cos) so model understands circularity
    feat["hour_sin"]     = float(np.sin(2 * np.pi * now.hour / 24))
    feat["hour_cos"]     = float(np.cos(2 * np.pi * now.hour / 24))
    feat["dow_sin"]      = float(np.sin(2 * np.pi * now.weekday() / 7))
    feat["dow_cos"]      = float(np.cos(2 * np.pi * now.weekday() / 7))
    feat["month_sin"]    = float(np.sin(2 * np.pi * now.month / 12))
    feat["month_cos"]    = float(np.cos(2 * np.pi * now.month / 12))

    # ── 7. QUALITATIVE INTELLIGENCE ENGINE ──
    # Scrapes Google News RSS (no API key needed) across every topic that
    # can move USDT/NGN, then uses Gemini to REASON about the headlines —
    # not just count them. Covers geopolitics, oil shocks, CBN actions,
    # Fed policy, crypto regulation, and anything else breaking right now.
    raw["news_intel"] = {}
    news_headlines = []

    # Google News RSS topics — free, no key, always fresh
    rss_topics = [
        ("Nigeria naira exchange rate",          "nigeria_fx"),
        ("CBN central bank Nigeria",             "cbn_policy"),
        ("Nigeria inflation economy",            "nigeria_macro"),
        ("crude oil price Brent OPEC",           "oil_markets"),
        ("Iran oil sanctions geopolitics",       "geopolitics"),
        ("US Federal Reserve interest rates",    "fed_usd"),
        ("dollar strength DXY emerging markets", "usd_em"),
        ("Nigeria crypto USDT P2P",              "crypto_nigeria"),
        ("Bitcoin crypto market",                "crypto_global"),
        ("Nigeria government fiscal policy",     "nigeria_fiscal"),
        ("Middle East conflict oil supply",      "mideast_oil"),
        ("Russia Ukraine war commodity",         "war_commodity"),
        ("China economy trade dollar",           "china_macro"),
        ("IMF World Bank Nigeria debt",          "intl_finance"),
        ("Nigeria election politics 2025",       "political_risk"),
        ("OPEC production cut oil output",       "opec"),
        ("Nigeria remittance diaspora dollar",   "remittances"),
        ("US inflation CPI jobs report",         "us_macro"),
        ("Nigeria foreign exchange reserves CBN", "fx_reserves"),
        ("Nigeria Eurobond yield investor sentiment", "capital_flows"),
        ("Nigeria petrol price fuel subsidy news", "inflation_trigger"),
        ("Naira devaluation news today", "devaluation_risk"),
        ("OPEC Nigeria oil production quota", "supply_side"),
        ("IMF World Bank Nigeria economic outlook", "institutional"),
        ("Nigeria inflation rate NBS statistics", "macro_data"),
        ("US Dollar Index DXY trend", "global_usd"),
        ("Binance Nigeria P2P regulation news", "p2p_legal"),
    ]

    for query, category in rss_topics:
        try:
            encoded = requests.utils.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                import re as _re
                # Extract titles and descriptions from RSS XML
                titles = _re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
                descs  = _re.findall(r"<description><!\[CDATA\[(.*?)\]\]></description>", r.text)
                for i, title in enumerate(titles[1:4]):  # skip feed title, get top 3
                    desc = descs[i] if i < len(descs) else ""
                    # Strip any remaining HTML tags
                    desc_clean = _re.sub(r"<[^>]+>", "", desc)[:150]
                    headline = f"[{category.upper()}] {title.strip()}"
                    if desc_clean:
                        headline += f" — {desc_clean.strip()}"
                    news_headlines.append(headline)
        except:
            pass

    # Also pull from NewsAPI if key is available (supplements RSS)
    if NEWS_KEY:
        extra_topics = [
            "Nigeria naira dollar",
            "oil price Iran sanctions",
            "CBN forex intervention",
            "Nigeria economy 2025",
        ]
        for q in extra_topics:
            try:
                url = (f"https://newsapi.org/v2/everything"
                       f"?q={requests.utils.quote(q)}"
                       f"&sortBy=publishedAt&pageSize=3&language=en&apiKey={NEWS_KEY}")
                r = requests.get(url, timeout=8)
                if r.status_code == 200:
                    for a in r.json().get("articles", [])[:2]:
                        t = a.get("title", "")
                        d = a.get("description", "") or ""
                        if t:
                            news_headlines.append(f"[NEWSAPI] {t.strip()} — {d[:120].strip()}")
            except:
                pass

    raw["news_headlines_count"] = len(news_headlines)
    raw["news_headlines"]       = news_headlines[:40]  # store for display

    # ── GEMINI DEEP QUALITATIVE ANALYSIS ──
    # Feed ALL headlines to Gemini and ask it to REASON about each factor,
    # not just score them. This is where geopolitics like Iran/oil get analysed.
    if news_headlines:
        headlines_block = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(news_headlines[:35]))
        now_str = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M WAT")

        qualitative_prompt = f"""You are a senior FX strategist analysing the USDT/NGN black market rate.
Today is {now_str}. The current Binance P2P rate is approximately ₦{raw.get('p2p_mid', 0):,.0f}/USDT.

Below are {len(news_headlines[:35])} LIVE news headlines scraped right now from Google News and NewsAPI.
Read every single one carefully and reason about how each topic affects the USDT/NGN rate.

LIVE HEADLINES:
{headlines_block}

REASONING FRAMEWORK — think through each of these channels:
1. OIL PRICE: Nigeria earns ~90% of FX from crude. Rising oil = more USD inflows = NGN strengthens = USDT rate falls. Iran tensions, OPEC cuts, Russia sanctions all affect this.
2. USD STRENGTH: Strong DXY = harder for all EM currencies including NGN. Fed hawkishness = stronger USD = higher USDT/NGN.
3. CBN POLICY: Interventions, rate hikes, FX restrictions, or easing all directly move the rate.
4. GEOPOLITICS: Wars, sanctions, supply shocks affect oil and risk sentiment. Middle East tensions = oil spike.
5. NIGERIA POLITICS: Elections, fiscal policy, protests affect investor confidence and capital flight.
6. CRYPTO MARKET: BTC rally = more P2P activity in Nigeria = tighter spreads. BTC crash = less liquidity.
7. INFLATION: High Nigeria CPI = structural NGN weakness. US CPI = affects Fed decisions.
8. REMITTANCES: Festive seasons, diaspora policy changes affect USD supply into Nigeria.
9. INTERNATIONAL FINANCE: IMF reviews, World Bank loans, Eurobond issuance affect FX reserves.
10. GLOBAL RISK: EM selloffs, credit events, recessions reduce risk appetite and hurt NGN.

Return ONLY valid JSON, no markdown, no backticks:
{{
  "overall_score": <-100 to 100, positive=USDT rises/NGN weakens, negative=NGN strengthens>,
  "nigeria_macro": <-100 to 100>,
  "cbn_policy": <-100 to 100>,
  "oil_impact": <-100 to 100>,
  "usd_fed_impact": <-100 to 100>,
  "crypto_sentiment": <-100 to 100>,
  "geopolitical_risk": <-100 to 100>,
  "political_risk_nigeria": <-100 to 100>,
  "remittance_flow": <-100 to 100>,
  "global_em_risk": <-100 to 100>,
  "top_bullish_catalyst": "<single most important reason USDT/NGN could rise today — be specific, cite the actual news>",
  "top_bearish_catalyst": "<single most important reason USDT/NGN could fall today — be specific, cite the actual news>",
  "breaking_event": "<any breaking event right now that could cause sudden rate movement — or null if none>",
  "oil_analysis": "<2 sentences: what is happening with oil right now and how does it affect NGN specifically>",
  "geopolitical_analysis": "<2 sentences: any wars, sanctions, conflicts affecting oil supply or USD demand>",
  "cbn_analysis": "<2 sentences: what CBN is doing or likely to do based on current news>",
  "overall_qualitative_direction": "BULLISH_USDT|BEARISH_USDT|NEUTRAL",
  "qualitative_confidence": <0-100, how confident based on news signal strength>,
  "key_headlines_used": ["<headline 1>", "<headline 2>", "<headline 3>", "<headline 4>", "<headline 5>"]
}}"""

        try:
            raw_q = gemini(qualitative_prompt,
                           "You are a quantitative FX strategist. Analyse ALL headlines carefully. Return only valid JSON.")
            # Parse JSON robustly
            clean_q = raw_q.strip()
            if "```" in clean_q:
                parts_q = clean_q.split("```")
                for p in parts_q:
                    p = p.strip()
                    if p.startswith("json"): p = p[4:].strip()
                    if p.startswith("{"): clean_q = p; break
            if not clean_q.startswith("{"):
                idx = clean_q.find("{")
                if idx >= 0: clean_q = clean_q[idx:]
            last = clean_q.rfind("}")
            if last >= 0: clean_q = clean_q[:last+1]
            q_intel = json.loads(clean_q)

            # Feed numeric scores into ML features
            feat["news_overall"]        = float(q_intel.get("overall_score", 0))
            feat["news_nigeria"]        = float(q_intel.get("nigeria_macro", 0))
            feat["news_cbn"]            = float(q_intel.get("cbn_policy", 0))
            feat["news_oil"]            = float(q_intel.get("oil_impact", 0))
            feat["news_usd"]            = float(q_intel.get("usd_fed_impact", 0))
            feat["news_crypto"]         = float(q_intel.get("crypto_sentiment", 0))
            feat["news_geopolitics"]    = float(q_intel.get("geopolitical_risk", 0))
            feat["news_political_risk"] = float(q_intel.get("political_risk_nigeria", 0))
            feat["news_remittance"]     = float(q_intel.get("remittance_flow", 0))
            feat["news_em_risk"]        = float(q_intel.get("global_em_risk", 0))

            # Store full qualitative analysis for display in UI
            raw["news_intel"]     = q_intel
            raw["news_sentiment"] = q_intel  # backward compat
        except Exception as e:
            # Gemini parse failed — zero out but don't crash
            for k in ["news_overall","news_nigeria","news_cbn","news_oil","news_usd",
                      "news_crypto","news_geopolitics","news_political_risk",
                      "news_remittance","news_em_risk"]:
                feat[k] = 0.0
            raw["news_intel_error"] = str(e)[:100]
    else:
        # No headlines at all — zero features
        for k in ["news_overall","news_nigeria","news_cbn","news_oil","news_usd",
                  "news_crypto","news_geopolitics","news_political_risk",
                  "news_remittance","news_em_risk"]:
            feat[k] = 0.0

    # ── 8. HISTORICAL MOMENTUM FEATURES ──
    # Uses our in-session rate history stored in session_state
    hist = st.session_state.rate_history
    if len(hist) >= 2:
        recent_rates = [h["p2p_mid"] for h in hist[-10:] if h.get("p2p_mid")]
        if len(recent_rates) >= 2:
            feat["momentum_1"]    = recent_rates[-1] - recent_rates[-2]          # 1-step change
            feat["momentum_avg"]  = recent_rates[-1] - np.mean(recent_rates[:-1]) # vs rolling mean
            feat["volatility"]    = float(np.std(recent_rates))                   # recent std dev
            # Trend: slope of linear fit
            x = np.arange(len(recent_rates))
            slope = float(np.polyfit(x, recent_rates, 1)[0])
            feat["trend_slope"]   = slope
            feat["trend_accel"]   = feat["momentum_1"] - (recent_rates[-2] - recent_rates[-3]) if len(recent_rates) >= 3 else 0.0
        if len(recent_rates) >= 5:
            feat["rate_ma5"]      = float(np.mean(recent_rates[-5:]))
            feat["rate_ma5_dev"]  = recent_rates[-1] - feat["rate_ma5"]
    else:
        # Cold-start: no momentum yet
        feat["momentum_1"]   = 0.0
        feat["momentum_avg"] = 0.0
        feat["volatility"]   = 0.0
        feat["trend_slope"]  = 0.0
        feat["trend_accel"]  = 0.0

    # Replace any NaN with 0 for robustness
    for k, v in feat.items():
        if isinstance(v, float) and np.isnan(v):
            feat[k] = 0.0

    raw["timestamp"] = datetime.datetime.now().isoformat()
    raw["features"]  = feat
    return raw, feat


# ─────────────────────────────────────────────
# ── ML ENGINE ──
# ─────────────────────────────────────────────

# FIXED feature list — keeps model stable across runs
FEATURE_COLS = [
    # P2P / Rate
    "p2p_spread_abs", "p2p_spread_pct", "p2p_buy_std", "p2p_sell_std",
    "premium_pct", "premium_abs", "official_rate",
    # Crypto
    "btc_24h_change", "eth_24h_change", "usdt_ngn_cg",
    "chainlink_change", "uniswap_change",
    # FX / USD
    "eurusd", "dxy_proxy",
    "usd_zar", "usd_kes", "usd_ghs",
    # Time
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_business",
    # Qualitative / News (10 dimensions — all from Gemini deep analysis)
    "news_overall", "news_nigeria", "news_cbn", "news_oil", "news_usd",
    "news_crypto", "news_geopolitics", "news_political_risk",
    "news_remittance", "news_em_risk",
    # Momentum / Trend
    "momentum_1", "momentum_avg", "volatility", "trend_slope", "trend_accel",
    "rate_ma5_dev",
]

def features_to_vector(feat: dict) -> np.ndarray:
    """Convert feature dict → fixed-length numpy vector. Missing = 0."""
    return np.array([float(feat.get(c, 0.0)) for c in FEATURE_COLS], dtype=float)


def build_training_data() -> tuple:
    """
    Build X (features) and y (actual P2P mid rate) from session history.
    We use a 1-step-ahead target: predict next observation from current features.
    Returns (X, y, timestamps) or (None, None, None) if insufficient data.
    """
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
    """
    Train 3 models on session history, then predict the next rate.
    Returns full metrics, individual model predictions, ensemble, and confidence.
    """
    X, y, times = build_training_data()

    cold_start = (X is None)

    # ── COLD START: not enough history yet ──
    if cold_start:
        # Fall back to a physics-based estimate using signal scores
        score = 0.0
        feat = current_feat
        score += feat.get("btc_24h_change", 0) * 0.3         # BTC momentum
        score += -feat.get("dxy_proxy", 0) * 0.01            # USD strength
        score += feat.get("news_overall", 0) * 0.5           # news sentiment
        score += -feat.get("premium_pct", 0) * 0.2           # spread compression signal
        score += feat.get("momentum_1", 0) * 0.8             # recent momentum

        direction_factor = 1 + score / 5000.0
        est = round(current_rate * direction_factor, 2)
        low  = round(current_rate * 0.992, 2)
        high = round(current_rate * 1.008, 2)

        return {
            "cold_start": True,
            "n_training_points": 0,
            "ridge_pred":  est,
            "rf_pred":     est,
            "gb_pred":     est,
            "ensemble":    est,
            "pred_low":    low,
            "pred_high":   high,
            "confidence":  35,      # explicitly low — no historical data
            "direction":   "BULLISH" if est > current_rate else "BEARISH" if est < current_rate else "NEUTRAL",
            "model_agreement": 100.0,
            "ridge_cv_mae": None,
            "rf_cv_mae":    None,
            "gb_cv_mae":    None,
            "rf_feature_importance": {},
            "note": "COLD START — fewer than 5 historical data points. Confidence is intentionally low.",
        }

    # ── FULL TRAINING ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_pred   = scaler.transform(features_to_vector(current_feat).reshape(1, -1))

    # ── MODEL 1: Ridge Regression (regularised linear — trend baseline) ──
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_scaled, y)
    ridge_pred = float(ridge.predict(x_pred)[0])
    # Cross-validation MAE
    if len(X_scaled) >= 5:
        ridge_cv = cross_val_score(ridge, X_scaled, y,
                                   cv=min(5, len(X_scaled)),
                                   scoring="neg_mean_absolute_error")
        ridge_cv_mae = float(-ridge_cv.mean())
    else:
        ridge_cv_mae = None

    # ── MODEL 2: Random Forest (non-linear, captures regime changes) ──
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf.fit(X_scaled, y)
    rf_pred = float(rf.predict(x_pred)[0])
    if len(X_scaled) >= 5:
        rf_cv = cross_val_score(rf, X_scaled, y,
                                cv=min(5, len(X_scaled)),
                                scoring="neg_mean_absolute_error")
        rf_cv_mae = float(-rf_cv.mean())
    else:
        rf_cv_mae = None

    # Feature importance from RF
    imp = dict(zip(FEATURE_COLS, rf.feature_importances_))
    top_features = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:8])

    # ── MODEL 3: Gradient Boosting (sequential error correction) ──
    gb = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.08,
        max_depth=4, subsample=0.8, random_state=42
    )
    gb.fit(X_scaled, y)
    gb_pred = float(gb.predict(x_pred)[0])
    if len(X_scaled) >= 5:
        gb_cv = cross_val_score(gb, X_scaled, y,
                                cv=min(5, len(X_scaled)),
                                scoring="neg_mean_absolute_error")
        gb_cv_mae = float(-gb_cv.mean())
    else:
        gb_cv_mae = None

    # ── ENSEMBLE: Weighted average (GB weighted higher for recent data) ──
    # Weights: Ridge=0.25 (linear baseline), RF=0.35 (non-linear), GB=0.40 (best for sequences)
    weights = np.array([0.25, 0.35, 0.40])
    preds   = np.array([ridge_pred, rf_pred, gb_pred])
    ensemble = float(np.dot(weights, preds))

    # ── CONFIDENCE SCORE ──
    # Based on:
    # (a) Model agreement: 1 - (std of 3 predictions / mean) → higher agreement = higher confidence
    # (b) Cross-val MAE relative to rate level: lower MAE → higher confidence
    # (c) Sample size: more data = more confidence
    #
    pred_std  = float(np.std(preds))
    pred_mean = float(np.mean(preds))
    agreement_score = max(0.0, 1.0 - (pred_std / max(pred_mean, 1.0))) * 100  # 0-100

    # MAE-based confidence: if avg MAE < 0.5% of rate, that's strong
    maes = [m for m in [ridge_cv_mae, rf_cv_mae, gb_cv_mae] if m is not None]
    if maes:
        avg_mae = np.mean(maes)
        mae_pct = avg_mae / max(current_rate, 1.0) * 100
        mae_conf = max(0.0, min(100.0, 100.0 - mae_pct * 5))  # 1% MAE → 95 conf; 20% MAE → 0
    else:
        mae_conf = 50.0

    # Sample size bonus: 5 pts → +0, 20 pts → +15, 50+ pts → +20
    n = len(X_scaled)
    size_bonus = min(20.0, n * 0.4)

    # Combined confidence
    raw_conf = (agreement_score * 0.45 + mae_conf * 0.40 + size_bonus * 0.15)
    confidence = int(min(92, max(30, round(raw_conf))))  # cap at 92% — no model is perfect

    # ── PREDICTION INTERVAL ──
    # Use model disagreement + historical volatility for the range
    vol = current_feat.get("volatility", current_rate * 0.003)
    half_range = max(pred_std * 2, vol * 1.5, current_rate * 0.003)
    pred_low  = round(ensemble - half_range, 2)
    pred_high = round(ensemble + half_range, 2)

    direction = ("BULLISH" if ensemble > current_rate * 1.0005
                 else "BEARISH" if ensemble < current_rate * 0.9995
                 else "NEUTRAL")

    # ── IN-SAMPLE R² for display ──
    y_in_sample = []
    for i in range(len(X_scaled)):
        y_in_sample.append(float(gb.predict(X_scaled[i:i+1])[0]))
    r2 = float(r2_score(y, y_in_sample)) if len(y) > 1 else None

    # Store metrics for display
    metrics = {
        "n_training_points": n,
        "ridge_cv_mae": ridge_cv_mae,
        "rf_cv_mae":    rf_cv_mae,
        "gb_cv_mae":    gb_cv_mae,
        "r2_in_sample": r2,
        "pred_std":     pred_std,
        "agreement_score": agreement_score,
        "mae_conf":        mae_conf,
        "size_bonus":      size_bonus,
        "rf_feature_importance": top_features,
    }
    st.session_state.ml_metrics = metrics

    return {
        "cold_start": False,
        "n_training_points": n,
        "ridge_pred":  round(ridge_pred, 2),
        "rf_pred":     round(rf_pred, 2),
        "gb_pred":     round(gb_pred, 2),
        "ensemble":    round(ensemble, 2),
        "pred_low":    pred_low,
        "pred_high":   pred_high,
        "confidence":  confidence,
        "direction":   direction,
        "model_agreement": round(agreement_score, 1),
        "ridge_cv_mae": ridge_cv_mae,
        "rf_cv_mae":    rf_cv_mae,
        "gb_cv_mae":    gb_cv_mae,
        "r2_in_sample": r2,
        "rf_feature_importance": top_features,
        "note": f"Trained on {n} observations. Ensemble of Ridge + RF + GradientBoosting.",
    }


# ─────────────────────────────────────────────
# ── GEMINI: INTERPRET ML + QUALITATIVE OUTPUT ──
# ─────────────────────────────────────────────
def interpret_ml(ml: dict, raw: dict, feat: dict) -> dict:
    top_feat = ml.get("rf_feature_importance", {})
    top_str = "\n".join(f"  {k}: {v:.4f}" for k, v in list(top_feat.items())[:8])
    headlines = raw.get("news_headlines", [])
    headlines_sample = "\n".join(f"  • {h}" for h in headlines[:20])

    system = """You are a Chief Economist for a Tier-1 Investment Bank. 
    You translate complex ML data into high-level stakeholder insights. 
    Return ONLY valid JSON."""

    prompt = f"""
    Analyse the USDT/NGN market using these ML outputs and live news.
    
    ══ ML PREDICTIONS ══
    - Current Rate: ₦{raw.get('p2p_mid'):,.2f}
    - ML Target: ₦{ml['ensemble']:,.2f} ({ml['direction']})
    - Confidence: {ml['confidence']}%
    - Algorithmic Drivers: {top_str}
    
    ══ LIVE HEADLINES ══
    {headlines_sample}

    Provide a 'Stakeholder Intelligence Report' in JSON:
    {{
      "executive_summary": "3-sentence strategic overview of the market regime.",
      "price_movement_drivers": "Explicitly link specific news events to the predicted price change.",
      "stakeholder_insight": "Actionable advice for business owners (e.g., 'Hedge now', 'Wait for 48 hours').",
      "global_context": "How global events (Fed, Oil, DXY) are trickling down to the Naira today.",
      "risk_rating": "Low/Medium/High/Critical based on news volatility.",
      "trade_recommendation": "Specific timing and execution strategy.",
      "best_convert_time": "Optimal window for conversion.",
      "weekly_outlook": "7-day trend forecast.",
      "key_risks": ["Risk 1", "Risk 2"]
    }}
    """
    
    raw_out = gemini(prompt, system)
    try:
        # Robust JSON cleaning
        clean = raw_out.strip()
        if "```" in clean:
            clean = clean.split("```")[1].replace("json", "").strip()
        return json.loads(clean)
    except:
        return {"executive_summary": "Error parsing AI insights. Please rerun."}


# ─────────────────────────────────────────────
# ── FULL ANALYSIS ORCHESTRATOR ──
# ─────────────────────────────────────────────
def run_full_analysis() -> dict:
    # Step 1: Collect all features
    raw, feat = collect_features()

    p2p_mid = raw.get("p2p_mid") or raw.get("p2p_buy") or 1620.0
    raw["p2p_mid"] = p2p_mid

    # Step 2: Store in session history BEFORE training
    st.session_state.rate_history.append({
        "timestamp": raw.get("timestamp"),
        "p2p_mid":   p2p_mid,
        "p2p_buy":   raw.get("p2p_buy"),
        "p2p_sell":  raw.get("p2p_sell"),
        "official":  raw.get("official"),
        "features":  feat,
    })

    # Step 3: Train ML models and predict
    ml = train_and_predict(feat, p2p_mid)

    # Step 4: Gemini interprets (does NOT predict — just explains)
    interp = interpret_ml(ml, raw, feat)

    return {
        "success":   True,
        "raw":       raw,
        "features":  feat,
        "ml":        ml,
        "interp":    interp,
        "timestamp": raw.get("timestamp"),
        "n_history": len(st.session_state.rate_history),
    }


# ─────────────────────────────────────────────
# ── CHAT ──
# ─────────────────────────────────────────────
def chat_response(msg: str, result: dict) -> str:
    ml = result.get("ml", {})
    raw = result.get("raw", {})
    feat = result.get("features", {})
    interp = result.get("interp", {})

    ctx = f"""
CURRENT ML ANALYSIS:
- P2P Rate: ₦{raw.get("p2p_mid", 0):,.2f}
- Ensemble Prediction: ₦{ml.get("ensemble", 0):,.2f}
- Range: ₦{ml.get("pred_low",0):,.2f}–₦{ml.get("pred_high",0):,.2f}
- Direction: {ml.get("direction","N/A")}
- Confidence: {ml.get("confidence",0)}%
- Model Agreement: {ml.get("model_agreement",0):.1f}%
- Training Points: {ml.get("n_training_points",0)}
- Cold Start: {ml.get("cold_start", True)}
- Trade Recommendation: {interp.get("trade_recommendation","N/A")}
- Executive Summary: {interp.get("executive_summary","N/A")}
- Key Risks: {interp.get("key_risks",[])}
- Weekly Outlook: {interp.get("weekly_outlook","N/A")}

TOP ML FEATURES: {json.dumps(dict(list(ml.get("rf_feature_importance",{}).items())[:5]))}

LIVE QUALITATIVE INTELLIGENCE ({raw.get("news_headlines_count",0)} headlines):
- Overall score: {feat.get("news_overall","N/A")} | Oil: {feat.get("news_oil","N/A")} | Geopolitics: {feat.get("news_geopolitics","N/A")}
- CBN: {feat.get("news_cbn","N/A")} | USD/Fed: {feat.get("news_usd","N/A")} | Nigeria macro: {feat.get("news_nigeria","N/A")}
- Breaking event: {raw.get("news_intel",{}).get("breaking_event","None")}
- Oil analysis: {raw.get("news_intel",{}).get("oil_analysis","N/A")}
- Geopolitical analysis: {raw.get("news_intel",{}).get("geopolitical_analysis","N/A")}
- CBN analysis: {raw.get("news_intel",{}).get("cbn_analysis","N/A")}
- Top bullish catalyst: {raw.get("news_intel",{}).get("top_bullish_catalyst","N/A")}
- Top bearish catalyst: {raw.get("news_intel",{}).get("top_bearish_catalyst","N/A")}
- Models vs news alignment: {interp.get("qualitative_vs_quantitative","N/A")}
"""

    hist = ""
    for m in st.session_state.chat[-6:]:
        hist += f"\n{'User' if m['r']=='u' else 'Oracle'}: {m['c']}"

    system = """You are the USDT/NGN Oracle ML Edition — Nigeria's most rigorous AI FX analyst.
You base ALL answers on statistical model output (Ridge, Random Forest, Gradient Boosting).
You NEVER give a vague answer. If confidence is low, say so and explain why.
If the user asks "why", reference actual feature importances, model agreement, or historical data points.
Speak like a Bloomberg terminal analyst — precise, direct, no fluff."""

    return gemini(
        f"{ctx}\n\nConversation:{hist}\n\nUser: {msg}\n\nOracle:",
        system
    )


# ─────────────────────────────────────────────
# ── EMAIL ALERTS ──
# ─────────────────────────────────────────────
def send_email_alert(to_email: str, subject: str, html_body: str, resend_key: str) -> bool:
    if not to_email or not resend_key:
        return False
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {resend_key}",
                "Content-Type": "application/json"
            },
            json={
                "from": "USDT/NGN Oracle <onboarding@resend.dev>",
                "to": [to_email],
                "subject": subject,
                "html": html_body
            },
            timeout=15
        )
        return r.status_code == 200
    except:
        return False


def build_email_html(msg: str, rate: float, direction: str, confidence: int,
                     pred_low: float, pred_high: float, recommendation: str) -> str:
    dir_color = "#05d68a" if direction == "BULLISH" else "#f0455a" if direction == "BEARISH" else "#f5a623"
    dir_arrow = "▲" if direction == "BULLISH" else "▼" if direction == "BEARISH" else "◆"
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;
    background:#0c1220;color:#dce8f8;border-radius:16px;overflow:hidden;">
      <div style="background:linear-gradient(135deg,#111d2e,#0c1220);
      padding:28px 32px;border-bottom:2px solid #a78bfa;">
        <div style="font-size:11px;letter-spacing:3px;color:#a78bfa;
        text-transform:uppercase;margin-bottom:8px;">🇳🇬 USDT/NGN Oracle · ML Edition</div>
        <div style="font-size:24px;font-weight:700;color:#dce8f8;">Price Alert Triggered</div>
      </div>
      <div style="padding:28px 32px;">
        <div style="background:#111d2e;border:1px solid #1a2942;border-left:4px solid #f5a623;
        border-radius:10px;padding:16px 20px;margin-bottom:20px;font-size:16px;
        font-weight:600;color:#f5a623;">🔔 {msg}</div>
        <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;color:#6b84a0;font-size:13px;">ML Prediction</td>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;text-align:right;font-weight:700;color:{dir_color};">
              {dir_arrow} {direction}
            </td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;color:#6b84a0;font-size:13px;">Current Rate</td>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;text-align:right;font-weight:700;color:#05d68a;">
              ₦{rate:,.0f}
            </td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;color:#6b84a0;font-size:13px;">ML Range (next step)</td>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;text-align:right;font-weight:600;color:#dce8f8;">
              ₦{pred_low:,.0f} – ₦{pred_high:,.0f}
            </td>
          </tr>
          <tr>
            <td style="padding:10px 0;color:#6b84a0;font-size:13px;">Statistical Confidence</td>
            <td style="padding:10px 0;text-align:right;font-weight:700;color:#f5a623;">{confidence}%</td>
          </tr>
        </table>
        <div style="background:#111d2e;border:1px solid #1a2942;border-radius:10px;
        padding:16px 20px;margin-bottom:24px;">
          <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#a78bfa;margin-bottom:8px;">
            ML Recommendation
          </div>
          <div style="font-size:13px;line-height:1.7;color:#b0c8e8;">{recommendation}</div>
        </div>
        <div style="font-size:11px;color:#4a6080;text-align:center;padding-top:16px;
        border-top:1px solid #1a2942;line-height:1.6;">
          ⚠️ Statistical ML prediction. Not financial advice. Always DYOR.
        </div>
      </div>
    </div>"""


def check_and_trigger_alerts(rate: float, ml_result: dict, interp: dict):
    """Check all active alerts against current rate and fire emails if triggered."""
    triggered = []
    user_email = st.session_state.get("user_email", "")
    resend_key = ""
    try:
        resend_key = st.secrets.get("RESEND_API_KEY", "")
    except:
        pass

    direction    = ml_result.get("direction", "N/A")
    confidence   = ml_result.get("confidence", 0)
    pred_low     = ml_result.get("pred_low", 0)
    pred_high    = ml_result.get("pred_high", 0)
    recommendation = interp.get("trade_recommendation", "N/A")

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

        if msg and user_email and resend_key:
            html = build_email_html(msg, rate, direction, confidence, pred_low, pred_high, recommendation)
            send_email_alert(user_email, f"🔔 USDT/NGN Alert: {msg}", html, resend_key)

    return triggered


# ─────────────────────────────────────────────
# ── RENDER HELPERS ──
# ─────────────────────────────────────────────
def prog_bar(label, val, color, min_val=-100, max_val=100):
    norm = (val - min_val) / (max_val - min_val) * 100
    norm = max(0, min(100, norm))
    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-label">
        <span style="color:var(--muted2);font-size:12px;">{label}</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{color};">{val:+.1f}</span>
      </div>
      <div class="prog-track">
        <div class="prog-fill" style="width:{norm}%;background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── UI BEGINS ──
# ─────────────────────────────────────────────

# ── ACTION BAR ──
a1, a2, a3, a4 = st.columns([2, 1.5, 1.5, 6])
with a1:
    run_btn = st.button("🔍 Run ML Analysis", use_container_width=True, type="primary")
with a2:
    auto_ref = st.toggle("Auto-refresh", value=st.session_state.auto_refresh, key="ar_tog")
    st.session_state.auto_refresh = auto_ref
with a3:
    if auto_ref:
        iv = st.selectbox("Interval", [15, 30, 60, 120], index=2,
                          format_func=lambda x: f"{x}m", label_visibility="collapsed", key="ar_iv")
        st.session_state.refresh_interval = iv
with a4:
    if st.session_state.last_time:
        el = int((datetime.datetime.now() - st.session_state.last_time).total_seconds() // 60)
        pts = st.session_state.result.get("n_history", 0) if st.session_state.result else 0
        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;color:var(--muted2);'
            f'margin:10px 0 0 0;text-align:right;"><span class="live-dot"></span>'
            f'Updated {el}m ago · {pts} training points</p>',
            unsafe_allow_html=True
        )

# ── HEADER ──
st.markdown("""
<div style="padding:8px 0 18px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--amber);
  letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">
    <span class="live-dot"></span>STATISTICAL ML EDITION — BETA BRANCH
  </div>
  <h1 style="font-family:'IBM Plex Mono',monospace;font-size:30px;font-weight:700;
  margin:0 0 4px;background:linear-gradient(135deg,#dce8f8,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    USDT / NGN Oracle · ML
  </h1>
  <p style="color:var(--muted2);font-size:13px;margin:0;">
    Ridge · Random Forest · Gradient Boosting · Ensemble · Statistical confidence · Gemini interpretation
  </p>
</div>""", unsafe_allow_html=True)

# ── ML METHODOLOGY NOTE ──
st.markdown("""
<div class="alert-box alert-info" style="margin-bottom:18px;">
  <strong>🧪 How this works:</strong>
  Three statistical ML models (Ridge Regression, Random Forest, Gradient Boosting) are trained on
  your session's live rate history. Each run adds a new data point. Confidence is computed from
  <em>model agreement</em> and <em>cross-validated MAE</em> — not from guesswork.
  Gemini AI <em>interprets</em> the model output; it does NOT set the price.
  Run it at least 5–10 times to build training data for statistically meaningful predictions.
</div>""", unsafe_allow_html=True)


# ── RUN ──
if run_btn:
    with st.spinner("Collecting live features · Training ML models · Running Gemini interpretation..."):
        result = run_full_analysis()
        st.session_state.result    = result
        st.session_state.last_time = datetime.datetime.now()
    st.rerun()

# ── AUTO REFRESH ──
if auto_ref and st.session_state.last_time and GEMINI_KEY:
    elapsed_sec  = (datetime.datetime.now() - st.session_state.last_time).total_seconds()
    interval_sec = st.session_state.refresh_interval * 60
    if elapsed_sec >= interval_sec:
        with st.spinner("Auto-refreshing ML analysis..."):
            result = run_full_analysis()
            st.session_state.result    = result
            st.session_state.last_time = datetime.datetime.now()
        st.rerun()
    else:
        rem = int((interval_sec - elapsed_sec) // 60)
        st.markdown(
            f'<p style="font-size:10px;color:var(--green);text-align:right;">'
            f'🔄 Next refresh in {rem}m</p>', unsafe_allow_html=True
        )


# ── MAIN DISPLAY ──
# ── MAIN DISPLAY ──
if not st.session_state.result:
    # ... (Keep your existing "Empty State" code here) ...
    pts = len(st.session_state.rate_history)
    needed = max(0, 5 - pts)
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px 40px;">
      <div style="font-size:48px;margin-bottom:16px;">🤖</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:700;
      background:linear-gradient(135deg,#dce8f8,#a78bfa);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;margin-bottom:12px;">ML Oracle — Ready</div>
      <p style="color:var(--muted2);max-width:500px;margin:0 auto 16px;line-height:1.8;font-size:14px;">
        This engine uses <strong style="color:var(--amber);">Ridge + Random Forest + Gradient Boosting</strong>
        trained on your live session data.
      </p>
      <div style="background:var(--card);border:1px solid var(--border2);border-radius:12px;
      padding:18px 24px;max-width:420px;margin:0 auto;">
        <div style="font-size:11px;color:var(--muted);margin-bottom:8px;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;text-transform:uppercase;">Training Data Progress</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;color:var(--amber);">{pts}/5 runs</div>
      </div>
    </div>""", unsafe_allow_html=True)

else:
    # 1. UNPACK ALL VARIABLES FIRST (This prevents the NameError)
    r = st.session_state.result
    ml = r.get("ml", {})
    raw = r.get("raw", {})
    feat = r.get("features", {})
    interp = r.get("interp", {})  # <--- THIS DEFINES THE VARIABLE
    metrics = st.session_state.ml_metrics

    p2p_mid = raw.get("p2p_mid", 0)
    official = raw.get("official", 0)
    ensemble = ml.get("ensemble", 0)
    direction = ml.get("direction", "NEUTRAL")
    conf = ml.get("confidence", 0)
    pred_low = ml.get("pred_low", 0)
    pred_high = ml.get("pred_high", 0)
    n_pts = ml.get("n_training_points", 0)
    cold = ml.get("cold_start", True)

    # 2. RENDER THE TOP METRIC CARDS
    dc = "var(--green)" if direction == "BULLISH" else "var(--red)" if direction == "BEARISH" else "var(--amber)"
    da = "▲" if direction == "BULLISH" else "▼" if direction == "BEARISH" else "◆"
    cc = "var(--green)" if conf >= 65 else "var(--amber)" if conf >= 45 else "var(--red)"
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="mcard mcard-green"><div class="mcard-label">Live Rate</div><div class="mcard-value">₦{p2p_mid:,.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="mcard mcard-purple"><div class="mcard-label">ML Target</div><div class="mcard-value">₦{ensemble:,.0f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="mcard mcard-blue"><div class="mcard-label">Direction</div><div class="mcard-value" style="color:{dc};">{da} {direction}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="mcard mcard-purple"><div class="mcard-label">Confidence</div><div class="mcard-value" style="color:{cc};">{conf}%</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="mcard mcard-blue"><div class="mcard-label">Data Points</div><div class="mcard-value">{n_pts}</div></div>', unsafe_allow_html=True)

    # 3. 💎 THE NEW STAKEHOLDER INTELLIGENCE BRIEF (Now interp is defined!)
    st.markdown("""<div style="margin-top:25px; margin-bottom:15px;">
        <h3 style="font-family:'IBM Plex Mono'; color:var(--purple); font-size:18px; letter-spacing:2px;">💎 STAKEHOLDER INTELLIGENCE BRIEF</h3>
    </div>""", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown(f"""
        <div class="ocard" style="border-left: 4px solid var(--purple); height: 100%;">
            <div class="ocard-title">Executive Strategy</div>
            <p style="font-size:16px; line-height:1.6; color:var(--text); font-weight:500;">
                {interp.get('executive_summary', 'Analysing current trends...')}
            </p>
            <div style="background:var(--bg2); padding:15px; border-radius:10px; margin-top:15px;">
                <div class="mcard-label" style="color:var(--amber);">Strategic Advice</div>
                <p style="font-size:14px; color:var(--text); margin:0;">{interp.get('stakeholder_insight', 'Observe market liquidity before large entries.')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        risk = str(interp.get('risk_rating', 'Medium')).upper()
        r_color = "var(--red)" if "HIGH" in risk or "CRITICAL" in risk else "var(--green)" if "LOW" in risk else "var(--amber)"
        
        st.markdown(f"""
        <div class="ocard" style="text-align:center; height: 100%;">
            <div class="ocard-title">Volatility Risk Rating</div>
            <div style="font-size:38px; font-weight:800; color:{r_color}; margin:15px 0;">{risk}</div>
            <p style="font-size:12px; color:var(--muted2);">{interp.get('global_context', 'Monitoring global FX flows...')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ocard" style="background: linear-gradient(90deg, var(--card), var(--bg2)); border-top: 1px solid var(--purple); margin-top:15px;">
        <div class="ocard-title">🗞️ Why is the price moving? (News Correlation)</div>
        <p style="font-size:14px; color:var(--blue); line-height:1.6;">{interp.get('price_movement_drivers', 'Identifying correlation between recent headlines and predicted price shifts...')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. ORIGINAL TABS START HERE
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 ML Results", "📐 Model Metrics", "📊 History & Chart", "🌍 Features", "💬 Chat", "🔔 Alerts"
    ])
    
    # ... (The rest of your code inside tab1, tab2, etc. follows here) ...
    

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 ML Results", "📐 Model Metrics", "📊 History & Chart", "🌍 Features", "💬 Chat", "🔔 Alerts"
    ])

    # ════════ TAB 1: ML RESULTS ════════
    with tab1:
        left, right = st.columns([3, 2])

        with left:
            # Model predictions breakdown
            st.markdown("""<div class="ocard">
            <div class="ocard-title">Individual Model Predictions</div>""", unsafe_allow_html=True)

            for badge, label, pred, desc in [
                ("badge-ridge", "Ridge Regression", ml.get("ridge_pred", 0),
                 "Regularised linear model — captures long-term trend direction. Best for stable regimes."),
                ("badge-rf", "Random Forest", ml.get("rf_pred", 0),
                 "Ensemble of decision trees — detects non-linear patterns & regime shifts. Most robust."),
                ("badge-gb", "Gradient Boosting", ml.get("gb_pred", 0),
                 "Sequential error-correction model — best for time-series patterns. Highest weight in ensemble."),
                ("badge-ens", "Weighted Ensemble", ml.get("ensemble", 0),
                 "Ridge×0.25 + RF×0.35 + GB×0.40 — final prediction. Agreement between models drives confidence."),
            ]:
                diff  = pred - p2p_mid
                color = "var(--green)" if diff >= 0 else "var(--red)"
                arrow = "▲" if diff >= 0 else "▼"
                st.markdown(f"""
                <div style="padding:14px 0;border-bottom:1px solid var(--border);">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                      <span class="model-badge {badge}">{label}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                      font-weight:700;color:{'var(--amber)' if 'Ensemble' in label else 'var(--text)'};">
                        ₦{pred:,.2f}
                      </span>
                      <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{color};">
                        {arrow} ₦{abs(diff):,.2f}
                      </span>
                    </div>
                  </div>
                  <div style="font-size:11px;color:var(--muted2);">{desc}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # AI interpretation
            st.markdown(f"""
            <div class="ocard">
              <div class="ocard-title">🤖 AI Interpretation of ML Output</div>
              <p style="color:#b0c8e8;font-size:14px;line-height:1.75;margin-bottom:14px;">
                {interp.get("executive_summary","")}
              </p>
              <div style="background:var(--bg2);border-radius:10px;padding:14px 16px;
              border:1px solid var(--border2);margin-bottom:12px;">
                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                color:var(--amber);margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
                  WHY THIS DIRECTION
                </div>
                <p style="margin:0;font-size:13px;color:var(--text);line-height:1.6;">
                  {interp.get("why_this_direction","")}
                </p>
              </div>
              <div style="background:var(--bg2);border-radius:10px;padding:14px 16px;
              border:1px solid var(--border2);margin-bottom:12px;">
                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                color:var(--blue);margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
                  TRADE RECOMMENDATION
                </div>
                <p style="margin:0;font-size:13px;color:var(--text);line-height:1.6;">
                  {interp.get("trade_recommendation","")}
                </p>
              </div>
              <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <div style="flex:1;background:var(--bg2);border-radius:8px;padding:10px 14px;
                border:1px solid var(--border);">
                  <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px;
                  text-transform:uppercase;font-family:'IBM Plex Mono',monospace;">Best Convert Time</div>
                  <div style="font-size:13px;color:var(--amber);margin-top:4px;">
                    {interp.get("best_convert_time","N/A")}
                  </div>
                </div>
                <div style="flex:1;background:var(--bg2);border-radius:8px;padding:10px 14px;
                border:1px solid var(--border);">
                  <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px;
                  text-transform:uppercase;font-family:'IBM Plex Mono',monospace;">Weekly Outlook</div>
                  <div style="font-size:13px;color:var(--text);margin-top:4px;">
                    {interp.get("weekly_outlook","N/A")}
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with right:
            # Confidence breakdown
            st.markdown(f"""
            <div class="ocard">
              <div class="ocard-title">📊 Confidence Breakdown</div>
              <p style="font-size:12px;color:var(--muted2);margin-bottom:12px;">
                {interp.get("confidence_explanation","")}
              </p>""", unsafe_allow_html=True)

            if not cold:
                prog_bar("Model Agreement",
                         metrics.get("agreement_score", 0), "var(--green)", 0, 100)
                prog_bar("CV MAE Confidence",
                         metrics.get("mae_conf", 0), "var(--blue)", 0, 100)
                prog_bar("Sample Size Bonus",
                         metrics.get("size_bonus", 0), "var(--purple)", 0, 20)

            st.markdown(f"""
              <div style="margin-top:10px;text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:38px;
                font-weight:700;color:{cc};">{conf}%</div>
                <div style="font-size:11px;color:var(--muted2);margin-top:4px;">
                  Statistical Confidence
                </div>
                <div style="font-size:10px;color:var(--muted);margin-top:6px;">
                  {interp.get("model_agreement_meaning","")}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Key risks
            risks = interp.get("key_risks", [])
            if risks:
                st.markdown('<div class="ocard"><div class="ocard-title">⚠️ Key Risks</div>',
                            unsafe_allow_html=True)
                for rk in risks:
                    st.markdown(
                        f'<div class="alert-box alert-warn" style="margin-bottom:8px;">⚡ {rk}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            # Qualitative vs Quantitative alignment
            qvq = interp.get("qualitative_vs_quantitative", "")
            if qvq:
                q_intel = raw.get("news_intel", {})
                q_dir   = q_intel.get("overall_qualitative_direction", "")
                q_conf  = q_intel.get("qualitative_confidence", 0)
                align_color = "var(--green)" if (
                    (direction == "BULLISH" and "BULLISH" in q_dir) or
                    (direction == "BEARISH" and "BEARISH" in q_dir)
                ) else "var(--amber)" if "NEUTRAL" in q_dir else "var(--red)"
                align_label = "✅ ALIGNED" if (
                    (direction == "BULLISH" and "BULLISH" in q_dir) or
                    (direction == "BEARISH" and "BEARISH" in q_dir)
                ) else "⚠️ DIVERGING"
                st.markdown(f"""
                <div class="ocard">
                  <div class="ocard-title">🔀 Models vs News Alignment</div>
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                    color:{align_color};font-weight:700;">{align_label}</div>
                    <div style="font-size:11px;color:var(--muted2);">
                      ML: <strong style="color:var(--amber);">{direction}</strong> &nbsp;|&nbsp;
                      News: <strong style="color:{align_color};">{q_dir.replace("_USDT","")}</strong>
                    </div>
                  </div>
                  <p style="font-size:12px;color:var(--muted2);margin:0;line-height:1.6;">{qvq}</p>
                </div>""", unsafe_allow_html=True)

            # Rate table
            st.markdown("""
            <div class="ocard"><div class="ocard-title">Rate Summary</div>
            <table class="spread-table">
            <tr><th>Metric</th><th>Value</th></tr>""", unsafe_allow_html=True)

            for label, val, color in [
                ("P2P Live Rate",    f"₦{p2p_mid:,.2f}",  "var(--green)"),
                ("Official Rate",   f"₦{official:,.2f}",  "var(--blue)"),
                ("B.M. Premium",    f"+{prem:.2f}%",       "var(--amber)"),
                ("Ridge Target",    f"₦{ml.get('ridge_pred',0):,.2f}", "var(--blue)"),
                ("RF Target",       f"₦{ml.get('rf_pred',0):,.2f}",   "var(--green)"),
                ("GB Target",       f"₦{ml.get('gb_pred',0):,.2f}",   "var(--purple)"),
                ("Ensemble Target", f"₦{ensemble:,.2f}",  "var(--amber)"),
                ("Range Low",       f"₦{pred_low:,.2f}",  "var(--muted2)"),
                ("Range High",      f"₦{pred_high:,.2f}", "var(--muted2)"),
            ]:
                st.markdown(
                    f'<tr><td style="font-size:12px;">{label}</td>'
                    f'<td style="font-family:IBM Plex Mono,monospace;color:{color};">{val}</td></tr>',
                    unsafe_allow_html=True
                )
            st.markdown('</table></div>', unsafe_allow_html=True)

        # ── QUALITATIVE INTELLIGENCE SECTION (full width below columns) ──
        q_intel      = raw.get("news_intel", {})
        headlines_all = raw.get("news_headlines", [])
        n_headlines  = raw.get("news_headlines_count", 0)

        if q_intel or headlines_all:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:3px;
            text-transform:uppercase;color:var(--purple);margin-bottom:14px;">
              🌐 Live World Intelligence — {n_headlines} headlines scraped & analysed
            </div>""", unsafe_allow_html=True)

            # Breaking event banner
            breaking = q_intel.get("breaking_event")
            if breaking and breaking not in ("null", "None", "N/A", None):
                st.markdown(f"""
                <div style="background:rgba(240,69,90,0.12);border:1px solid var(--red);
                border-left:4px solid var(--red);border-radius:10px;padding:14px 18px;
                margin-bottom:16px;">
                  <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                  color:var(--red);margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
                    ⚡ BREAKING EVENT DETECTED
                  </div>
                  <p style="margin:0;font-size:13px;color:var(--text);line-height:1.6;">{breaking}</p>
                </div>""", unsafe_allow_html=True)

            # Three analysis cards: Oil / Geopolitics / CBN
            qa1, qa2, qa3 = st.columns(3)
            with qa1:
                oil_score = feat.get("news_oil", 0)
                oil_color = "var(--red)" if oil_score > 20 else "var(--green)" if oil_score < -20 else "var(--amber)"
                oil_arrow = "▲ BEARISH FOR NGN" if oil_score > 20 else "▼ BULLISH FOR NGN" if oil_score < -20 else "◆ NEUTRAL"
                st.markdown(f"""
                <div class="ocard" style="height:100%;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                    <div class="ocard-title" style="margin:0;">🛢️ Oil Markets</div>
                    <div style="font-size:10px;color:{oil_color};font-family:'IBM Plex Mono',monospace;
                    font-weight:700;">{oil_arrow}</div>
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                  font-weight:700;color:{oil_color};margin-bottom:8px;">{oil_score:+.0f}</div>
                  <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin:0;">
                    {q_intel.get("oil_analysis","No oil analysis available.")}
                  </p>
                </div>""", unsafe_allow_html=True)
            with qa2:
                geo_score = feat.get("news_geopolitics", 0)
                geo_color = "var(--red)" if geo_score > 20 else "var(--green)" if geo_score < -20 else "var(--amber)"
                st.markdown(f"""
                <div class="ocard" style="height:100%;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                    <div class="ocard-title" style="margin:0;">🌍 Geopolitics</div>
                    <div style="font-size:10px;color:{geo_color};font-family:'IBM Plex Mono',monospace;
                    font-weight:700;">Score: {geo_score:+.0f}</div>
                  </div>
                  <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin:0;">
                    {q_intel.get("geopolitical_analysis","No geopolitical analysis available.")}
                  </p>
                  <div style="margin-top:10px;font-size:11px;color:var(--muted);">
                    <strong style="color:var(--text);">📈 Bullish catalyst:</strong><br>
                    {q_intel.get("top_bullish_catalyst","N/A")[:160]}
                  </div>
                </div>""", unsafe_allow_html=True)
            with qa3:
                cbn_score = feat.get("news_cbn", 0)
                cbn_color = "var(--green)" if cbn_score < -20 else "var(--red)" if cbn_score > 20 else "var(--amber)"
                st.markdown(f"""
                <div class="ocard" style="height:100%;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                    <div class="ocard-title" style="margin:0;">🏦 CBN Watch</div>
                    <div style="font-size:10px;color:{cbn_color};font-family:'IBM Plex Mono',monospace;
                    font-weight:700;">Score: {cbn_score:+.0f}</div>
                  </div>
                  <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin:0;">
                    {q_intel.get("cbn_analysis","No CBN analysis available.")}
                  </p>
                  <div style="margin-top:10px;font-size:11px;color:var(--muted);">
                    <strong style="color:var(--text);">📉 Bearish catalyst:</strong><br>
                    {q_intel.get("top_bearish_catalyst","N/A")[:160]}
                  </div>
                </div>""", unsafe_allow_html=True)

            # Qualitative scores heatmap row
            st.markdown("""
            <div class="ocard" style="margin-top:16px;">
              <div class="ocard-title">📡 10-Dimension Qualitative Signal Scores</div>
              <div style="font-size:11px;color:var(--muted2);margin-bottom:14px;">
                Scores range from -100 (very bullish NGN) to +100 (very bullish USDT).
                Generated by Gemini reasoning over live headlines.
              </div>""", unsafe_allow_html=True)

            score_dims = [
                ("Overall",          feat.get("news_overall", 0),        "🧭"),
                ("Nigeria Macro",    feat.get("news_nigeria", 0),        "🇳🇬"),
                ("CBN Policy",       feat.get("news_cbn", 0),            "🏦"),
                ("Oil Markets",      feat.get("news_oil", 0),            "🛢️"),
                ("USD / Fed",        feat.get("news_usd", 0),            "🇺🇸"),
                ("Crypto",           feat.get("news_crypto", 0),         "₿"),
                ("Geopolitics",      feat.get("news_geopolitics", 0),    "🌍"),
                ("Nigeria Politics", feat.get("news_political_risk", 0), "🗳️"),
                ("Remittances",      feat.get("news_remittance", 0),     "💸"),
                ("Global EM Risk",   feat.get("news_em_risk", 0),        "📊"),
            ]
            cols_scores = st.columns(5)
            for i, (label, score, icon) in enumerate(score_dims):
                col = cols_scores[i % 5]
                bar_pct = int(abs(score))
                bar_color = "var(--red)" if score > 15 else "var(--green)" if score < -15 else "var(--amber)"
                direction_label = "↑ USDT" if score > 15 else "↓ USDT" if score < -15 else "≈ FLAT"
                with col:
                    st.markdown(f"""
                    <div style="background:var(--bg2);border:1px solid var(--border2);
                    border-radius:10px;padding:12px 14px;margin-bottom:10px;text-align:center;">
                      <div style="font-size:18px;margin-bottom:4px;">{icon}</div>
                      <div style="font-size:10px;color:var(--muted2);margin-bottom:6px;
                      font-family:'IBM Plex Mono',monospace;">{label}</div>
                      <div style="font-size:22px;font-weight:700;font-family:'IBM Plex Mono',monospace;
                      color:{bar_color};">{score:+.0f}</div>
                      <div style="font-size:9px;color:{bar_color};margin-top:2px;">{direction_label}</div>
                      <div style="background:var(--border);border-radius:4px;height:4px;
                      margin-top:8px;overflow:hidden;">
                        <div style="background:{bar_color};height:100%;width:{bar_pct}%;
                        border-radius:4px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Headlines used
            key_headlines = interp.get("key_headlines_driving_prediction", [])
            if key_headlines:
                st.markdown("""
                <div class="ocard" style="margin-top:8px;">
                  <div class="ocard-title">📰 Key Headlines Driving This Prediction</div>""",
                unsafe_allow_html=True)
                for h in key_headlines:
                    st.markdown(
                        f'<div style="padding:8px 0;border-bottom:1px solid var(--border);'
                        f'font-size:12px;color:var(--text);line-height:1.5;">📌 {h}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            # Full headlines expander
            if headlines_all:
                with st.expander(f"📋 View all {len(headlines_all)} scraped headlines"):
                    for i, h in enumerate(headlines_all, 1):
                        color = "var(--red)" if "IRAN" in h.upper() or "OIL" in h.upper() or "SANCTION" in h.upper() else \
                                "var(--green)" if "NGN" in h.upper() or "NAIRA" in h.upper() else \
                                "var(--amber)" if "CBN" in h.upper() or "FED" in h.upper() else "var(--muted2)"
                        st.markdown(
                            f'<div style="padding:5px 0;font-size:11px;color:{color};'
                            f'border-bottom:1px solid var(--border);font-family:IBM Plex Mono,monospace;">'
                            f'{i:02d}. {h}</div>',
                            unsafe_allow_html=True
                        )

    # ════════ TAB 2: MODEL METRICS ════════
    with tab2:
        st.markdown("""
        <div class="alert-box alert-info" style="margin-bottom:16px;">
          <strong>📐 What these metrics mean:</strong>
          <strong>CV MAE</strong> = average prediction error across k-fold cross-validation (lower = better).
          <strong>R²</strong> = how much variance the model explains (1.0 = perfect, 0 = baseline).
          <strong>Model Agreement</strong> = how closely all 3 models agree (higher = more reliable ensemble).
        </div>""", unsafe_allow_html=True)

        if cold:
            st.markdown("""<div class="ocard" style="text-align:center;padding:40px;">
            <div style="font-size:32px;margin-bottom:12px;">🌱</div>
            <p style="color:var(--muted2);">Full model metrics available after 5 runs.<br>
            Keep running to build training data.</p>
            </div>""", unsafe_allow_html=True)
        else:
            m1, m2, m3 = st.columns(3)
            for col, badge, label, mae_key in [
                (m1, "badge-ridge", "Ridge Regression", "ridge_cv_mae"),
                (m2, "badge-rf",    "Random Forest",    "rf_cv_mae"),
                (m3, "badge-gb",    "Gradient Boosting","gb_cv_mae"),
            ]:
                mae = metrics.get(mae_key)
                with col:
                    st.markdown(f"""
                    <div class="mcard mcard-blue" style="text-align:center;">
                      <span class="model-badge {badge}" style="margin-bottom:10px;display:inline-block;">{label}</span>
                      <div class="mcard-label">Cross-Val MAE</div>
                      <div class="mcard-value" style="color:var(--amber);">
                        {'₦' + f'{mae:,.2f}' if mae else 'N/A'}
                      </div>
                      <div class="mcard-sub">
                        {f'±{mae/p2p_mid*100:.3f}% of rate' if mae and p2p_mid else '—'}
                      </div>
                    </div>""", unsafe_allow_html=True)

            # R² and agreement
            mc1, mc2 = st.columns(2)
            with mc1:
                r2 = metrics.get("r2_in_sample")
                r2_color = "var(--green)" if (r2 and r2 > 0.7) else "var(--amber)" if (r2 and r2 > 0.4) else "var(--red)"
                st.markdown(f"""
                <div class="mcard mcard-green" style="text-align:center;">
                  <div class="mcard-label">Gradient Boost In-Sample R²</div>
                  <div class="mcard-value" style="color:{r2_color};">{'N/A' if r2 is None else f'{r2:.4f}'}</div>
                  <div class="mcard-sub">
                    {'Excellent fit' if r2 and r2>0.85 else 'Good fit' if r2 and r2>0.65 else 'Moderate — need more data'}
                  </div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                agree = metrics.get("agreement_score", 0)
                agree_color = "var(--green)" if agree > 80 else "var(--amber)" if agree > 60 else "var(--red)"
                st.markdown(f"""
                <div class="mcard mcard-purple" style="text-align:center;">
                  <div class="mcard-label">Model Agreement Score</div>
                  <div class="mcard-value" style="color:{agree_color};">{agree:.1f}%</div>
                  <div class="mcard-sub">
                    {'Strong consensus' if agree>80 else 'Moderate agreement' if agree>60 else 'Models diverging — treat with caution'}
                  </div>
                </div>""", unsafe_allow_html=True)

            # Feature importances
            fi = metrics.get("rf_feature_importance", {})
            if fi:
                st.markdown("""<div class="ocard">
                <div class="ocard-title">🔬 Top Feature Importances (Random Forest)</div>
                <p style="font-size:12px;color:var(--muted2);margin-bottom:14px;">
                  These are the features the model actually uses most — not chosen by the AI, computed by the algorithm.
                </p>""", unsafe_allow_html=True)
                max_imp = max(fi.values()) if fi else 1.0
                for fname, imp in list(fi.items())[:8]:
                    bar_w = int(imp / max_imp * 100)
                    st.markdown(f"""
                    <div class="prog-wrap">
                      <div class="prog-label">
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--text);">
                          {fname}
                        </span>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--amber);">
                          {imp:.4f}
                        </span>
                      </div>
                      <div class="prog-track">
                        <div class="prog-fill" style="width:{bar_w}%;background:var(--purple);"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Explain #1 feature
                top_feature = list(fi.keys())[0]
                st.markdown(f"""
                <div class="ocard">
                  <div class="ocard-title">🔍 Most Important Feature Explained</div>
                  <div style="font-size:13px;color:#b0c8e8;line-height:1.7;">
                    <strong style="color:var(--amber);">#{top_feature}</strong>
                    {interp.get("top_signal_explained","")}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ════════ TAB 3: HISTORY & CHART ════════
    with tab3:
        hist = st.session_state.rate_history
        if len(hist) < 2:
            st.markdown("""<div class="ocard" style="text-align:center;padding:40px;">
            <p style="color:var(--muted2);">Run analysis at least twice to see charts.</p>
            </div>""", unsafe_allow_html=True)
        else:
            import json as _json

            labels    = [h.get("timestamp","")[:16].replace("T"," ") for h in hist]
            p2p_vals  = [h.get("p2p_mid", 0) for h in hist]
            buy_vals  = [h.get("p2p_buy", 0) for h in hist]
            sell_vals = [h.get("p2p_sell", 0) for h in hist]

            # Prediction history (from model_history)
            pred_vals = []
            for h in hist:
                if h.get("features"):
                    pred_vals.append(None)  # placeholder — predictions are next-step
                else:
                    pred_vals.append(None)

            chart_html = f"""
            <div class="ocard">
              <div class="ocard-title">Live P2P Rate History</div>
              <canvas id="rateChart" style="max-height:280px;"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
              const ctx = document.getElementById('rateChart').getContext('2d');
              new Chart(ctx, {{
                type: 'line',
                data: {{
                  labels: {_json.dumps(labels)},
                  datasets: [
                    {{
                      label: 'P2P Mid Rate',
                      data: {_json.dumps(p2p_vals)},
                      borderColor: '#05d68a', backgroundColor: 'rgba(5,214,138,0.08)',
                      borderWidth: 2.5, pointRadius: 5, pointBackgroundColor: '#05d68a',
                      tension: 0.3, fill: true
                    }},
                    {{
                      label: 'P2P Buy',
                      data: {_json.dumps(buy_vals)},
                      borderColor: '#4f8ef7', backgroundColor: 'transparent',
                      borderWidth: 1.5, borderDash: [4,3], pointRadius: 2, tension: 0.3
                    }},
                    {{
                      label: 'P2P Sell',
                      data: {_json.dumps(sell_vals)},
                      borderColor: '#f0455a', backgroundColor: 'transparent',
                      borderWidth: 1.5, borderDash: [4,3], pointRadius: 2, tension: 0.3
                    }}
                  ]
                }},
                options: {{
                  responsive: true,
                  plugins: {{
                    legend: {{ labels: {{ color: '#6b84a0', font: {{ size: 11 }} }} }},
                    tooltip: {{
                      backgroundColor: '#111d2e', titleColor: '#dce8f8', bodyColor: '#6b84a0',
                      borderColor: '#1a2942', borderWidth: 1,
                      callbacks: {{ label: ctx => ctx.dataset.label + ': ₦' + ctx.parsed.y.toLocaleString() }}
                    }}
                  }},
                  scales: {{
                    x: {{ ticks: {{ color: '#4a6080', font: {{ size: 10 }} }}, grid: {{ color: '#1a2942' }} }},
                    y: {{
                      ticks: {{ color: '#4a6080', font: {{ size: 10 }},
                        callback: v => '₦' + v.toLocaleString() }},
                      grid: {{ color: '#1a2942' }}
                    }}
                  }}
                }}
              }});
            </script>"""
            st.components.v1.html(chart_html, height=340)

            # Volatility chart
            if len(hist) >= 3:
                vol_vals = []
                window = 3
                for i in range(len(hist)):
                    if i < window - 1:
                        vol_vals.append(0)
                    else:
                        window_rates = [hist[j].get("p2p_mid", 0) for j in range(i-window+1, i+1)]
                        vol_vals.append(round(float(np.std(window_rates)), 2))

                vol_html = f"""
                <div class="ocard" style="margin-top:16px;">
                  <div class="ocard-title">Rolling Volatility (σ, window=3)</div>
                  <canvas id="volChart" style="max-height:160px;"></canvas>
                </div>
                <script>
                  const vctx = document.getElementById('volChart').getContext('2d');
                  new Chart(vctx, {{
                    type: 'bar',
                    data: {{
                      labels: {_json.dumps(labels)},
                      datasets: [{{
                        label: 'Rolling Std Dev (₦)',
                        data: {_json.dumps(vol_vals)},
                        backgroundColor: 'rgba(245,166,35,0.3)',
                        borderColor: '#f5a623', borderWidth: 1, borderRadius: 4
                      }}]
                    }},
                    options: {{
                      responsive: true,
                      plugins: {{
                        legend: {{ labels: {{ color: '#6b84a0', font: {{ size: 11 }} }} }},
                        tooltip: {{
                          backgroundColor: '#111d2e', titleColor: '#dce8f8',
                          callbacks: {{ label: ctx => 'Volatility: ₦' + ctx.parsed.y.toFixed(2) }}
                        }}
                      }},
                      scales: {{
                        x: {{ ticks: {{ color: '#4a6080', font: {{ size: 10 }} }}, grid: {{ color: '#1a2942' }} }},
                        y: {{ ticks: {{ color: '#4a6080', font: {{ size: 10 }} }}, grid: {{ color: '#1a2942' }} }}
                      }}
                    }}
                  }});
                </script>"""
                st.components.v1.html(vol_html, height=220)

            # History table
            st.markdown("""<div class="ocard"><div class="ocard-title">Full History Log</div>
            <table class="spread-table">
            <tr><th>#</th><th>Timestamp</th><th>P2P Mid</th><th>P2P Buy</th><th>P2P Sell</th><th>Official</th><th>Premium</th></tr>""",
                        unsafe_allow_html=True)
            for i, h in enumerate(reversed(hist)):
                prem_val = ((h.get("p2p_mid",0) - h.get("official",0)) / max(h.get("official",1),1) * 100
                            if h.get("official") else 0)
                st.markdown(f"""
                <tr>
                  <td style="color:var(--muted);font-family:'IBM Plex Mono',monospace;">{len(hist)-i}</td>
                  <td style="font-size:11px;color:var(--muted2);">{str(h.get("timestamp",""))[:19]}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;color:var(--green);">₦{h.get("p2p_mid",0):,.2f}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;color:var(--blue);">{f"₦{h.get('p2p_buy',0):,.2f}" if h.get("p2p_buy") else "—"}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;color:var(--red);">{f"₦{h.get('p2p_sell',0):,.2f}" if h.get("p2p_sell") else "—"}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;color:var(--muted2);">{"₦"+f"{h.get('official',0):,.2f}" if h.get("official") else "—"}</td>
                  <td style="font-family:'IBM Plex Mono',monospace;color:var(--amber);">{f"+{prem_val:.2f}%"}</td>
                </tr>""", unsafe_allow_html=True)
            st.markdown('</table></div>', unsafe_allow_html=True)

    # ════════ TAB 4: FEATURES ════════
    with tab4:
        st.markdown("""<div class="ocard">
        <div class="ocard-title">🔬 All Live Feature Values</div>
        <p style="font-size:12px;color:var(--muted2);margin-bottom:14px;">
          Every numeric signal fed into the ML models. These are the raw inputs — transparent and auditable.
        </p>""", unsafe_allow_html=True)

        feat_groups = {
            "P2P / Rate": ["p2p_spread_abs", "p2p_spread_pct", "p2p_buy_std", "p2p_sell_std",
                           "premium_pct", "premium_abs", "official_rate"],
            "Crypto Market": ["btc_24h_change", "eth_24h_change", "usdt_ngn_cg",
                              "chainlink_change", "uniswap_change"],
            "USD / FX": ["eurusd", "dxy_proxy", "usd_zar", "usd_kes", "usd_ghs"],
            "Time Features": ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                               "is_weekend", "is_business"],
            "News Sentiment": ["news_overall", "news_nigeria", "news_cbn",
                               "news_oil", "news_usd", "news_crypto"],
            "Momentum / Trend": ["momentum_1", "momentum_avg", "volatility",
                                  "trend_slope", "trend_accel", "rate_ma5_dev"],
        }

        for group_name, keys in feat_groups.items():
            st.markdown(f"""
            <div style="margin-bottom:16px;">
              <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
              color:var(--blue);margin-bottom:8px;font-family:'IBM Plex Mono',monospace;">
                {group_name}
              </div>
              <table class="spread-table" style="font-size:12px;">
                <tr><th>Feature</th><th>Value</th><th>Importance (RF)</th></tr>""",
                        unsafe_allow_html=True)
            fi = metrics.get("rf_feature_importance", {})
            for k in keys:
                v = feat.get(k)
                imp = fi.get(k, 0)
                vstr = f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "—"
                imp_bar = f'<div style="width:{int(imp*500)}px;max-width:80px;height:4px;background:var(--purple);border-radius:2px;"></div>'
                st.markdown(
                    f'<tr><td style="font-family:IBM Plex Mono,monospace;font-size:11px;">{k}</td>'
                    f'<td style="color:var(--amber);">{vstr}</td>'
                    f'<td>{imp_bar if imp > 0 else "—"}</td></tr>',
                    unsafe_allow_html=True
                )
            st.markdown('</table></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ════════ TAB 5: CHAT ════════
    with tab5:
        st.markdown('<div class="ocard"><div class="ocard-title">💬 Ask the ML Oracle</div>',
                    unsafe_allow_html=True)

        if not st.session_state.chat:
            st.markdown("""
            <div class="chat-a">
              <div class="chat-badge">⬡ ML ORACLE</div>
              I'm the ML-powered USDT/NGN Oracle. My answers are grounded in statistical model outputs —
              Ridge, Random Forest, and Gradient Boosting. Ask me about predictions, feature importance,
              model confidence, or market direction. I'll tell you <em>why</em> the numbers say what they say.
            </div>""", unsafe_allow_html=True)

        for m in st.session_state.chat:
            if m["r"] == "u":
                st.markdown(f'<div class="chat-u">🧑 {m["c"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-a"><div class="chat-badge">⬡ ML ORACLE</div>{m["c"]}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

        # Quick prompts
        qcols = st.columns(2)
        qs = [
            "Why did the model predict this direction?",
            "What feature is most important right now?",
            "Should I convert USDT to naira today?",
            "How reliable is the current prediction?",
        ]
        clicked = None
        for i, q in enumerate(qs):
            with qcols[i % 2]:
                if st.button(q, key=f"mlq{i}", use_container_width=True):
                    clicked = q

        col1, col2 = st.columns([5, 1])
        with col1:
            user_msg = st.text_input(
                "msg", placeholder="Ask anything about the ML predictions...",
                label_visibility="collapsed", key="ml_chat_in"
            )
        with col2:
            send = st.button("Send →", use_container_width=True, key="ml_send")

        question = user_msg if (send and user_msg) else clicked
        if question:
            st.session_state.chat.append({"r": "u", "c": question})
            with st.spinner("ML Oracle thinking..."):
                reply = chat_response(question, r)
            st.session_state.chat.append({"r": "a", "c": reply})
            st.rerun()

        if st.session_state.chat:
            if st.button("🗑 Clear Chat", key="ml_clear"):
                st.session_state.chat = []
                st.rerun()

    # ════════ TAB 6: ALERTS ════════
    with tab6:
        al_left, al_right = st.columns([1, 1])

        with al_left:
            st.markdown("""
            <div class="ocard">
              <div class="ocard-title">📧 Email Alerts</div>
              <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin-bottom:4px;">
                Get emailed the moment the rate crosses your target price —
                with the latest ML prediction and recommendation included.
              </p>
            </div>""", unsafe_allow_html=True)

            user_email = st.text_input(
                "Your Email Address",
                value=st.session_state.user_email,
                placeholder="yourname@gmail.com",
                key="alert_email_input"
            )
            st.session_state.user_email = user_email

            if user_email:
                if st.button("🧪 Send Test Email", use_container_width=True, key="test_email_btn"):
                    rk = RESEND_KEY
                    if rk:
                        test_html = build_email_html(
                            "This is a test — your email is connected!",
                            p2p_mid, direction, conf, pred_low, pred_high,
                            "This is a test message. Real alerts include live ML analysis."
                        )
                        ok = send_email_alert(user_email, "✅ USDT/NGN Oracle ML — Test Alert", test_html, rk)
                        if ok:
                            st.success("✅ Test email sent! Check your inbox (and spam folder).")
                        else:
                            st.error("❌ Send failed. Check your email and RESEND_API_KEY in secrets.")
                    else:
                        st.error("❌ RESEND_API_KEY not found in secrets.toml. Add it to enable emails.")

                # Show email service status
                if RESEND_KEY:
                    st.markdown(
                        '<div class="alert-box alert-bull" style="margin-top:8px;">'
                        '✅ Email service connected (Resend API key found)</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alert-box alert-warn" style="margin-top:8px;">'
                        '⚠️ Add RESEND_API_KEY to .streamlit/secrets.toml to enable email alerts.<br>'
                        'Get a free key at <strong>resend.com</strong></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("Enter your email above to receive price alerts.")

        with al_right:
            st.markdown("""
            <div class="ocard">
              <div class="ocard-title">🔔 Set Price Alert</div>
              <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin-bottom:4px;">
                Alert fires on the next analysis run when the live P2P rate crosses your level.
              </p>
            </div>""", unsafe_allow_html=True)

            a_level = st.number_input(
                "Alert price (₦)", min_value=100.0, max_value=9999.0,
                value=float(round(p2p_mid * 1.01)),
                step=10.0, key="alert_price_input"
            )
            a_type = st.selectbox("Alert when rate goes:", ["above", "below"], key="alert_type_select")

            if st.button("+ Add Alert", use_container_width=True, key="add_alert_btn"):
                st.session_state.alerts.append({"level": a_level, "type": a_type})
                em_icon = "📧" if st.session_state.user_email else "🔕"
                direction_word = "above" if a_type == "above" else "below"
                st.success(f"Alert set: {em_icon} Notify when rate goes {direction_word} ₦{a_level:,.0f}")

            # Show current rate for reference
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;
            padding:12px 14px;margin-top:8px;text-align:center;">
              <div style="font-size:10px;color:var(--muted);letter-spacing:1px;
              text-transform:uppercase;font-family:IBM Plex Mono,monospace;">Current P2P Rate</div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:22px;
              font-weight:700;color:var(--green);margin-top:4px;">₦{p2p_mid:,.2f}</div>
              <div style="font-size:11px;color:var(--muted2);">Source: {raw.get("rate_source","—")}</div>
            </div>""", unsafe_allow_html=True)

        # Active alerts list
        if st.session_state.alerts:
            st.markdown('<div class="ocard" style="margin-top:16px;"><div class="ocard-title">Active Alerts</div>',
                        unsafe_allow_html=True)
            user_email_set = bool(st.session_state.user_email)
            for i, a in enumerate(st.session_state.alerts):
                already_triggered = i in st.session_state.alert_triggered
                ac1, ac2, ac3 = st.columns([4, 2, 1])
                with ac1:
                    arrow = "▲" if a["type"] == "above" else "▼"
                    em_icon = "📧" if user_email_set else "🔕"
                    status_badge = (
                        '<span style="color:var(--green);font-size:10px;">✅ Triggered</span>'
                        if already_triggered else
                        '<span style="color:var(--amber);font-size:10px;">⏳ Watching</span>'
                    )
                    st.markdown(
                        f'<span style="font-size:13px;color:var(--text);">'
                        f'{em_icon} {arrow} ₦{a["level"]:,} ({a["type"]})</span> {status_badge}',
                        unsafe_allow_html=True
                    )
                with ac2:
                    # Show distance from current rate
                    dist = a["level"] - p2p_mid
                    dist_color = "var(--green)" if (
                        (a["type"] == "above" and dist > 0) or
                        (a["type"] == "below" and dist < 0)
                    ) else "var(--red)"
                    st.markdown(
                        f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;'
                        f'color:{dist_color};">{dist:+,.0f} from now</span>',
                        unsafe_allow_html=True
                    )
                with ac3:
                    if st.button("✕", key=f"del_alert_{i}"):
                        st.session_state.alerts.pop(i)
                        if i in st.session_state.alert_triggered:
                            st.session_state.alert_triggered.remove(i)
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ocard" style="text-align:center;padding:24px;margin-top:16px;">
              <div style="font-size:24px;margin-bottom:8px;opacity:0.3;">🔔</div>
              <p style="color:var(--muted2);font-size:12px;">No active alerts. Set one above.</p>
            </div>""", unsafe_allow_html=True)

    # ── DISCLAIMER ──
    st.markdown("""
    <div style="font-size:10px;color:var(--muted);margin-top:20px;line-height:1.7;
    padding:14px 16px;border-top:1px solid var(--border);text-align:center;">
      ⚠️ Statistical models predict based on historical patterns — they cannot guarantee future prices.
      Confidence scores reflect model agreement and cross-validated accuracy, not certainty.
      Not financial advice. Always do your own research.
    </div>""", unsafe_allow_html=True)
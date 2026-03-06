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
        "rate_history": [],
        "model_history": [],
        "alerts": [], "alert_triggered": [],
        "auto_refresh": False, "refresh_interval": 60,
        "prev_rate": None,
        "ml_metrics": {},
        "user_email": "",
        # ── Global signals cache (refreshes every 5 mins independently) ──
        "global_signals": None,          # full signals dict
        "global_signals_time": None,     # datetime of last fetch
        "global_signals_loading": False, # prevent double-fetch
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


# ═══════════════════════════════════════════════════════════════
# ── GLOBAL SIGNALS ENGINE ──
# Fetches 15+ live data streams every 5 minutes, completely
# independently of the ML analysis. Shows a live ticker panel.
# ═══════════════════════════════════════════════════════════════
SIGNALS_TTL = 300   # seconds — refresh every 5 minutes

def _signals_stale() -> bool:
    """True if signals are missing or older than TTL."""
    t = st.session_state.global_signals_time
    if t is None:
        return True
    return (datetime.datetime.now() - t).total_seconds() > SIGNALS_TTL

def fetch_global_signals() -> dict:
    """
    Pulls 15+ live data feeds and returns a single structured dict.
    Runs in ~8-12 seconds total (parallel topics, short timeouts).
    All errors are caught — partial data is fine.
    """
    sig = {
        "fetched_at": datetime.datetime.now().isoformat(),
        "sources":    [],
        "errors":     [],
    }

    # ── 1. CRYPTO PRICES ──
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,tether,binancecoin,solana,ripple"
            "&vs_currencies=usd,ngn&include_24hr_change=true&include_market_cap=true",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            sig["btc_usd"]       = d.get("bitcoin",{}).get("usd")
            sig["btc_24h"]       = d.get("bitcoin",{}).get("usd_24h_change")
            sig["btc_mcap"]      = d.get("bitcoin",{}).get("usd_market_cap")
            sig["eth_usd"]       = d.get("ethereum",{}).get("usd")
            sig["eth_24h"]       = d.get("ethereum",{}).get("usd_24h_change")
            sig["bnb_usd"]       = d.get("binancecoin",{}).get("usd")
            sig["bnb_24h"]       = d.get("binancecoin",{}).get("usd_24h_change")
            sig["sol_usd"]       = d.get("solana",{}).get("usd")
            sig["sol_24h"]       = d.get("solana",{}).get("usd_24h_change")
            sig["xrp_usd"]       = d.get("ripple",{}).get("usd")
            sig["xrp_24h"]       = d.get("ripple",{}).get("usd_24h_change")
            sig["usdt_ngn_cg"]   = d.get("tether",{}).get("ngn")
            sig["sources"].append("CoinGecko (crypto)")
    except Exception as e:
        sig["errors"].append(f"CoinGecko: {str(e)[:60]}")

    # ── 2. FX RATES (USD vs multiple currencies) ──
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
        if r.status_code == 200:
            rates = r.json().get("rates", {})
            sig["usd_ngn_official"] = rates.get("NGN")
            sig["usd_eur"]          = rates.get("EUR")
            sig["usd_gbp"]          = rates.get("GBP")
            sig["usd_zar"]          = rates.get("ZAR")
            sig["usd_kes"]          = rates.get("KES")
            sig["usd_ghs"]          = rates.get("GHS")
            sig["usd_egp"]          = rates.get("EGP")
            sig["usd_cny"]          = rates.get("CNY")
            sig["usd_jpy"]          = rates.get("JPY")
            sig["eurusd"]           = round(1 / rates["EUR"], 5) if rates.get("EUR") else None
            sig["dxy_proxy"]        = round(rates["EUR"] * 100, 3) if rates.get("EUR") else None
            sig["sources"].append("ExchangeRate-API (FX)")
    except Exception as e:
        sig["errors"].append(f"FX API: {str(e)[:60]}")

    # ── 3. COMMODITY PROXY: Oil (via alternative free endpoint) ──
    # CoinGecko doesn't have oil, but we can get a rough oil proxy from
    # commodity-linked tokens and supplement with RSS headlines
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=petro,oilcoin&vs_currencies=usd&include_24hr_change=true",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            sig["oil_token_change"] = (
                d.get("petro",{}).get("usd_24h_change") or
                d.get("oilcoin",{}).get("usd_24h_change")
            )
    except:
        pass

    # Also scrape oil price headline from Google Finance via RSS
    try:
        r = requests.get(
            "https://news.google.com/rss/search?q=Brent+crude+oil+price+today&hl=en-US&gl=US&ceid=US:en",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
            sig["oil_headline"] = titles[1] if len(titles) > 1 else None
    except:
        pass

    # ── 4. FEAR & GREED INDEX ──
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r.status_code == 200:
            fng = r.json().get("data", [{}])[0]
            sig["fear_greed_value"]      = int(fng.get("value", 50))
            sig["fear_greed_label"]      = fng.get("value_classification", "N/A")
            sig["sources"].append("Alternative.me (Fear & Greed)")
    except Exception as e:
        sig["errors"].append(f"FnG: {str(e)[:40]}")

    # ── 5. GLOBAL NEWS HEADLINES (18 RSS feeds — parallel) ──
    rss_topics = [
        ("Nigeria naira exchange rate",           "🇳🇬 NGN"),
        ("CBN central bank Nigeria forex",        "🏦 CBN"),
        ("crude oil price Brent OPEC today",      "🛢️ Oil"),
        ("Iran oil sanctions",                    "⚠️ Iran"),
        ("US Federal Reserve interest rates",     "🇺🇸 Fed"),
        ("Bitcoin crypto market today",           "₿ BTC"),
        ("Nigeria economy inflation 2025",        "📉 NG Macro"),
        ("Middle East conflict oil supply",       "🌍 MidEast"),
        ("dollar index DXY strength",             "💵 DXY"),
        ("OPEC production output cut",            "🛢️ OPEC"),
        ("Russia Ukraine war commodity",          "⚡ Russia"),
        ("Nigeria crypto P2P USDT",               "💱 NG Crypto"),
        ("emerging markets currency selloff",     "📊 EM FX"),
        ("US inflation CPI report",               "📈 US CPI"),
        ("IMF World Bank Nigeria",                "🏛️ IMF"),
        ("China economy trade slowdown",          "🇨🇳 China"),
        ("Nigeria remittance diaspora",           "💸 Remittance"),
        ("gold price safe haven",                 "🥇 Gold"),
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
                    headlines_raw.append({
                        "tag":   tag,
                        "title": title.strip(),
                        "desc":  desc.strip(),
                        "full":  f"{tag} | {title.strip()}",
                    })
        except:
            pass

    sig["headlines"]       = headlines_raw
    sig["headline_count"]  = len(headlines_raw)
    if headlines_raw:
        sig["sources"].append(f"Google News RSS ({len(headlines_raw)} headlines)")

    # ── 6. SUPPLEMENTAL: NewsAPI headlines (if key available) ──
    if NEWS_KEY:
        extra = []
        for q in ["Nigeria naira USDT", "oil price Iran", "CBN forex"]:
            try:
                r = requests.get(
                    f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}"
                    f"&sortBy=publishedAt&pageSize=3&language=en&apiKey={NEWS_KEY}",
                    timeout=7
                )
                if r.status_code == 200:
                    for a in r.json().get("articles", [])[:2]:
                        t = a.get("title","")
                        d = (a.get("description") or "")[:120]
                        if t:
                            extra.append({"tag":"📰 NewsAPI","title":t,"desc":d,"full":f"📰 NewsAPI | {t}"})
            except:
                pass
        sig["headlines"] = headlines_raw + extra
        sig["headline_count"] = len(sig["headlines"])
        if extra:
            sig["sources"].append(f"NewsAPI ({len(extra)} articles)")

    # ── 7. GEMINI QUALITATIVE SCORING of all signals ──
    all_headlines = sig.get("headlines", [])
    if all_headlines:
        now_str  = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M WAT")
        btc_str  = f"${sig.get('btc_usd',0):,.0f} ({sig.get('btc_24h',0):+.1f}%)" if sig.get("btc_usd") else "N/A"
        fng_str  = f"{sig.get('fear_greed_value','?')} — {sig.get('fear_greed_label','N/A')}"
        oil_hl   = sig.get("oil_headline","N/A")
        dxy_str  = str(sig.get("dxy_proxy","N/A"))
        headlines_block = "\n".join(f"  {i+1}. {h['full']}" for i, h in enumerate(all_headlines[:40]))

        q_prompt = f"""You are a senior FX strategist. Today is {now_str}.

LIVE MARKET SNAPSHOT:
- BTC: {btc_str}
- Crypto Fear & Greed: {fng_str}
- EUR/USD: {sig.get('eurusd','N/A')} | DXY proxy: {dxy_str}
- Oil headline: {oil_hl}
- USD/ZAR: {sig.get('usd_zar','N/A')} | USD/KES: {sig.get('usd_kes','N/A')} | USD/GHS: {sig.get('usd_ghs','N/A')}

LIVE HEADLINES ({len(all_headlines[:40])} total):
{headlines_block}

TASK: Reason about how ALL of the above affects the USDT/NGN P2P black market rate right now.
Think through: oil → Nigeria FX earnings → NGN supply, Fed/DXY → EM pressure, CBN actions,
geopolitical risk → capital flight, crypto sentiment → P2P liquidity, remittances → USD supply.

Return ONLY valid JSON, no markdown, no backticks:
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
  "top_mover_today": "<the single biggest market-moving event happening right now — specific>",
  "breaking_event": "<any breaking event causing sudden rate movement — or null>",
  "oil_analysis": "<2 sentences: current oil situation and NGN impact>",
  "geopolitical_analysis": "<2 sentences: active conflicts/sanctions affecting oil or USD>",
  "cbn_analysis": "<2 sentences: CBN's current stance and likely near-term action>",
  "crypto_analysis": "<1 sentence: crypto market mood and P2P Nigeria impact>",
  "em_analysis": "<1 sentence: broader EM currency pressure today>",
  "top_bullish_catalyst": "<most important reason USDT/NGN could RISE today — cite actual news>",
  "top_bearish_catalyst": "<most important reason USDT/NGN could FALL today — cite actual news>",
  "overall_qualitative_direction": "BULLISH_USDT|BEARISH_USDT|NEUTRAL",
  "qualitative_confidence": <0-100>,
  "30min_bias": "BUY|SELL|HOLD",
  "key_watch_items": ["<thing to watch 1>", "<thing to watch 2>", "<thing to watch 3>"]
}}"""

        try:
            raw_q = gemini(q_prompt,
                "You are a quantitative FX strategist. Return only valid JSON. No markdown.")
            clean_q = raw_q.strip()
            if "```" in clean_q:
                for p in clean_q.split("```"):
                    p = p.strip()
                    if p.startswith("json"): p = p[4:].strip()
                    if p.startswith("{"): clean_q = p; break
            if not clean_q.startswith("{"):
                idx = clean_q.find("{")
                if idx >= 0: clean_q = clean_q[idx:]
            last = clean_q.rfind("}")
            if last >= 0: clean_q = clean_q[:last+1]
            sig["analysis"] = json.loads(clean_q)
            sig["sources"].append("Gemini deep analysis")
        except Exception as e:
            sig["analysis"] = {}
            sig["errors"].append(f"Gemini analysis: {str(e)[:80]}")

    return sig


def maybe_refresh_signals(force: bool = False):
    """Refresh global signals if stale or forced. Stores in session_state."""
    if force or _signals_stale():
        st.session_state.global_signals_loading = True
        try:
            sig = fetch_global_signals()
            st.session_state.global_signals      = sig
            st.session_state.global_signals_time = datetime.datetime.now()
        except Exception as e:
            pass
        finally:
            st.session_state.global_signals_loading = False


def _sig_badge(score, invert=False):
    """Return (label, fg_color, bg_color) for a score -100..+100.
    Positive = USDT bullish (NGN weakens). invert=True flips logic."""
    try: score = float(score)
    except: score = 0
    if invert: score = -score
    if score > 15:   return "BEARISH", "#f0455a", "rgba(240,69,90,0.13)"
    if score < -15:  return "BULLISH", "#05d68a", "rgba(5,214,138,0.13)"
    return "NEUTRAL", "#f5a623", "rgba(245,166,35,0.13)"

def _signal_card(icon, title, badge_label, badge_fg, badge_bg,
                 source_label, source_tag, metrics_line, body):
    """Renders one signal card exactly matching the reference design."""
    return f"""
<div style="background:var(--card);border:1px solid var(--border);border-radius:14px;
padding:16px 18px;margin-bottom:10px;">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
  margin-bottom:8px;gap:8px;flex-wrap:wrap;">
    <div style="display:flex;align-items:center;gap:7px;">
      <span style="font-size:14px;">{icon}</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;
      letter-spacing:.5px;color:var(--text);">{title}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
      <span style="background:{badge_bg};color:{badge_fg};border:1px solid {badge_fg}44;
      border-radius:5px;padding:2px 9px;font-size:10px;font-family:'IBM Plex Mono',monospace;
      font-weight:700;letter-spacing:.8px;">{badge_label}</span>
      <span style="font-size:9px;color:var(--muted);font-family:'IBM Plex Mono',monospace;">
        <span class="live-dot" style="width:5px;height:5px;display:inline-block;
        border-radius:50%;background:var(--green);animation:blink 2s infinite;
        margin-right:3px;vertical-align:middle;"></span>{source_label}
        &nbsp;<span style="color:var(--border2);">·</span>&nbsp;
        <span style="color:var(--blue);">{source_tag}</span>
      </span>
    </div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--amber);
  margin-bottom:6px;">{metrics_line}</div>
  <div style="font-size:12px;color:var(--muted2);line-height:1.65;">{body}</div>
</div>"""


def render_global_signals_tab():
    """Global Signals tab — structured signal cards, refreshes every 5 min."""
    maybe_refresh_signals()

    sig  = st.session_state.global_signals or {}
    anal = sig.get("analysis", {})
    now  = st.session_state.global_signals_time

    # ── Status bar ──
    age_str, next_str = "never", "5m 0s"
    if now:
        age_s   = int((datetime.datetime.now() - now).total_seconds())
        age_str = f"{age_s}s ago" if age_s < 60 else f"{age_s//60}m {age_s%60}s ago"
        rem     = max(0, SIGNALS_TTL - age_s)
        next_str = f"{rem//60}m {rem%60}s"

    n_hl = sig.get("headline_count", 0)
    n_src = len(sig.get("sources", []))

    hcol1, hcol2 = st.columns([6, 1])
    with hcol1:
        st.markdown(f"""
        <div style="margin-bottom:14px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;
          text-transform:uppercase;color:var(--purple);margin-bottom:4px;">🌐 LIVE GLOBAL SIGNALS</div>
          <div style="font-size:11px;color:var(--muted2);">
            {n_hl} headlines &nbsp;·&nbsp; {n_src} sources &nbsp;·&nbsp;
            <span style="color:var(--green);">Updated {age_str}</span> &nbsp;·&nbsp;
            <span style="color:var(--muted);">Next refresh in {next_str}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with hcol2:
        if st.button("↻ Refresh", key="gs_refresh_btn", use_container_width=True):
            maybe_refresh_signals(force=True)
            st.rerun()

    if not sig:
        st.markdown("""<div class="ocard" style="text-align:center;padding:40px;">
        <div style="font-size:32px;margin-bottom:12px;">📡</div>
        <p style="color:var(--muted2);">Press ↻ Refresh to load live global signals.</p>
        </div>""", unsafe_allow_html=True)
        return

    # ── Breaking event banner ──
    breaking = anal.get("breaking_event","")
    if breaking and str(breaking).lower() not in ("null","none","n/a",""):
        st.markdown(f"""
        <div style="background:rgba(240,69,90,0.1);border:1px solid #f0455a;
        border-left:4px solid #f0455a;border-radius:10px;padding:11px 16px;margin-bottom:12px;">
          <span style="font-size:9px;color:#f0455a;font-family:'IBM Plex Mono',monospace;
          letter-spacing:2px;text-transform:uppercase;">⚡ BREAKING EVENT</span>
          <div style="font-size:12px;color:#dce8f8;margin-top:4px;line-height:1.5;">{breaking}</div>
        </div>""", unsafe_allow_html=True)

    # ── Top mover ──
    top_mover = anal.get("top_mover_today","")
    if top_mover:
        st.markdown(f"""
        <div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.25);
        border-radius:10px;padding:9px 16px;margin-bottom:12px;">
          <span style="font-size:9px;color:var(--purple);font-family:'IBM Plex Mono',monospace;
          letter-spacing:2px;text-transform:uppercase;">🎯 TOP MOVER TODAY</span>
          <div style="font-size:12px;color:#dce8f8;margin-top:4px;line-height:1.5;">{top_mover}</div>
        </div>""", unsafe_allow_html=True)

    # ── Extract values ──
    btc_usd  = sig.get("btc_usd");  btc_24h  = sig.get("btc_24h") or 0
    eth_usd  = sig.get("eth_usd");  eth_24h  = sig.get("eth_24h") or 0
    bnb_usd  = sig.get("bnb_usd")
    eurusd   = sig.get("eurusd");   gbp_r    = sig.get("usd_gbp") or 0
    gbpusd   = round(1/gbp_r, 4) if gbp_r else None
    zar      = sig.get("usd_zar");  kes      = sig.get("usd_kes")
    official = sig.get("usd_ngn_official")
    fng_val  = sig.get("fear_greed_value", 50)
    fng_lbl  = sig.get("fear_greed_label","N/A")
    oil_hl   = sig.get("oil_headline","")
    now_dt   = datetime.datetime.now()
    hour     = now_dt.hour; dow = now_dt.weekday()
    dow_name = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][dow]
    session  = ("Lagos/London Overlap" if 8<=hour<=17
                else "US Session" if 18<=hour<=22 else "Off-hours")

    # ─── CARD 1: Crypto Market ───
    crypto_b, crypto_fg, crypto_bg = _sig_badge(
        anal.get("crypto_sentiment", -btc_24h * 2))  # positive BTC = bullish sentiment = USDT bearish
    btc_str = f"BTC ${btc_usd:,.0f} ({btc_24h:+.1f}% 24h)" if btc_usd else "BTC N/A"
    eth_str = f"ETH ${eth_usd:,.0f}" if eth_usd else "ETH N/A"
    mood_txt = ("RISK-OFF (bearish sentiment)" if btc_24h < -1
                else "RISK-ON (bullish sentiment)" if btc_24h > 1 else "Neutral")
    crypto_body = (f"Crypto market sentiment: {mood_txt}. BTC moves affect P2P crypto volume in Nigeria. "
                   f"{anal.get('crypto_analysis','')}")
    st.markdown(_signal_card("📡","Crypto Market", crypto_b, crypto_fg, crypto_bg,
        "CoinGecko","Live", f"{btc_str} | {eth_str}", crypto_body), unsafe_allow_html=True)

    # ─── CARD 2: USD Strength ───
    dxy = sig.get("dxy_proxy",0) or 0
    usd_b, usd_fg, usd_bg = _sig_badge(dxy - 91)   # above 91 = strong USD = bearish NGN
    eur_str = f"EUR/USD: {eurusd:.4f}" if eurusd else "EUR/USD: N/A"
    gbp_str = f"GBP/USD: {gbpusd:.4f}" if gbpusd else "GBP/USD: N/A"
    usd_body = ("Strong USD puts pressure on NGN — harder for CBN to defend. "
                f"{anal.get('em_analysis','')}")
    st.markdown(_signal_card("📡","USD Strength (DXY Proxy)", usd_b, usd_fg, usd_bg,
        "ExchangeRate-API","Live", f"{eur_str} | {gbp_str}", usd_body), unsafe_allow_html=True)

    # ─── CARD 3: EM Africa FX ───
    em_score = anal.get("global_em_risk",0) or 0
    em_b, em_fg, em_bg = _sig_badge(em_score)
    zar_str = f"USD/ZAR: {zar:.2f}" if zar else "USD/ZAR: N/A"
    kes_str = f"USD/KES: {kes:.2f}" if kes else "USD/KES: N/A"
    st.markdown(_signal_card("📡","EM Africa FX Context", em_b, em_fg, em_bg,
        "ExchangeRate-API","Live", f"{zar_str} | {kes_str}",
        "South African Rand and Kenyan Shilling as African EM peers. "
        "Broad EM selloff tends to drag NGN down too. Tracks global risk appetite."),
        unsafe_allow_html=True)

    # ─── CARD 4: Market Microstructure ───
    vol_note = ("High volume — tighter spreads likely." if 9<=hour<=17 and dow<5
                else "Off-hours — lower volume, price may drift.")
    micro_b  = "BULLISH" if 9<=hour<=16 and dow<5 else "NEUTRAL"
    micro_fg = "#05d68a" if micro_b=="BULLISH" else "#f5a623"
    micro_bg = "rgba(5,214,138,0.1)" if micro_b=="BULLISH" else "rgba(245,166,35,0.1)"
    st.markdown(_signal_card("📡","Market Microstructure", micro_b, micro_fg, micro_bg,
        "Market Structure","",
        f"Session: {dow_name} {now_dt.strftime('%H:%M')} WAT",
        f"{vol_note} P2P spreads widen after 11 PM WAT and on weekends. "
        "Best execution: 9 AM–5 PM WAT weekdays."), unsafe_allow_html=True)

    # ─── CARD 5: CBN Policy ───
    cbn_score = anal.get("cbn_policy",0) or 0
    cbn_b, cbn_fg, cbn_bg = _sig_badge(cbn_score)
    cbn_body  = anal.get("cbn_analysis",
        "CBN unified official and I&E window rates in 2023. "
        "Periodic FX auctions to authorised dealers. Diaspora remittance policy eased. "
        "Watch for surprise interventions.")
    st.markdown(_signal_card("📡","CBN Policy Framework", cbn_b, cbn_fg, cbn_bg,
        "Policy Context","",
        "CBN Managed Float + FX Unification (2023–present)",
        cbn_body), unsafe_allow_html=True)

    # ─── CARD 6: Oil Revenue ───
    oil_score = anal.get("oil_impact",0) or 0
    oil_b, oil_fg, oil_bg = _sig_badge(oil_score)
    oil_body  = anal.get("oil_analysis",
        "Nigeria earns most of its USD from crude oil exports. "
        "Brent crude above $80/bbl is generally supportive of NGN. "
        "OPEC+ cuts affect volume. Pipeline vandalism reduces output.")
    if oil_hl:
        oil_body += f" Latest: {oil_hl[:120]}"
    st.markdown(_signal_card("📡","Oil Revenue", oil_b, oil_fg, oil_bg,
        "Google News RSS","Live",
        "Nigeria Oil Dependency: ~90% of FX Earnings",
        oil_body), unsafe_allow_html=True)

    # ─── CARD 7: Crypto P2P Dynamics ───
    ng_score = anal.get("crypto_sentiment",0) or 0
    ng_b, ng_fg, ng_bg = _sig_badge(ng_score)
    st.markdown(_signal_card("📡","Crypto P2P Dynamics", ng_b, ng_fg, ng_bg,
        "Crypto Context","",
        "Binance P2P + KuCoin + LocalBitcoins Nigeria",
        "After CBN Binance ban in 2024, P2P shifted to other platforms. "
        "USDT demand surged as Nigerians hedge against inflation. "
        "P2P rate = best proxy for true black market rate."),
        unsafe_allow_html=True)

    # ─── CARD 8: Inflation Differential ───
    inf_score = anal.get("nigeria_macro",0) or 0
    inf_b, inf_fg, inf_bg = _sig_badge(inf_score)
    st.markdown(_signal_card("📡","Inflation Differential", inf_b, inf_fg, inf_bg,
        "Macro Structural","",
        "Nigeria Inflation ~28–32% vs US ~3% (2025 est.)",
        "High inflation differential means NGN structurally weakens over time vs USD. "
        "This creates long-term USDT demand pressure. "
        "Short-term volatility around MPC meetings."),
        unsafe_allow_html=True)

    # ─── CARD 9: Remittances ───
    rem_score = anal.get("remittance_flow",0) or 0
    rem_b, rem_fg, rem_bg = _sig_badge(rem_score, invert=True)  # high flow = bullish NGN
    st.markdown(_signal_card("📡","Remittances", rem_b, rem_fg, rem_bg,
        "Macro Structural","",
        "Diaspora Remittances ~$20–25B/Year",
        "Remittances are Nigeria's second-largest FX inflow. "
        "Festive seasons (Dec, Eid) see USD supply spikes → NGN strengthens. "
        "Policy on transfer fees and receiving channels affects volumes."),
        unsafe_allow_html=True)

    # ─── CARD 10: Geopolitical Risk ───
    geo_score = anal.get("geopolitical_risk",0) or 0
    geo_b, geo_fg, geo_bg = _sig_badge(geo_score)
    geo_body  = anal.get("geopolitical_analysis",
        "Active conflicts and sanctions affect oil supply and USD safe-haven demand.")
    st.markdown(_signal_card("📡","Geopolitical Risk", geo_b, geo_fg, geo_bg,
        "Google News RSS","Live",
        f"Risk Score: {geo_score:+.0f}/100",
        geo_body), unsafe_allow_html=True)

    # ── Bullish / Bearish summary ──
    bull = anal.get("top_bullish_catalyst","")
    bear = anal.get("top_bearish_catalyst","")
    if bull or bear:
        st.markdown("<br>", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if bull:
                st.markdown(f"""
                <div style="background:rgba(5,214,138,0.07);border:1px solid rgba(5,214,138,0.28);
                border-radius:12px;padding:14px 16px;">
                  <div style="font-size:9px;color:#05d68a;letter-spacing:1.5px;text-transform:uppercase;
                  font-family:'IBM Plex Mono',monospace;margin-bottom:6px;">📈 Top Bullish Catalyst for USDT</div>
                  <div style="font-size:12px;color:#b0c8e8;line-height:1.65;">{bull}</div>
                </div>""", unsafe_allow_html=True)
        with cc2:
            if bear:
                st.markdown(f"""
                <div style="background:rgba(240,69,90,0.07);border:1px solid rgba(240,69,90,0.28);
                border-radius:12px;padding:14px 16px;">
                  <div style="font-size:9px;color:#f0455a;letter-spacing:1.5px;text-transform:uppercase;
                  font-family:'IBM Plex Mono',monospace;margin-bottom:6px;">📉 Top Bearish Catalyst for USDT</div>
                  <div style="font-size:12px;color:#b0c8e8;line-height:1.65;">{bear}</div>
                </div>""", unsafe_allow_html=True)

    # ── Watch items ──
    watch = anal.get("key_watch_items",[])
    if watch:
        st.markdown(f"""
        <div style="margin-top:14px;font-size:11px;color:#6b84a0;border-top:1px solid var(--border);
        padding-top:12px;">
          <strong style="color:#4a6080;font-family:'IBM Plex Mono',monospace;font-size:9px;
          letter-spacing:1.5px;text-transform:uppercase;">👁 WATCH THIS WEEK:</strong>
          {"  ·  ".join(f'<span style="color:#f5a623;">{w}</span>' for w in watch)}
        </div>""", unsafe_allow_html=True)

    # ── Full headlines expander ──
    headlines = sig.get("headlines",[])
    if headlines:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(f"📋 All {len(headlines)} live scraped headlines"):
            for h in headlines:
                c = get_headline_color(h.get("tag",""))
                desc_part = f'<br><span style="color:#4a6080;font-size:10px;">{h["desc"]}</span>' if h.get("desc") else ""
                st.markdown(
                    f'<div style="padding:5px 0;border-bottom:1px solid #1a2942;font-size:11px;">'
                    f'<span style="color:{c};font-family:IBM Plex Mono,monospace;">{h.get("tag","")}</span> '
                    f'<span style="color:#dce8f8;">{h.get("title","")}</span>{desc_part}</div>',
                    unsafe_allow_html=True)


def get_headline_color(tag: str) -> str:
    t = tag.upper()
    if any(x in t for x in ["IRAN","MIDEAST","RUSSIA","WAR","CONFLICT","SANCTION"]): return "#f0455a"
    if any(x in t for x in ["OIL","OPEC","BRENT","🛢"]): return "#f5a623"
    if any(x in t for x in ["NGN","NAIRA","CBN","🇳🇬","🏦"]): return "#05d68a"
    if any(x in t for x in ["BTC","CRYPTO","BITCOIN","₿"]): return "#a78bfa"
    if any(x in t for x in ["FED","USD","DXY","CPI","🇺🇸"]): return "#4f8ef7"
    return "#6b84a0"
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
    # Reuse the cached global signals if available and fresh.
    # This avoids double-fetching headlines on every analysis run.
    cached_sig  = st.session_state.global_signals or {}
    cached_anal = cached_sig.get("analysis", {})
    headlines_all = [h.get("full","") for h in cached_sig.get("headlines", [])]

    # If cache is fresh, pull scores directly from it
    if cached_anal and not _signals_stale():
        feat["news_overall"]        = float(cached_anal.get("overall_score", cached_anal.get("overall_score", 0)))
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
        raw["news_sentiment"]       = cached_anal
        raw["news_headlines"]       = headlines_all[:40]
        raw["news_headlines_count"] = len(headlines_all)
    else:
        # Cache miss — fetch fresh (this also updates the global signals cache)
        maybe_refresh_signals(force=True)
        fresh_sig  = st.session_state.global_signals or {}
        fresh_anal = fresh_sig.get("analysis", {})
        fresh_headlines = [h.get("full","") for h in fresh_sig.get("headlines", [])]

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
            raw["news_sentiment"]       = fresh_anal
            raw["news_headlines"]       = fresh_headlines[:40]
            raw["news_headlines_count"] = len(fresh_headlines)
        else:
            for k in ["news_overall","news_nigeria","news_cbn","news_oil","news_usd",
                      "news_crypto","news_geopolitics","news_political_risk",
                      "news_remittance","news_em_risk"]:
                feat[k] = 0.0
            raw["news_intel"] = {}
            raw["news_headlines"] = []
            raw["news_headlines_count"] = 0

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
    """
    Combines ML numbers + qualitative news intelligence into one unified interpretation.
    Gemini explains the ML result AND integrates real-world events like oil shocks,
    geopolitics, CBN actions etc. Does NOT re-predict — explains what the models say
    and why, in the context of what's actually happening in the world right now.
    """
    top_feat   = ml.get("rf_feature_importance", {})
    top_str    = "\n".join(f"  {k}: {v:.4f}" for k, v in list(top_feat.items())[:8])
    q_intel    = raw.get("news_intel", {})
    headlines  = raw.get("news_headlines", [])
    headlines_sample = "\n".join(f"  • {h}" for h in headlines[:15])

    system = """You are a senior FX strategist and quantitative analyst for a Nigeria-focused trading desk.
You combine statistical ML model outputs with real-world qualitative intelligence.
Be specific — cite actual news events, name the mechanisms, give real numbers.
Return ONLY valid JSON. No markdown, no backticks."""

    prompt = f"""Interpret these combined ML + qualitative results for USDT/NGN:

══ STATISTICAL ML RESULTS ══
- Current P2P Rate: ₦{raw.get("p2p_mid", 0):,.2f}
- Ridge Prediction: ₦{ml["ridge_pred"]:,.2f}
- Random Forest Prediction: ₦{ml["rf_pred"]:,.2f}
- Gradient Boosting Prediction: ₦{ml["gb_pred"]:,.2f}
- Ensemble Prediction: ₦{ml["ensemble"]:,.2f}
- Range: ₦{ml["pred_low"]:,.2f} – ₦{ml["pred_high"]:,.2f}
- Direction: {ml["direction"]}
- Statistical Confidence: {ml["confidence"]}%
- Model Agreement: {ml["model_agreement"]}%
- Training Points: {ml["n_training_points"]}
- Cold Start: {ml["cold_start"]}
- CV MAE (Ridge/RF/GB): {ml.get("ridge_cv_mae") or "N/A"} / {ml.get("rf_cv_mae") or "N/A"} / {ml.get("gb_cv_mae") or "N/A"}

TOP ML FEATURE IMPORTANCES:
{top_str}

══ LIVE QUALITATIVE INTELLIGENCE ({raw.get("news_headlines_count", 0)} headlines scraped) ══
Gemini's qualitative scores from real news (-100 to +100, positive = USDT rises):
- Overall news score:      {feat.get("news_overall", 0):+.0f}
- Nigeria macro:           {feat.get("news_nigeria", 0):+.0f}
- CBN policy:              {feat.get("news_cbn", 0):+.0f}
- Oil market impact:       {feat.get("news_oil", 0):+.0f}
- USD / Fed impact:        {feat.get("news_usd", 0):+.0f}
- Crypto sentiment:        {feat.get("news_crypto", 0):+.0f}
- Geopolitical risk:       {feat.get("news_geopolitics", 0):+.0f}
- Nigeria political risk:  {feat.get("news_political_risk", 0):+.0f}
- Remittance flows:        {feat.get("news_remittance", 0):+.0f}
- Global EM risk:          {feat.get("news_em_risk", 0):+.0f}

KEY QUALITATIVE FINDINGS:
- Top Bullish Catalyst: {q_intel.get("top_bullish_catalyst", "N/A")}
- Top Bearish Catalyst: {q_intel.get("top_bearish_catalyst", "N/A")}
- Breaking Event: {q_intel.get("breaking_event", "None")}
- Oil Analysis: {q_intel.get("oil_analysis", "N/A")}
- Geopolitical Analysis: {q_intel.get("geopolitical_analysis", "N/A")}
- CBN Analysis: {q_intel.get("cbn_analysis", "N/A")}
- Qualitative Direction: {q_intel.get("overall_qualitative_direction", "N/A")}

SAMPLE LIVE HEADLINES USED:
{headlines_sample}

LIVE MARKET DATA:
- BTC 24h: {feat.get("btc_24h_change", "N/A"):+.2f}% | EUR/USD: {feat.get("eurusd", "N/A"):.4f}
- P2P Spread: ₦{feat.get("p2p_spread_abs", 0):.0f} | B.M. Premium: {feat.get("premium_pct", 0):+.2f}%
- Trend slope: {feat.get("trend_slope", 0):+.2f}/interval | Volatility: ₦{feat.get("volatility", 0):.2f}

Return ONLY this JSON (no markdown):
{{
  "executive_summary": "<3-4 sentences combining ML output AND real news events — e.g. mention oil price, Iran, CBN, etc. by name if relevant. What does everything together say?>",
  "why_this_direction": "<2-3 sentences: reference BOTH the top ML features AND the key qualitative drivers — be specific>",
  "top_signal_explained": "<explain the #1 most important ML feature AND the most important qualitative event together>",
  "model_agreement_meaning": "<plain English: what {ml['model_agreement']:.0f}% agreement means for reliability>",
  "confidence_explanation": "<why {ml['confidence']}% confidence — what is limiting it, what would push it higher>",
  "qualitative_vs_quantitative": "<1-2 sentences: do the news signals AGREE or CONTRADICT the ML models? What does that mean?>",
  "trade_recommendation": "<specific actionable recommendation integrating BOTH ML and news — mention timing, amount if relevant>",
  "best_convert_time": "<specific time window based on all signals>",
  "weekly_outlook": "<1-2 sentences: 7-day view combining trend + qualitative events expected this week>",
  "oil_impact_today": "<specific: what is oil doing today and how does that feed through to NGN?>",
  "geopolitical_impact_today": "<specific: any conflicts, sanctions, elections affecting rate today?>",
  "cbn_watch": "<what should we watch from CBN this week?>",
  "key_risks": ["<specific risk citing actual current event>", "<risk 2>", "<risk 3>", "<risk 4>"],
  "key_headlines_driving_prediction": ["<most impactful headline 1>", "<headline 2>", "<headline 3>"]
}}"""

    raw_out = gemini(prompt, system)
    try:
        clean = raw_out.strip()
        if "```" in clean:
            parts = clean.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"): p = p[4:].strip()
                if p.startswith("{"): clean = p; break
        if not clean.startswith("{"):
            idx = clean.find("{"); clean = clean[idx:] if idx >= 0 else clean
        last = clean.rfind("}")
        if last >= 0: clean = clean[:last+1]
        return json.loads(clean)
    except:
        return {
            "executive_summary": raw_out[:500] if raw_out else "Interpretation unavailable.",
            "why_this_direction": "See executive summary.",
            "top_signal_explained": "N/A",
            "model_agreement_meaning": "N/A",
            "confidence_explanation": "N/A",
            "qualitative_vs_quantitative": "N/A",
            "trade_recommendation": "Insufficient data.",
            "best_convert_time": "N/A",
            "weekly_outlook": "N/A",
            "oil_impact_today": "N/A",
            "geopolitical_impact_today": "N/A",
            "cbn_watch": "N/A",
            "key_risks": ["Parse error — retry analysis"],
            "key_headlines_driving_prediction": []
        }


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

# ── GLOBAL SIGNALS: loaded on demand inside its tab ──


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
if not st.session_state.result:
    # Empty state
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
        trained on your live session data. The more times you run it, the smarter it gets.
      </p>
      <div style="background:var(--card);border:1px solid var(--border2);border-radius:12px;
      padding:18px 24px;max-width:420px;margin:0 auto;">
        <div style="font-size:11px;color:var(--muted);margin-bottom:8px;font-family:'IBM Plex Mono',monospace;letter-spacing:1px;text-transform:uppercase;">Training Data Progress</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;color:var(--amber);">{pts}/5 runs</div>
        <div style="font-size:12px;color:var(--muted2);margin-top:4px;">
          {'✅ Enough for ML predictions!' if pts >= 5 else f'Need {needed} more run(s) for full ML predictions'}
        </div>
      </div>
      <p style="color:var(--muted);font-size:12px;margin-top:24px;">
        Press <strong style="color:var(--green);">Run ML Analysis</strong> to start.
        Cold-start estimates are available from run 1 — full ML kicks in after run 5.
      </p>
    </div>""", unsafe_allow_html=True)

else:
    r     = st.session_state.result
    ml    = r.get("ml", {})
    raw   = r.get("raw", {})
    feat  = r.get("features", {})
    interp = r.get("interp", {})
    metrics = st.session_state.ml_metrics

    p2p_mid   = raw.get("p2p_mid", 0)
    official  = raw.get("official", 0)
    ensemble  = ml.get("ensemble", 0)
    direction = ml.get("direction", "NEUTRAL")
    conf      = ml.get("confidence", 0)
    pred_low  = ml.get("pred_low", 0)
    pred_high = ml.get("pred_high", 0)
    n_pts     = ml.get("n_training_points", 0)
    cold      = ml.get("cold_start", True)
    prem      = feat.get("premium_pct", 0)
    p2p_buy   = raw.get("p2p_buy", 0)
    p2p_sell  = raw.get("p2p_sell", 0)
    spread    = (p2p_buy - p2p_sell) if p2p_buy and p2p_sell else 0

    # Pull cached signal data for top cards
    _sig   = st.session_state.global_signals or {}
    _anal  = _sig.get("analysis", {})
    _qdir  = _anal.get("overall_qualitative_direction","NEUTRAL")
    _bias  = _anal.get("30min_bias","—")
    _fng   = _sig.get("fear_greed_value","—")
    _fng_l = _sig.get("fear_greed_label","N/A")

    dc = "var(--green)" if direction=="BULLISH" else "var(--red)" if direction=="BEARISH" else "var(--amber)"
    da = "▲" if direction=="BULLISH" else "▼" if direction=="BEARISH" else "◆"
    cc = "var(--green)" if conf>=65 else "var(--amber)" if conf>=45 else "var(--red)"
    prem_col = "var(--red)" if prem>8 else "var(--amber)" if prem>4 else "var(--green)"
    qd_col = ("var(--green)" if "BEARISH" in _qdir   # bearish USDT = NGN strengthening
              else "var(--red)" if "BULLISH" in _qdir else "var(--amber)")
    qd_lbl = _qdir.replace("_USDT","").replace("_"," ")
    bias_col = "var(--green)" if _bias=="BUY" else "var(--red)" if _bias=="SELL" else "var(--amber)"
    fng_col  = ("var(--green)" if _fng!="—" and int(_fng)>=60
                else "var(--red)" if _fng!="—" and int(_fng)<=30 else "var(--amber)")

    # ── 6 USDT/NGN-focused top cards ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="mcard mcard-green">
        <div class="mcard-label">P2P Buy / Sell</div>
        <div class="mcard-value" style="color:var(--green);">₦{p2p_buy:,.0f}</div>
        <div class="mcard-sub">Sell ₦{p2p_sell:,.0f} · Spread ₦{spread:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="mcard mcard-amber">
        <div class="mcard-label">Black Mkt Premium</div>
        <div class="mcard-value" style="color:{prem_col};">{prem:+.2f}%</div>
        <div class="mcard-sub">vs CBN ₦{official:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="mcard mcard-{'green' if direction=='BULLISH' else 'red' if direction=='BEARISH' else 'amber'}">
        <div class="mcard-label">ML Prediction</div>
        <div class="mcard-value" style="color:{dc};">{da} ₦{ensemble:,.0f}</div>
        <div class="mcard-sub">₦{pred_low:,.0f} – ₦{pred_high:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="mcard mcard-purple">
        <div class="mcard-label">ML Confidence</div>
        <div class="mcard-value" style="color:{cc};">{conf}%</div>
        <div class="mcard-sub">{"⚠️ Cold start" if cold else f"✅ {n_pts} training pts"}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="mcard mcard-blue">
        <div class="mcard-label">News Signal</div>
        <div class="mcard-value" style="color:{qd_col};font-size:16px;">{qd_lbl}</div>
        <div class="mcard-sub">F&G: {_fng} — {_fng_l}</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="mcard mcard-{'green' if _bias=='BUY' else 'red' if _bias=='SELL' else 'amber'}">
        <div class="mcard-label">30-Min Bias</div>
        <div class="mcard-value" style="color:{bias_col};">⚡ {_bias}</div>
        <div class="mcard-sub">Qualitative signal</div>
        </div>""", unsafe_allow_html=True)

    if cold:
        st.markdown(f"""
        <div class="alert-box alert-warn" style="margin-top:12px;">
          <strong>⚠️ Cold Start Mode</strong> — {ml.get("note","")}
          Confidence capped at 35%. Run analysis 5+ times to unlock full ML predictions.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ALERT BANNERS ──
    triggered_alerts = check_and_trigger_alerts(p2p_mid, ml, interp)
    for _, msg in triggered_alerts:
        st.markdown(f'<div class="alert-box alert-warn">{msg}</div>', unsafe_allow_html=True)

    # ── RATE SOURCE BADGE ──
    src = raw.get("rate_source", "unknown")
    status = raw.get("rate_status", "unknown")
    src_color = "var(--green)" if status=="live" else "var(--amber)" if status=="estimated" else "var(--red)"
    st.markdown(
        f'<p style="font-size:11px;font-family:IBM Plex Mono,monospace;color:{src_color};">'
        f'<span class="live-dot" style="background:{src_color};"></span>'
        f'Rate source: {src} &nbsp;·&nbsp; '
        f'Buy ₦{p2p_buy:,.0f} &nbsp;|&nbsp; Sell ₦{p2p_sell:,.0f}'
        f'</p>', unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Analysis", "🌍 Global Signals", "💱 Converter",
        "📈 History", "💬 Chat", "🔔 Alerts", "📐 Model Metrics"
    ])


    # ════════ TAB 1: ANALYSIS ════════
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

    # ════════ TAB 2: GLOBAL SIGNALS ════════
    with tab2:
        render_global_signals_tab()

    # ════════ TAB 3: CONVERTER ════════
    with tab3:
        st.markdown("""<div class="ocard">
        <div class="ocard-title">💱 USDT / NGN Converter</div>""", unsafe_allow_html=True)
        cv1, cv2 = st.columns(2)
        with cv1:
            usdt_in = st.number_input("USDT Amount", min_value=0.0, value=100.0, step=10.0, key="conv_usdt")
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid var(--border2);border-radius:10px;
            padding:14px 18px;margin-top:8px;text-align:center;">
              <div style="font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;
              font-family:IBM Plex Mono,monospace;margin-bottom:6px;">USDT → NGN (P2P mid)</div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:26px;font-weight:700;
              color:var(--green);">₦{usdt_in * p2p_mid:,.2f}</div>
              <div style="font-size:11px;color:var(--muted2);margin-top:4px;">At ₦{p2p_mid:,.2f}/USDT</div>
            </div>""", unsafe_allow_html=True)
        with cv2:
            ngn_in = st.number_input("NGN Amount (₦)", min_value=0.0, value=100000.0, step=1000.0, key="conv_ngn")
            usdt_out = ngn_in / p2p_mid if p2p_mid else 0
            st.markdown(f"""
            <div style="background:var(--bg2);border:1px solid var(--border2);border-radius:10px;
            padding:14px 18px;margin-top:8px;text-align:center;">
              <div style="font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;
              font-family:IBM Plex Mono,monospace;margin-bottom:6px;">NGN → USDT (P2P mid)</div>
              <div style="font-family:IBM Plex Mono,monospace;font-size:26px;font-weight:700;
              color:var(--purple);">{usdt_out:,.4f} USDT</div>
              <div style="font-size:11px;color:var(--muted2);margin-top:4px;">At ₦{p2p_mid:,.2f}/USDT</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<table class="spread-table">
        <tr><th>Rate</th><th>₦/USDT</th><th>For $100 USDT</th><th>Note</th></tr>""",
            unsafe_allow_html=True)
        for lbl, rate, note in [
            ("P2P Buy",     p2p_buy,          "You pay this to get USDT"),
            ("P2P Sell",    p2p_sell,         "You receive this selling USDT"),
            ("P2P Mid",     p2p_mid,          "Fair value midpoint"),
            ("ML Target",   ensemble,         "Model next-step prediction"),
            ("Official",    official or 0,    "CBN interbank benchmark"),
        ]:
            st.markdown(
                f'<tr><td style="font-size:12px;">{lbl}</td>'
                f'<td style="font-family:IBM Plex Mono,monospace;color:var(--amber);">₦{rate:,.2f}</td>'
                f'<td style="font-family:IBM Plex Mono,monospace;color:var(--green);">₦{rate*100:,.0f}</td>'
                f'<td style="font-size:11px;color:var(--muted2);">{note}</td></tr>',
                unsafe_allow_html=True)
        st.markdown('</table></div>', unsafe_allow_html=True)

    # ════════ TAB 7: MODEL METRICS ════════
    with tab7:
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

    # ════════ TAB 4: HISTORY ════════
    with tab4:
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

    # ── FEATURES: shown as expander inside Analysis tab (tab1) ──
    # (rendered in tab1 at the bottom — see "Feature Values" expander in qualitative section)

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
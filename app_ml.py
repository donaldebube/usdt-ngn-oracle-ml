"""
USD/NGN Oracle — Unified Prediction Engine v4.0
═════════════════════════════════════════════════
CHANGES FROM v3:
  ✅ Uses CBN OFFICIAL USD/NGN rate as primary (not P2P / parallel market)
  ✅ Walk-forward backtesting: every prediction is stored then resolved
     against the next real CBN rate, producing genuine out-of-sample MAE
  ✅ Confidence % is ONLY shown once walk-forward data exists (≥2 pairs)
  ✅ P2P rate retained as a secondary spread-signal feature only
  ✅ Auto-refresh uses st_autorefresh (no screen dim/flash)

ML Engine:
  • Ridge Regression (trend baseline)           weight 0.25
  • Random Forest (non-linear regime detection) weight 0.35
  • Gradient Boosting (sequential correction)   weight 0.40

Qualitative Engine:
  • 18 RSS feeds → 40+ live headlines
  • 10-dimension Gemini AI scoring

Multi-Timeframe Forecasting:
  • 24H · 7D · 30D · 3M · 6M · 12M · 2YR
  • Confidence derived from real walk-forward error only
  • Gemini AI narrative for each horizon

Run:
  pip install streamlit scikit-learn numpy pandas requests streamlit-autorefresh
  streamlit run app_oracle.py
"""

import streamlit as st
import requests
import json
import os
import datetime
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="USD/NGN Oracle · CBN Official",
    page_icon="₦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
:root {
  --bg:#060912;--bg2:#0a1020;--bg3:#0d1628;
  --card:#0f1d30;--card2:#132236;
  --border:#172440;--border2:#1e3054;--border3:#274070;
  --green:#00e5a0;--green2:rgba(0,229,160,0.10);--green3:rgba(0,229,160,0.04);
  --red:#ff4466;--red2:rgba(255,68,102,0.10);
  --amber:#ffb020;--amber2:rgba(255,176,32,0.10);
  --blue:#4488ff;--blue2:rgba(68,136,255,0.10);
  --purple:#b060ff;--purple2:rgba(176,96,255,0.10);
  --cyan:#00d4ff;--gold:#ffd700;
  --text:#cfe0f5;--text2:#9ab0cc;--muted:#4a6080;--muted2:#3a4e66;
  --font-mono:'JetBrains Mono',monospace;
  --font-body:'Space Grotesk',sans-serif;
  --r-sm:8px;--r-md:12px;--r-lg:16px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:var(--font-body)!important;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
.block-container{padding:1rem 1.8rem 2rem!important;max-width:1500px!important;}
section[data-testid="stSidebar"]{display:none!important;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3,h4{font-family:var(--font-mono)!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg2);}
::-webkit-scrollbar-thumb{background:var(--border3);border-radius:4px;}
.stTextInput>div>div>input,.stTextArea textarea,.stNumberInput>div>div>input,.stSelectbox>div>div>div{
  background:var(--card)!important;border:1px solid var(--border2)!important;
  color:var(--text)!important;border-radius:var(--r-sm)!important;}
.stButton>button{
  background:linear-gradient(135deg,#1a3060,#2a50a0)!important;color:#fff!important;
  border:1px solid var(--border3)!important;border-radius:var(--r-sm)!important;
  font-family:var(--font-body)!important;font-weight:600!important;transition:all 0.2s ease!important;}
.stButton>button:hover{background:linear-gradient(135deg,#204080,#3060c0)!important;
  border-color:var(--blue)!important;transform:translateY(-1px)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;border-radius:var(--r-md)!important;
  border:1px solid var(--border)!important;padding:4px!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  border-radius:var(--r-sm)!important;font-family:var(--font-mono)!important;
  font-size:11px!important;letter-spacing:0.5px!important;padding:8px 14px!important;}
.stTabs [aria-selected="true"]{background:var(--card2)!important;color:var(--text)!important;
  border:1px solid var(--border2)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:20px!important;}
.stExpander{border:1px solid var(--border)!important;border-radius:var(--r-md)!important;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
@keyframes ticker{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.live-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);
  animation:pulse 2s ease-in-out infinite;margin-right:5px;vertical-align:middle;}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r-lg);
  padding:20px 22px;position:relative;overflow:hidden;animation:fadeIn 0.3s ease;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--border3),transparent);}
.card-green{border-top:2px solid var(--green);}
.card-red{border-top:2px solid var(--red);}
.card-amber{border-top:2px solid var(--amber);}
.card-blue{border-top:2px solid var(--blue);}
.card-purple{border-top:2px solid var(--purple);}
.card-cyan{border-top:2px solid var(--cyan);}
.card-gold{border-top:2px solid var(--gold);}
.card-label{font-family:var(--font-mono);font-size:9px;letter-spacing:2.5px;
  text-transform:uppercase;color:var(--muted);margin-bottom:8px;}
.card-value{font-family:var(--font-mono);font-size:24px;font-weight:700;line-height:1.1;margin-bottom:4px;}
.card-sub{font-size:11px;color:var(--text2);margin-top:5px;}
.sec-header{font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;
  color:var(--muted);padding-bottom:10px;border-bottom:1px solid var(--border);margin-bottom:16px;}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:100px;
  font-family:var(--font-mono);font-size:10px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;}
.badge-bull{background:var(--green2);color:var(--green);border:1px solid rgba(0,229,160,0.3);}
.badge-bear{background:var(--red2);color:var(--red);border:1px solid rgba(255,68,102,0.3);}
.badge-neu{background:var(--amber2);color:var(--amber);border:1px solid rgba(255,176,32,0.3);}
.tf-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r-lg);
  padding:16px 18px;position:relative;overflow:hidden;transition:all 0.2s ease;}
.tf-card:hover{border-color:var(--border3);transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,0.4);}
.tf-card-accent{position:absolute;top:0;left:0;right:0;height:2px;}
.tf-label{font-family:var(--font-mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;
  color:var(--muted);margin-bottom:8px;}
.tf-value{font-family:var(--font-mono);font-size:20px;font-weight:700;line-height:1.2;}
.tf-range{font-size:10px;color:var(--text2);margin-top:4px;font-family:var(--font-mono);}
.tf-change{font-size:11px;font-weight:600;font-family:var(--font-mono);margin-top:6px;}
.tf-conf{font-size:9px;color:var(--muted);margin-top:4px;font-family:var(--font-mono);}
.prog-wrap{margin-bottom:12px;}
.prog-label{display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px;}
.prog-track{background:var(--border);border-radius:3px;height:5px;overflow:hidden;}
.prog-fill{height:100%;border-radius:3px;transition:width 0.5s ease;}
.alert-box{border-radius:var(--r-md);padding:12px 16px;font-size:13px;
  margin-bottom:10px;border-left:3px solid;line-height:1.5;}
.alert-bull{background:var(--green2);border-color:var(--green);color:#a0ead4;}
.alert-bear{background:var(--red2);border-color:var(--red);color:#ffaabb;}
.alert-info{background:var(--blue2);border-color:var(--blue);color:#99bbff;}
.alert-warn{background:var(--amber2);border-color:var(--amber);color:#ffd580;}
.model-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:100px;
  font-family:var(--font-mono);font-size:10px;font-weight:700;}
.badge-ridge{background:rgba(68,136,255,0.15);color:#4488ff;border:1px solid rgba(68,136,255,0.3);}
.badge-rf{background:rgba(0,229,160,0.10);color:#00e5a0;border:1px solid rgba(0,229,160,0.3);}
.badge-gb{background:rgba(176,96,255,0.12);color:#b060ff;border:1px solid rgba(176,96,255,0.3);}
.badge-ens{background:rgba(255,176,32,0.15);color:#ffb020;border:1px solid rgba(255,176,32,0.3);}
.ticker-wrap{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r-sm);
  overflow:hidden;padding:8px 0;margin-bottom:16px;white-space:nowrap;}
.ticker-inner{display:inline-flex;gap:40px;animation:ticker 35s linear infinite;padding:0 20px;}
.ticker-item{display:inline-flex;align-items:center;gap:6px;font-family:var(--font-mono);font-size:11px;color:var(--muted);}
.ticker-item .val{color:var(--text);font-weight:600;}
.ticker-item .up{color:var(--green);}
.ticker-item .dn{color:var(--red);}
.ticker-sep{color:var(--border3);}
.spread-table{width:100%;border-collapse:collapse;font-size:12px;}
.spread-table th{font-family:var(--font-mono);font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--muted);padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);}
.spread-table td{padding:9px 12px;border-bottom:1px solid var(--border);color:var(--text);}
.spread-table tr:last-child td{border-bottom:none;}
.spread-table tr:hover td{background:rgba(255,255,255,0.02);}
.chat-u{background:var(--blue2);border:1px solid rgba(68,136,255,0.2);
  border-radius:14px 14px 3px 14px;padding:12px 16px;margin:8px 0;margin-left:15%;font-size:13px;line-height:1.6;}
.chat-a{background:var(--card);border:1px solid var(--border);
  border-radius:14px 14px 14px 3px;padding:12px 16px;margin:8px 0;margin-right:15%;font-size:13px;line-height:1.6;}
.chat-badge{font-family:var(--font-mono);font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--green);margin-bottom:5px;}
.hl-row{padding:7px 0;border-bottom:1px solid var(--border);font-size:11px;line-height:1.5;}
.hl-row:last-child{border-bottom:none;}
.scenario-card{border-radius:var(--r-md);padding:14px 16px;border:1px solid;margin-bottom:10px;}
.scenario-bull{background:var(--green3);border-color:rgba(0,229,160,0.3);}
.scenario-bear{background:rgba(255,68,102,0.05);border-color:rgba(255,68,102,0.25);}
.hdivider{height:1px;background:linear-gradient(90deg,transparent,var(--border3),transparent);margin:18px 0;}
.wf-table{width:100%;border-collapse:collapse;font-size:11px;}
.wf-table th{font-family:var(--font-mono);font-size:9px;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--muted);padding:7px 10px;text-align:left;border-bottom:1px solid var(--border);}
.wf-table td{padding:7px 10px;border-bottom:1px solid var(--border);font-family:var(--font-mono);}
.wf-table tr:last-child td{border-bottom:none;}
.wf-table tr:hover td{background:rgba(255,255,255,0.02);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SESSION STATE + PERSISTENCE
# ══════════════════════════════════════════════════════
HISTORY_FILE  = "oracle_rate_history.json"
BACKTEST_FILE = "oracle_backtest.json"
MAX_HISTORY   = 2000

def _save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state.rate_history[-MAX_HISTORY:], f, default=str)
    except Exception:
        pass

def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        # Accept both old (p2p_mid) and new (cbn_rate) keys
        valid = []
        for d in data:
            if not isinstance(d, dict): continue
            if not d.get("timestamp"): continue
            if d.get("cbn_rate") or d.get("p2p_mid"):
                if not d.get("cbn_rate") and d.get("p2p_mid"):
                    d["cbn_rate"] = d["p2p_mid"]   # migrate old records
                valid.append(d)
        return valid[-MAX_HISTORY:]
    except Exception:
        return []

def _save_backtest():
    try:
        with open(BACKTEST_FILE, "w") as f:
            json.dump(st.session_state.backtest_log[-500:], f, default=str)
    except Exception:
        pass

def _load_backtest() -> list:
    if not os.path.exists(BACKTEST_FILE):
        return []
    try:
        with open(BACKTEST_FILE) as f:
            return json.load(f)[-500:]
    except Exception:
        return []

def init():
    defaults = {
        "chat": [],
        "result": None,
        "last_time": None,
        "rate_history": [],
        "backtest_log": [],        # [{timestamp, predicted, actual, error_abs, error_pct, direction_correct}]
        "pending_pred": None,      # {ensemble, timestamp, prev_rate} — resolved on next run
        "ml_metrics": {},
        "alerts": [],
        "alert_triggered": [],
        "auto_refresh": False,
        "refresh_interval": 60,
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
            existing = {d.get("timestamp") for d in st.session_state.rate_history}
            new_pts  = [d for d in saved if d.get("timestamp") not in existing]
            st.session_state.rate_history = new_pts + st.session_state.rate_history
        bt = _load_backtest()
        if bt:
            st.session_state.backtest_log = bt
        st.session_state.history_loaded = True

init()


# ══════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════
try:
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
    NEWS_KEY   = st.secrets.get("NEWS_KEY", "")
    GNEWS_KEY  = st.secrets.get("GNEWS_KEY", "")
    RESEND_KEY = st.secrets.get("RESEND_API_KEY", "")
except Exception:
    GEMINI_KEY = ""
    NEWS_KEY   = ""
    GNEWS_KEY  = ""
    RESEND_KEY = ""

if not GEMINI_KEY:
    st.error("⚠️ **GEMINI_KEY not configured.** Add it to `.streamlit/secrets.toml`.")
    st.stop()

# ══════════════════════════════════════════════════════
# GEMINI ENGINE
# ══════════════════════════════════════════════════════
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]

@st.cache_data(ttl=300)
def check_gemini_key(key: str) -> tuple:
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        try:
            r = requests.post(url, json={"contents":[{"parts":[{"text":"Reply: OK"}]}],
                              "generationConfig":{"maxOutputTokens":5}}, timeout=12)
            if r.status_code == 200: return True, model, ""
            if r.status_code == 403: return False, "", "API key invalid"
        except: continue
    return False, "", "No working Gemini model found"

_key_ok, _working_model, _key_err = check_gemini_key(GEMINI_KEY)
if not _key_ok:
    st.error(f"❌ Gemini key error: {_key_err}")
    st.stop()

def gemini(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 4096) -> str:
    parts = []
    if system:
        parts.append({"text": f"SYSTEM:\n{system}\n\n---\n\n"})
    parts.append({"text": prompt})
    payload = {"contents":[{"parts":parts}],
                "generationConfig":{"temperature":temperature,"maxOutputTokens":max_tokens}}
    errors = []
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        try:
            r = requests.post(url, json=payload, timeout=45)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 429:
                return "❌ Rate limit. Wait 60s."
            elif r.status_code == 403:
                return "❌ API key invalid."
            else:
                errors.append(f"{model}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{model}: {str(e)[:50]}")
    return f"❌ All Gemini models failed: {' | '.join(errors)}"

def _parse_json(raw: str) -> dict:
    if not raw or raw.startswith("❌"):
        return {}
    clean = raw.strip()
    if "```" in clean:
        for p in clean.split("```"):
            p = p.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("{"): clean = p; break
    if not clean.startswith("{"):
        idx = clean.find("{")
        if idx >= 0: clean = clean[idx:]
        else: return {}
    last = clean.rfind("}")
    if last >= 0: clean = clean[:last+1]
    try:
        return json.loads(clean)
    except:
        pass
    try:
        fixed = re.sub(r",\s*([}\]])", r"", clean)
        fixed = re.sub(r":\s*<[^>]+>", ': null', fixed)
        return json.loads(fixed)
    except:
        return {}


# ══════════════════════════════════════════════════════
# GLOBAL SIGNALS ENGINE  (unchanged from v3)
# ══════════════════════════════════════════════════════
SIGNALS_TTL = 300

def _signals_stale() -> bool:
    t = st.session_state.global_signals_time
    if t is None: return True
    return (datetime.datetime.now() - t).total_seconds() > SIGNALS_TTL

_BH = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def fetch_global_signals() -> dict:
    sig = {"fetched_at": datetime.datetime.now().isoformat(), "sources": [], "errors": []}

    # Step 0: Gemini knowledge-based scoring (no network needed)
    try:
        now_str = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M WAT")
        early_prompt = (
            f"You are a senior FX strategist for Nigeria/NGN. Today is {now_str}.\n"
            "Score CURRENT market conditions for USD/NGN OFFICIAL CBN rate prediction.\n"
            "Positive score = NGN weakens (USD rises). Negative = NGN strengthens.\n\n"
            "Use your knowledge of CBN policy, oil, USD/Fed, crypto, geopolitics, EM FX, remittances.\n"
            "NEVER return 0 unless genuinely neutral. Return honest non-zero scores.\n\n"
            "Return ONLY valid JSON. No markdown. No backticks. Start with {\n"
            '{"overall_score":15,"nigeria_macro":20,"cbn_policy":-10,"oil_impact":5,'
            '"usd_fed_impact":10,"crypto_sentiment":8,"geopolitical_risk":12,'
            '"political_risk_nigeria":15,"remittance_flow":-5,"global_em_risk":10,'
            '"market_mood":"RISK_ON","top_mover_today":"USD strength","breaking_event":null,'
            '"oil_analysis":"Oil rangebound.","geopolitical_analysis":"Tensions elevated.",'
            '"cbn_analysis":"CBN maintaining managed float.","crypto_analysis":"Risk-on.",'
            '"em_analysis":"EM under pressure.","top_bullish_catalyst":"USD strength",'
            '"top_bearish_catalyst":"CBN intervention","overall_qualitative_direction":"BULLISH_USD",'
            '"qualitative_confidence":65,"30min_bias":"HOLD",'
            '"key_watch_items":["CBN policy","Brent crude","US jobs data"],'
            '"medium_term_outlook":"NGN faces structural pressure.",'
            '"long_term_outlook":"Reform path key.",'
            '"structural_ngn_risks":["Oil dependence","CBN reserves","political risk"]}\n\n'
            "ABOVE IS EXAMPLE FORMAT ONLY. Return YOUR real scored JSON for today. Start with { immediately:"
        )
        raw_early = gemini(early_prompt,
            "Quantitative FX strategist. Return ONLY valid JSON. Start with {.",
            temperature=0.3, max_tokens=1500)
        sig["gemini_raw_response"] = raw_early[:600]
        if raw_early.startswith("❌"):
            sig["errors"].append(f"Gemini early: {raw_early[:200]}")
        else:
            parsed_early = _parse_json(raw_early)
            if parsed_early and "overall_score" in parsed_early:
                sig["analysis"] = parsed_early
                sig["sources"].append("Gemini AI (knowledge-based)")
            else:
                sig["errors"].append(f"Gemini early parse failed. Raw: {raw_early[:200]}")
    except Exception as e:
        sig["errors"].append(f"Gemini early FAILED: {str(e)[:200]}")

    # 1. Crypto prices
    crypto_ok = False
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,tether&vs_currencies=usd&include_24hr_change=true",
            timeout=12, headers=_BH)
        if r.status_code == 200:
            d = r.json()
            sig["btc_usd"] = d.get("bitcoin",{}).get("usd")
            sig["btc_24h"] = d.get("bitcoin",{}).get("usd_24h_change")
            sig["eth_usd"] = d.get("ethereum",{}).get("usd")
            sig["eth_24h"] = d.get("ethereum",{}).get("usd_24h_change")
            sig["sources"].append("CoinGecko")
            crypto_ok = True
        else:
            sig["errors"].append(f"CoinGecko HTTP {r.status_code}")
    except Exception as e:
        sig["errors"].append(f"CoinGecko: {str(e)[:100]}")
    if not crypto_ok:
        try:
            br = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
                              timeout=8, headers=_BH)
            if br.status_code == 200:
                bd = br.json()
                sig["btc_usd"] = float(bd.get("lastPrice",0))
                sig["btc_24h"] = float(bd.get("priceChangePercent",0))
                sig["sources"].append("Binance BTC fallback")
        except: pass

    # 2. FX rates
    def _apply_fx(rates, sig):
        sig["usd_ngn_official"] = rates.get("NGN")
        sig["usd_eur"]  = rates.get("EUR")
        sig["usd_gbp"]  = rates.get("GBP")
        sig["usd_zar"]  = rates.get("ZAR")
        sig["usd_ghs"]  = rates.get("GHS")
        sig["eurusd"]   = round(1/rates["EUR"],5) if rates.get("EUR") else None
        sig["dxy_proxy"]= round(rates["EUR"]*100,3) if rates.get("EUR") else None

    fx_ok = False
    for fx_url in ["https://open.er-api.com/v6/latest/USD",
                   "https://api.frankfurter.app/latest?from=USD"]:
        try:
            r = requests.get(fx_url, timeout=8, headers=_BH)
            if r.status_code == 200:
                _apply_fx(r.json().get("rates",{}), sig)
                sig["sources"].append("ExchangeRate-API")
                fx_ok = True
                break
        except Exception as e:
            sig["errors"].append(f"FX ({fx_url[:30]}): {str(e)[:80]}")

    # 3. Fear & Greed
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8, headers=_BH)
        if r.status_code == 200:
            fng = r.json().get("data",[{}])[0]
            sig["fear_greed_value"] = int(fng.get("value",50))
            sig["fear_greed_label"] = fng.get("value_classification","N/A")
            sig["sources"].append("Fear&Greed Index")
    except: pass

    # 4. News headlines (18 RSS topics)
    rss_topics = [
        ("Nigeria naira exchange rate CBN","🇳🇬 NGN"),
        ("CBN central bank Nigeria forex","🏦 CBN"),
        ("crude oil price Brent OPEC today","🛢️ Oil"),
        ("Iran oil sanctions","⚠️ Iran"),
        ("US Federal Reserve interest rates","🇺🇸 Fed"),
        ("Bitcoin crypto market today","₿ BTC"),
        ("Nigeria economy inflation 2025","📉 NG Macro"),
        ("Middle East conflict oil supply","🌍 MidEast"),
        ("dollar index DXY strength","💵 DXY"),
        ("OPEC production output cut","🛢️ OPEC"),
        ("Russia Ukraine war commodity","⚡ Russia"),
        ("Nigeria crypto USDT","💱 NG Crypto"),
        ("emerging markets currency selloff","📊 EM FX"),
        ("US inflation CPI report","📈 US CPI"),
        ("IMF World Bank Nigeria","🏛️ IMF/WB"),
        ("China economy trade slowdown","🇨🇳 China"),
        ("Nigeria remittance diaspora","💸 Remittance"),
        ("gold price safe haven","🥇 Gold"),
    ]

    def _parse_rss_titles(xml_text):
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", xml_text, re.DOTALL)
        if titles:
            return [t.strip() for t in titles[1:] if t.strip()]
        titles = re.findall(r"<title>(.*?)</title>", xml_text, re.DOTALL)
        cleaned = []
        for t in titles[1:]:
            t = re.sub(r"<[^>]+>","",t).strip()
            t = (t.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
                  .replace("&#39;","'").replace("&quot;",'"'))
            if t and len(t) > 8:
                cleaned.append(t)
        return cleaned

    headlines_raw = []
    rss_errors    = []
    rss_ok_count  = 0
    rss_first_snippet = None

    for query, tag in rss_topics:
        try:
            encoded = requests.utils.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(url, timeout=8, headers={
                "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept":"application/rss+xml, application/xml, text/xml, */*"})
            if rss_first_snippet is None:
                rss_first_snippet = f"HTTP {r.status_code} | {r.headers.get('Content-Type','?')} | {r.text[:200]}"
            if r.status_code == 200:
                titles = _parse_rss_titles(r.text)
                added = 0
                for title in titles[:2]:
                    headlines_raw.append({"tag":tag,"title":title,"full":f"{tag} | {title}"})
                    added += 1
                if added > 0: rss_ok_count += 1
                else: rss_errors.append(f"{tag}: 200 but 0 titles parsed")
            else:
                rss_errors.append(f"{tag}: HTTP {r.status_code}")
        except Exception as e:
            rss_errors.append(f"{tag}: {str(e)[:80]}")

    sig["rss_ok_count"]      = rss_ok_count
    sig["rss_errors"]        = rss_errors
    sig["rss_first_response"]= rss_first_snippet

    if GNEWS_KEY:
        gnews_queries = [
            ("Nigeria naira CBN exchange rate","🇳🇬 GNews"),
            ("Nigeria economy CBN policy","🏦 GNews"),
            ("crude oil price OPEC Brent","🛢️ GNews"),
            ("US Federal Reserve dollar","🇺🇸 GNews"),
            ("emerging markets currency","📊 GNews"),
        ]
        gnews_count = 0
        for q, tag in gnews_queries:
            try:
                r = requests.get("https://gnews.io/api/v4/search",
                    params={"q":q,"lang":"en","max":3,"sortby":"publishedAt","apikey":GNEWS_KEY},
                    timeout=8, headers={"User-Agent":"Mozilla/5.0"})
                if r.status_code == 200:
                    for a in r.json().get("articles",[])[:2]:
                        t = (a.get("title") or "").strip()
                        if t and len(t)>8:
                            headlines_raw.append({"tag":tag,"title":t,"full":f"{tag} | {t}"})
                            gnews_count += 1
                elif r.status_code in (429,403,401):
                    sig["errors"].append(f"GNews {r.status_code}")
                    break
            except: pass
        if gnews_count > 0:
            sig["sources"].append(f"GNews ({gnews_count} headlines)")
        sig["gnews_count"] = gnews_count

    if NEWS_KEY:
        for q in ["Nigeria naira CBN","oil price Iran","Nigeria economy"]:
            try:
                r = requests.get(
                    f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}"
                    f"&sortBy=publishedAt&pageSize=3&language=en&apiKey={NEWS_KEY}", timeout=7)
                if r.status_code == 200:
                    for a in r.json().get("articles",[])[:2]:
                        t = a.get("title","")
                        if t: headlines_raw.append({"tag":"📰 NewsAPI","title":t,"full":f"📰 NewsAPI | {t}"})
            except: pass

    sig["headlines"]      = headlines_raw
    sig["headline_count"] = len(headlines_raw)
    if headlines_raw:
        sig["sources"].append(f"Google News RSS ({rss_ok_count}/{len(rss_topics)} feeds)")

    # Step 6: Gemini enhanced pass with live market data + headlines
    has_market = bool(sig.get("btc_usd") or sig.get("usd_ngn_official"))
    has_hl     = bool(headlines_raw)
    already    = bool(sig.get("analysis"))

    if has_market or has_hl:
        now_str2  = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M WAT")
        btc_str   = f"${sig.get('btc_usd',0):,.0f} ({sig.get('btc_24h',0):+.1f}%)" if sig.get("btc_usd") else "N/A"
        fng_str   = f"{sig.get('fear_greed_value','?')} — {sig.get('fear_greed_label','N/A')}"
        hl_block  = "\n".join(f"  {i+1}. {h['full']}" for i,h in enumerate(headlines_raw[:35]))
        enhanced  = f"""You are a senior FX strategist. Today: {now_str2}
Score USD/NGN OFFICIAL CBN rate. Positive = NGN weakens.

LIVE DATA:
- BTC: {btc_str} | F&G: {fng_str}
- EUR/USD: {sig.get('eurusd','N/A')} | USD/NGN Official: {sig.get('usd_ngn_official','N/A')}

HEADLINES ({len(headlines_raw)}):
{hl_block if has_hl else "(none)"}

Return ONLY valid JSON. Start with {{"overall_score":
ALL fields required: overall_score, nigeria_macro, cbn_policy, oil_impact, usd_fed_impact,
crypto_sentiment, geopolitical_risk, political_risk_nigeria, remittance_flow, global_em_risk
(integers -100 to 100), market_mood, top_mover_today, breaking_event,
oil_analysis, geopolitical_analysis, cbn_analysis, crypto_analysis, em_analysis,
top_bullish_catalyst, top_bearish_catalyst, overall_qualitative_direction,
qualitative_confidence, 30min_bias, key_watch_items (array 3),
medium_term_outlook, long_term_outlook, structural_ngn_risks (array 3).
START NOW with {{"""
        try:
            raw_q = gemini(enhanced,
                "FX strategist. Return ONLY valid JSON. Start with {.", temperature=0.3, max_tokens=1800)
            sig["gemini_raw_response"] = raw_q[:600]
            if not raw_q.startswith("❌"):
                parsed = _parse_json(raw_q)
                if parsed and "overall_score" in parsed:
                    sig["analysis"] = parsed
                    sig["sources"].append("Gemini AI (enhanced with live data)")
                elif already:
                    sig["errors"].append("Gemini enhanced: bad parse, keeping Step 0 result.")
                else:
                    sig["errors"].append(f"Gemini enhanced: empty parse. Raw: {raw_q[:150]}")
            else:
                sig["errors"].append(f"Gemini enhanced error: {raw_q[:100]}")
        except Exception as e:
            sig["errors"].append(f"Gemini enhanced: {str(e)[:150]}")

    return sig


def maybe_refresh_signals(force: bool = False):
    if force or _signals_stale():
        try:
            sig = fetch_global_signals()
            st.session_state.global_signals      = sig
            st.session_state.global_signals_time = datetime.datetime.now()
        except Exception as e:
            crash = st.session_state.global_signals or {}
            if isinstance(crash, dict):
                crash["errors"] = crash.get("errors",[]) + [f"CRASHED: {str(e)[:200]}"]
                st.session_state.global_signals = crash


# ══════════════════════════════════════════════════════
# CBN OFFICIAL RATE FETCHER
# ══════════════════════════════════════════════════════
def fetch_cbn_rate() -> dict:
    """
    Fetch the CBN official USD/NGN exchange rate.
    Primary sources: open.er-api.com, frankfurter.app, exchangerate-api.com
    These pull from IMF/central bank data and update once per business day.
    P2P rate is also fetched as a supplementary signal/feature — never as primary.
    Returns: {cbn_rate, source, status, eur, gbp, zar, ghs, eurusd, dxy_proxy,
              p2p_mid (optional), btc_24h (optional), eth_24h (optional)}
    """
    result = {
        "cbn_rate": None, "source": "unknown", "status": "fetching",
        "eur": None, "gbp": None, "zar": None, "ghs": None,
        "eurusd": None, "dxy_proxy": None, "p2p_mid": None,
        "btc_24h": None, "eth_24h": None,
    }

    # Primary: free FX APIs that source from CBN/IMF
    for url in [
        "https://open.er-api.com/v6/latest/USD",
        "https://api.frankfurter.app/latest?from=USD",
        "https://api.exchangerate-api.com/v4/latest/USD",
    ]:
        try:
            r = requests.get(url, timeout=10, headers=_BH)
            if r.status_code == 200:
                rates = r.json().get("rates", {})
                ngn   = rates.get("NGN")
                if ngn and float(ngn) > 100:
                    result["cbn_rate"]  = float(ngn)
                    result["source"]    = f"ExchangeRate-API ({url.split('/')[2]})"
                    result["status"]    = "live"
                    result["eur"]       = rates.get("EUR")
                    result["gbp"]       = rates.get("GBP")
                    result["zar"]       = rates.get("ZAR")
                    result["ghs"]       = rates.get("GHS")
                    if rates.get("EUR"):
                        result["eurusd"]    = round(1/rates["EUR"], 5)
                        result["dxy_proxy"] = round(rates["EUR"]*100, 3)
                    break
        except: continue

    # Fallback: CoinGecko USDT/NGN — rough proxy (parallel rate, warn user)
    if result["cbn_rate"] is None:
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=ngn",
                timeout=10, headers=_BH)
            if r.status_code == 200:
                ngn_val = r.json().get("tether",{}).get("ngn")
                if ngn_val and float(ngn_val) > 100:
                    result["cbn_rate"] = float(ngn_val)
                    result["source"]   = "CoinGecko USDT/NGN ⚠️ proxy (not official CBN)"
                    result["status"]   = "proxy"
        except: pass

    # Hard fallback
    if result["cbn_rate"] is None:
        result["cbn_rate"] = 1580.0
        result["source"]   = "⚠️ Fallback estimate (all APIs failed)"
        result["status"]   = "estimated"

    # Crypto 24h changes (used as ML features)
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true",
            timeout=10, headers=_BH)
        if r.status_code == 200:
            d = r.json()
            result["btc_24h"] = d.get("bitcoin",{}).get("usd_24h_change")
            result["eth_24h"] = d.get("ethereum",{}).get("usd_24h_change")
    except: pass

    # P2P rate as secondary signal feature (Binance P2P)
    try:
        r = requests.post(
            "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search",
            json={"asset":"USDT","fiat":"NGN","merchantCheck":False,"page":1,
                  "payTypes":[],"publisherType":None,"rows":10,"tradeType":"BUY"},
            headers={"Content-Type":"application/json","User-Agent":"Mozilla/5.0"},
            timeout=15)
        if r.status_code == 200:
            prices = [float(item["adv"]["price"])
                      for item in r.json().get("data",[])
                      if float(item.get("adv",{}).get("price",0)) > 100]
            if prices:
                result["p2p_mid"] = round(sum(prices)/len(prices), 2)
    except: pass

    return result


# ══════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST ENGINE
# ══════════════════════════════════════════════════════
def resolve_pending_prediction(actual_cbn_rate: float):
    """
    Called at the START of every analysis run.
    Compares the prediction stored last run against today's real CBN rate.
    Appends a validated row to backtest_log and clears pending_pred.
    This is the ONLY source of confidence numbers shown to users.
    """
    pending = st.session_state.pending_pred
    if pending is None:
        return
    pred      = pending.get("ensemble")
    prev_rate = pending.get("prev_rate")
    ts        = pending.get("timestamp")
    if pred is None:
        st.session_state.pending_pred = None
        return

    error_abs = abs(actual_cbn_rate - pred)
    error_pct = error_abs / max(actual_cbn_rate, 1) * 100

    direction_correct = None
    if prev_rate:
        actual_moved_up = actual_cbn_rate > prev_rate
        pred_moved_up   = pred > prev_rate
        direction_correct = (actual_moved_up == pred_moved_up)

    st.session_state.backtest_log.append({
        "timestamp":         ts,
        "predicted":         round(pred, 2),
        "actual":            round(actual_cbn_rate, 2),
        "error_abs":         round(error_abs, 2),
        "error_pct":         round(error_pct, 4),
        "direction_correct": direction_correct,
    })
    _save_backtest()
    st.session_state.pending_pred = None


def compute_backtest_stats() -> dict:
    """
    Compute genuine out-of-sample accuracy from the walk-forward log.
    Returns ready=False until we have at least 2 validated pairs.
    Confidence is derived from real MAPE — NOT from in-sample cross-validation.
    """
    log = st.session_state.backtest_log
    n   = len(log)

    if n < 2:
        return {
            "n": n, "mae": None, "mape": None, "dir_accuracy": None,
            "conf_from_bt": None, "ready": False,
            "message": (
                f"Need at least 2 resolved predictions ({n} so far). "
                "Run the analysis on two separate occasions to build real backtest data."
            ),
        }

    errors_abs = [r["error_abs"] for r in log]
    errors_pct = [r["error_pct"] for r in log]
    dir_log    = [r for r in log if r.get("direction_correct") is not None]

    mae  = round(float(np.mean(errors_abs)), 2)
    mape = round(float(np.mean(errors_pct)), 4)
    dir_acc = (
        round(sum(1 for r in dir_log if r["direction_correct"]) / len(dir_log) * 100, 1)
        if dir_log else None
    )

    # Derive confidence from real MAPE:
    # MAPE ≤ 0.1%  → 85%  confidence (tight)
    # MAPE ≥ 2.0%  → 25%  confidence (loose)
    # Linear interpolation in between
    if mape <= 0.10:
        conf = 85
    elif mape >= 2.0:
        conf = 25
    else:
        conf = int(85 - (mape - 0.10) / (2.0 - 0.10) * 60)

    return {
        "n": n, "mae": mae, "mape": mape,
        "dir_accuracy": dir_acc, "conf_from_bt": conf, "ready": True,
        "message": f"Based on {n} real walk-forward observations.",
    }


# ══════════════════════════════════════════════════════
# FEATURE ENGINEERING  (CBN rate as primary signal)
# ══════════════════════════════════════════════════════
FEATURE_COLS = [
    # P2P premium as a spread / black-market pressure signal
    "p2p_premium_pct",
    # Crypto
    "btc_24h_change", "eth_24h_change",
    # FX
    "eurusd", "dxy_proxy", "usd_zar", "usd_ghs",
    # Temporal
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_business",
    # Qualitative Gemini scores (10 dimensions)
    "news_overall", "news_nigeria", "news_cbn", "news_oil", "news_usd",
    "news_crypto", "news_geopolitics", "news_political_risk",
    "news_remittance", "news_em_risk",
    # Historical momentum (computed on CBN rate history)
    "momentum_1", "momentum_avg", "volatility", "trend_slope", "trend_accel",
    "rate_ma5_dev",
]

def features_to_vector(feat: dict) -> np.ndarray:
    return np.array([float(feat.get(c, 0.0)) for c in FEATURE_COLS], dtype=float)


def collect_features() -> tuple:
    """
    Fetch the CBN official rate (primary) + all auxiliary features.
    Returns (raw_dict, features_dict).
    """
    feat = {}
    raw  = {}

    # ── CBN Official Rate ──
    rate_data = fetch_cbn_rate()
    cbn_rate  = rate_data["cbn_rate"]
    raw.update(rate_data)
    raw["cbn_rate"] = cbn_rate

    # P2P premium as a spread-pressure feature (positive = parallel > official)
    if raw.get("p2p_mid") and cbn_rate:
        feat["p2p_premium_pct"] = round((raw["p2p_mid"] - cbn_rate) / cbn_rate * 100, 4)
    else:
        feat["p2p_premium_pct"] = 0.0

    # ── FX Features ──
    if raw.get("eur"):
        feat["eurusd"]    = round(1 / raw["eur"], 6)
        feat["dxy_proxy"] = round(raw["eur"] * 100, 4)
    if raw.get("zar"):  feat["usd_zar"] = float(raw["zar"])
    if raw.get("ghs"):  feat["usd_ghs"] = float(raw["ghs"])

    # ── Crypto Features ──
    feat["btc_24h_change"] = raw.get("btc_24h") or 0.0
    feat["eth_24h_change"] = raw.get("eth_24h") or 0.0

    # ── Temporal Features ──
    now = datetime.datetime.now()
    feat["is_weekend"]  = int(now.weekday() >= 5)
    feat["is_business"] = int(8 <= now.hour <= 17 and now.weekday() < 5)
    feat["hour_sin"]    = float(np.sin(2 * np.pi * now.hour / 24))
    feat["hour_cos"]    = float(np.cos(2 * np.pi * now.hour / 24))
    feat["dow_sin"]     = float(np.sin(2 * np.pi * now.weekday() / 7))
    feat["dow_cos"]     = float(np.cos(2 * np.pi * now.weekday() / 7))
    feat["month_sin"]   = float(np.sin(2 * np.pi * now.month / 12))
    feat["month_cos"]   = float(np.cos(2 * np.pi * now.month / 12))

    # ── Qualitative Intelligence ──
    cached_sig  = st.session_state.global_signals or {}
    cached_anal = cached_sig.get("analysis", {})

    if cached_anal and not _signals_stale():
        anal = cached_anal
    else:
        maybe_refresh_signals(force=True)
        anal = (st.session_state.global_signals or {}).get("analysis", {})

    if anal:
        feat["news_overall"]        = float(anal.get("overall_score", 0))
        feat["news_nigeria"]        = float(anal.get("nigeria_macro", 0))
        feat["news_cbn"]            = float(anal.get("cbn_policy", 0))
        feat["news_oil"]            = float(anal.get("oil_impact", 0))
        feat["news_usd"]            = float(anal.get("usd_fed_impact", 0))
        feat["news_crypto"]         = float(anal.get("crypto_sentiment", 0))
        feat["news_geopolitics"]    = float(anal.get("geopolitical_risk", 0))
        feat["news_political_risk"] = float(anal.get("political_risk_nigeria", 0))
        feat["news_remittance"]     = float(anal.get("remittance_flow", 0))
        feat["news_em_risk"]        = float(anal.get("global_em_risk", 0))
        raw["news_intel"]           = anal
        raw["news_headlines"]       = [h.get("full","") for h in cached_sig.get("headlines",[])[:40]]
        raw["news_headlines_count"] = len(raw["news_headlines"])
    else:
        for k in ["news_overall","news_nigeria","news_cbn","news_oil","news_usd",
                  "news_crypto","news_geopolitics","news_political_risk","news_remittance","news_em_risk"]:
            feat[k] = 0.0
        raw["news_intel"]           = {}
        raw["news_headlines"]       = []
        raw["news_headlines_count"] = 0

    # ── Historical Momentum (on CBN rate) ──
    hist = st.session_state.rate_history
    if len(hist) >= 2:
        recent = [h["cbn_rate"] for h in hist[-10:] if h.get("cbn_rate")]
        if len(recent) >= 2:
            feat["momentum_1"]   = recent[-1] - recent[-2]
            feat["momentum_avg"] = recent[-1] - float(np.mean(recent[:-1]))
            feat["volatility"]   = float(np.std(recent))
            x = np.arange(len(recent))
            feat["trend_slope"]  = float(np.polyfit(x, recent, 1)[0])
            feat["trend_accel"]  = (feat["momentum_1"] - (recent[-2] - recent[-3])
                                    if len(recent) >= 3 else 0.0)
        if len(recent) >= 5:
            feat["rate_ma5_dev"] = recent[-1] - float(np.mean(recent[-5:]))
    else:
        for k in ["momentum_1","momentum_avg","volatility","trend_slope","trend_accel","rate_ma5_dev"]:
            feat[k] = 0.0

    # Sanitise NaN / Inf
    for k, v in feat.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
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
        feat      = hist[i].get("features", {})
        next_rate = hist[i+1].get("cbn_rate")   # ← predict next CBN rate
        if next_rate and feat:
            X_rows.append(features_to_vector(feat))
            y_vals.append(float(next_rate))
            times.append(hist[i].get("timestamp",""))
    if len(X_rows) < 4:
        return None, None, None
    return np.array(X_rows), np.array(y_vals), times


def train_and_predict(current_feat: dict, cbn_rate: float) -> dict:
    """
    Train ensemble on historical CBN rates and return next-observation prediction.
    Confidence is NEVER fabricated — it is either:
      (a) None  → not yet validated (fewer than 2 walk-forward pairs)
      (b) int   → derived from real walk-forward MAPE
    """
    X, y, times = build_training_data()
    cold_start  = (X is None)
    bt_stats    = compute_backtest_stats()

    if cold_start:
        # Simple heuristic — directional nudge only, no confidence claimed
        score  = 0.0
        score += feat.get("btc_24h_change", 0) * 0.3 if (feat := current_feat) else 0
        score += -current_feat.get("dxy_proxy", 0) * 0.01
        score += current_feat.get("news_overall", 0) * 0.5
        score += current_feat.get("momentum_1", 0) * 0.8
        est = round(cbn_rate * (1 + score / 5000.0), 2)
        return {
            "cold_start": True,
            "n_training_points": 0,
            "ridge_pred": est, "rf_pred": est, "gb_pred": est, "ensemble": est,
            "pred_low":   round(cbn_rate * 0.992, 2),
            "pred_high":  round(cbn_rate * 1.008, 2),
            "confidence":        None,   # ← intentionally None
            "confidence_source": "cold start — no history yet",
            "direction": "BULLISH" if est > cbn_rate else "BEARISH" if est < cbn_rate else "NEUTRAL",
            "model_agreement": 100.0,
            "rf_feature_importance": {},
            "backtest": bt_stats,
            "note": "COLD START — fewer than 5 data points. Run analysis daily to build history.",
        }

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_pred   = scaler.transform(features_to_vector(current_feat).reshape(1, -1))
    cv_k     = min(5, len(X_scaled))

    # Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_scaled, y)
    ridge_pred    = float(ridge.predict(x_pred)[0])
    ridge_cv_mae  = float(-cross_val_score(ridge, X_scaled, y, cv=cv_k,
                          scoring="neg_mean_absolute_error").mean()) if len(X_scaled) >= 5 else None

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_pred    = float(rf.predict(x_pred)[0])
    rf_cv_mae  = float(-cross_val_score(rf, X_scaled, y, cv=cv_k,
                       scoring="neg_mean_absolute_error").mean()) if len(X_scaled) >= 5 else None
    top_features = dict(sorted(
        dict(zip(FEATURE_COLS, rf.feature_importances_)).items(),
        key=lambda x: x[1], reverse=True)[:10])

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                    max_depth=4, subsample=0.8, random_state=42)
    gb.fit(X_scaled, y)
    gb_pred    = float(gb.predict(x_pred)[0])
    gb_cv_mae  = float(-cross_val_score(gb, X_scaled, y, cv=cv_k,
                       scoring="neg_mean_absolute_error").mean()) if len(X_scaled) >= 5 else None

    # Ensemble
    preds    = np.array([ridge_pred, rf_pred, gb_pred])
    ensemble = float(np.dot([0.25, 0.35, 0.40], preds))
    pred_std = float(np.std(preds))
    agreement_score = max(0.0, 1.0 - pred_std / max(abs(float(np.mean(preds))), 1)) * 100

    # ── CONFIDENCE: walk-forward validated ONLY ──
    if bt_stats["ready"]:
        # Decay confidence with mape
        conf      = bt_stats["conf_from_bt"]
        conf_src  = f"walk-forward backtest ({bt_stats['n']} real observations, MAPE={bt_stats['mape']}%)"
    else:
        conf      = None   # Do NOT invent a number
        conf_src  = bt_stats["message"]

    # Prediction interval
    vol        = current_feat.get("volatility", cbn_rate * 0.003) or cbn_rate * 0.003
    half_range = max(pred_std * 2, vol * 1.5, cbn_rate * 0.003)
    pred_low   = round(ensemble - half_range, 2)
    pred_high  = round(ensemble + half_range, 2)

    direction = ("BULLISH" if ensemble > cbn_rate * 1.0005
                 else "BEARISH" if ensemble < cbn_rate * 0.9995
                 else "NEUTRAL")

    y_hat = [float(gb.predict(X_scaled[i:i+1])[0]) for i in range(len(X_scaled))]
    r2    = float(r2_score(y, y_hat)) if len(y) > 1 else None

    st.session_state.ml_metrics = {
        "n_training_points": len(X_scaled),
        "ridge_cv_mae": ridge_cv_mae, "rf_cv_mae": rf_cv_mae, "gb_cv_mae": gb_cv_mae,
        "r2_in_sample": r2, "pred_std": pred_std, "agreement_score": agreement_score,
        "rf_feature_importance": top_features,
    }

    return {
        "cold_start": False,
        "n_training_points": len(X_scaled),
        "ridge_pred": round(ridge_pred, 2),
        "rf_pred":    round(rf_pred, 2),
        "gb_pred":    round(gb_pred, 2),
        "ensemble":   round(ensemble, 2),
        "pred_low":   pred_low, "pred_high": pred_high,
        "confidence":        conf,
        "confidence_source": conf_src,
        "direction":         direction,
        "model_agreement":   round(agreement_score, 1),
        "ridge_cv_mae": ridge_cv_mae, "rf_cv_mae": rf_cv_mae, "gb_cv_mae": gb_cv_mae,
        "r2_in_sample": r2, "rf_feature_importance": top_features,
        "backtest": bt_stats,
        "note": f"Ensemble trained on {len(X_scaled)} CBN-rate observations.",
    }


# ══════════════════════════════════════════════════════
# MULTI-TIMEFRAME FORECASTING ENGINE
# ══════════════════════════════════════════════════════
TIMEFRAMES = [
    {"label":"24H",    "hours":24,    "key":"24h"},
    {"label":"7 DAY",  "hours":168,   "key":"7d"},
    {"label":"30 DAY", "hours":720,   "key":"30d"},
    {"label":"3 MO",   "hours":2160,  "key":"3m"},
    {"label":"6 MO",   "hours":4320,  "key":"6m"},
    {"label":"12 MO",  "hours":8760,  "key":"12m"},
    {"label":"2 YR",   "hours":17520, "key":"2yr"},
]

def build_multi_timeframe_forecast(cbn_rate: float, ml: dict, feat: dict) -> dict:
    ensemble   = ml.get("ensemble", cbn_rate)
    vol        = feat.get("volatility", cbn_rate*0.005) or cbn_rate*0.005
    news_score = feat.get("news_overall", 0)
    cbn_score  = feat.get("news_cbn", 0) or 0
    oil_impact = feat.get("news_oil", 0) or 0
    em_risk    = feat.get("news_em_risk", 0) or 0
    trend_slope= feat.get("trend_slope", 0)
    step_24h   = ensemble - cbn_rate

    annual_depr = 0.08
    if abs(news_score) > 30:
        annual_depr += news_score * 0.001
    if cbn_score > 20:
        annual_depr -= 0.02
    annual_depr = max(0.02, min(0.25, annual_depr))

    bt_stats = ml.get("backtest", {})
    bt_ready = bt_stats.get("ready", False)
    bt_conf  = bt_stats.get("conf_from_bt")  # None if not ready

    forecasts = {}
    for tf in TIMEFRAMES:
        hours = tf["hours"]
        years = hours / 8760.0
        days  = hours / 24.0

        if hours <= 24:
            central = ensemble
        elif hours <= 168:
            central = cbn_rate + step_24h + trend_slope * days * 4
        elif hours <= 720:
            central = cbn_rate * (1 + annual_depr) ** years
            central += (news_score / 100) * cbn_rate * 0.015
        else:
            central  = cbn_rate * (1 + annual_depr) ** years
            central += (oil_impact / 100) * cbn_rate * 0.02 * (1 / (1 + years))
            central += (em_risk / 100)    * cbn_rate * 0.03 * min(years, 1)

        base_vol   = max(vol, cbn_rate * 0.003)
        half_range = (base_vol * np.sqrt(days) * 0.4
                      + cbn_rate * years * 0.04
                      + cbn_rate * 0.005 * np.sqrt(days))
        low     = round(central - half_range, 0)
        high    = round(central + half_range, 0)
        central = round(central, 0)

        # Confidence ONLY from walk-forward backtest, decayed by horizon
        if bt_ready and bt_conf is not None:
            decayed = bt_conf * (1 / (1 + np.sqrt(years) * 0.8))
            conf_num = int(max(15, min(bt_conf, round(decayed))))
            conf_str = f"{conf_num}% (walk-forward validated)"
        else:
            conf_num = None
            conf_str = "— (awaiting backtest data)"

        pct_change    = round((central - cbn_rate) / cbn_rate * 100, 1)
        direction_lbl = ("▲ HIGHER" if central > cbn_rate*1.002
                         else "▼ LOWER" if central < cbn_rate*0.998
                         else "◆ STABLE")

        forecasts[tf["key"]] = {
            "label": tf["label"], "hours": hours,
            "central": central, "low": low, "high": high,
            "bull_case": round(low * 0.95, 0),
            "bear_case": round(high * 1.12, 0),
            "pct_change": pct_change,
            "direction":  direction_lbl,
            "confidence": conf_num,
            "confidence_str": conf_str,
        }
    return forecasts


def build_forecast_narratives(cbn_rate: float, forecasts: dict,
                               ml: dict, feat: dict, raw: dict) -> dict:
    q_intel = raw.get("news_intel", {})
    sig     = st.session_state.global_signals or {}
    anal    = sig.get("analysis", {})
    bt      = ml.get("backtest", {})
    bt_note = (
        f"Walk-forward backtest: {bt.get('n',0)} obs · MAE=₦{bt.get('mae','N/A')} · "
        f"MAPE={bt.get('mape','N/A')}% · Direction accuracy={bt.get('dir_accuracy','N/A')}%."
        if bt.get("ready") else
        f"Backtest not yet ready: {bt.get('message','run analysis on multiple days.')}"
    )

    tf_lines = "\n".join(
        f"- {forecasts.get(k,{}).get('label','')}: ₦{forecasts.get(k,{}).get('central',0):,.0f} "
        f"({forecasts.get(k,{}).get('pct_change',0):+.1f}%) Conf: {forecasts.get(k,{}).get('confidence_str','—')}"
        for k in ["24h","7d","30d","3m","6m","12m","2yr"]
    )

    prompt = f"""You are Nigeria's premier FX strategist. Focus on the USD/NGN OFFICIAL CBN rate.

CURRENT DATA:
- CBN Official Rate: ₦{cbn_rate:,.2f}
- ML Ensemble (24h): ₦{ml.get('ensemble',0):,.0f}
- Training Points: {ml.get('n_training_points',0)}
- Backtest Status: {bt_note}

QUALITATIVE SCORES (+ = NGN weakens):
- Overall: {feat.get('news_overall',0):+.0f} | CBN Policy: {feat.get('news_cbn',0):+.0f}
- Oil: {feat.get('news_oil',0):+.0f} | USD/Fed: {feat.get('news_usd',0):+.0f}
- EM Risk: {feat.get('news_em_risk',0):+.0f} | Geopolitics: {feat.get('news_geopolitics',0):+.0f}
CBN Analysis: {q_intel.get('cbn_analysis','N/A')}
Oil Analysis: {q_intel.get('oil_analysis','N/A')}
Medium-term: {anal.get('medium_term_outlook','N/A')}
Long-term:   {anal.get('long_term_outlook','N/A')}

STATISTICAL FORECASTS:
{tf_lines}

Return ONLY valid JSON (no markdown):
{{
  "exec_summary": "<4-5 sentences: CBN rate context, ML signal, key drivers>",
  "trade_recommendation": "<Specific actionable recommendation for USD/NGN conversion>",
  "best_convert_time": "<Best timing window based on signals>",
  "n24h_narrative": "<2 sentences: 24h CBN rate outlook>",
  "n24h_drivers": ["<driver 1>", "<driver 2>"],
  "n24h_risk": "<Main risk to 24h forecast>",
  "n7d_narrative": "<2 sentences: 7-day outlook>",
  "n7d_drivers": ["<driver 1>", "<driver 2>"],
  "n7d_risk": "<Main risk>",
  "n30d_narrative": "<2 sentences: 30-day macro outlook>",
  "n30d_bull_scenario": "<What causes NGN to strengthen>",
  "n30d_bear_scenario": "<What causes NGN to weaken further>",
  "n3m_narrative": "<2 sentences>",
  "n6m_narrative": "<2 sentences>",
  "n12m_narrative": "<2 sentences: CBN structural trajectory>",
  "n2yr_narrative": "<2 sentences: Nigeria reform trajectory>",
  "key_risks": ["<risk 1>","<risk 2>","<risk 3>","<risk 4>"],
  "key_upside_catalysts": ["<catalyst 1>","<catalyst 2>","<catalyst 3>"],
  "cbn_watch": "<Key CBN signals to monitor>",
  "oil_impact_summary": "<How oil feeds through to CBN rate>",
  "data_quality_note": "<Honest assessment of model data quality and backtest status>",
  "disclaimer_note": "<Risk disclaimer — note confidence is only shown when walk-forward validated>"
}}"""

    try:
        raw_out = gemini(prompt, "Nigeria FX strategist. Return only valid JSON.")
        return _parse_json(raw_out)
    except:
        return {
            "exec_summary": f"CBN rate at ₦{cbn_rate:,.0f}. ML ensemble on {ml.get('n_training_points',0)} observations.",
            "trade_recommendation": "Confidence metrics not yet walk-forward validated. Exercise caution.",
            "best_convert_time": "Monitor next 24-48 hours for direction confirmation.",
            "n24h_narrative": f"ML projects ₦{ml.get('ensemble',0):,.0f} for next observation.",
            "n7d_narrative": "7-day path driven by CBN policy and global USD dynamics.",
            "n30d_narrative": "30-day view shaped by oil, CBN reserves, and EM risk appetite.",
            "n3m_narrative": "3-month view depends on Fed rate trajectory and Nigeria fiscal position.",
            "n6m_narrative": "6-month outlook reflects structural NGN depreciation pressures.",
            "n12m_narrative": "12-month view tied to CBN reform execution and oil revenue.",
            "n2yr_narrative": "2-year trajectory hinges on Nigeria economic reform credibility.",
            "key_risks": ["CBN policy surprise","Oil price shock","USD strength","Nigeria political risk"],
            "key_upside_catalysts": ["Oil price surge","CBN rate hike","Strong remittances"],
            "cbn_watch": "Watch CBN MPC meetings, intervention volumes, and reserve levels.",
            "oil_impact_summary": "Oil revenue underpins Nigeria FX earnings and NGN support.",
            "data_quality_note": bt_note,
            "disclaimer_note": "Confidence % not shown until walk-forward backtest has ≥2 validated predictions."
        }


# ══════════════════════════════════════════════════════
# MAIN ANALYSIS ORCHESTRATOR
# ══════════════════════════════════════════════════════
def run_full_analysis() -> dict:
    raw, feat    = collect_features()
    cbn_rate     = raw.get("cbn_rate", 1580.0)
    raw["cbn_rate"] = cbn_rate

    # Step 1: Resolve previous prediction against today's actual CBN rate
    resolve_pending_prediction(cbn_rate)

    # Step 2: Store this observation in history
    st.session_state.rate_history.append({
        "timestamp":   raw.get("timestamp"),
        "cbn_rate":    cbn_rate,
        "p2p_mid":     raw.get("p2p_mid"),   # retained as supplementary signal
        "rate_source": raw.get("source", ""),
        "rate_status": raw.get("status", ""),
        "features":    feat,
    })
    _save_history()

    # Step 3: Train ML and predict
    ml        = train_and_predict(feat, cbn_rate)
    forecasts = build_multi_timeframe_forecast(cbn_rate, ml, feat)
    narratives= build_forecast_narratives(cbn_rate, forecasts, ml, feat, raw)

    # Step 4: Store this prediction as pending — resolved on the NEXT run
    st.session_state.pending_pred = {
        "ensemble":  ml.get("ensemble"),
        "timestamp": raw.get("timestamp"),
        "prev_rate": cbn_rate,
    }

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
    ml         = result.get("ml", {})
    raw        = result.get("raw", {})
    feat       = result.get("features", {})
    forecasts  = result.get("forecasts", {})
    narratives = result.get("narratives", {})
    bt         = ml.get("backtest", {})

    ctx = f"""USD/NGN ORACLE — CBN OFFICIAL RATE CONTEXT
CBN Rate: ₦{raw.get('cbn_rate',0):,.2f} | Source: {raw.get('source','')}
P2P (supplementary signal): ₦{raw.get('p2p_mid','N/A')}
ML Ensemble (24h): ₦{ml.get('ensemble',0):,.2f} | Direction: {ml.get('direction','N/A')}
Confidence: {ml.get('confidence','Not yet validated')} — {ml.get('confidence_source','')}
Training Points: {ml.get('n_training_points',0)} | Cold Start: {ml.get('cold_start',True)}

WALK-FORWARD BACKTEST:
- Observations: {bt.get('n',0)}
- Real MAE: ₦{bt.get('mae','N/A')}
- Real MAPE: {bt.get('mape','N/A')}%
- Direction Accuracy: {bt.get('dir_accuracy','N/A')}%
- Status: {bt.get('message','')}

FORECASTS:
{json.dumps({k:{
    "central":v.get("central"),"low":v.get("low"),"high":v.get("high"),
    "pct_change":v.get("pct_change"),"confidence":v.get("confidence_str")
} for k,v in forecasts.items()}, indent=2)}

EXECUTIVE SUMMARY: {narratives.get('exec_summary','N/A')}
TRADE REC: {narratives.get('trade_recommendation','N/A')}
KEY RISKS: {narratives.get('key_risks',[])}

QUALITATIVE (+ = NGN weakens):
Overall:{feat.get('news_overall',0):+.0f} | Nigeria:{feat.get('news_nigeria',0):+.0f} | CBN:{feat.get('news_cbn',0):+.0f}
Oil:{feat.get('news_oil',0):+.0f} | USD/Fed:{feat.get('news_usd',0):+.0f}
"""
    hist = "".join(f"\n{'User' if m['r']=='u' else 'Oracle'}: {m['c']}"
                   for m in st.session_state.chat[-6:])
    system = """You are the USD/NGN Oracle — Nigeria's AI FX analyst focused on the CBN OFFICIAL rate.
Always be honest about uncertainty. Distinguish between validated confidence (walk-forward backtest)
and unvalidated model output. Never invent confidence numbers. Be direct and data-driven."""
    return gemini(f"{ctx}\n\nConversation:{hist}\n\nUser: {msg}\n\nOracle:", system, max_tokens=1500)


# ══════════════════════════════════════════════════════
# EMAIL ALERTS
# ══════════════════════════════════════════════════════
def send_email_alert(to_email, subject, html_body, key) -> bool:
    if not to_email or not key: return False
    try:
        r = requests.post("https://api.resend.com/emails",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"from":"USD/NGN Oracle <onboarding@resend.dev>",
                  "to":[to_email],"subject":subject,"html":html_body},timeout=15)
        return r.status_code == 200
    except: return False

def check_and_trigger_alerts(cbn_rate, ml, narratives):
    triggered = []
    user_email = st.session_state.get("user_email","")
    try: rk = st.secrets.get("RESEND_API_KEY","")
    except: rk = ""
    for i, a in enumerate(st.session_state.alerts):
        msg = ""
        if a["type"]=="above" and cbn_rate >= a["level"] and i not in st.session_state.alert_triggered:
            msg = f"CBN rate crossed ABOVE ₦{a['level']:,} — now ₦{cbn_rate:,.0f}"
            triggered.append((i, f"🔔 {msg}"))
            st.session_state.alert_triggered.append(i)
        elif a["type"]=="below" and cbn_rate <= a["level"] and i not in st.session_state.alert_triggered:
            msg = f"CBN rate dropped BELOW ₦{a['level']:,} — now ₦{cbn_rate:,.0f}"
            triggered.append((i, f"🔔 {msg}"))
            st.session_state.alert_triggered.append(i)
        if msg and user_email and rk:
            conf_disp = f"{ml.get('confidence')}% validated" if ml.get('confidence') else "Not yet validated"
            html = f"""<div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;
            background:#060912;color:#cfe0f5;border-radius:16px;border:1px solid #172440;padding:24px;">
            <div style="font-size:10px;letter-spacing:3px;color:#b060ff;text-transform:uppercase;margin-bottom:8px;">
            🇳🇬 USD/NGN Oracle · CBN Official Rate</div>
            <h2 style="font-size:20px;font-weight:700;margin-bottom:16px;">Price Alert Triggered</h2>
            <div style="background:#0f1d30;border-left:4px solid #ffb020;border-radius:8px;
            padding:12px 16px;margin-bottom:16px;font-size:14px;color:#ffb020;">🔔 {msg}</div>
            <div style="font-size:12px;color:#9ab0cc;margin-bottom:8px;">Direction: {ml.get('direction','N/A')}</div>
            <div style="font-size:12px;color:#9ab0cc;margin-bottom:8px;">ML 24h Target: ₦{ml.get('ensemble',0):,.0f}</div>
            <div style="font-size:12px;color:#9ab0cc;margin-bottom:16px;">Confidence: {conf_disp}</div>
            <div style="font-size:10px;color:#4a6080;">⚠️ Not financial advice. Always DYOR.</div>
            </div>"""
            send_email_alert(user_email, f"🔔 USD/NGN Oracle: {msg}", html, rk)
    return triggered


# ══════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════
def prog_bar(label, val, color, min_val=-100, max_val=100):
    norm = max(0, min(100, (val-min_val)/(max_val-min_val)*100))
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

def tf_accent(key):
    return {"24h":"#00e5a0","7d":"#4488ff","30d":"#b060ff","3m":"#ffb020",
            "6m":"#ff4466","12m":"#00d4ff","2yr":"#ffd700"}.get(key,"#4488ff")

def metric_card(col, accent, label, value, sub, val_color=None):
    vc = val_color or "var(--text)"
    col.markdown(f"""<div class="card card-{accent}">
    <div class="card-label">{label}</div>
    <div class="card-value" style="color:{vc};font-size:22px;">{value}</div>
    <div class="card-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# ── UI: ACTION BAR + HEADER ──
# ══════════════════════════════════════════════════════
ab1, ab2, ab3, ab4, ab5 = st.columns([2.5, 1.2, 1.2, 1.2, 5])
with ab1:
    run_btn = st.button("⚡ Run Full Analysis", use_container_width=True, type="primary")
with ab2:
    auto_ref = st.toggle("Auto-refresh", value=st.session_state.auto_refresh, key="ar_tog")
    st.session_state.auto_refresh = auto_ref
with ab3:
    if auto_ref:
        iv = st.selectbox("Interval",[15,30,60,120],index=2,
                          format_func=lambda x:f"{x}m",label_visibility="collapsed",key="ar_iv")
        st.session_state.refresh_interval = iv
with ab5:
    if st.session_state.last_time:
        el  = int((datetime.datetime.now()-st.session_state.last_time).total_seconds()//60)
        pts = len(st.session_state.rate_history)
        bt_n= len(st.session_state.backtest_log)
        st.markdown(
            f'<p style="font-family:var(--font-mono);font-size:10px;color:var(--text2);'
            f'margin:10px 0 0;text-align:right;">'
            f'<span class="live-dot"></span>Last run {el}m ago &nbsp;·&nbsp; {pts} training pts'
            f' &nbsp;·&nbsp; {bt_n} backtest obs</p>',
            unsafe_allow_html=True)

st.markdown("""
<div style="padding:10px 0 20px;">
  <div style="font-family:var(--font-mono);font-size:9px;color:var(--cyan);
  letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;">
    <span class="live-dot"></span>CBN OFFICIAL RATE · ML + GEMINI AI · WALK-FORWARD VALIDATED
  </div>
  <h1 style="font-family:var(--font-mono);font-size:28px;font-weight:700;margin:0 0 6px;
  background:linear-gradient(135deg,#cfe0f5 0%,#4488ff 50%,#b060ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;">
    USD / NGN Oracle
  </h1>
  <p style="color:var(--text2);font-size:12px;margin:0;font-family:var(--font-mono);">
    CBN Official Rate · Ridge · Random Forest · Gradient Boosting · Walk-Forward Validated Confidence
  </p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="alert-box alert-info" style="margin-bottom:18px;font-size:12px;">
  <strong style="color:var(--blue);">🏦 v4.0 — What changed:</strong>
  This Oracle now tracks the <strong>CBN official USD/NGN rate</strong> (not P2P / parallel market).
  P2P data is retained only as a secondary spread-signal feature.
  <strong>Confidence % is only displayed after walk-forward validation</strong> — each prediction is stored
  and compared against the next real CBN rate before any confidence number is shown.
  In-sample cross-validation is shown for diagnostics only and never used as a headline confidence figure.
</div>""", unsafe_allow_html=True)

# ── RUN + AUTO-REFRESH ──
if run_btn:
    with st.spinner("Fetching CBN rate · Resolving backtest · Training ML · Generating forecasts..."):
        result = run_full_analysis()
        st.session_state.result    = result
        st.session_state.last_time = datetime.datetime.now()
    st.rerun()

if auto_ref and st.session_state.last_time and GEMINI_KEY:
    elapsed_sec  = (datetime.datetime.now()-st.session_state.last_time).total_seconds()
    interval_sec = st.session_state.refresh_interval * 60
    if elapsed_sec >= interval_sec:
        with st.spinner("Auto-refreshing Oracle..."):
            result = run_full_analysis()
            st.session_state.result    = result
            st.session_state.last_time = datetime.datetime.now()
        st.rerun()
    else:
        rem_sec = int(interval_sec - elapsed_sec)
        st.markdown(
            f'<p style="font-size:10px;color:var(--green);text-align:right;margin-bottom:0;">'
            f'🔄 Auto-refresh in {rem_sec//60}m {rem_sec%60}s</p>',
            unsafe_allow_html=True)
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=10_000, limit=None, key="oracle_autorefresh")


# ══════════════════════════════════════════════════════
# ── EMPTY STATE ──
# ══════════════════════════════════════════════════════
if not st.session_state.result:
    pts    = len(st.session_state.rate_history)
    bt_n   = len(st.session_state.backtest_log)
    needed = max(0, 5-pts)
    persisted = (
        f'<div class="alert-box alert-bull" style="max-width:520px;margin:0 auto 12px;">'
        f'📂 Loaded <strong>{pts}</strong> CBN-rate observations · '
        f'<strong>{bt_n}</strong> walk-forward backtest pairs. '
        f'{"✅ ML ready!" if pts >= 5 else f"Need {needed} more run(s) for ML."}'
        f'</div>'
    ) if pts > 0 else ""
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px 40px;">
      <div style="font-size:50px;margin-bottom:20px;">🏦</div>
      <h2 style="font-family:var(--font-mono);font-size:26px;font-weight:700;
      background:linear-gradient(135deg,#cfe0f5,#4488ff,#b060ff);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;margin-bottom:14px;">Oracle Ready</h2>
      <p style="color:var(--text2);max-width:600px;margin:0 auto 20px;line-height:1.8;font-size:14px;">
        Tracks the <strong style="color:var(--amber);">CBN Official USD/NGN Rate</strong>.
        Confidence % is only shown after <strong style="color:var(--cyan);">walk-forward validation</strong> —
        predictions are stored and compared against subsequent real rates before any number is displayed.
      </p>
      {persisted}
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;max-width:780px;margin:24px auto;">
        <div class="card card-green" style="padding:18px 14px;">
          <div style="font-size:26px;margin-bottom:8px;">🏦</div>
          <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;">CBN Official Rate</div>
          <div style="font-size:10px;color:var(--muted);">ExchangeRate-API · Frankfurter</div>
        </div>
        <div class="card card-blue" style="padding:18px 14px;">
          <div style="font-size:26px;margin-bottom:8px;">🌍</div>
          <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;">40+ Headlines</div>
          <div style="font-size:10px;color:var(--muted);">18 RSS feeds · Gemini scoring</div>
        </div>
        <div class="card card-purple" style="padding:18px 14px;">
          <div style="font-size:26px;margin-bottom:8px;">🤖</div>
          <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;">ML Ensemble</div>
          <div style="font-size:10px;color:var(--muted);">Ridge · RF · GradBoost</div>
        </div>
        <div class="card card-amber" style="padding:18px 14px;">
          <div style="font-size:26px;margin-bottom:8px;">🔬</div>
          <div style="font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;">Walk-Forward BT</div>
          <div style="font-size:10px;color:var(--muted);">Real out-of-sample MAE</div>
        </div>
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

    cbn_rate  = raw.get("cbn_rate", 0)
    p2p_mid   = raw.get("p2p_mid")
    ensemble  = ml.get("ensemble", 0)
    direction = ml.get("direction", "NEUTRAL")
    conf      = ml.get("confidence")        # None until walk-forward validated
    pred_low  = ml.get("pred_low", 0)
    pred_high = ml.get("pred_high", 0)
    n_pts     = ml.get("n_training_points", 0)
    cold      = ml.get("cold_start", True)
    bt_stats  = ml.get("backtest", {})
    prem      = feat.get("p2p_premium_pct", 0) or 0

    _sig  = st.session_state.global_signals or {}
    _anal = _sig.get("analysis", {})
    _bias = _anal.get("30min_bias", "—")
    _fng  = _sig.get("fear_greed_value", "—")
    _fng_l= _sig.get("fear_greed_label", "N/A")

    dc       = "var(--green)" if direction=="BULLISH" else "var(--red)" if direction=="BEARISH" else "var(--amber)"
    da       = "▲" if direction=="BULLISH" else "▼" if direction=="BEARISH" else "◆"
    bias_col = "var(--green)" if _bias=="BUY" else "var(--red)" if _bias=="SELL" else "var(--amber)"
    prem_col = "var(--red)" if prem>8 else "var(--amber)" if prem>4 else "var(--green)"

    # ── LIVE TICKER ──
    btc_u = _sig.get("btc_usd"); btc_c = _sig.get("btc_24h") or 0
    eth_u = _sig.get("eth_usd"); eth_c = _sig.get("eth_24h") or 0
    eur   = _sig.get("eurusd");  dxy   = _sig.get("dxy_proxy")
    zar   = _sig.get("usd_zar");

    def ticker_item(label, val, change=None, fmt=""):
        if val is None: return ""
        v_str  = f"{fmt}{val:,.2f}" if isinstance(val, float) else f"{val}"
        ch_str = ""
        if change is not None:
            cls    = "up" if float(change) >= 0 else "dn"
            ch_str = f' <span class="{cls}">{float(change):+.2f}%</span>'
        return (f'<span class="ticker-item">{label} <span class="val">{v_str}</span>{ch_str}</span>'
                f'<span class="ticker-sep">·</span>')

    ti = "".join([
        ticker_item("CBN USD/NGN", cbn_rate, None, "₦"),
        ticker_item("P2P (signal)", p2p_mid, None, "₦") if p2p_mid else "",
        ticker_item("BTC", btc_u, btc_c, "$"),
        ticker_item("ETH", eth_u, eth_c, "$"),
        ticker_item("EUR/USD", eur, None),
        ticker_item("DXY Proxy", dxy, None),
        ticker_item("USD/ZAR", zar, None),
        f'<span class="ticker-item">F&G <span class="val">{_fng} — {_fng_l}</span></span><span class="ticker-sep">·</span>',
        f'<span class="ticker-item">30m Bias <span style="color:{bias_col};font-weight:700;">⚡{_bias}</span></span>',
    ])
    st.markdown(f'<div class="ticker-wrap"><div class="ticker-inner">{ti}{ti}</div></div>',
                unsafe_allow_html=True)

    # ── TOP METRIC CARDS (7) ──
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)

    with c1:
        src_stat = raw.get("status","")
        sc = "var(--green)" if src_stat=="live" else "var(--amber)" if src_stat=="proxy" else "var(--red)"
        metric_card(c1,"green","CBN Official Rate",f"₦{cbn_rate:,.2f}",
                    f"{raw.get('source','')[:30]}", sc)
    with c2:
        p2p_disp = f"₦{p2p_mid:,.0f}" if p2p_mid else "N/A"
        metric_card(c2,"amber","P2P Premium (signal)",f"{prem:+.2f}%",
                    f"P2P ≈ {p2p_disp}", prem_col)
    with c3:
        metric_card(c3,
            "green" if direction=="BULLISH" else "red" if direction=="BEARISH" else "amber",
            "ML Prediction 24H", f"{da} ₦{ensemble:,.0f}",
            f"₦{pred_low:,.0f} – ₦{pred_high:,.0f}", dc)
    with c4:
        # Walk-forward confidence — ONLY shown when validated
        if conf is not None:
            cc = "var(--green)" if conf>=65 else "var(--amber)" if conf>=45 else "var(--red)"
            metric_card(c4,"purple","Validated Confidence",f"{conf}%",
                        f"✅ {bt_stats.get('n',0)} real obs · MAE ₦{bt_stats.get('mae','—')}", cc)
        else:
            metric_card(c4,"purple","Confidence",
                        "—",
                        f"⏳ {bt_stats.get('message','Run on 2+ separate days')[:45]}",
                        "var(--muted)")
    with c5:
        mae_disp  = f"₦{bt_stats.get('mae',0):,.2f}" if bt_stats.get("mae") else "—"
        mape_disp = f"{bt_stats.get('mape',0):.3f}%" if bt_stats.get("mape") else "not ready"
        metric_card(c5,"blue","Real Out-of-Sample MAE",mae_disp,
                    f"MAPE: {mape_disp} · {bt_stats.get('n',0)} obs","var(--blue)")
    with c6:
        dir_acc  = bt_stats.get("dir_accuracy")
        da_disp  = f"{dir_acc}%" if dir_acc is not None else "—"
        metric_card(c6,
            "green" if _bias=="BUY" else "red" if _bias=="SELL" else "amber",
            "Direction Accuracy", da_disp,
            f"30m bias: ⚡{_bias}", bias_col)
    with c7:
        f7d = forecasts.get("7d",{})
        c7c = "var(--green)" if f7d.get("central",0) > cbn_rate else "var(--red)" if f7d.get("central",0) < cbn_rate else "var(--amber)"
        metric_card(c7,"cyan","7-Day Forecast",
                    f"₦{f7d.get('central',0):,.0f}",
                    f"{f7d.get('pct_change',0):+.1f}% from now", c7c)

    # ── STATUS NOTICES ──
    if cold:
        st.markdown(f'<div class="alert-box alert-warn" style="margin-top:12px;">⚠️ <strong>Cold Start</strong> — {ml.get("note","")} Run 5+ times to train ML.</div>',
                    unsafe_allow_html=True)
    if not bt_stats.get("ready"):
        st.markdown(
            f'<div class="alert-box alert-info" style="margin-top:8px;font-size:12px;">'
            f'🔬 <strong>Confidence not yet shown</strong> — {bt_stats.get("message","")} '
            f'Each run stores your prediction; the <em>next</em> run resolves it against the real CBN rate.</div>',
            unsafe_allow_html=True)

    triggered_alerts = check_and_trigger_alerts(cbn_rate, ml, narratives)
    for _, msg in triggered_alerts:
        st.markdown(f'<div class="alert-box alert-warn">{msg}</div>', unsafe_allow_html=True)

    ts = raw.get("timestamp","")[:19].replace("T"," ")
    st.markdown(
        f'<p style="font-size:10px;font-family:var(--font-mono);color:var(--green);margin-bottom:0;">'
        f'<span class="live-dot"></span>CBN Rate: ₦{cbn_rate:,.2f} &nbsp;·&nbsp; '
        f'Source: {raw.get("source","")} &nbsp;·&nbsp; '
        f'<span style="color:var(--muted);">Updated {ts}</span></p>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
        "📊 Analysis", "📈 Forecasts", "🔬 Backtest",
        "🌍 Global Signals", "💱 Converter",
        "📉 History", "💬 Chat", "🔔 Alerts", "🩺 Diagnostics",
    ])


    # ══════ TAB 1: ANALYSIS ══════
    with tab1:
        left, right = st.columns([3,2])
        with left:
            exec_sum = narratives.get("exec_summary","")
            if exec_sum:
                st.markdown(f"""<div class="card card-purple" style="margin-bottom:16px;">
                <div class="sec-header">🧠 EXECUTIVE SUMMARY</div>
                <p style="font-size:13px;color:var(--text);line-height:1.75;margin:0;">{exec_sum}</p>
                </div>""", unsafe_allow_html=True)

            trade_rec = narratives.get("trade_recommendation","")
            best_time = narratives.get("best_convert_time","")
            if trade_rec:
                db = f'<span class="badge badge-{"bull" if direction=="BULLISH" else "bear" if direction=="BEARISH" else "neu"}">{da} {direction}</span>'
                st.markdown(
                    f'<div class="card card-{"green" if direction=="BULLISH" else "red" if direction=="BEARISH" else "amber"}" style="margin-bottom:16px;">'
                    f'<div class="sec-header">⚡ TRADE RECOMMENDATION &nbsp;&nbsp;{db}</div>'
                    f'<p style="font-size:13px;color:var(--text);line-height:1.7;margin:0 0 10px;">{trade_rec}</p>'
                    + (f'<div style="font-size:11px;color:var(--text2);border-top:1px solid var(--border);padding-top:8px;">⏰ Best time: {best_time}</div>' if best_time else "")
                    + '</div>', unsafe_allow_html=True)

            # Individual model predictions
            st.markdown('<div class="card" style="margin-bottom:16px;"><div class="sec-header">🤖 INDIVIDUAL MODEL PREDICTIONS (vs CBN Official)</div>',
                        unsafe_allow_html=True)
            for badge_cls,lbl,pred,desc in [
                ("badge-ridge","Ridge Regression",  ml.get("ridge_pred",0),"Regularised linear — captures trend direction."),
                ("badge-rf",   "Random Forest",       ml.get("rf_pred",0),  "Decision tree ensemble — non-linear patterns."),
                ("badge-gb",   "Gradient Boosting",   ml.get("gb_pred",0),  "Sequential error-correction."),
                ("badge-ens",  "Weighted Ensemble",   ml.get("ensemble",0), "Ridge×0.25 + RF×0.35 + GB×0.40"),
            ]:
                diff  = pred - cbn_rate
                color = "var(--green)" if diff >= 0 else "var(--red)"
                arr   = "▲" if diff >= 0 else "▼"
                pct   = diff / max(cbn_rate, 1) * 100
                st.markdown(f"""
                <div style="padding:12px 0;border-bottom:1px solid var(--border);">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
                    <span class="model-badge {badge_cls}">{lbl}</span>
                    <div style="display:flex;align-items:center;gap:12px;">
                      <span style="font-family:var(--font-mono);font-size:17px;font-weight:700;color:{color};">₦{pred:,.0f}</span>
                      <span style="font-family:var(--font-mono);font-size:11px;color:{color};">{arr} {diff:+.0f} ({pct:+.2f}%)</span>
                    </div>
                  </div>
                  <p style="font-size:11px;color:var(--muted);margin:0;">{desc}</p>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            # Confidence display — validated vs unvalidated
            if conf is not None:
                cc = "var(--green)" if conf>=65 else "var(--amber)" if conf>=45 else "var(--red)"
                st.markdown(f"""<div class="card" style="margin-bottom:16px;text-align:center;padding:24px 22px;">
                <div class="sec-header" style="text-align:left;">📊 VALIDATED CONFIDENCE</div>
                <div style="display:flex;align-items:center;justify-content:center;gap:24px;margin:16px 0;">
                  <div style="border:6px solid {cc};border-radius:50%;width:100px;height:100px;
                  display:flex;flex-direction:column;align-items:center;justify-content:center;">
                    <div style="font-family:var(--font-mono);font-size:24px;font-weight:700;color:{cc};">{conf}%</div>
                    <div style="font-size:9px;color:var(--muted);letter-spacing:1px;">REAL</div>
                  </div>
                  <div style="text-align:left;">
                    <div style="font-size:10px;color:var(--text2);">From walk-forward backtest</div>
                    <div style="font-family:var(--font-mono);font-size:14px;color:var(--amber);margin-top:4px;">
                    MAE ₦{bt_stats.get('mae','—')} ({bt_stats.get('mape','—')}%)</div>
                    <div style="font-size:10px;color:var(--muted);margin-top:6px;">Direction Accuracy</div>
                    <div style="font-family:var(--font-mono);font-size:18px;color:var(--green);">{bt_stats.get('dir_accuracy','—')}%</div>
                  </div>
                </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="card card-amber" style="margin-bottom:16px;padding:24px 22px;">
                <div class="sec-header">⏳ CONFIDENCE NOT YET VALIDATED</div>
                <p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0;">
                  {bt_stats.get('message','')}
                </p>
                <p style="font-size:11px;color:var(--muted);margin-top:10px;line-height:1.5;">
                  Each run stores the current ML prediction. The <em>next</em> run compares it to
                  the real CBN rate and adds to the backtest log. Confidence is only shown once real
                  prediction error has been measured.
                </p>
                </div>""", unsafe_allow_html=True)

            # Rate summary table
            st.markdown('<div class="card" style="margin-bottom:16px;"><div class="sec-header">💹 RATE SUMMARY</div>'
                        '<table class="spread-table"><tr><th>Metric</th><th>Value</th></tr>',
                        unsafe_allow_html=True)
            for lbl, val, clr in [
                ("CBN Official",        f"₦{cbn_rate:,.2f}",              "var(--green)"),
                ("P2P (signal only)",   f"₦{p2p_mid:,.0f}" if p2p_mid else "N/A","var(--text2)"),
                ("P2P Premium",         f"{prem:+.2f}%",                   "var(--amber)"),
                ("Ridge Target",        f"₦{ml.get('ridge_pred',0):,.0f}", "var(--blue)"),
                ("RF Target",           f"₦{ml.get('rf_pred',0):,.0f}",    "var(--green)"),
                ("GB Target",           f"₦{ml.get('gb_pred',0):,.0f}",    "var(--purple)"),
                ("Ensemble (24H)",      f"₦{ensemble:,.0f}",               "var(--amber)"),
                ("Range Low",           f"₦{pred_low:,.0f}",               "var(--text2)"),
                ("Range High",          f"₦{pred_high:,.0f}",              "var(--text2)"),
                ("Real MAE",            f"₦{bt_stats.get('mae','—')}",     "var(--cyan)"),
                ("Real MAPE",           f"{bt_stats.get('mape','—')}%",    "var(--cyan)"),
            ]:
                st.markdown(f'<tr><td style="font-size:11px;color:var(--text2);">{lbl}</td>'
                            f'<td style="font-family:var(--font-mono);color:{clr};font-size:12px;">{val}</td></tr>',
                            unsafe_allow_html=True)
            st.markdown('</table></div>', unsafe_allow_html=True)

            risks     = narratives.get("key_risks",[])
            catalysts = narratives.get("key_upside_catalysts",[])
            if risks or catalysts:
                st.markdown('<div class="card"><div class="sec-header">⚠️ KEY RISKS & CATALYSTS</div>',
                            unsafe_allow_html=True)
                for ri in risks[:4]:
                    if ri: st.markdown(f'<div style="font-size:11px;color:#ffaabb;padding:5px 0;border-bottom:1px solid var(--border);">🔻 {ri}</div>',unsafe_allow_html=True)
                for ci in catalysts[:3]:
                    if ci: st.markdown(f'<div style="font-size:11px;color:#a0ead4;padding:5px 0;border-bottom:1px solid var(--border);">🔺 {ci}</div>',unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Full-width qualitative section
        q_intel = raw.get("news_intel",{})
        n_hl    = raw.get("news_headlines_count",0)
        if q_intel:
            st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--purple);margin-bottom:16px;">🌐 LIVE WORLD INTELLIGENCE — {n_hl} headlines</div>',
                        unsafe_allow_html=True)
            breaking = q_intel.get("breaking_event","")
            if breaking and str(breaking).lower() not in ("null","none","n/a",""):
                st.markdown(f'<div style="background:rgba(255,68,102,0.1);border:1px solid var(--red);border-left:4px solid var(--red);border-radius:10px;padding:12px 16px;margin-bottom:14px;"><span style="font-size:9px;color:var(--red);font-family:var(--font-mono);letter-spacing:2px;text-transform:uppercase;">⚡ BREAKING</span><div style="font-size:13px;color:var(--text);margin-top:5px;">{breaking}</div></div>',
                            unsafe_allow_html=True)
            qa1,qa2,qa3 = st.columns(3)
            for col,icon_lbl,score_key,analysis_key in [
                (qa1,"🛢️ Oil Markets","news_oil","oil_analysis"),
                (qa2,"🌍 Geopolitics","news_geopolitics","geopolitical_analysis"),
                (qa3,"🏦 CBN Policy","news_cbn","cbn_analysis"),
            ]:
                s  = feat.get(score_key,0); sc = signal_color(s)
                col.markdown(f"""<div class="card" style="height:100%;margin-bottom:0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                  <div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);">{icon_lbl}</div>
                  <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{sc};">{s:+.0f}</div>
                </div>
                <p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0;">{q_intel.get(analysis_key,"N/A")}</p>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:14px;"><div class="sec-header">📡 10-DIMENSION QUALITATIVE SCORES (+ = NGN weakens)</div>',
                        unsafe_allow_html=True)
            dims = [
                ("Overall Signal",   feat.get("news_overall",0)),
                ("Nigeria Macro",    feat.get("news_nigeria",0)),
                ("CBN Policy",       feat.get("news_cbn",0)),
                ("Oil Markets",      feat.get("news_oil",0)),
                ("USD / Fed",        feat.get("news_usd",0)),
                ("Crypto Sentiment", feat.get("news_crypto",0)),
                ("Geopolitics",      feat.get("news_geopolitics",0)),
                ("Nigeria Politics", feat.get("news_political_risk",0)),
                ("Remittances",      feat.get("news_remittance",0)),
                ("Global EM Risk",   feat.get("news_em_risk",0)),
            ]
            cols_dim = st.columns(2)
            for i,(lbl,score) in enumerate(dims):
                with cols_dim[i%2]:
                    prog_bar(lbl, score, signal_color(score))
            st.markdown('</div>', unsafe_allow_html=True)


    # ══════ TAB 2: FORECASTS ══════
    with tab2:
        st.markdown(f"""<div class="card card-purple" style="margin-bottom:20px;">
        <div class="sec-header">📈 MULTI-TIMEFRAME CBN RATE FORECASTS</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-size:12px;color:var(--text2);">
          <div>📌 <strong style="color:var(--text);">CBN Rate:</strong> <span style="font-family:var(--font-mono);color:var(--green);">₦{cbn_rate:,.0f}</span></div>
          <div>🤖 <strong style="color:var(--text);">Training Points:</strong> <span style="font-family:var(--font-mono);color:var(--amber);">{n_pts}</span></div>
          <div>🔬 <strong style="color:var(--text);">Backtest Obs:</strong> <span style="font-family:var(--font-mono);color:var(--cyan);">{bt_stats.get('n',0)}</span></div>
        </div>
        <div style="margin-top:10px;font-size:11px;color:var(--muted);">
          ⚠️ Confidence shown only after walk-forward backtest has ≥2 validated predictions.
          Uncertainty ranges scale with √time (statistical law).
        </div></div>""", unsafe_allow_html=True)

        # Row 1: 24H 7D 30D
        tf_row1 = st.columns(3)
        for col,key in zip(tf_row1,["24h","7d","30d"]):
            f   = forecasts.get(key,{})
            ac  = tf_accent(key)
            ctr = f.get("central",0); pct = f.get("pct_change",0)
            fc  = "var(--green)" if "HIGHER" in f.get("direction","") else "var(--red)" if "LOWER" in f.get("direction","") else "var(--amber)"
            col.markdown(f"""<div class="tf-card">
              <div class="tf-card-accent" style="background:{ac};"></div>
              <div class="tf-label">{f.get("label","")}</div>
              <div class="tf-value" style="color:{ac};">₦{ctr:,.0f}</div>
              <div class="tf-change" style="color:{fc};">{f.get("direction","")}</div>
              <div class="tf-range">₦{f.get("low",0):,.0f} – ₦{f.get("high",0):,.0f}</div>
              <div style="font-family:var(--font-mono);font-size:12px;color:{fc};margin-top:4px;">{pct:+.1f}% from now</div>
              <div class="tf-conf">Confidence: {f.get("confidence_str","—")}</div>
              <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:10px;color:var(--muted);">
                Bull ₦{f.get("bull_case",0):,.0f} &nbsp;|&nbsp; Bear ₦{f.get("bear_case",0):,.0f}</div>
            </div>""", unsafe_allow_html=True)

        # Row 2: 3M 6M 12M 2YR
        tf_row2 = st.columns(4)
        for col,key in zip(tf_row2,["3m","6m","12m","2yr"]):
            f   = forecasts.get(key,{})
            ac  = tf_accent(key)
            ctr = f.get("central",0); pct = f.get("pct_change",0)
            fc  = "var(--green)" if "HIGHER" in f.get("direction","") else "var(--red)" if "LOWER" in f.get("direction","") else "var(--amber)"
            col.markdown(f"""<div class="tf-card">
              <div class="tf-card-accent" style="background:{ac};"></div>
              <div class="tf-label">{f.get("label","")}</div>
              <div class="tf-value" style="color:{ac};">₦{ctr:,.0f}</div>
              <div class="tf-change" style="color:{fc};">{f.get("direction","")}</div>
              <div class="tf-range">₦{f.get("low",0):,.0f} – ₦{f.get("high",0):,.0f}</div>
              <div style="font-family:var(--font-mono);font-size:12px;color:{fc};margin-top:4px;">{pct:+.1f}% from now</div>
              <div class="tf-conf">Confidence: {f.get("confidence_str","—")}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:16px;">💬 TIMEFRAME NARRATIVES</div>',
                    unsafe_allow_html=True)

        nar_data = [
            ("24H","24h","n24h_narrative","n24h_risk","n24h_drivers"),
            ("7D", "7d", "n7d_narrative", "n7d_risk", "n7d_drivers"),
            ("30D","30d","n30d_narrative",None,        None),
            ("3M", "3m", "n3m_narrative", None,        None),
            ("6M", "6m", "n6m_narrative", None,        None),
            ("12M","12m","n12m_narrative",None,        None),
            ("2YR","2yr","n2yr_narrative",None,        None),
        ]
        for lbl,key,nar_key,risk_key,drivers_key in nar_data:
            nar   = narratives.get(nar_key,"")
            risk  = narratives.get(risk_key,"") if risk_key else ""
            drvrs = narratives.get(drivers_key,[]) if drivers_key else []
            ac    = tf_accent(key)
            fd    = forecasts.get(key,{})
            fpct  = fd.get("pct_change",0); fctr = fd.get("central",0)
            fconf_s = fd.get("confidence_str","—")
            fclr  = "var(--green)" if fpct>0.5 else "var(--red)" if fpct<-0.5 else "var(--amber)"
            drvrs_html = "".join([
                f'<span style="background:rgba(68,136,255,0.12);color:var(--blue);border:1px solid rgba(68,136,255,0.25);border-radius:4px;padding:2px 8px;font-size:10px;font-family:var(--font-mono);margin-right:6px;">{d}</span>'
                for d in drvrs if d
            ])
            risk_row = f'<div style="margin-top:8px;font-size:11px;color:#ffaabb;"><strong>⚠️ Risk:</strong> {risk}</div>' if risk else ""
            st.markdown(
                f'<div class="card" style="margin-bottom:10px;border-left:3px solid {ac};">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<div style="font-family:var(--font-mono);font-size:11px;font-weight:700;color:{ac};">{lbl} HORIZON</div>'
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f'<span style="font-family:var(--font-mono);font-size:13px;color:{fclr};font-weight:700;">₦{fctr:,.0f} ({fpct:+.1f}%)</span>'
                f'<span style="font-family:var(--font-mono);font-size:9px;color:var(--muted);">{fconf_s}</span>'
                f'</div></div>'
                + (f'<div style="margin-bottom:6px;">{drvrs_html}</div>' if drvrs_html else "")
                + f'<p style="font-size:12px;color:var(--text2);line-height:1.65;margin:0;">{nar if nar else f"CBN Rate target: ₦{fctr:,.0f} ({fpct:+.1f}%)"}</p>'
                + risk_row + '</div>', unsafe_allow_html=True)
            if key == "30d":
                bull_s = narratives.get("n30d_bull_scenario","")
                bear_s = narratives.get("n30d_bear_scenario","")
                if bull_s or bear_s:
                    st.markdown(
                        '<div style="display:flex;gap:10px;margin-top:-6px;margin-bottom:10px;">'
                        '<div class="scenario-card scenario-bull" style="flex:1;">'
                        '<div style="font-size:9px;color:var(--green);font-family:var(--font-mono);letter-spacing:1px;margin-bottom:5px;">📈 BULL SCENARIO</div>'
                        f'<div style="font-size:11px;color:var(--text2);">{bull_s}</div></div>'
                        '<div class="scenario-card scenario-bear" style="flex:1;">'
                        '<div style="font-size:9px;color:var(--red);font-family:var(--font-mono);letter-spacing:1px;margin-bottom:5px;">📉 BEAR SCENARIO</div>'
                        f'<div style="font-size:11px;color:var(--text2);">{bear_s}</div></div>'
                        '</div>', unsafe_allow_html=True)

        oil_sum = narratives.get("oil_impact_summary","")
        cbn_watch = narratives.get("cbn_watch","")
        if oil_sum or cbn_watch:
            oi1,oi2 = st.columns(2)
            if oil_sum:
                oi1.markdown(f'<div class="card"><div class="sec-header">🛢️ OIL → CBN RATE TRANSMISSION</div><p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{oil_sum}</p></div>',
                             unsafe_allow_html=True)
            if cbn_watch:
                oi2.markdown(f'<div class="card"><div class="sec-header">🏦 CBN WATCH LIST</div><p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{cbn_watch}</p></div>',
                             unsafe_allow_html=True)

        dq = narratives.get("data_quality_note",""); disc = narratives.get("disclaimer_note","")
        if dq:
            st.markdown(f'<div class="alert-box alert-info" style="margin-top:14px;font-size:11px;"><strong>🔬 Data Quality:</strong> {dq}</div>',
                        unsafe_allow_html=True)
        if disc:
            st.markdown(f'<div style="font-size:10px;color:var(--muted);margin-top:8px;line-height:1.6;">⚠️ {disc}</div>',
                        unsafe_allow_html=True)


    # ══════ TAB 3: WALK-FORWARD BACKTEST ══════
    with tab3:
        bt_log   = st.session_state.backtest_log
        bt_stats = compute_backtest_stats()

        st.markdown("""<div class="card card-cyan" style="margin-bottom:18px;">
        <div class="sec-header">🔬 WALK-FORWARD BACKTEST ENGINE</div>
        <p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">
          Every time you run the Oracle, the current ML ensemble prediction is stored.
          On the <em>next</em> run, it is resolved against the actual CBN rate — producing a genuine
          out-of-sample prediction error. <strong style="color:var(--amber);">Confidence % is only shown
          once at least 2 pairs have been validated.</strong> This is the only honest approach.
        </p>
        </div>""", unsafe_allow_html=True)

        # Stats cards
        bts1, bts2, bts3, bts4 = st.columns(4)
        def bt_card(col, label, val, sub, color):
            col.markdown(f"""<div class="card" style="text-align:center;padding:16px 12px;">
            <div class="card-label">{label}</div>
            <div style="font-family:var(--font-mono);font-size:22px;font-weight:700;color:{color};">{val}</div>
            <div style="font-size:10px;color:var(--muted);margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        n_obs = bt_stats.get("n", 0)
        mae_d = f"₦{bt_stats['mae']:,.2f}" if bt_stats.get("mae") else "—"
        mape_d = f"{bt_stats['mape']:.3f}%" if bt_stats.get("mape") else "—"
        dir_d  = f"{bt_stats['dir_accuracy']}%" if bt_stats.get("dir_accuracy") is not None else "—"
        conf_d = f"{bt_stats['conf_from_bt']}%" if bt_stats.get("conf_from_bt") else "—"

        bt_card(bts1, "Observations", str(n_obs), "Walk-forward validated pairs", "var(--cyan)")
        bt_card(bts2, "Real MAE", mae_d, "Mean Absolute Error (out-of-sample)", "var(--amber)")
        bt_card(bts3, "Real MAPE", mape_d, "Mean Absolute Percentage Error", "var(--blue)")
        bt_card(bts4, "Direction Accuracy", dir_d, "% correct up/down calls", "var(--green)")

        if not bt_stats.get("ready"):
            st.markdown(f"""<div class="alert-box alert-info" style="margin-top:14px;">
            ⏳ <strong>Backtest not yet ready:</strong> {bt_stats.get('message','')}
            <br><br>
            How walk-forward validation works:
            <ol style="margin-top:8px;padding-left:18px;font-size:12px;line-height:1.8;">
              <li>Run the Oracle now → prediction is stored as "pending"</li>
              <li>Run it again later (same day or next day) → that run fetches the real CBN rate and resolves the previous prediction</li>
              <li>Error metrics accumulate over time → confidence % becomes available</li>
            </ol>
            </div>""", unsafe_allow_html=True)
        else:
            # Confidence derivation explanation
            c_val = bt_stats["conf_from_bt"]
            c_col = "var(--green)" if c_val>=65 else "var(--amber)" if c_val>=45 else "var(--red)"
            st.markdown(f"""<div class="card card-purple" style="margin-top:14px;">
            <div class="sec-header">📊 VALIDATED CONFIDENCE DERIVATION</div>
            <div style="display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:center;">
              <div style="text-align:center;">
                <div style="border:5px solid {c_col};border-radius:50%;width:90px;height:90px;
                display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0 auto;">
                  <div style="font-family:var(--font-mono);font-size:22px;font-weight:700;color:{c_col};">{c_val}%</div>
                  <div style="font-size:9px;color:var(--muted);letter-spacing:1px;">REAL</div>
                </div>
              </div>
              <div style="font-size:12px;color:var(--text2);line-height:1.7;">
                Derived from <strong style="color:var(--text);">real walk-forward MAPE = {bt_stats.get('mape','?')}%</strong>
                on <strong style="color:var(--text);">{n_obs} observations</strong>.<br>
                Formula: MAPE ≤ 0.1% → 85% · MAPE ≥ 2.0% → 25% · Linear interpolation between.<br>
                Direction accuracy: <strong style="color:var(--green);">{dir_d}</strong> of up/down calls were correct.
              </div>
            </div>
            </div>""", unsafe_allow_html=True)

        # Walk-forward log table
        if bt_log:
            st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:14px;">📋 WALK-FORWARD LOG (last {min(len(bt_log),30)} entries)</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="card"><table class="wf-table">'
                        '<tr><th>Timestamp</th><th>Predicted</th><th>Actual</th><th>Error (₦)</th><th>MAPE</th><th>Direction</th></tr>',
                        unsafe_allow_html=True)
            for row in reversed(bt_log[-30:]):
                ts    = str(row.get("timestamp",""))[:16].replace("T"," ")
                pred  = row.get("predicted", 0)
                actual= row.get("actual", 0)
                err   = row.get("error_abs", 0)
                mape_r= row.get("error_pct", 0)
                dir_ok= row.get("direction_correct")
                err_c = "var(--green)" if err < actual*0.005 else "var(--amber)" if err < actual*0.015 else "var(--red)"
                dir_sym = ("✅" if dir_ok else "❌") if dir_ok is not None else "—"
                st.markdown(f"""<tr>
                  <td style="color:var(--muted);">{ts}</td>
                  <td>₦{pred:,.0f}</td>
                  <td style="color:var(--green);">₦{actual:,.0f}</td>
                  <td style="color:{err_c};">₦{err:,.2f}</td>
                  <td style="color:{err_c};">{mape_r:.3f}%</td>
                  <td>{dir_sym}</td>
                </tr>""", unsafe_allow_html=True)
            st.markdown('</table></div>', unsafe_allow_html=True)

            # SVG error-over-time chart
            errors_over_time = [r.get("error_abs", 0) for r in bt_log[-30:]]
            if len(errors_over_time) >= 2:
                max_err = max(errors_over_time) * 1.1 or 1
                n_e     = len(errors_over_time)
                path_pts = " ".join(
                    f"{'M' if i==0 else 'L'} {i*(560/(n_e-1)):.1f} {((max_err-v)/max_err*70):.1f}"
                    for i, v in enumerate(errors_over_time)
                )
                avg_e = sum(errors_over_time)/len(errors_over_time)
                avg_y = (max_err - avg_e) / max_err * 70
                st.markdown(f"""<div class="card" style="margin-top:12px;">
                <div class="sec-header">📉 PREDICTION ERROR OVER TIME</div>
                <svg width="100%" viewBox="0 0 600 90" style="overflow:visible;">
                  <defs>
                    <linearGradient id="err_grad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stop-color="#ffb020" stop-opacity="0.3"/>
                      <stop offset="100%" stop-color="#ffb020" stop-opacity="0"/>
                    </linearGradient>
                  </defs>
                  <path d="{path_pts} L {560:.1f} 70 L 0 70 Z" fill="url(#err_grad)"/>
                  <path d="{path_pts}" fill="none" stroke="#ffb020" stroke-width="2"/>
                  <line x1="0" y1="{avg_y:.1f}" x2="560" y2="{avg_y:.1f}"
                        stroke="#4488ff" stroke-width="1" stroke-dasharray="4,4"/>
                  <text x="565" y="{avg_y+4:.1f}" fill="#4488ff" font-size="9" font-family="JetBrains Mono">avg</text>
                  <text x="0" y="85" fill="var(--muted)" font-size="9" font-family="JetBrains Mono">₦0</text>
                  <text x="0" y="12" fill="var(--muted)" font-size="9" font-family="JetBrains Mono">₦{max_err:,.0f}</text>
                  <text x="490" y="85" fill="#ffb020" font-size="9" font-family="JetBrains Mono">latest ₦{errors_over_time[-1]:,.2f}</text>
                </svg>
                </div>""", unsafe_allow_html=True)

        # Pending prediction box
        pend = st.session_state.pending_pred
        if pend:
            ts_p   = str(pend.get("timestamp",""))[:16].replace("T"," ")
            ens_p  = pend.get("ensemble", 0)
            prev_p = pend.get("prev_rate", 0)
            st.markdown(f"""<div class="alert-box alert-bull" style="margin-top:14px;font-size:12px;">
            ⏳ <strong>Pending prediction:</strong> ₦{ens_p:,.0f} (stored at {ts_p}, from rate ₦{prev_p:,.0f}).
            This will be resolved against the CBN rate on the <strong>next</strong> run and added to the backtest log.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="alert-box alert-info" style="margin-top:14px;font-size:12px;">
            No pending prediction. Run the Oracle to store one.
            </div>""", unsafe_allow_html=True)


    # ══════ TAB 4: GLOBAL SIGNALS ══════
    with tab4:
        sig  = st.session_state.global_signals or {}
        anal = sig.get("analysis", {})
        now_gs = st.session_state.global_signals_time

        age_str, next_str = "never", f"{SIGNALS_TTL//60}m 0s"
        if now_gs:
            age_s    = int((datetime.datetime.now() - now_gs).total_seconds())
            age_str  = f"{age_s}s ago" if age_s < 60 else f"{age_s//60}m {age_s%60}s ago"
            rem      = max(0, SIGNALS_TTL - age_s)
            next_str = f"{rem//60}m {rem%60}s"

        gs1, gs2 = st.columns([6,1])
        with gs1:
            n_hl  = sig.get("headline_count", 0)
            n_src = len(sig.get("sources", []))
            st.markdown(f"""<div style="margin-bottom:14px;">
            <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;
            text-transform:uppercase;color:var(--purple);margin-bottom:4px;">🌐 LIVE GLOBAL SIGNALS — CBN RATE CONTEXT</div>
            <div style="font-size:11px;color:var(--text2);">
              {n_hl} headlines &nbsp;·&nbsp; {n_src} sources &nbsp;·&nbsp;
              <span style="color:var(--green);">Updated {age_str}</span> &nbsp;·&nbsp;
              Next refresh: {next_str}
            </div></div>""", unsafe_allow_html=True)
        with gs2:
            if st.button("↻ Refresh", key="gs_refresh", use_container_width=True):
                maybe_refresh_signals(force=True)
                st.rerun()

        if not sig:
            st.markdown('<div class="card" style="text-align:center;padding:40px;"><div style="font-size:32px;margin-bottom:12px;">📡</div><p style="color:var(--text2);">Press ↻ Refresh or run the Oracle to load live signals.</p></div>',
                        unsafe_allow_html=True)
        else:
            breaking = anal.get("breaking_event","")
            if breaking and str(breaking).lower() not in ("null","none","n/a",""):
                st.markdown(f"""<div style="background:rgba(255,68,102,0.1);border:1px solid var(--red);
                border-left:4px solid var(--red);border-radius:10px;padding:12px 16px;margin-bottom:12px;">
                <span style="font-size:9px;color:var(--red);font-family:var(--font-mono);
                letter-spacing:2px;text-transform:uppercase;">⚡ BREAKING EVENT</span>
                <div style="font-size:13px;color:var(--text);margin-top:5px;line-height:1.6;">{breaking}</div>
                </div>""", unsafe_allow_html=True)

            top_mover = anal.get("top_mover_today","")
            if top_mover:
                st.markdown(f"""<div style="background:var(--purple2);border:1px solid rgba(176,96,255,0.3);
                border-radius:10px;padding:10px 16px;margin-bottom:14px;">
                <span style="font-size:9px;color:var(--purple);font-family:var(--font-mono);
                letter-spacing:2px;text-transform:uppercase;">🎯 TOP MOVER TODAY</span>
                <div style="font-size:12px;color:var(--text);margin-top:5px;">{top_mover}</div>
                </div>""", unsafe_allow_html=True)

            btc_u = sig.get("btc_usd"); btc_c = sig.get("btc_24h") or 0
            eth_u = sig.get("eth_usd"); eth_c = sig.get("eth_24h") or 0
            eur   = sig.get("eurusd");  dxy   = sig.get("dxy_proxy")
            fng_v = sig.get("fear_greed_value","—")
            fng_l = sig.get("fear_greed_label","N/A")

            gm = st.columns(4)
            for col, lbl, val, sub, clr in [
                (gm[0], "BTC/USD",      f"${btc_u:,.0f}" if btc_u else "N/A", f"{btc_c:+.2f}% 24h", "var(--green)" if btc_c>=0 else "var(--red)"),
                (gm[1], "ETH/USD",      f"${eth_u:,.0f}" if eth_u else "N/A", f"{eth_c:+.2f}% 24h", "var(--green)" if eth_c>=0 else "var(--red)"),
                (gm[2], "EUR/USD",      f"{eur:.4f}" if eur else "N/A",       f"DXY proxy: {dxy}", "var(--blue)"),
                (gm[3], "Fear & Greed", f"{fng_v}", fng_l, "var(--green)" if isinstance(fng_v,int) and fng_v>60 else "var(--red)" if isinstance(fng_v,int) and fng_v<30 else "var(--amber)"),
            ]:
                col.markdown(f"""<div class="card" style="margin-bottom:12px;">
                <div class="card-label">{lbl}</div>
                <div style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:{clr};">{val}</div>
                <div class="card-sub">{sub}</div></div>""", unsafe_allow_html=True)

            signal_cards = [
                ("🛢️ Oil Markets",    "oil_impact",        "oil_analysis",          "Oil-CBN"),
                ("🌍 Geopolitics",    "geopolitical_risk", "geopolitical_analysis", "GEO"),
                ("🏦 CBN Policy",     "cbn_policy",        "cbn_analysis",          "CBN"),
                ("₿ Crypto Sentiment","crypto_sentiment",  "crypto_analysis",       "Crypto"),
                ("📊 EM FX Risk",     "global_em_risk",    "em_analysis",           "EM FX"),
                ("💸 Remittances",    "remittance_flow",   None,                    "Remit"),
            ]
            for icon_lbl, score_key, analysis_key, _ in signal_cards:
                score = anal.get(score_key, 0) or 0
                body  = anal.get(analysis_key, "") if analysis_key else f"Score: {score:+.0f}"
                sc    = signal_color(score)
                score_lbl = "BEARISH NGN" if score > 15 else "BULLISH NGN" if score < -15 else "NEUTRAL"
                st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:var(--r-lg);
                padding:14px 18px;margin-bottom:10px;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
                    <span style="font-family:var(--font-mono);font-size:11px;font-weight:700;color:var(--text);">{icon_lbl}</span>
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:{sc};">{score:+.0f}</span>
                      <span style="background:rgba(255,255,255,0.05);color:{sc};border:1px solid {sc}44;
                      border-radius:4px;padding:2px 8px;font-size:9px;font-family:var(--font-mono);font-weight:700;">{score_lbl}</span>
                    </div>
                  </div>
                  <div style="font-size:12px;color:var(--text2);line-height:1.65;">{body or "No analysis available."}</div>
                </div>""", unsafe_allow_html=True)

            bull = anal.get("top_bullish_catalyst","")
            bear = anal.get("top_bearish_catalyst","")
            bb1, bb2 = st.columns(2)
            if bull:
                bb1.markdown(f"""<div style="background:var(--green3);border:1px solid rgba(0,229,160,0.3);
                border-radius:var(--r-md);padding:14px 16px;">
                <div style="font-size:9px;color:var(--green);font-family:var(--font-mono);
                letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">📈 Top Bullish Catalyst (USD)</div>
                <div style="font-size:12px;color:var(--text2);line-height:1.65;">{bull}</div></div>""",
                unsafe_allow_html=True)
            if bear:
                bb2.markdown(f"""<div style="background:rgba(255,68,102,0.05);border:1px solid rgba(255,68,102,0.25);
                border-radius:var(--r-md);padding:14px 16px;">
                <div style="font-size:9px;color:var(--red);font-family:var(--font-mono);
                letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">📉 Top Bearish Catalyst (USD)</div>
                <div style="font-size:12px;color:var(--text2);line-height:1.65;">{bear}</div></div>""",
                unsafe_allow_html=True)

            watch = anal.get("key_watch_items", [])
            if watch:
                st.markdown(f"""<div style="margin-top:14px;font-size:11px;border-top:1px solid var(--border);padding-top:12px;color:var(--muted);">
                <strong style="font-family:var(--font-mono);font-size:9px;letter-spacing:1.5px;text-transform:uppercase;">👁 WATCH THIS WEEK: </strong>
                {"  &nbsp;·&nbsp;  ".join(f'<span style="color:var(--amber);">{w}</span>' for w in watch)}
                </div>""", unsafe_allow_html=True)

            mt = anal.get("medium_term_outlook",""); lt = anal.get("long_term_outlook","")
            if mt or lt:
                mo1, mo2 = st.columns(2)
                if mt: mo1.markdown(f'<div class="card" style="margin-top:14px;"><div class="sec-header">📅 30–90 DAY OUTLOOK</div><p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{mt}</p></div>', unsafe_allow_html=True)
                if lt: mo2.markdown(f'<div class="card" style="margin-top:14px;"><div class="sec-header">📆 6–12 MONTH OUTLOOK</div><p style="font-size:12px;color:var(--text2);line-height:1.7;margin:0;">{lt}</p></div>', unsafe_allow_html=True)

            headlines = sig.get("headlines", [])
            if headlines:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander(f"📋 All {len(headlines)} live headlines"):
                    for h in headlines:
                        clr = headline_color(h.get("tag",""))
                        st.markdown(
                            f'<div class="hl-row"><span style="color:{clr};font-family:var(--font-mono);">{h.get("tag","")}</span> '
                            f'<span style="color:var(--text);">{h.get("title","")}</span></div>',
                            unsafe_allow_html=True)


    # ══════ TAB 5: CONVERTER ══════
    with tab5:
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:16px;">💱 USD / NGN CONVERTER</div>',
                    unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        conv_rates = {
            "CBN Official":        cbn_rate,
            "P2P (signal only)":   raw.get("p2p_mid") or cbn_rate,
            "ML Prediction (24H)": ensemble,
            "7D Forecast":         forecasts.get("7d",{}).get("central", cbn_rate),
            "30D Forecast":        forecasts.get("30d",{}).get("central", cbn_rate),
        }
        with cc1:
            usd_in = st.number_input("USD Amount", min_value=0.0, value=100.0, step=10.0)
            st.markdown('<div class="card"><div class="sec-header">USD → NGN</div>', unsafe_allow_html=True)
            for lbl, rv in conv_rates.items():
                if rv:
                    ngn_out = usd_in * rv
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);">'
                                f'<span style="font-size:12px;color:var(--text2);">{lbl}</span>'
                                f'<span style="font-family:var(--font-mono);font-size:13px;font-weight:700;color:var(--green);">₦{ngn_out:,.2f}</span>'
                                f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            ngn_in = st.number_input("NGN Amount", min_value=0.0, value=100000.0, step=1000.0)
            st.markdown('<div class="card"><div class="sec-header">NGN → USD</div>', unsafe_allow_html=True)
            for lbl, rv in conv_rates.items():
                if rv:
                    usd_out = ngn_in / rv
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);">'
                                f'<span style="font-size:12px;color:var(--text2);">{lbl}</span>'
                                f'<span style="font-family:var(--font-mono);font-size:13px;font-weight:700;color:var(--cyan);">${usd_out:,.4f}</span>'
                                f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-info" style="margin-top:12px;font-size:11px;">ℹ️ CBN Official rate is sourced from ExchangeRate-API / Frankfurter (IMF/central bank data). P2P is the parallel market rate shown for reference only.</div>',
                    unsafe_allow_html=True)

    # ══════ TAB 6: HISTORY ══════
    with tab6:
        hist_data = st.session_state.rate_history
        if len(hist_data) < 2:
            st.markdown('<div class="card" style="text-align:center;padding:40px;"><p style="color:var(--text2);">Run the analysis at least twice to see CBN rate history.</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:10px;font-family:var(--font-mono);color:var(--text2);margin-bottom:16px;"><span class="live-dot"></span>{len(hist_data)} CBN-rate observations in history</div>',
                        unsafe_allow_html=True)
            chart_vals = [d.get("cbn_rate") for d in hist_data[-50:] if d.get("cbn_rate")]
            if len(chart_vals) >= 2:
                chart_min = min(chart_vals) * 0.998
                chart_max = max(chart_vals) * 1.002
                n_c = len(chart_vals)
                path_d = " ".join(
                    f"{'M' if i==0 else 'L'} {i*(560/(n_c-1)):.1f} {((chart_max - v)/(chart_max - chart_min)*80):.1f}"
                    for i, v in enumerate(chart_vals)
                )
                trend_up  = chart_vals[-1] >= chart_vals[0]
                line_color= "#00e5a0" if trend_up else "#ff4466"
                st.markdown(f"""<div class="card" style="margin-bottom:16px;">
                <div class="sec-header">📉 CBN OFFICIAL RATE CHART (last {n_c} observations)</div>
                <svg width="100%" viewBox="0 0 600 100" style="overflow:visible;">
                  <defs>
                    <linearGradient id="cg2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stop-color="{line_color}" stop-opacity="0.2"/>
                      <stop offset="100%" stop-color="{line_color}" stop-opacity="0"/>
                    </linearGradient>
                  </defs>
                  <path d="{path_d} L 560 80 L 0 80 Z" fill="url(#cg2)"/>
                  <path d="{path_d}" fill="none" stroke="{line_color}" stroke-width="2"/>
                  <text x="0" y="90" fill="var(--muted)" font-size="9" font-family="JetBrains Mono">₦{chart_min:,.0f}</text>
                  <text x="0" y="8"  fill="var(--muted)" font-size="9" font-family="JetBrains Mono">₦{chart_max:,.0f}</text>
                  <text x="470" y="90" fill="{line_color}" font-size="9" font-family="JetBrains Mono">₦{chart_vals[-1]:,.0f}</text>
                </svg></div>""", unsafe_allow_html=True)

            df_hist = pd.DataFrame([{
                "Time":     d.get("timestamp","")[:16].replace("T"," "),
                "CBN Rate": d.get("cbn_rate"),
                "P2P Mid":  d.get("p2p_mid"),
                "Source":   d.get("rate_source",""),
            } for d in hist_data[-50:]]).dropna(subset=["CBN Rate"])

            st.markdown('<div class="card"><div class="sec-header">DATA TABLE</div>', unsafe_allow_html=True)
            st.dataframe(df_hist.tail(30).sort_values("Time", ascending=False),
                         use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if len(chart_vals) >= 3:
                s1,s2,s3,s4 = st.columns(4)
                for col, lbl, val, clr in [
                    (s1,"Avg CBN Rate", f"₦{np.mean(chart_vals):,.0f}", "var(--text)"),
                    (s2,"Volatility (σ)",f"₦{np.std(chart_vals):,.0f}","var(--amber)"),
                    (s3,"Session High",  f"₦{max(chart_vals):,.0f}",   "var(--green)"),
                    (s4,"Session Low",   f"₦{min(chart_vals):,.0f}",   "var(--red)"),
                ]:
                    col.markdown(f"""<div class="card" style="margin-top:12px;padding:14px 16px;">
                    <div class="card-label">{lbl}</div>
                    <div style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:{clr};">{val}</div>
                    </div>""", unsafe_allow_html=True)

    # ══════ TAB 7: CHAT ══════
    with tab7:
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:12px;">💬 ASK THE ORACLE (CBN RATE FOCUS)</div>',
                    unsafe_allow_html=True)
        quick_prompts = [
            "What's the 7-day CBN rate forecast and why?",
            "How reliable are the current confidence numbers?",
            "What are the biggest risks to NGN right now?",
            "Should I convert USD to NGN today?",
            "How does oil price affect the CBN official rate?",
            "What's the 2-year structural outlook for NGN?",
        ]
        clicked = None
        qcols = st.columns(3)
        for i, qp in enumerate(quick_prompts):
            if qcols[i%3].button(qp, key=f"qp_{i}", use_container_width=True):
                clicked = qp
        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)
        for msg_data in st.session_state.chat:
            if msg_data["r"] == "u":
                st.markdown(f'<div class="chat-u">👤 {msg_data["c"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-a"><div class="chat-badge">🏦 CBN ORACLE</div>{msg_data["c"]}</div>',
                            unsafe_allow_html=True)
        inp_col, btn_col = st.columns([5,1])
        with inp_col:
            user_msg = st.text_input("Ask the Oracle about USD/NGN...", key="chat_input",
                                     label_visibility="collapsed",
                                     placeholder="e.g. What is driving the CBN rate today?")
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

    # ══════ TAB 8: ALERTS ══════
    with tab8:
        al1, al2 = st.columns(2)
        with al1:
            st.markdown("""<div class="card"><div class="sec-header">📧 EMAIL ALERTS</div>
            <p style="font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:12px;">
            Get notified when the CBN official rate crosses your target.</p></div>""",
            unsafe_allow_html=True)
            user_email = st.text_input("Your Email", value=st.session_state.user_email,
                                       placeholder="yourname@gmail.com", key="alert_email_input")
            st.session_state.user_email = user_email
            if user_email:
                if st.button("🧪 Send Test Email", use_container_width=True):
                    rk = RESEND_KEY
                    if rk:
                        html = (f'<div style="font-family:Arial;padding:20px;background:#060912;color:#cfe0f5;border-radius:10px;">'
                                f'<h2>✅ CBN Rate Oracle — Test Email</h2>'
                                f'<p>CBN Rate: ₦{cbn_rate:,.2f} | Direction: {direction}</p></div>')
                        ok = send_email_alert(user_email, "✅ CBN Rate Oracle — Test Alert", html, rk)
                        if ok: st.success("✅ Test email sent!")
                        else:  st.error("❌ Failed. Check RESEND_API_KEY in secrets.")
                    else:
                        st.error("❌ RESEND_API_KEY not configured.")
                if RESEND_KEY:
                    st.markdown('<div class="alert-box alert-bull" style="margin-top:8px;font-size:11px;">✅ Email service connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-box alert-warn" style="margin-top:8px;font-size:11px;">⚠️ Add RESEND_API_KEY to secrets.toml (free at resend.com)</div>', unsafe_allow_html=True)
        with al2:
            st.markdown(f"""<div class="card"><div class="sec-header">🔔 SET PRICE ALERT (CBN RATE)</div></div>""",
                        unsafe_allow_html=True)
            a_level = st.number_input("Alert price (₦)", min_value=100.0, max_value=9999.0,
                                      value=float(round(cbn_rate * 1.01)), step=10.0, key="alert_price_input")
            a_type  = st.selectbox("Alert when CBN rate goes:", ["above","below"], key="alert_type_select")
            if st.button("+ Add Alert", use_container_width=True, key="add_alert_btn"):
                st.session_state.alerts.append({"level": a_level, "type": a_type})
                st.success(f"Alert set: notify when CBN rate goes {a_type} ₦{a_level:,.0f}")
            st.markdown(f"""<div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--r-sm);
            padding:12px 14px;margin-top:8px;text-align:center;">
            <div style="font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;font-family:var(--font-mono);">CBN Official Rate</div>
            <div style="font-family:var(--font-mono);font-size:20px;font-weight:700;color:var(--green);margin-top:4px;">₦{cbn_rate:,.2f}</div>
            <div style="font-size:10px;color:var(--muted2);">{raw.get("source","—")[:35]}</div>
            </div>""", unsafe_allow_html=True)
        if st.session_state.alerts:
            st.markdown('<div class="card" style="margin-top:16px;"><div class="sec-header">ACTIVE ALERTS</div>',
                        unsafe_allow_html=True)
            for i, a in enumerate(st.session_state.alerts):
                triggered_flag = i in st.session_state.alert_triggered
                ac1, ac2, ac3 = st.columns([4,2,1])
                with ac1:
                    arrow = "▲" if a["type"]=="above" else "▼"
                    status = '<span style="color:var(--green);font-size:10px;">✅ Triggered</span>' if triggered_flag else '<span style="color:var(--amber);font-size:10px;">⏳ Watching</span>'
                    st.markdown(f'<span style="font-size:13px;">{arrow} ₦{a["level"]:,} ({a["type"]})</span> {status}', unsafe_allow_html=True)
                with ac2:
                    dist = a["level"] - cbn_rate
                    dc2 = "var(--green)" if ((a["type"]=="above" and dist>0) or (a["type"]=="below" and dist<0)) else "var(--red)"
                    st.markdown(f'<span style="font-family:var(--font-mono);font-size:11px;color:{dc2};">{dist:+,.0f} from now</span>', unsafe_allow_html=True)
                with ac3:
                    if st.button("✕", key=f"del_alert_{i}"):
                        st.session_state.alerts.pop(i)
                        if i in st.session_state.alert_triggered:
                            st.session_state.alert_triggered.remove(i)
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════ TAB 9: DIAGNOSTICS ══════
    with tab9:
        st.markdown('<div style="font-family:var(--font-mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--cyan);margin-bottom:16px;">🩺 SYSTEM DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        sig_diag = st.session_state.global_signals or {}
        c_btn, _ = st.columns([1,3])
        with c_btn:
            if st.button("🔄 Force Refresh All Signals", key="diag_refresh"):
                maybe_refresh_signals(force=True)
                st.rerun()

        gen_errs = sig_diag.get("errors", [])
        rss_errs = sig_diag.get("rss_errors", [])
        all_errs = gen_errs + rss_errs
        if all_errs:
            st.markdown('<div class="alert-box alert-warn"><strong>🚨 Pipeline errors:</strong></div>', unsafe_allow_html=True)
            for e in all_errs[:20]:
                st.markdown(f'<div style="font-size:11px;font-family:var(--font-mono);color:var(--red);background:rgba(255,68,102,0.07);padding:4px 10px;border-radius:4px;margin-bottom:3px;">❌ {e}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-header" style="margin-top:12px;">⚙️ PIPELINE STATUS</div>', unsafe_allow_html=True)
        pipe_items = [
            ("CBN Rate Fetched",        bool(raw.get("cbn_rate")),          f"₦{raw.get('cbn_rate',0):,.2f} | {raw.get('source','')}"),
            ("P2P Rate (signal)",       bool(raw.get("p2p_mid")),           f"₦{raw.get('p2p_mid','N/A')}"),
            ("ExchangeRate-API (FX)",   bool(sig_diag.get("usd_ngn_official")), f"USD/NGN={sig_diag.get('usd_ngn_official','missing')}"),
            ("CoinGecko (BTC/ETH)",     bool(sig_diag.get("btc_usd")),      f"BTC=${sig_diag.get('btc_usd','missing')}"),
            ("Fear & Greed",            bool(sig_diag.get("fear_greed_value")), f"{sig_diag.get('fear_greed_value','missing')}"),
            ("Google News RSS",         sig_diag.get("rss_ok_count",0) > 0, f"{sig_diag.get('rss_ok_count',0)}/18 feeds"),
            ("GNews API",               sig_diag.get("gnews_count",0) > 0,  f"{sig_diag.get('gnews_count',0)} headlines"),
            ("Gemini AI",               bool(sig_diag.get("analysis")),     "JSON parsed OK" if sig_diag.get("analysis") else "FAILED"),
            ("Walk-Forward Backtest",   bt_stats.get("ready",False),        f"{bt_stats.get('n',0)} obs · MAE ₦{bt_stats.get('mae','—')}"),
        ]
        for lbl, ok, detail in pipe_items:
            icon  = "✅" if ok else "❌"
            color = "var(--green)" if ok else "var(--red)"
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 12px;border:1px solid var(--border);border-radius:6px;margin-bottom:6px;background:rgba(255,255,255,0.02);">'
                        f'<span style="font-size:12px;color:var(--text2);">{icon} {lbl}</span>'
                        f'<span style="font-family:var(--font-mono);font-size:11px;color:{color};">{detail}</span>'
                        f'</div>', unsafe_allow_html=True)

        # ML Diagnostics
        st.markdown('<div class="sec-header" style="margin-top:14px;">🤖 ML MODEL DIAGNOSTICS</div>', unsafe_allow_html=True)
        if cold:
            st.markdown(f'<div class="alert-box alert-warn">⚠️ Cold Start — {ml.get("note","")} Run analysis 5+ times.</div>', unsafe_allow_html=True)
        else:
            mm1,mm2,mm3,mm4 = st.columns(4)
            for col, lbl, val, clr in [
                (mm1,"Training Points",str(metrics.get("n_training_points",0)),"var(--amber)"),
                (mm2,"R² In-Sample",   f"{metrics.get('r2_in_sample',0):.4f}" if metrics.get("r2_in_sample") else "N/A","var(--blue)"),
                (mm3,"Model Agreement",f"{metrics.get('agreement_score',0):.1f}%","var(--green)"),
                (mm4,"In-Sample MAE",  f"₦{min(x for x in [metrics.get('ridge_cv_mae'),metrics.get('rf_cv_mae'),metrics.get('gb_cv_mae')] if x) :,.2f}" if any(metrics.get(k) for k in ['ridge_cv_mae','rf_cv_mae','gb_cv_mae']) else "N/A","var(--amber)"),
            ]:
                col.markdown(f"""<div class="card" style="margin-top:6px;padding:12px 14px;">
                <div class="card-label">{lbl}</div>
                <div style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:{clr};">{val}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("""<div class="alert-box alert-info" style="margin-top:12px;font-size:11px;">
            ℹ️ <strong>In-sample metrics (above) are shown for diagnostics only</strong> and are NOT used to compute
            the headline confidence %. Confidence is derived exclusively from the walk-forward backtest
            to prevent data leakage and overconfident reporting.</div>""", unsafe_allow_html=True)

            top_f = ml.get("rf_feature_importance", {})
            if top_f:
                st.markdown('<div class="card" style="margin-top:12px;"><div class="sec-header">RF FEATURE IMPORTANCES (TOP 10)</div>', unsafe_allow_html=True)
                max_imp = max(top_f.values()) if top_f else 1
                for fname, fval in list(top_f.items())[:10]:
                    pct = fval / max_imp * 100
                    st.markdown(f"""<div class="prog-wrap">
                    <div class="prog-label">
                      <span style="font-size:10px;color:var(--text2);">{fname.replace("_"," ").upper()}</span>
                      <span style="font-family:var(--font-mono);font-size:10px;color:var(--amber);">{fval:.4f}</span>
                    </div>
                    <div class="prog-track"><div class="prog-fill" style="width:{pct}%;background:var(--blue);"></div></div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if sig_diag.get("gemini_raw_response"):
            st.markdown('<div class="sec-header" style="margin-top:16px;">🤖 GEMINI RAW RESPONSE</div>', unsafe_allow_html=True)
            st.code(sig_diag["gemini_raw_response"], language="json")

        headlines_diag = sig_diag.get("headlines", [])
        st.markdown(f'<div class="sec-header" style="margin-top:16px;">📰 SAMPLE HEADLINES ({len(headlines_diag)} total)</div>', unsafe_allow_html=True)
        if headlines_diag:
            for h in headlines_diag[:12]:
                st.markdown(f'<div style="font-size:11px;color:var(--text2);padding:4px 0;border-bottom:1px solid var(--border);">'
                            f'<span style="color:var(--cyan);font-family:var(--font-mono);font-size:9px;">{h.get("tag","")}</span> '
                            f'{h.get("title","")}</div>', unsafe_allow_html=True)

    # ── FOOTER DISCLAIMER ──
    st.markdown("""
    <div style="font-size:10px;color:var(--muted);margin-top:24px;line-height:1.7;
    padding:14px 16px;border-top:1px solid var(--border);text-align:center;">
      ⚠️ CBN official rate sourced from ExchangeRate-API / Frankfurter (IMF/central bank data, updates once per business day).
      ML predictions use 3 models trained on historical CBN observations.
      <strong>Confidence % is only shown once walk-forward backtest has ≥2 validated prediction pairs.</strong>
      Not financial advice. Always DYOR before converting funds.
    </div>""", unsafe_allow_html=True)
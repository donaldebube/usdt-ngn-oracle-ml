import streamlit as st
import requests
import json
import datetime
import random

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="USDT/NGN Oracle",
    page_icon="₦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080c14;
    --bg2: #0c1220;
    --bg3: #101828;
    --card: #111d2e;
    --border: #1a2942;
    --border2: #243550;
    --green: #05d68a;
    --green2: rgba(5,214,138,0.12);
    --red: #f0455a;
    --red2: rgba(240,69,90,0.12);
    --amber: #f5a623;
    --amber2: rgba(245,166,35,0.12);
    --blue: #4f8ef7;
    --blue2: rgba(79,142,247,0.12);
    --purple: #a78bfa;
    --text: #dce8f8;
    --muted: #4a6080;
    --muted2: #6b84a0;
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

/* ── TICKER BAR ── */
.ticker-wrap {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    padding: 10px 0;
    margin-bottom: 20px;
    white-space: nowrap;
}
.ticker-inner {
    display: inline-flex;
    gap: 48px;
    animation: ticker 30s linear infinite;
    padding: 0 24px;
}
@keyframes ticker { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted2);
}
.ticker-item .val { color: var(--text); font-weight: 600; }
.ticker-item .up { color: var(--green); }
.ticker-item .dn { color: var(--red); }

/* ── METRIC CARDS ── */
.mcard {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.mcard::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 14px;
    pointer-events: none;
}
.mcard-green { border-top: 2px solid var(--green); }
.mcard-red   { border-top: 2px solid var(--red); }
.mcard-amber { border-top: 2px solid var(--amber); }
.mcard-blue  { border-top: 2px solid var(--blue); }
.mcard-purple{ border-top: 2px solid var(--purple); }

.mcard-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
.mcard-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
}
.mcard-sub {
    font-size: 12px;
    color: var(--muted2);
    margin-top: 4px;
}

/* ── ORACLE CARD ── */
.ocard {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.ocard-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* ── DIRECTION BADGE ── */
.dir-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 18px;
    border-radius: 100px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.dir-bull { background: var(--green2); color: var(--green); border: 1px solid rgba(5,214,138,0.3); }
.dir-bear { background: var(--red2);   color: var(--red);   border: 1px solid rgba(240,69,90,0.3); }
.dir-neu  { background: var(--amber2); color: var(--amber); border: 1px solid rgba(245,166,35,0.3); }

/* ── SIGNAL ROWS ── */
.sig-row {
    padding: 11px 0;
    border-bottom: 1px solid var(--border);
}
.sig-row:last-child { border-bottom: none; }
.sig-tags { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; flex-wrap: wrap; }
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 100px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.tag-bull   { background: var(--green2); color: var(--green); }
.tag-bear   { background: var(--red2);   color: var(--red); }
.tag-neu    { background: var(--amber2); color: var(--amber); }
.tag-hi     { background: var(--blue2);  color: var(--blue); }
.tag-med    { background: rgba(167,139,250,0.1); color: var(--purple); }
.tag-lo     { background: rgba(255,255,255,0.05); color: var(--muted2); }
.sig-name   { font-size: 14px; font-weight: 600; color: var(--text); }
.sig-detail { font-size: 12px; color: var(--muted2); line-height: 1.5; margin-top: 3px; }

/* ── PROGRESS BARS ── */
.prog-wrap { margin-bottom: 14px; }
.prog-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; }
.prog-track { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 4px; }

/* ── SPREAD TABLE ── */
.spread-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.spread-table th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.spread-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.spread-table tr:last-child td { border-bottom: none; }
.spread-table tr:hover td { background: rgba(255,255,255,0.02); }

/* ── CHAT ── */
.chat-u {
    background: var(--blue2);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 14px 14px 3px 14px;
    padding: 12px 16px;
    margin: 10px 0;
    margin-left: 12%;
    font-size: 14px;
    line-height: 1.6;
}
.chat-a {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 3px;
    padding: 12px 16px;
    margin: 10px 0;
    margin-right: 12%;
    font-size: 14px;
    line-height: 1.6;
}
.chat-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 5px;
}

/* ── ALERT BOX ── */
.alert-box {
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    margin-bottom: 10px;
    border-left: 3px solid;
    line-height: 1.5;
}
.alert-bull { background: var(--green2); border-color: var(--green); color: #a7f3d0; }
.alert-bear { background: var(--red2);   border-color: var(--red);   color: #fca5a5; }
.alert-info { background: var(--blue2);  border-color: var(--blue);  color: #bfdbfe; }
.alert-warn { background: var(--amber2); border-color: var(--amber); color: #fde68a; }

/* ── LIVE DOT ── */
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: blink 2s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ── CONVERTER ── */
.conv-box {
    background: var(--bg2);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 18px;
}
.conv-result {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--green);
    margin-top: 10px;
    text-align: center;
}

/* ── INPUTS ── */
.stTextInput>div>div>input, .stTextArea textarea, .stNumberInput>div>div>input {
    background: var(--card) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}
.stTextInput>div>div>input:focus, .stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}
.stButton>button {
    background: linear-gradient(135deg, #1a3a6e, #2d5fb8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton>button:hover { filter: brightness(1.15) !important; transform: translateY(-1px) !important; }
.stSelectbox>div>div { background: var(--card) !important; border-color: var(--border2) !important; color: var(--text) !important; }

#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 768px) {

    /* Tighten main content padding */
    .block-container {
        padding: 0.8rem 0.8rem 2rem 0.8rem !important;
    }

    /* Stack metric cards into 2 columns on mobile */
    [data-testid="column"] {
        min-width: 48% !important;
        flex: 0 0 48% !important;
    }

    /* Shrink metric card text */
    .mcard-value { font-size: 18px !important; }
    .mcard-label { font-size: 9px !important; }
    .mcard { padding: 14px 12px !important; }

    /* Make tabs scrollable on small screens */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch;
    }
    .stTabs [data-baseweb="tab"] {
        white-space: nowrap !important;
        font-size: 12px !important;
        padding: 8px 12px !important;
    }

    /* Shrink hero title */
    .hero-title { font-size: 28px !important; }

    /* Make tables scrollable */
    .spread-table { font-size: 11px !important; }
    .ocard { overflow-x: auto !important; }
}

</style>
""", unsafe_allow_html=True)





# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init():
    for k, v in {
        "chat": [], "result": None, "last_time": None,
        "history": [],
        "alerts": [], "alert_triggered": [], "user_email": "",
        "auto_refresh": False, "refresh_interval": 30, "prev_rate": None
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# ─────────────────────────────────────────────
# API KEYS — loaded exclusively from Streamlit Secrets
# ─────────────────────────────────────────────
# ⚠️ No keys are hardcoded here for security.
# Add your keys in: Streamlit Cloud → App Settings → Secrets
# Format:
#   GEMINI_KEY = "your-gemini-key"
#   NEWS_KEY = "your-newsapi-key"
try:
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
    NEWS_KEY = st.secrets.get("NEWS_KEY", "")
except Exception:
    GEMINI_KEY = ""
    NEWS_KEY = ""

if not GEMINI_KEY:
    st.error("⚠️ Gemini API key not configured. Please contact the app administrator.")
    st.stop()




# ─────────────────────────────────────────────
# GEMINI CALL
# ─────────────────────────────────────────────
def gemini(prompt: str, key: str, system: str = "") -> str:
    models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ]
    parts = []
    if system:
        parts.append({"text": f"SYSTEM:\n{system}\n\n---\n\n"})
    parts.append({"text": prompt})
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 16384}
    }
    last_error = ""
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        try:
            r = requests.post(url, json=payload, timeout=45)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 429:
                return "❌ Rate limit hit. Wait 60 seconds and retry."
            elif r.status_code == 403:
                return "❌ API key invalid. Go to aistudio.google.com to verify your key."
            elif r.status_code == 404:
                last_error = f"Model {model} not found, trying next..."
                continue
            else:
                last_error = f"HTTP {r.status_code}"
                continue
        except Exception as e:
            last_error = f"Connection error: {e}"
            continue
    return f"❌ All models failed. Last error: {last_error}"


# ─────────────────────────────────────────────
# FETCH BLACK MARKET + RATES
# ─────────────────────────────────────────────
def fetch_rates() -> dict:
    """
    Fetch USDT/NGN rates from multiple real sources:
    1. Binance P2P — real live peer-to-peer USDT/NGN trades (TRUE black market)
    2. Bybit P2P   — secondary P2P source
    3. CoinGecko   — aggregated USDT/NGN fallback
    4. OpenExchangeRates — official CBN/interbank USD/NGN rate
    """
    result = {
        "official": None,
        "black_market": None,
        "black_market_source": "",
        "source_official": "",
        "p2p_buy": None,   # what sellers are asking (you pay this to buy USDT)
        "p2p_sell": None,  # what buyers are bidding (you get this when selling USDT)
        "p2p_spread": None,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "partial",
        "all_sources": {}
    }

    # ── SOURCE 1: BINANCE P2P (undocumented public endpoint — no key needed) ──
    # This is the REAL rate Nigerians actually trade USDT at
    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        # BUY side — what you pay per USDT when buying
        buy_payload = {
            "asset": "USDT",
            "fiat": "NGN",
            "merchantCheck": False,
            "page": 1,
            "payTypes": [],
            "publisherType": None,
            "rows": 5,
            "tradeType": "BUY",  # BUY = sellers listing USDT — their ask price
        }
        r = requests.post(
            "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search",
            json=buy_payload,
            headers=headers,
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            prices = []
            for item in data.get("data", []):
                price = float(item.get("adv", {}).get("price", 0))
                if price > 0:
                    prices.append(price)
            if prices:
                result["p2p_buy"] = round(sum(prices) / len(prices), 2)
                result["all_sources"]["binance_p2p_buy"] = result["p2p_buy"]

        # SELL side — what you receive per USDT when selling
        sell_payload = {**buy_payload, "tradeType": "SELL"}
        r2 = requests.post(
            "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search",
            json=sell_payload,
            headers=headers,
            timeout=15
        )
        if r2.status_code == 200:
            data2 = r2.json()
            prices2 = []
            for item in data2.get("data", []):
                price = float(item.get("adv", {}).get("price", 0))
                if price > 0:
                    prices2.append(price)
            if prices2:
                result["p2p_sell"] = round(sum(prices2) / len(prices2), 2)
                result["all_sources"]["binance_p2p_sell"] = result["p2p_sell"]

        # Use midpoint of buy/sell as the true P2P black market rate
        if result["p2p_buy"] and result["p2p_sell"]:
            result["black_market"] = round((result["p2p_buy"] + result["p2p_sell"]) / 2, 2)
            result["p2p_spread"] = round(result["p2p_buy"] - result["p2p_sell"], 2)
            result["black_market_source"] = f"Binance P2P live (Buy: ₦{result['p2p_buy']:,.0f} | Sell: ₦{result['p2p_sell']:,.0f})"
            result["status"] = "live"
        elif result["p2p_buy"]:
            result["black_market"] = result["p2p_buy"]
            result["black_market_source"] = f"Binance P2P live (ask: ₦{result['p2p_buy']:,.0f})"
            result["status"] = "live"
    except Exception as e:
        result["all_sources"]["binance_p2p_error"] = str(e)[:80]

    # ── SOURCE 2: BYBIT P2P (fallback if Binance fails) ──
    if not result["black_market"]:
        try:
            r = requests.get(
                "https://api2.bybit.com/fiat/otc/item/list?userId=&tokenId=USDT&currencyId=NGN&payment=&side=1&size=5&page=1&amount=&authMaker=false&canTrade=false",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12
            )
            if r.status_code == 200:
                items = r.json().get("result", {}).get("items", [])
                prices = [float(i.get("price", 0)) for i in items if float(i.get("price", 0)) > 0]
                if prices:
                    result["black_market"] = round(sum(prices) / len(prices), 2)
                    result["black_market_source"] = "Bybit P2P live"
                    result["status"] = "live"
                    result["all_sources"]["bybit_p2p"] = result["black_market"]
        except Exception as e:
            result["all_sources"]["bybit_p2p_error"] = str(e)[:80]

    # ── SOURCE 3: COINGECKO USDT/NGN (second fallback) ──
    if not result["black_market"]:
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=ngn",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if r.status_code == 200:
                ngn = r.json().get("tether", {}).get("ngn")
                if ngn and float(ngn) > 0:
                    result["black_market"] = float(ngn)
                    result["black_market_source"] = "CoinGecko USDT/NGN (aggregated)"
                    result["status"] = "live"
                    result["all_sources"]["coingecko"] = result["black_market"]
        except: pass

    # ── SOURCE 4: OFFICIAL CBN/INTERBANK RATE ──
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        if r.status_code == 200:
            ngn = r.json().get("rates", {}).get("NGN")
            if ngn:
                result["official"] = float(ngn)
                result["source_official"] = "OpenExchangeRates (interbank)"
                result["all_sources"]["official_rate"] = result["official"]
    except: pass

    # Fallback official rate source
    if not result["official"]:
        try:
            r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
            if r.status_code == 200:
                ngn = r.json().get("rates", {}).get("NGN")
                if ngn:
                    result["official"] = float(ngn)
                    result["source_official"] = "ExchangeRateAPI (interbank)"
        except: pass

    # ── FINAL FALLBACK ──
    if not result["black_market"]:
        result["black_market"] = 1620.0
        result["black_market_source"] = "Fallback estimate (all live sources failed)"
        result["status"] = "estimated"

    if not result["official"]:
        result["official"] = result["black_market"] * 0.93  # official typically ~7% below P2P
        result["source_official"] = "Estimated from P2P rate"

    result["primary"] = result["black_market"]

    # Calculate official-to-blackmarket spread
    if result["official"] and result["black_market"]:
        result["spread_pct"] = round(
            ((result["black_market"] - result["official"]) / result["official"]) * 100, 2
        )
    else:
        result["spread_pct"] = 0.0

    return result


# ─────────────────────────────────────────────
# FETCH GLOBAL SIGNALS
# ─────────────────────────────────────────────
def fetch_global_signals(news_key: str = "") -> list:
    """Comprehensive global signals that affect USDT/NGN."""
    signals = []

    # ── LIVE GLOBAL MACRO (free APIs) ──
    # BTC price (crypto sentiment proxy)
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true",
            timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            d = r.json()
            btc = d.get("bitcoin", {})
            eth = d.get("ethereum", {})
            btc_chg = btc.get("usd_24h_change", 0)
            signals.append({
                "category": "Crypto Market",
                "title": f"BTC ${btc.get('usd',0):,.0f} ({btc_chg:+.1f}% 24h) | ETH ${eth.get('usd',0):,.0f}",
                "detail": f"Crypto market sentiment: {'RISK-ON (bullish for NGN demand for USDT)' if btc_chg > 2 else 'RISK-OFF (bearish sentiment)' if btc_chg < -2 else 'NEUTRAL'}. BTC moves affect P2P crypto volume in Nigeria.",
                "impact": "BULLISH" if btc_chg > 2 else "BEARISH" if btc_chg < -2 else "NEUTRAL",
                "source": "CoinGecko Live"
            })
    except: pass

    # Oil price (Nigeria is oil-dependent, ~90% of forex earnings)
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=wrapped-bitcoin&vs_currencies=usd",
            timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
    except: pass

    # DXY / USD strength proxy via EUR/USD
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        if r.status_code == 200:
            rates = r.json().get("rates", {})
            eur = rates.get("EUR", 0)
            gbp = rates.get("GBP", 0)
            signals.append({
                "category": "USD Strength (DXY Proxy)",
                "title": f"EUR/USD: {eur:.4f} | GBP/USD: {gbp:.4f}",
                "detail": ("Strong USD puts pressure on NGN — harder for CBN to defend." if eur < 1.05 else "Weak USD gives NGN breathing room — reduces import pressure." if eur > 1.10 else "USD moderately strong — neutral effect on NGN."),
                "impact": "BEARISH" if eur < 1.05 else "BULLISH" if eur > 1.10 else "NEUTRAL",
                "source": "OpenExchangeRates Live"
            })
            # Emerging market risk gauge
            zar = rates.get("ZAR", 0)
            kes = rates.get("KES", 0)
            signals.append({
                "category": "EM Africa FX Context",
                "title": f"USD/ZAR: {zar:.2f} | USD/KES: {kes:.2f}",
                "detail": "South African Rand and Kenyan Shilling as African EM peers. Broad EM selloff tends to drag NGN down too. Tracks global risk appetite.",
                "impact": "NEUTRAL",
                "source": "OpenExchangeRates Live"
            })
    except: pass

    # ── LIVE NEWS via NewsAPI ──
    if news_key:
        topics = [
            ("Nigeria naira dollar exchange rate 2025", "Nigeria FX"),
            ("CBN central bank Nigeria forex intervention", "CBN Policy"),
            ("Nigeria inflation CPI economy", "Nigeria Macro"),
            ("crude oil price brent WTI today", "Oil Markets"),
            ("US Federal Reserve interest rates dollar", "Fed / USD"),
            ("USDT Tether stablecoin Nigeria crypto", "Crypto / USDT"),
            ("Nigeria remittance diaspora dollar", "Remittances"),
            ("IMF World Bank Nigeria economy", "International Finance"),
            ("Nigeria election politics economy", "Political Risk"),
            ("global risk sentiment emerging markets", "Global EM Risk"),
        ]
        for query, cat in topics[:6]:
            try:
                url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&sortBy=publishedAt&pageSize=3&language=en&apiKey={news_key}"
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    for a in r.json().get("articles", []):
                        t = a.get("title", "")
                        d = a.get("description", "")
                        if t and d:
                            signals.append({
                                "category": cat,
                                "title": t[:120],
                                "detail": d[:200],
                                "published": a.get("publishedAt", "")[:10],
                                "source": a.get("source", {}).get("name", "News"),
                                "url": a.get("url", ""),
                                "impact": "UNKNOWN"
                            })
            except: pass

    # ── STRUCTURAL SIGNALS (always included) ──
    now = datetime.datetime.now()
    dow = now.weekday()  # 0=Mon
    hour = now.hour

    signals += [
        {
            "category": "Market Microstructure",
            "title": f"Session: {now.strftime('%A')} {hour:02d}:00 WAT",
            "detail": f"{'Business hours — high P2P volume, more price movement.' if 8<=hour<=17 and dow<5 else 'Evening session — moderate retail P2P activity.' if 18<=hour<=22 else 'Weekend — lower liquidity, wider spreads.' if dow>=5 else 'Off-hours — lower volume, price may drift.'}",
            "impact": "NEUTRAL",
            "source": "Market Structure"
        },
        {
            "category": "CBN Policy Framework",
            "title": "CBN Managed Float + FX Unification (2023–present)",
            "detail": "CBN unified official and I&E window rates in 2023. Periodic FX auctions to authorised dealers. Diaspora remittance policy eased. Watch for surprise interventions.",
            "impact": "NEUTRAL",
            "source": "Policy Context"
        },
        {
            "category": "Oil Revenue",
            "title": "Nigeria Oil Dependency: ~90% of FX Earnings",
            "detail": "Nigeria earns most of its USD from crude oil exports. Brent crude above $80/bbl is generally supportive of NGN. OPEC+ cuts affect volume. Pipeline vandalism reduces output.",
            "impact": "NEUTRAL",
            "source": "Macro Structural"
        },
        {
            "category": "Crypto P2P Dynamics",
            "title": "Binance P2P + KuCoin + LocalBitcoins Nigeria",
            "detail": "After CBN Binance ban in 2024, P2P shifted to other platforms. USDT demand surged as Nigerians hedge against inflation. P2P rate = best proxy for true black market rate.",
            "impact": "BULLISH",
            "source": "Crypto Context"
        },
        {
            "category": "Inflation Differential",
            "title": f"Nigeria Inflation ~28–32% vs US ~3% (2025 est.)",
            "detail": "High inflation differential means NGN structurally weakens over time relative to USD. This creates long-term USDT demand pressure. Short-term volatility around MPC meetings.",
            "impact": "BEARISH",
            "source": "Macro Structural"
        },
        {
            "category": "Remittances",
            "title": "Diaspora Remittances ~$20–25B/Year",
            "detail": "Nigerian diaspora sends billions in USD annually. High during festive seasons (Dec, Easter). Increased remittances = more USD supply = NGN strength. IMTOs and fintechs compete on rates.",
            "impact": "NEUTRAL",
            "source": "Macro Structural"
        },
    ]

    return signals


# ─────────────────────────────────────────────
# FULL ANALYSIS
# ─────────────────────────────────────────────
def run_analysis(api_key: str, news_key: str, rates: dict) -> dict:
    signals = fetch_global_signals(news_key)
    now = datetime.datetime.now()

    ctx = ""
    for i, s in enumerate(signals[:25]):
        ctx += f"\n[{i+1}] CATEGORY: {s.get('category','')}\nTITLE: {s.get('title','')}\nDETAIL: {s.get('detail','')[:220]}\nSOURCE: {s.get('source','')}\n"

    official = rates.get("official")
    black = rates.get("black_market") or rates.get("primary", 0)
    spread = rates.get("spread_pct")

    system = """You are a senior FX strategist and crypto market analyst specializing in Nigerian naira (NGN) and emerging market currencies.

You analyze ALL global factors: US Fed policy, oil markets, crypto sentiment, EM contagion, CBN interventions, political risk, inflation dynamics, remittance flows, and P2P crypto market microstructure.

The BLACK MARKET / P2P rate is the MOST IMPORTANT rate for your analysis — this is the real price Nigerians pay for USDT.

Return ONLY valid raw JSON. No markdown. No backticks. No explanation outside the JSON object."""

    prompt = f"""Analyze the USDT/NGN market comprehensively and produce a 24-hour prediction.

RATE DATA:
- Black Market / P2P USDT Rate: ₦{black:,.2f} (PRIMARY — most important)
- Official CBN / I&E Rate: ₦{f"{official:,.2f}" if official else "N/A"}
- Official-to-Black-Market Spread: {f'{spread:.1f}%' if spread else 'N/A'}
- Data Timestamp: {now.strftime('%A %d %B %Y, %H:%M WAT')}
- Market Session: {'Weekday active hours' if 8<=now.hour<=18 and now.weekday()<5 else 'Off-peak / weekend'}

GLOBAL SIGNALS ({len(signals)} total):
{ctx}

ANALYSIS INSTRUCTIONS:
1. Weight the signals by relevance: Oil price > CBN policy > USD strength > crypto sentiment > political risk > remittances
2. Consider both the direction AND magnitude of the predicted move
3. Provide specific price targets based on the black market rate
4. Be honest — if signals are mixed, say NEUTRAL with a tight range
5. Factor in time-of-day and day-of-week liquidity effects

Return ONLY this exact JSON (no markdown, no backticks):
{{
  "black_market_rate": {black},
  "official_rate": {official or 0},
  "spread_pct": {spread or 0},
  "prediction_direction": "BULLISH|BEARISH|NEUTRAL",
  "predicted_low": <number>,
  "predicted_high": <number>,
  "predicted_midpoint": <number>,
  "confidence_score": <0-100>,
  "accuracy_basis": "<why this confidence level — be specific about signal quality>",
  "time_horizon": "24 hours",
  "timestamp": "{now.isoformat()}",
  "executive_summary": "<3 sentence outlook covering direction, key catalyst, and key risk>",
  "trade_recommendation": "<specific recommendation: hold USDT / convert to NGN / wait / buy USDT — with specific reasoning>",
  "best_time_to_convert": "<e.g. Morning weekday session / After CBN close / Weekend>",
  "key_drivers": [
    {{"signal": "<name>", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW", "detail": "<2 sentence explanation>", "category": "<Oil|CBN|USD|Crypto|Political|Remittance|Inflation|EM Risk>"}},
    {{"signal": "<name>", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW", "detail": "<2 sentence explanation>", "category": "<category>"}},
    {{"signal": "<name>", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW", "detail": "<2 sentence explanation>", "category": "<category>"}},
    {{"signal": "<name>", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW", "detail": "<2 sentence explanation>", "category": "<category>"}},
    {{"signal": "<name>", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW", "detail": "<2 sentence explanation>", "category": "<category>"}},
    {{"signal": "<name>", "impact": "BULLISH|BEARISH|NEUTRAL", "weight": "HIGH|MEDIUM|LOW", "detail": "<2 sentence explanation>", "category": "<category>"}}
  ],
  "risk_factors": ["<specific risk 1>", "<specific risk 2>", "<specific risk 3>", "<specific risk 4>"],
  "news_sentiment_score": <-100 to 100>,
  "oil_score": <-100 to 100>,
  "usd_strength_score": <-100 to 100>,
  "cbn_policy_score": <-100 to 100>,
  "crypto_sentiment_score": <-100 to 100>,
  "political_risk_score": <-100 to 100>,
  "weekly_outlook": "<brief 1-week directional view>",
  "sources_analyzed": {len(signals)},
  "black_market_premium_analysis": "<analysis of the current spread between official and black market and what it signals>",
  "model": "USDT-NGN-Oracle-v2"
}}"""

    raw = gemini(prompt, api_key, system)

    try:
        clean = raw.strip()
        # Strip markdown code fences
        if "```" in clean:
            parts = clean.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    clean = p
                    break
        # Find JSON start
        if not clean.startswith("{"):
            idx = clean.find("{")
            if idx >= 0:
                clean = clean[idx:]
        # Find last complete closing brace
        last = clean.rfind("}")
        if last >= 0:
            clean = clean[:last+1]
        # Try to parse — if truncated, patch it closed
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Response was cut off — try to close open arrays/objects
            open_braces = clean.count("{") - clean.count("}")
            open_brackets = clean.count("[") - clean.count("]")
            # Close any open string first
            if clean.count('"') % 2 != 0:
                clean += '"'
            # Close open arrays and objects
            clean += "]" * max(0, open_brackets)
            clean += "}" * max(0, open_braces)
            try:
                parsed = json.loads(clean)
            except:
                # Last resort: extract what we can with regex
                import re
                result = {}
                for field in ["black_market_rate","official_rate","spread_pct",
                              "prediction_direction","predicted_low","predicted_high",
                              "predicted_midpoint","confidence_score","executive_summary",
                              "trade_recommendation","accuracy_basis","weekly_outlook"]:
                    # Try number
                    m = re.search(rf'"{field}"\s*:\s*([0-9.]+)', clean)
                    if m:
                        result[field] = float(m.group(1))
                        continue
                    # Try string
                    m = re.search(rf'"{field}"\s*:\s*"([^"]+)', clean)
                    if m:
                        result[field] = m.group(1)
                parsed = result

        parsed["fetch_success"] = True
        parsed["raw_signals"] = signals
        parsed["rates"] = rates
        # Fill in defaults for any missing fields
        parsed.setdefault("prediction_direction", "NEUTRAL")
        parsed.setdefault("confidence_score", 50)
        parsed.setdefault("executive_summary", "Analysis partially available — response was truncated.")
        parsed.setdefault("key_drivers", [])
        parsed.setdefault("risk_factors", [])
        parsed.setdefault("trade_recommendation", "Insufficient data for recommendation.")
        parsed.setdefault("weekly_outlook", "N/A")
        parsed.setdefault("best_time_to_convert", "N/A")
        parsed.setdefault("black_market_premium_analysis", "N/A")
        parsed.setdefault("accuracy_basis", "N/A")
        parsed.setdefault("sources_analyzed", len(signals))
        parsed.setdefault("news_sentiment_score", 0)
        parsed.setdefault("oil_score", 0)
        parsed.setdefault("usd_strength_score", 0)
        parsed.setdefault("cbn_policy_score", 0)
        parsed.setdefault("crypto_sentiment_score", 0)
        parsed.setdefault("political_risk_score", 0)
        return parsed
    except Exception as e:
        return {
            "fetch_success": False,
            "error": "Could not parse AI response",
            "raw_response": raw,
            "rates": rates
        }


# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
def chat(msg: str, api_key: str, ctx: dict) -> str:
    ctx_str = ""
    if ctx:
        ctx_str = f"""
CURRENT ANALYSIS:
- Black Market Rate: ₦{ctx.get('black_market_rate',0):,}
- Official Rate: ₦{ctx.get('official_rate',0):,}
- Spread: {ctx.get('spread_pct',0):.1f}%
- Direction: {ctx.get('prediction_direction','N/A')}
- Range: ₦{ctx.get('predicted_low',0):,.0f}–₦{ctx.get('predicted_high',0):,.0f}
- Confidence: {ctx.get('confidence_score',0)}%
- Recommendation: {ctx.get('trade_recommendation','')}
- Weekly Outlook: {ctx.get('weekly_outlook','')}
- Executive Summary: {ctx.get('executive_summary','')}
"""
    hist = ""
    for m in st.session_state.chat[-8:]:
        hist += f"\n{'User' if m['r']=='u' else 'Oracle'}: {m['c']}"

    sys = """You are the USDT/NGN Oracle — Nigeria's sharpest AI FX analyst.
You speak like a market pro: direct, confident, no fluff. You know P2P rates, Binance dynamics, CBN policy, oil markets, and global macro cold.
Answer in 3-5 sentences unless a breakdown is asked for. Always give a real, specific answer — never hedge without reason.
If asked about the rate, give the number. If asked for advice, give it clearly with reasoning."""

    return gemini(f"{ctx_str}\n\nPrevious chat:{hist}\n\nUser: {msg}\n\nOracle:", api_key, sys)


# ─────────────────────────────────────────────
# CHECK ALERTS
# ─────────────────────────────────────────────
def send_email_alert(to_email: str, subject: str, html_body: str, resend_key: str) -> bool:
    """Send email alert via Resend.com API."""
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
    """Build a clean HTML email for price alerts."""
    dir_color = "#05d68a" if direction == "BULLISH" else "#f0455a" if direction == "BEARISH" else "#f5a623"
    dir_arrow = "▲" if direction == "BULLISH" else "▼" if direction == "BEARISH" else "◆"
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;
    background:#0c1220;color:#dce8f8;border-radius:16px;overflow:hidden;">
      <div style="background:linear-gradient(135deg,#111d2e,#0c1220);
      padding:28px 32px;border-bottom:2px solid #05d68a;">
        <div style="font-size:11px;letter-spacing:3px;color:#4f8ef7;
        text-transform:uppercase;margin-bottom:8px;">🇳🇬 USDT/NGN Oracle</div>
        <div style="font-size:24px;font-weight:700;color:#dce8f8;">
          Price Alert Triggered
        </div>
      </div>
      <div style="padding:28px 32px;">
        <div style="background:#111d2e;border:1px solid #1a2942;border-left:4px solid #f5a623;
        border-radius:10px;padding:16px 20px;margin-bottom:20px;font-size:16px;
        font-weight:600;color:#f5a623;">
          🔔 {msg}
        </div>
        <table style="width:100%;border-collapse:collapse;margin-bottom:20px;">
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;
            color:#6b84a0;font-size:13px;">AI Prediction</td>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;
            text-align:right;font-weight:700;color:{dir_color};">
              {dir_arrow} {direction}
            </td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;
            color:#6b84a0;font-size:13px;">Current Rate</td>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;
            text-align:right;font-weight:700;color:#05d68a;">
              ₦{rate:,.0f}
            </td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;
            color:#6b84a0;font-size:13px;">Predicted Range (24H)</td>
            <td style="padding:10px 0;border-bottom:1px solid #1a2942;
            text-align:right;font-weight:600;color:#dce8f8;">
              ₦{pred_low:,.0f} – ₦{pred_high:,.0f}
            </td>
          </tr>
          <tr>
            <td style="padding:10px 0;color:#6b84a0;font-size:13px;">
              AI Confidence</td>
            <td style="padding:10px 0;text-align:right;font-weight:700;
            color:#f5a623;">{confidence}%</td>
          </tr>
        </table>
        <div style="background:#111d2e;border:1px solid #1a2942;border-radius:10px;
        padding:16px 20px;margin-bottom:24px;">
          <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
          color:#4f8ef7;margin-bottom:8px;">AI Recommendation</div>
          <div style="font-size:13px;line-height:1.7;color:#b0c8e8;">
            {recommendation}
          </div>
        </div>
        <div style="font-size:11px;color:#4a6080;text-align:center;
        padding-top:16px;border-top:1px solid #1a2942;line-height:1.6;">
          ⚠️ This alert is for informational purposes only.<br>
          Not financial advice. Always do your own research.
        </div>
      </div>
    </div>
    """


def check_alerts(rate: float, prediction: dict = {}):
    """Check alerts and send email notifications if triggered."""
    triggered = []
    user_email = st.session_state.get("user_email", "")
    resend_key = ""
    try:
        resend_key = st.secrets.get("RESEND_API_KEY", "")
    except:
        pass

    for i, a in enumerate(st.session_state.alerts):
        msg = ""
        direction = prediction.get("prediction_direction", "N/A")
        confidence = prediction.get("confidence_score", 0)
        pred_low = prediction.get("predicted_low", 0)
        pred_high = prediction.get("predicted_high", 0)
        recommendation = prediction.get("trade_recommendation", "N/A")

        if a["type"] == "above" and rate > a["level"] and i not in st.session_state.alert_triggered:
            msg = f"Rate crossed ABOVE ₦{a['level']:,} — now at ₦{rate:,.0f}"
            triggered.append((i, "🔔 " + msg))
            st.session_state.alert_triggered.append(i)

        elif a["type"] == "below" and rate < a["level"] and i not in st.session_state.alert_triggered:
            msg = f"Rate dropped BELOW ₦{a['level']:,} — now at ₦{rate:,.0f}"
            triggered.append((i, "🔔 " + msg))
            st.session_state.alert_triggered.append(i)

        if msg and user_email and resend_key:
            subject = f"🔔 USDT/NGN Alert: {msg}"
            html = build_email_html(
                msg, rate, direction, confidence,
                pred_low, pred_high, recommendation
            )
            send_email_alert(user_email, subject, html, resend_key)

    return triggered


# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
def score_bar(label, score, color):
    norm = (score + 100) / 2
    st.markdown(f"""
    <div class="prog-wrap">
      <div class="prog-label">
        <span style="color:var(--muted2);font-size:12px;">{label}</span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{color};">{score:+d}</span>
      </div>
      <div class="prog-track">
        <div class="prog-fill" style="width:{norm}%;background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── ACTION BAR (top of main page) ──
# ─────────────────────────────────────────────
act_c1, act_c2, act_c3, act_c4 = st.columns([2, 1.5, 1.5, 6])
with act_c1:
    run_btn = st.button("🔍 Run Full Analysis", use_container_width=True, type="primary")
with act_c2:
    auto_refresh = st.toggle("Auto-refresh", value=st.session_state.auto_refresh, key="auto_refresh_toggle")
    st.session_state.auto_refresh = auto_refresh
with act_c3:
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Interval",
            options=[15, 30, 60, 120, 180],
            index=[15, 30, 60, 120, 180].index(st.session_state.refresh_interval),
            format_func=lambda x: f"{x}m",
            label_visibility="collapsed"
        )
        st.session_state.refresh_interval = refresh_interval
with act_c4:
    if st.session_state.last_time:
        elapsed = int((datetime.datetime.now() - st.session_state.last_time).total_seconds() // 60)
        status_text = f'<span class="live-dot"></span> Updated {elapsed}m ago'
        if auto_refresh:
            status_text += f' · Auto-refresh every {st.session_state.refresh_interval}m'
        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            f'color:var(--muted2);margin:10px 0 0 0;text-align:right;">{status_text}</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            'color:var(--muted);margin:10px 0 0 0;text-align:right;">Ready to analyze</p>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# (Sidebar removed — controls now in action bar + Alerts tab)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# ── RUN ANALYSIS ──
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching black market rate · Gathering global signals · Running AI analysis..."):
        rates = fetch_rates()
        result = run_analysis(GEMINI_KEY, NEWS_KEY, rates)
        result["rate_data"] = rates
        st.session_state.result = result
        # Save previous rate for change tracking
        if st.session_state.result and st.session_state.result.get("fetch_success"):
            st.session_state.prev_rate = st.session_state.result.get("black_market_rate", None)
        st.session_state.last_time = datetime.datetime.now()
        if result.get("fetch_success"):
            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%H:%M"),
                "date": datetime.datetime.now().strftime("%d/%m"),
                "rate": rates.get("primary", 0),
                "dir": result.get("prediction_direction", "N/A"),
                "conf": result.get("confidence_score", 0),
                "p2p_buy": rates.get("p2p_buy", 0),
                "p2p_sell": rates.get("p2p_sell", 0),
            })
    st.rerun()

# ── AUTO REFRESH TRIGGER ──
if st.session_state.auto_refresh and st.session_state.last_time and GEMINI_KEY:
    elapsed_sec = (datetime.datetime.now() - st.session_state.last_time).total_seconds()
    interval_sec = st.session_state.refresh_interval * 60
    if elapsed_sec >= interval_sec:
        with st.spinner(f"Auto-refreshing... (every {st.session_state.refresh_interval} min)"):
            rates = fetch_rates()
            if st.session_state.result and st.session_state.result.get("fetch_success"):
                st.session_state.prev_rate = st.session_state.result.get("black_market_rate", None)
            result = run_analysis(GEMINI_KEY, NEWS_KEY, rates)
            result["rate_data"] = rates
            st.session_state.result = result
            st.session_state.last_time = datetime.datetime.now()
            if result.get("fetch_success"):
                st.session_state.history.append({
                    "time": datetime.datetime.now().strftime("%H:%M"),
                    "date": datetime.datetime.now().strftime("%d/%m"),
                    "rate": rates.get("primary", 0),
                    "dir": result.get("prediction_direction", "N/A"),
                    "conf": result.get("confidence_score", 0),
                    "p2p_buy": rates.get("p2p_buy", 0),
                    "p2p_sell": rates.get("p2p_sell", 0),
                })
        st.rerun()
    else:
        # Show time since last refresh — no sleep, no blocking
        remaining = int((interval_sec - elapsed_sec) // 60)
        secs = int((interval_sec - elapsed_sec) % 60)
        st.markdown(
            f'<p style="font-size:10px;color:var(--green);text-align:right;">🔄 Next auto-refresh in {remaining}m {secs}s</p>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# ── HEADER ──
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:8px 0 18px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--blue);
  letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">
    <span class="live-dot"></span>REAL-TIME MARKET INTELLIGENCE
  </div>
  <h1 style="font-family:'IBM Plex Mono',monospace;font-size:30px;font-weight:700;
  margin:0 0 4px;background:linear-gradient(135deg,#dce8f8,#6b84a0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    USDT / NGN Oracle
  </h1>
  <p style="color:var(--muted2);font-size:13px;margin:0;">
    Black market · P2P · Global macro · AI-powered · Gemini 2.5
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ── MAIN CONTENT ──
# ─────────────────────────────────────────────
if st.session_state.result:
    p = st.session_state.result
    rd = p.get("rates", {})

    # ── ALERT BANNERS ──
    if rd.get("primary"):
        triggered = check_alerts(rd["primary"], p)
        for _, msg in triggered:
            st.markdown(f'<div class="alert-box alert-warn">{msg}</div>', unsafe_allow_html=True)

    if not p.get("fetch_success"):
        st.error(f"Analysis failed: {p.get('error','Unknown error')}")
        with st.expander("Debug: Raw AI Response"):
            st.text(p.get("raw_response", "")[:3000])
    else:
        direction = p.get("prediction_direction", "NEUTRAL")
        conf = p.get("confidence_score", 0)
        bm_rate = p.get("black_market_rate", rd.get("primary", 0))
        official = p.get("official_rate", rd.get("official", 0))
        spread = p.get("spread_pct", rd.get("spread_pct", 0))
        pred_low = p.get("predicted_low", 0)
        pred_high = p.get("predicted_high", 0)
        pred_mid = p.get("predicted_midpoint", 0)

        # ── TICKER BAR ──
        items = [
            f'<span class="ticker-item">USDT/NGN (P2P) <span class="val">₦{bm_rate:,.0f}</span></span>',
            f'<span class="ticker-item">OFFICIAL <span class="val">₦{official:,.0f}</span></span>',
            f'<span class="ticker-item">SPREAD <span class="{"up" if spread>0 else "dn"}">+{spread:.1f}%</span></span>',
            f'<span class="ticker-item">24H TARGET <span class="{"up" if direction=="BULLISH" else "dn" if direction=="BEARISH" else "val"}">₦{pred_mid:,.0f}</span></span>',
            f'<span class="ticker-item">CONFIDENCE <span class="val">{conf}%</span></span>',
            f'<span class="ticker-item">DIRECTION <span class="{"up" if direction=="BULLISH" else "dn" if direction=="BEARISH" else "val"}">{direction}</span></span>',
            f'<span class="ticker-item">SIGNALS <span class="val">{p.get("sources_analyzed",0)}</span></span>',
        ]
        ticker_html = "".join(items * 2)
        st.markdown(f'<div class="ticker-wrap"><div class="ticker-inner">{ticker_html}</div></div>', unsafe_allow_html=True)

        # ── TOP METRIC CARDS ──
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            p2p_buy = rd.get("p2p_buy")
            p2p_sell = rd.get("p2p_sell")
            p2p_detail = f"Buy ₦{p2p_buy:,.0f} · Sell ₦{p2p_sell:,.0f}" if p2p_buy and p2p_sell else rd.get("black_market_source","P2P")
            prev = st.session_state.prev_rate
            if prev and prev > 0:
                chg = bm_rate - prev
                chg_pct = (chg / prev) * 100
                chg_color = "var(--green)" if chg >= 0 else "var(--red)"
                chg_arrow = "▲" if chg >= 0 else "▼"
                chg_str = f'<span style="color:{chg_color};font-size:12px;">{chg_arrow} ₦{abs(chg):,.1f} ({chg_pct:+.2f}%) since last run</span>'
            else:
                chg_str = '<span style="font-size:11px;color:var(--muted);">First analysis run</span>'
            st.markdown(f"""<div class="mcard mcard-green">
            <div class="mcard-label">Black Market Rate (P2P)</div>
            <div class="mcard-value" style="color:var(--green);">₦{bm_rate:,.0f}</div>
            <div class="mcard-sub" style="font-size:11px;">{p2p_detail}</div>
            <div style="margin-top:6px;">{chg_str}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="mcard mcard-blue">
            <div class="mcard-label">Official Rate</div>
            <div class="mcard-value" style="color:var(--blue);">₦{official:,.0f}</div>
            <div class="mcard-sub">{rd.get("source_official","CBN/NAFEX")}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            sc = "var(--red)" if spread > 10 else "var(--amber)" if spread > 5 else "var(--green)"
            st.markdown(f"""<div class="mcard mcard-amber">
            <div class="mcard-label">B.Market Premium</div>
            <div class="mcard-value" style="color:{sc};">+{spread:.1f}%</div>
            <div class="mcard-sub">Over official rate</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            dc = "var(--green)" if direction=="BULLISH" else "var(--red)" if direction=="BEARISH" else "var(--amber)"
            darr = "▲" if direction=="BULLISH" else "▼" if direction=="BEARISH" else "◆"
            st.markdown(f"""<div class="mcard mcard-{'green' if direction=='BULLISH' else 'red' if direction=='BEARISH' else 'amber'}">
            <div class="mcard-label">24H Prediction</div>
            <div class="mcard-value" style="color:{dc};">{darr} {direction}</div>
            <div class="mcard-sub">₦{pred_low:,.0f} – ₦{pred_high:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            cc = "var(--green)" if conf >= 65 else "var(--amber)" if conf >= 45 else "var(--red)"
            st.markdown(f"""<div class="mcard mcard-purple">
            <div class="mcard-label">AI Confidence</div>
            <div class="mcard-value" style="color:{cc};">{conf}%</div>
            <div class="mcard-sub">{p.get("sources_analyzed",0)} signals analyzed</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TABS ──
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Analysis", "🌍 Global Signals", "💱 Converter", "📈 History", "💬 Chat", "🔔 Alerts"
        ])

        # ══════ TAB 1: ANALYSIS ══════
        with tab1:
            left, right = st.columns([3, 2])

            with left:
                # Summary card
                badge_cls = "dir-bull" if direction=="BULLISH" else "dir-bear" if direction=="BEARISH" else "dir-neu"
                darr2 = "▲" if direction=="BULLISH" else "▼" if direction=="BEARISH" else "◆"
                st.markdown(f"""
                <div class="ocard">
                  <div class="ocard-title">Market Outlook</div>
                  <span class="dir-badge {badge_cls}">{darr2} {direction} — ₦{pred_mid:,.0f} target</span>
                  <p style="color:#b0c8e8;line-height:1.75;font-size:14px;margin:0 0 14px;">
                    {p.get("executive_summary","")}
                  </p>
                  <div style="background:var(--bg2);border-radius:10px;padding:14px 16px;border:1px solid var(--border2);">
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--blue);margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">TRADE RECOMMENDATION</div>
                    <p style="margin:0;font-size:13px;color:var(--text);line-height:1.6;">{p.get("trade_recommendation","")}</p>
                  </div>
                  <div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;">
                    <div style="background:var(--bg2);border-radius:8px;padding:10px 14px;border:1px solid var(--border);flex:1;">
                      <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'IBM Plex Mono',monospace;">Best Convert Time</div>
                      <div style="font-size:13px;color:var(--amber);margin-top:4px;">{p.get("best_time_to_convert","N/A")}</div>
                    </div>
                    <div style="background:var(--bg2);border-radius:8px;padding:10px 14px;border:1px solid var(--border);flex:1;">
                      <div style="font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'IBM Plex Mono',monospace;">Weekly Outlook</div>
                      <div style="font-size:13px;color:var(--text);margin-top:4px;">{p.get("weekly_outlook","N/A")}</div>
                    </div>
                  </div>
                  <p style="margin:12px 0 0;font-size:11px;color:var(--muted);">
                    <span style="font-family:'IBM Plex Mono',monospace;color:var(--amber);">{conf}% confidence</span> — {p.get("accuracy_basis","")}
                  </p>
                </div>""", unsafe_allow_html=True)

                # Black market premium analysis
                st.markdown(f"""
                <div class="ocard">
                  <div class="ocard-title">Black Market Premium Analysis</div>
                  <p style="font-size:13px;color:#b0c8e8;line-height:1.7;margin:0;">{p.get("black_market_premium_analysis","")}</p>
                </div>""", unsafe_allow_html=True)

                # Key drivers
                st.markdown('<div class="ocard"><div class="ocard-title">Key Signal Drivers</div>', unsafe_allow_html=True)
                for d in p.get("key_drivers", []):
                    impact = d.get("impact","NEUTRAL")
                    wt = d.get("weight","MEDIUM")
                    tc = "tag-bull" if impact=="BULLISH" else "tag-bear" if impact=="BEARISH" else "tag-neu"
                    wc = "tag-hi" if wt=="HIGH" else "tag-med" if wt=="MEDIUM" else "tag-lo"
                    st.markdown(f"""
                    <div class="sig-row">
                      <div class="sig-tags">
                        <span class="tag {tc}">{impact}</span>
                        <span class="tag {wc}">{wt}</span>
                        <span class="tag" style="background:rgba(255,255,255,0.04);color:var(--muted2);">{d.get("category","")}</span>
                        <span class="sig-name">{d.get("signal","")}</span>
                      </div>
                      <div class="sig-detail">{d.get("detail","")}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                # Score breakdown
                st.markdown('<div class="ocard"><div class="ocard-title">Signal Scores</div>', unsafe_allow_html=True)
                scores = [
                    ("News Sentiment", p.get("news_sentiment_score",0)),
                    ("Oil Markets", p.get("oil_score",0)),
                    ("USD Strength", p.get("usd_strength_score",0)),
                    ("CBN Policy", p.get("cbn_policy_score",0)),
                    ("Crypto Sentiment", p.get("crypto_sentiment_score",0)),
                    ("Political Risk", p.get("political_risk_score",0)),
                ]
                for lbl, sc in scores:
                    color = "var(--green)" if sc > 20 else "var(--red)" if sc < -20 else "var(--amber)"
                    score_bar(lbl, sc, color)
                st.markdown('</div>', unsafe_allow_html=True)

                # Risk factors
                risks = p.get("risk_factors", [])
                if risks:
                    st.markdown('<div class="ocard"><div class="ocard-title">⚠️ Risk Factors</div>', unsafe_allow_html=True)
                    for r in risks:
                        st.markdown(f'<div class="alert-box alert-warn" style="margin-bottom:8px;">⚡ {r}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Rate comparison table
                st.markdown("""
                <div class="ocard">
                  <div class="ocard-title">Rate Comparison</div>
                  <table class="spread-table">
                    <tr><th>Market</th><th>Rate (₦)</th><th>Status</th></tr>""", unsafe_allow_html=True)

                rows = [
                    ("P2P / Black Market", bm_rate, "PRIMARY", "var(--green)"),
                    ("Official CBN/NAFEX", official, "REFERENCE", "var(--blue)"),
                    ("24H Predicted Low", pred_low, "FORECAST", "var(--muted2)"),
                    ("24H Predicted High", pred_high, "FORECAST", "var(--muted2)"),
                ]
                for name, val, status, color in rows:
                    st.markdown(f'<tr><td>{name}</td><td style="font-family:\'IBM Plex Mono\',monospace;color:{color};">₦{val:,.0f}</td><td style="font-size:11px;color:var(--muted);">{status}</td></tr>', unsafe_allow_html=True)
                st.markdown('</table></div>', unsafe_allow_html=True)

        # ══════ TAB 2: GLOBAL SIGNALS ══════
        with tab2:
            signals = p.get("raw_signals", [])
            categories = list(dict.fromkeys(s.get("category","") for s in signals))

            for cat in categories:
                cat_signals = [s for s in signals if s.get("category") == cat]
                st.markdown(f'<div class="ocard"><div class="ocard-title">📡 {cat}</div>', unsafe_allow_html=True)
                for s in cat_signals:
                    impact = s.get("impact","NEUTRAL")
                    tc = "tag-bull" if impact=="BULLISH" else "tag-bear" if impact=="BEARISH" else "tag-neu" if impact in ("NEUTRAL","UNKNOWN") else "tag-neu"
                    pub = s.get("published","")[:10] if s.get("published") else ""
                    src = s.get("source","")
                    url = s.get("url","")
                    link = f'<a href="{url}" target="_blank" style="color:var(--blue);font-size:11px;">→ Read</a>' if url else ""
                    st.markdown(f"""
                    <div class="sig-row">
                      <div class="sig-tags">
                        <span class="tag {tc}">{impact}</span>
                        <span style="font-size:11px;color:var(--muted);">{src}{' · '+pub if pub else ''}</span>
                        {link}
                      </div>
                      <div class="sig-name" style="font-size:13px;margin-bottom:4px;">{s.get("title","")}</div>
                      <div class="sig-detail">{s.get("detail","")[:250]}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ══════ TAB 3: CONVERTER ══════
        with tab3:
            st.markdown('<div class="ocard"><div class="ocard-title">💱 Currency Converter</div>', unsafe_allow_html=True)

            conv_dir = st.radio("Convert:", ["USDT → NGN", "NGN → USDT"], horizontal=True)
            amt = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
            rate_choice = st.radio("Use rate:", ["Black Market (P2P)", "Official Rate", "Predicted Midpoint"], horizontal=True)

            rate_map = {
                "Black Market (P2P)": bm_rate,
                "Official Rate": official,
                "Predicted Midpoint": pred_mid
            }
            used_rate = rate_map[rate_choice]

            if conv_dir == "USDT → NGN":
                converted = amt * used_rate
                st.markdown(f"""
                <div class="conv-box" style="margin-top:12px;">
                  <div style="text-align:center;color:var(--muted2);font-size:13px;margin-bottom:6px;">{amt:,.2f} USDT at ₦{used_rate:,.2f}</div>
                  <div class="conv-result">₦{converted:,.2f}</div>
                  <div style="text-align:center;font-size:11px;color:var(--muted);margin-top:8px;">Using {rate_choice} rate</div>
                </div>""", unsafe_allow_html=True)
            else:
                converted = amt / used_rate if used_rate else 0
                st.markdown(f"""
                <div class="conv-box" style="margin-top:12px;">
                  <div style="text-align:center;color:var(--muted2);font-size:13px;margin-bottom:6px;">₦{amt:,.2f} NGN at ₦{used_rate:,.2f}/USDT</div>
                  <div class="conv-result">{converted:,.4f} USDT</div>
                  <div style="text-align:center;font-size:11px;color:var(--muted);margin-top:8px;">Using {rate_choice} rate</div>
                </div>""", unsafe_allow_html=True)

            # Rate comparison for converter
            st.markdown(f"""
            <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
              <div style="background:var(--bg2);border-radius:8px;padding:12px;text-align:center;border:1px solid var(--border);">
                <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">If Black Market</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--green);margin-top:4px;">
                  {'₦'+f'{amt*bm_rate:,.0f}' if conv_dir=='USDT → NGN' else f'{amt/bm_rate:.4f} USDT'}
                </div>
              </div>
              <div style="background:var(--bg2);border-radius:8px;padding:12px;text-align:center;border:1px solid var(--border);">
                <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">If Official</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--blue);margin-top:4px;">
                  {'₦'+f'{amt*official:,.0f}' if conv_dir=='USDT → NGN' else f'{amt/official:.4f} USDT'}
                </div>
              </div>
              <div style="background:var(--bg2);border-radius:8px;padding:12px;text-align:center;border:1px solid var(--border);">
                <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">If Predicted</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:var(--amber);margin-top:4px;">
                  {'₦'+f'{amt*pred_mid:,.0f}' if conv_dir=='USDT → NGN' else f'{amt/pred_mid:.4f} USDT' if pred_mid else 'N/A'}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ══════ TAB 4: HISTORY ══════
        with tab4:
            hist = st.session_state.history
            if len(hist) < 2:
                st.markdown("""
                <div class="ocard" style="text-align:center;padding:40px;">
                  <div style="font-size:32px;margin-bottom:12px;opacity:0.3;">📈</div>
                  <p style="color:var(--muted2);">Run analysis at least twice to build your rate history chart.<br>
                  Each run is tracked and plotted automatically.</p>
                </div>""", unsafe_allow_html=True)
            else:
                import json as _json

                # ── RATE HISTORY CHART (pure HTML/JS with Chart.js) ──
                labels = [f"{h['date']} {h['time']}" for h in hist]
                rates_data = [h["rate"] for h in hist]
                buy_data = [h.get("p2p_buy", 0) for h in hist]
                sell_data = [h.get("p2p_sell", 0) for h in hist]

                chart_html = f"""
                <div class="ocard">
                  <div class="ocard-title">Rate History Chart</div>
                  <canvas id="rateChart" style="max-height:300px;"></canvas>
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
                          label: 'Black Market Rate',
                          data: {_json.dumps(rates_data)},
                          borderColor: '#05d68a',
                          backgroundColor: 'rgba(5,214,138,0.08)',
                          borderWidth: 2,
                          pointRadius: 4,
                          pointBackgroundColor: '#05d68a',
                          tension: 0.3,
                          fill: true
                        }},
                        {{
                          label: 'P2P Buy Price',
                          data: {_json.dumps(buy_data)},
                          borderColor: '#4f8ef7',
                          backgroundColor: 'transparent',
                          borderWidth: 1.5,
                          borderDash: [5,3],
                          pointRadius: 2,
                          tension: 0.3
                        }},
                        {{
                          label: 'P2P Sell Price',
                          data: {_json.dumps(sell_data)},
                          borderColor: '#f0455a',
                          backgroundColor: 'transparent',
                          borderWidth: 1.5,
                          borderDash: [5,3],
                          pointRadius: 2,
                          tension: 0.3
                        }}
                      ]
                    }},
                    options: {{
                      responsive: true,
                      plugins: {{
                        legend: {{
                          labels: {{ color: '#6b84a0', font: {{ size: 11 }} }}
                        }},
                        tooltip: {{
                          backgroundColor: '#111d2e',
                          titleColor: '#dce8f8',
                          bodyColor: '#6b84a0',
                          borderColor: '#1a2942',
                          borderWidth: 1,
                          callbacks: {{
                            label: function(ctx) {{
                              return ctx.dataset.label + ': ₦' + ctx.parsed.y.toLocaleString();
                            }}
                          }}
                        }}
                      }},
                      scales: {{
                        x: {{
                          ticks: {{ color: '#4a6080', font: {{ size: 10 }} }},
                          grid: {{ color: '#1a2942' }}
                        }},
                        y: {{
                          ticks: {{
                            color: '#4a6080',
                            font: {{ size: 10 }},
                            callback: function(v) {{ return '₦' + v.toLocaleString(); }}
                          }},
                          grid: {{ color: '#1a2942' }}
                        }}
                      }}
                    }}
                  }});
                </script>
                """
                st.components.v1.html(chart_html, height=360)

                # ── P2P SPREAD CHART ──
                spread_data = []
                for h in hist:
                    buy = h.get("p2p_buy", 0)
                    sell = h.get("p2p_sell", 0)
                    spread_data.append(round(buy - sell, 2) if buy and sell else 0)

                if any(s > 0 for s in spread_data):
                    spread_html = f"""
                    <div class="ocard" style="margin-top:16px;">
                      <div class="ocard-title">Binance P2P Spread History (Buy − Sell)</div>
                      <canvas id="spreadChart" style="max-height:180px;"></canvas>
                    </div>
                    <script>
                      const ctx2 = document.getElementById('spreadChart').getContext('2d');
                      new Chart(ctx2, {{
                        type: 'bar',
                        data: {{
                          labels: {_json.dumps(labels)},
                          datasets: [{{
                            label: 'P2P Spread (₦)',
                            data: {_json.dumps(spread_data)},
                            backgroundColor: 'rgba(245,166,35,0.3)',
                            borderColor: '#f5a623',
                            borderWidth: 1,
                            borderRadius: 4
                          }}]
                        }},
                        options: {{
                          responsive: true,
                          plugins: {{
                            legend: {{ labels: {{ color: '#6b84a0', font: {{ size: 11 }} }} }},
                            tooltip: {{
                              backgroundColor: '#111d2e',
                              titleColor: '#dce8f8',
                              bodyColor: '#6b84a0',
                              callbacks: {{
                                label: function(ctx) {{
                                  return 'Spread: ₦' + ctx.parsed.y.toLocaleString();
                                }}
                              }}
                            }}
                          }},
                          scales: {{
                            x: {{ ticks: {{ color: '#4a6080', font: {{ size: 10 }} }}, grid: {{ color: '#1a2942' }} }},
                            y: {{ ticks: {{ color: '#4a6080', font: {{ size: 10 }}, callback: function(v) {{ return '₦' + v; }} }}, grid: {{ color: '#1a2942' }} }}
                          }}
                        }}
                      }});
                    </script>
                    """
                    st.components.v1.html(spread_html, height=240)

                # ── HISTORY TABLE ──
                st.markdown('<div class="ocard"><div class="ocard-title">Full History Log</div>', unsafe_allow_html=True)
                st.markdown("""<table class="spread-table">
                <tr><th>Date</th><th>Time</th><th>Rate</th><th>P2P Buy</th><th>P2P Sell</th><th>Spread</th><th>Direction</th><th>Confidence</th></tr>""", unsafe_allow_html=True)
                for h in reversed(hist):
                    dc = "var(--green)" if h["dir"]=="BULLISH" else "var(--red)" if h["dir"]=="BEARISH" else "var(--amber)"
                    da = "▲" if h["dir"]=="BULLISH" else "▼" if h["dir"]=="BEARISH" else "◆"
                    buy = h.get("p2p_buy", 0)
                    sell = h.get("p2p_sell", 0)
                    sprd = f"₦{buy-sell:,.0f}" if buy and sell else "—"
                    st.markdown(f"""<tr>
                    <td style="color:var(--muted2);">{h['date']}</td>
                    <td style="font-family:'IBM Plex Mono',monospace;">{h['time']}</td>
                    <td style="font-family:'IBM Plex Mono',monospace;color:var(--green);">₦{h['rate']:,.0f}</td>
                    <td style="font-family:'IBM Plex Mono',monospace;color:var(--blue);">{f"₦{buy:,.0f}" if buy else "—"}</td>
                    <td style="font-family:'IBM Plex Mono',monospace;color:var(--red);">{f"₦{sell:,.0f}" if sell else "—"}</td>
                    <td style="font-family:'IBM Plex Mono',monospace;color:var(--amber);">{sprd}</td>
                    <td style="color:{dc};font-weight:600;">{da} {h['dir']}</td>
                    <td style="font-family:'IBM Plex Mono',monospace;color:var(--amber);">{h['conf']}%</td>
                    </tr>""", unsafe_allow_html=True)
                st.markdown('</table></div>', unsafe_allow_html=True)

        # ══════ TAB 5: CHAT ══════
        with tab5:
            st.markdown('<div class="ocard"><div class="ocard-title">💬 Ask the Oracle Anything</div>', unsafe_allow_html=True)

            if not st.session_state.chat:
                st.markdown("""
                <div class="chat-a">
                  <div class="chat-badge">⬡ ORACLE</div>
                  I'm your USDT/NGN Oracle. Ask me anything — current rate outlook, when to convert, what's driving the market, CBN news, oil impact, or how to read the signals. I'm grounded in the latest analysis data.
                </div>""", unsafe_allow_html=True)

            for m in st.session_state.chat:
                if m["r"] == "u":
                    st.markdown(f'<div class="chat-u">🧑 {m["c"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-a"><div class="chat-badge">⬡ ORACLE</div>{m["c"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Quick prompts
            st.markdown('<p style="font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-top:12px;">Quick Questions</p>', unsafe_allow_html=True)
            qcols = st.columns(2)
            qs = [
                "Should I convert my USDT to naira today?",
                "What's the best time this week to convert?",
                "How is oil price affecting the naira right now?",
                "What would make the naira strengthen suddenly?"
            ]
            clicked = None
            for i, q in enumerate(qs):
                with qcols[i % 2]:
                    if st.button(q, key=f"q{i}", use_container_width=True):
                        clicked = q

            col1, col2 = st.columns([5, 1])
            with col1:
                user_msg = st.text_input("msg", placeholder="Type your question here...", label_visibility="collapsed", key="chat_in")
            with col2:
                send = st.button("Send →", use_container_width=True)

            question = user_msg if (send and user_msg) else clicked

            if question and GEMINI_KEY:
                st.session_state.chat.append({"r": "u", "c": question})
                with st.spinner("Oracle thinking..."):
                    reply = chat(question, GEMINI_KEY, p)
                st.session_state.chat.append({"r": "a", "c": reply})
                st.rerun()
            elif question and not GEMINI_KEY:
                st.warning("Gemini API key not configured. Please add it to your Streamlit secrets.")

            if st.session_state.chat:
                if st.button("🗑 Clear Chat"):
                    st.session_state.chat = []
                    st.rerun()

        # ══════ TAB 6: ALERTS ══════
        with tab6:
            alert_left, alert_right = st.columns([1, 1])

            with alert_left:
                st.markdown("""
                <div class="ocard">
                  <div class="ocard-title">📧 Email Alerts</div>
                  <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin-bottom:12px;">
                    Enter your email to receive notifications when the rate crosses your target price.
                  </p>
                </div>""", unsafe_allow_html=True)

                user_email = st.text_input(
                    "Your Email Address",
                    value=st.session_state.user_email,
                    placeholder="yourname@gmail.com",
                    help="Enter the email where you want to receive price alerts",
                    key="alert_email_input"
                )
                st.session_state.user_email = user_email

                if user_email:
                    if st.button("🧪 Send Test Email", use_container_width=True, key="test_email_btn"):
                        try:
                            resend_key = st.secrets.get("RESEND_API_KEY", "")
                        except Exception:
                            resend_key = ""
                        if resend_key:
                            test_html = build_email_html(
                                "This is a test alert — your email is connected!",
                                1650.0, "NEUTRAL", 70, 1640.0, 1660.0,
                                "This is a test. Real alerts will include live AI predictions."
                            )
                            ok = send_email_alert(user_email, "✅ USDT/NGN Oracle — Test Alert", test_html, resend_key)
                            if ok:
                                st.success("✅ Test email sent! Check your inbox (and spam folder).")
                            else:
                                st.error("❌ Failed to send. Check your email address.")
                        else:
                            st.error("❌ Email service not configured on this deployment.")
                else:
                    st.caption("Enter your email above to get alerted when your target price is hit.")

            with alert_right:
                st.markdown("""
                <div class="ocard">
                  <div class="ocard-title">🔔 Price Alerts</div>
                  <p style="font-size:12px;color:var(--muted2);line-height:1.6;margin-bottom:12px;">
                    Set alerts to be notified when the USDT/NGN rate crosses a specific level.
                  </p>
                </div>""", unsafe_allow_html=True)

                a_level = st.number_input("Alert price (₦)", min_value=100.0, max_value=9999.0,
                                           value=1700.0, step=10.0, key="alert_price_input")
                a_type = st.selectbox("Alert when rate goes:", ["above", "below"], key="alert_type_select")
                if st.button("+ Add Alert", use_container_width=True, key="add_alert_btn"):
                    st.session_state.alerts.append({"level": a_level, "type": a_type})
                    em_icon = "📧" if user_email else "🔕"
                    st.success(f"Alert set: {em_icon} notify when rate goes {a_type} ₦{a_level:,.0f}")

                if st.session_state.alerts:
                    st.markdown('<div class="ocard"><div class="ocard-title">Active Alerts</div>', unsafe_allow_html=True)
                    for i, a in enumerate(st.session_state.alerts):
                        acol1, acol2 = st.columns([4, 1])
                        with acol1:
                            em_icon = "📧" if user_email else "🔕"
                            arrow = "\u25b2" if a["type"] == "above" else "\u25bc"
                            st.markdown(
                                f'<span style="font-size:13px;color:var(--text);">'
                                f'{em_icon} {arrow} ₦{a["level"]:,.0f} ({a["type"]})</span>',
                                unsafe_allow_html=True
                            )
                        with acol2:
                            if st.button("✕", key=f"del_alert_{i}"):
                                st.session_state.alerts.pop(i)
                                st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="ocard" style="text-align:center;padding:24px;">
                      <div style="font-size:24px;margin-bottom:8px;opacity:0.3;">🔔</div>
                      <p style="color:var(--muted2);font-size:12px;">No active alerts. Add one above to get started.</p>
                    </div>""", unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div style="font-size:10px;color:var(--muted);margin-top:16px;line-height:1.7;
            padding:14px 16px;border-top:1px solid var(--border);text-align:center;">
              ⚠️ Not financial advice. AI predictions carry uncertainty. Always DYOR before converting.
            </div>""", unsafe_allow_html=True)

# ── EMPTY STATE ──
else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 40px;">

      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--blue);
      letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;">
        <span class="live-dot"></span>NIGERIA FX INTELLIGENCE
      </div>

      <div style="font-family:'IBM Plex Mono',monospace;font-size:42px;font-weight:700;
      background:linear-gradient(135deg,#dce8f8,#4f8ef7);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;margin-bottom:8px;line-height:1.1;">
        USDT / NGN Oracle
      </div>

      <p style="color:var(--muted2);max-width:560px;margin:0 auto 12px;
      line-height:1.8;font-size:15px;">
        Your real-time AI-powered edge on the Nigerian USDT market.
        Live black market & P2P rates, global macro signals, and
        Gemini AI predictions — all in one place.
      </p>

      <p style="color:var(--muted);max-width:480px;margin:0 auto 36px;
      font-size:13px;line-height:1.7;">
        Click <strong style="color:var(--green);">Run Full Analysis</strong> above
        to get your latest 24-hour prediction, confidence score, and trade recommendation.
      </p>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
      max-width:640px;margin:0 auto 40px;">
        <div style="background:var(--card);border:1px solid var(--border);
        border-radius:14px;padding:22px 14px;border-top:2px solid var(--green);">
          <div style="font-size:28px;margin-bottom:8px;">📡</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;">Live P2P Rates</div>
          <div style="font-size:11px;color:var(--muted);">Binance P2P · Bybit · CoinGecko</div>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);
        border-radius:14px;padding:22px 14px;border-top:2px solid var(--blue);">
          <div style="font-size:28px;margin-bottom:8px;">🌍</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;">Global Signals</div>
          <div style="font-size:11px;color:var(--muted);">Oil · USD · CBN · Crypto · Politics</div>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);
        border-radius:14px;padding:22px 14px;border-top:2px solid var(--purple);">
          <div style="font-size:28px;margin-bottom:8px;">🤖</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;">AI Prediction</div>
          <div style="font-size:11px;color:var(--muted);">Gemini 2.5 · 24H forecast · Confidence score</div>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);
        border-radius:14px;padding:22px 14px;border-top:2px solid var(--amber);">
          <div style="font-size:28px;margin-bottom:8px;">💱</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;">Smart Converter</div>
          <div style="font-size:11px;color:var(--muted);">USDT ↔ NGN · Compare all rates</div>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);
        border-radius:14px;padding:22px 14px;border-top:2px solid var(--red);">
          <div style="font-size:28px;margin-bottom:8px;">🔔</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;">Price Alerts</div>
          <div style="font-size:11px;color:var(--muted);">Get notified when rate hits your target</div>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);
        border-radius:14px;padding:22px 14px;border-top:2px solid var(--green);">
          <div style="font-size:28px;margin-bottom:8px;">💬</div>
          <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px;">Ask the Oracle</div>
          <div style="font-size:11px;color:var(--muted);">Chat with AI about the market anytime</div>
        </div>
      </div>

      <div style="background:var(--card);border:1px solid var(--border2);border-radius:12px;
      padding:16px 24px;max-width:480px;margin:0 auto;display:flex;align-items:center;gap:12px;">
        <div style="font-size:22px;">⚠️</div>
        <div style="font-size:11px;color:var(--muted);text-align:left;line-height:1.6;">
          For informational purposes only. AI predictions carry uncertainty.
          Always do your own research before converting funds.
        </div>
      </div>

    </div>""", unsafe_allow_html=True)
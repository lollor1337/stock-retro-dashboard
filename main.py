import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from textblob import TextBlob
import finnhub
from datetime import datetime, timedelta
import numpy as np
import time

# --- Konfiguration ---
st.set_page_config(page_title="Profi Aktien-Analyse 41.4 (Unstoppable)", layout="wide")

# --- RETRO CSS INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Fira+Code:wght@500&display=swap');

    .stApp {
        background-color: #D499FF;
        font-family: 'Fira Code', monospace;
        color: #000000;
    }

    h1, h2, h3, .stMetricLabel, .stMarkdown h5 {
        font-family: 'VT323', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #000000 !important;
        text-shadow: 2px 2px 0px #FFFFFF;
    }
    h1 { font-size: 4rem !important; color: #FFFF00 !important; text-shadow: 3px 3px 0px #000000 !important; }
    h2, h3 { font-size: 2rem !important; }

    div[data-testid="stMetric"], .stTabs div[data-baseweb="tab-panel"], .stExpander {
        background-color: #FFFFFF;
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px 0px #000000;
        border-radius: 0px;
        padding: 15px !important;
        margin-bottom: 10px;
    }
    
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stRadio label {
        background-color: #FFFFFF;
        border: 2px solid #000000;
        box-shadow: 3px 3px 0px 0px #000000;
        border-radius: 0px;
        font-family: 'Fira Code', monospace;
    }

    div.stButton > button {
        background-color: #FF0055 !important;
        color: #FFFFFF !important;
        border: 3px solid #000000 !important;
        box-shadow: 4px 4px 0px 0px #000000 !important;
        font-family: 'VT323', monospace !important;
        font-size: 24px !important;
        border-radius: 0px !important;
        transition: all 0.1s ease-in-out;
    }
    div.stButton > button:active {
        box-shadow: 0px 0px 0px 0px #000000 !important;
        transform: translate(4px, 4px);
    }

    .js-plotly-plot .plotly {
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px 0px #000000;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'VT323', monospace;
        font-size: 1.2rem;
        border: 2px solid #000000;
        margin-right: 5px;
        background-color: #FFFF00;
        box-shadow: 3px 3px 0px 0px #000000;
        color: black;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00FFFF !important;
        box-shadow: none !important;
        transform: translate(3px, 3px);
    }
</style>
""", unsafe_allow_html=True)

# --- Titel & Header ---
st.title("🚦 Intelligente Aktien-Analyse")
st.markdown("### Goldstandard: Trend, Momentum, Smart Money & News")

# --- Sidebar ---
with st.sidebar:
    st.header("Konfiguration")
    finnhub_api_key = st.text_input("Finnhub API Key", value="d5dgfp9r01qur4is43o0d5dgfp9r01qur4is43og", type="password")
    st.write("---")
    search_query = st.text_input("Firma suchen", value="SAP", help="Gib den Firmennamen ein (z.B. Allianz, Tesla, BYD).")
    
    selected_ticker = None
    selected_name = None
    
    if finnhub_api_key and search_query:
        try:
            fh_client = finnhub.Client(api_key=finnhub_api_key)
            lookup = fh_client.symbol_lookup(search_query)
            results = lookup.get('result', [])
            
            if results:
                options = {f"{r['description']} ({r['symbol']})": r for r in results[:15]}
                selection = st.selectbox("Gefundene Wertpapiere:", options.keys())
                chosen = options[selection]
                selected_ticker = chosen['symbol']
                selected_name = chosen['description']
                st.caption(f"Gewählt: **{selected_ticker}**")
            else:
                st.error("Keine Treffer gefunden.")
                selected_ticker = search_query.upper()
        except Exception as e:
            st.error(f"API Fehler: {e}")
            selected_ticker = search_query.upper()
    else:
        selected_ticker = search_query.upper()

    st.write("---")
    view_option = st.selectbox("Chart-Zeitraum", ["1 Tag", "1 Woche", "1 Monat", "1 Jahr", "3 Jahre", "5 Jahre", "Max (All Time)"], index=3)
    st.write("---")
    strategy_type = st.radio("Strategie (Szenario):", ["Swing Trading (Kurz)", "Trendfolge (Mittel)", "Investment (Lang)"], index=0)

# --- Funktionen ---

def get_news_from_finnhub(api_key, ticker, company_name):
    if not api_key: return [], ticker
    try:
        finnhub_client = finnhub.Client(api_key=api_key)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        final_news = []
        used_ticker = ticker
        
        try:
            # Versuch 1: Original Ticker
            news = finnhub_client.company_news(ticker, _from=start, to=end)
            if news: 
                final_news = news
            else:
                # Versuch 2: Suffix entfernen für US-News
                clean_ticker = ticker.split(".")[0]
                news_us = finnhub_client.company_news(clean_ticker, _from=start, to=end)
                if news_us:
                    final_news = news_us
                    used_ticker = clean_ticker
        except:
            pass

        processed_news = []
        if final_news:
            for n in final_news[:5]: 
                headline = n.get('headline'); url = n.get('url'); summary = n.get('summary', '')
                if headline:
                    blob = TextBlob(f"{headline}. {summary}")
                    processed_news.append({"title": headline, "link": url, "score": blob.sentiment.polarity})
                    
        return processed_news, used_ticker
    except: return [], ticker

# --- NEU: FALLBACK FUNCTION ---
def get_finnhub_candles(ticker, api_key):
    """Holt Kursdaten von Finnhub wenn Yahoo klemmt"""
    try:
        fc = finnhub.Client(api_key=api_key)
        # Wir holen ca 1 Jahr an Daten (Resolution D = Day)
        # Finnhub braucht Unix Timestamps
        end = int(time.time())
        start = end - (365 * 24 * 60 * 60) # 1 Jahr zurück
        
        res = fc.stock_candles(ticker, 'D', start, end)
        
        if res and res.get('s') == 'ok':
            df = pd.DataFrame({
                'Open': res['o'],
                'High': res['h'],
                'Low': res['l'],
                'Close': res['c'],
                'Volume': res['v'],
                'Date': pd.to_datetime(res['t'], unit='s')
            })
            df.set_index('Date', inplace=True)
            return df
    except Exception as e:
        print(f"Finnhub Fallback failed: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=14400, show_spinner=False)
def get_data_and_indicators(ticker, api_key, company_name):
    info_source = "Yahoo"
    stock_info = {}
    
    # 1. VERSUCH: YAHOO FINANCE
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="max", interval="1d")
        stock_info = stock.info
        
        if df.empty: raise Exception("Empty Yahoo Data")
        
    except Exception as e:
        # 2. VERSUCH: FINNHUB FALLBACK
        # Wenn Yahoo blockt (Rate Limit) oder leer ist, nehmen wir Finnhub
        df = get_finnhub_candles(ticker, api_key)
        info_source = "Finnhub (Backup)"
        # Dummy Info bauen, da Finnhub Candles keine Metadaten haben
        stock_info = {'currency': '?', 'longName': ticker, 'shortPercentOfFloat': None}

    if df is None or df.empty:
        return None, "Keine Daten (Weder Yahoo noch Finnhub)", None, None

    # --- Indikatoren Berechnung (Identisch für beide Quellen) ---
    try:
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14'] 
        
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
            df.rename(columns={df.columns[-3]: 'MACD', df.columns[-1]: 'MACD_Signal'}, inplace=True)

        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            df = pd.concat([df, bb], axis=1)
            df['BB_Lower'] = df.iloc[:, -5]; df['BB_Upper'] = df.iloc[:, -3]

        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV_EMA'] = ta.ema(df['OBV'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    except: pass # Falls Indikatoren failen, geben wir rohe Daten zurück

    news_list, used_ticker_for_news = get_news_from_finnhub(api_key, ticker, company_name)
    
    # Packe die Info-Quelle in die stock_info, damit wir es anzeigen können
    stock_info['source_label'] = info_source
    
    return df, stock_info, news_list, used_ticker_for_news

def calculate_smart_score(df, news_list):
    score = 0; cat_trend = 0; cat_momentum = 0; cat_volume = 0; reasons = [] 
    if df.empty: return 0, 0, [], {}
    curr = df.iloc[-1]; close = curr['Close']
    
    # Trend
    if not pd.isna(curr.get('SMA_200')):
        if close > curr['SMA_200']: score += 2; cat_trend += 1; reasons.append({"sig": "🟢 Trend: Über SMA 200", "desc": "Langfristiger Aufwärtstrend.", "tech": "Close > SMA200"})
        else: score -= 2; cat_trend -= 1; reasons.append({"sig": "🔴 Trend: Unter SMA 200", "desc": "Langfristiger Abwärtstrend.", "tech": "Close < SMA200"})
    
    if not pd.isna(curr.get('EMA_50')):
        if close > curr['EMA_50']: score += 1; cat_trend += 1
        else: score -= 1; cat_trend -= 1

    if not pd.isna(curr.get('EMA_50')) and not pd.isna(curr.get('SMA_200')):
        if curr['EMA_50'] > curr['SMA_200']: score += 1; cat_trend += 1; reasons.append({"sig": "🌟 Trend: Golden Cross", "desc": "Starkes Kaufsignal.", "tech": "EMA50 > SMA200"})
        else: score -= 1

    if not pd.isna(curr.get('ADX')):
        if curr['ADX'] > 25: score += 1; cat_trend += 1
    
    # Momentum
    if not pd.isna(curr.get('RSI')):
        if curr['RSI'] < 30: score += 2; cat_momentum += 1; reasons.append({"sig": "🟢 RSI Oversold", "desc": "Überverkauft - Rebound möglich.", "tech": "RSI < 30"})
        elif curr['RSI'] > 70: score -= 2; cat_momentum -= 1; reasons.append({"sig": "🔴 RSI Overbought", "desc": "Überkauft - Korrektur möglich.", "tech": "RSI > 70"})
    
    if 'MACD' in df.columns and not pd.isna(curr.get('MACD')):
        if curr['MACD'] > curr['MACD_Signal']: score += 1; cat_momentum += 1
        else: score -= 1; cat_momentum -= 1

    # Volume
    if not pd.isna(curr.get('MFI')):
        if curr['MFI'] > 80: score -= 1; cat_volume -= 1
        elif curr['MFI'] < 20: score += 2; cat_volume += 1; reasons.append({"sig": "🟢 MFI Panik", "desc": "Washout am Markt.", "tech": "MFI < 20"})
        elif curr['MFI'] > 50: score += 1; cat_volume += 1

    # News
    if news_list:
        avg_sent = sum([n['score'] for n in news_list]) / len(news_list)
        if avg_sent > 0.15: score += 1
        elif avg_sent < -0.15: score -= 1

    norm_score = max(0, min(100, int(((score + 8) / 20) * 100)))
    return score, norm_score, reasons, {"Trend": cat_trend, "Momentum": cat_momentum, "Volumen": cat_volume}

def find_swing_points(df, window=5):
    if len(df) < window * 2: return df 
    df['Swing_High'] = df['High'][(df['High'].shift(window) < df['High']) & (df['High'].shift(-window) < df['High'])]
    df['Swing_Low'] = df['Low'][(df['Low'].shift(window) > df['Low']) & (df['Low'].shift(-window) > df['Low'])]
    return df

# --- Main ---
if 'analysis_active' not in st.session_state: st.session_state.analysis_active = False

motorhaube_text = """
DATENBASIS & INDIKATOREN
• Trend: SMA 200 / EMA 50
• Momentum: RSI (14) / MACD
• Smart Money: MFI (14) / OBV
• Fallback: Finnhub API (falls Yahoo Rate Limit)
"""

c_btn, c_help, c_space = st.columns([0.12, 0.02, 0.86]) 
with c_btn:
    if st.button("Analyse starten 🚀", type="primary"): st.session_state.analysis_active = True
with c_help:
    st.markdown(" ", help=motorhaube_text)

if selected_ticker and st.session_state.analysis_active:
    with st.spinner(f"Analysiere {selected_ticker} (Versuche Yahoo, Fallback Finnhub)..."):
        
        df, info, news_data, used_news_ticker = get_data_and_indicators(selected_ticker, finnhub_api_key, selected_name)
        
        if isinstance(df, str): # Error Message
             st.error(df)
        elif df is not None and not df.empty:
            raw_score, confidence, pattern_list, cats = calculate_smart_score(df, news_data)
            current_price = df['Close'].iloc[-1]
            currency = info.get('currency', '?')
            long_name = info.get('longName', selected_ticker)
            data_source = info.get('source_label', 'Unknown')

            # Alert wenn Backup Mode
            if "Finnhub" in data_source:
                st.warning("⚠️ **Yahoo Rate Limit aktiv.** Daten geladen via **Finnhub Backup**. (Short-Daten evtl. unvollständig).")

            view_df = df.copy()
            if view_option == "1 Tag": view_df = df.tail(1) 
            elif view_option == "1 Woche": view_df = df.tail(5)
            elif view_option == "1 Monat": view_df = df.tail(22) 
            elif view_option == "1 Jahr": view_df = df.tail(252)
            elif view_option == "3 Jahre": view_df = df.tail(252*3)
            elif view_option == "5 Jahre": view_df = df.tail(252*5)
            
            view_df = find_swing_points(view_df)
            
            tab1, tab2, tab3 = st.tabs(["📊 Chart", "📈 Momentum", "📰 News"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=view_df.index, open=view_df['Open'], high=view_df['High'], low=view_df['Low'], close=view_df['Close'], name='Kurs'))
                if len(view_df) > 20: fig.add_trace(go.Scatter(x=view_df.index, y=view_df['SMA_200'], line=dict(color='blue'), name='SMA 200'))
                sh = view_df[view_df['Swing_High'].notna()]; sl = view_df[view_df['Swing_Low'].notna()]
                fig.add_trace(go.Scatter(x=sh.index, y=sh['Swing_High'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=8), name='High'))
                fig.add_trace(go.Scatter(x=sl.index, y=sl['Swing_Low'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=8), name='Low'))
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{long_name} ({data_source})")
                st.plotly_chart(fig, width="stretch")
            
            with tab2: st.line_chart(view_df[['MFI', 'RSI']])
            with tab3:
                 if news_data:
                    for n in news_data: 
                        icon = "🟢" if n['score'] > 0.15 else "🔴" if n['score'] < -0.15 else "⚪"
                        st.markdown(f"{icon} [{n['title']}]({n['link']})")
                 else: st.info("Keine News gefunden.")

            st.divider()
            c1, c2, c3 = st.columns([1, 2, 1.5])
            
            with c1:
                st.metric("Preis", f"{current_price:.2f} {currency}")
                st.caption(f"Quelle: {data_source}")
                st.write("---")
                adx_val = df['ADX'].iloc[-1]
                st.markdown(f"**ADX:** {adx_val:.0f} " + ("(Stark)" if adx_val > 25 else "(Schwach)"))
                
                # Short Interest nur wenn Yahoo da ist
                if "Yahoo" in data_source and info.get('shortPercentOfFloat'):
                    sf = info['shortPercentOfFloat'] * 100
                    st.markdown(f"**Short:** {sf:.2f}%")
                elif "Finnhub" in data_source:
                    st.caption("Short-Daten im Backup-Modus nicht verfügbar.")

            with c2:
                st.subheader(f"Score: {confidence}/100")
                st.progress(confidence, text="Bullishness")
                if confidence >= 55: st.success("KAUFEN / LONG")
                elif confidence <= 40: st.error("VERKAUFEN / SHORT")
                else: st.warning("HOLD / NEUTRAL")
                
                k1, k2, k3 = st.columns(3)
                k1.metric("Trend", f"{cats['Trend']}")
                k2.metric("Mom.", f"{cats['Momentum']}")
                k3.metric("Vol.", f"{cats['Volumen']}")

            with c3:
                st.markdown("##### Signale")
                if pattern_list:
                    for p in pattern_list[:3]: st.caption(f"{p['sig']}")
                else: st.caption("Keine klaren Signale.")

            st.divider()
            # Elliott / Szenario
            atr = df['ATR'].iloc[-1]; stop = current_price - (2*atr); target = current_price + (4*atr)
            st.subheader(f"Szenario ({strategy_type})")
            k1, k2, k3 = st.columns(3)
            k1.metric("Entry", f"{current_price:.2f}")
            k2.metric("Stop", f"{stop:.2f}")
            k3.metric("Target", f"{target:.2f}")

        else:
            st.error(f"Konnte {selected_ticker} nicht finden.")

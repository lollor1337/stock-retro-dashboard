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
st.set_page_config(page_title="Profi Aktien-Analyse 41.0 (RETRO MODE)", layout="wide")

# --- RETRO CSS INJECTION ---
# Das hier ist das Herzstück des neuen Designs
st.markdown("""
<style>
    /* Import von coolen Retro-Fonts */
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Fira+Code:wght@500&display=swap');

    /* 1. Generelles Layout & Hintergrund */
    .stApp {
        background-color: #D499FF; /* Das Retro-Lila */
        font-family: 'Fira Code', monospace; /* Monospace für alle Daten */
        color: #000000;
    }

    /* 2. Überschriften im Pixel-Look */
    h1, h2, h3, .stMetricLabel, .stMarkdown h5 {
        font-family: 'VT323', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #000000 !important;
        text-shadow: 2px 2px 0px #FFFFFF; /* Weißer Schatten für Lesbarkeit */
    }
    h1 { font-size: 4rem !important; color: #FFFF00 !important; text-shadow: 3px 3px 0px #000000 !important; }
    h2, h3 { font-size: 2rem !important; }

    /* 3. Boxen (Metrics, Tabs, Expander) - Der Sticker-Look */
    div[data-testid="stMetric"], .stTabs div[data-baseweb="tab-panel"], .stExpander {
        background-color: #FFFFFF;
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px 0px #000000;
        border-radius: 0px; /* Eckig! */
        padding: 15px !important;
        margin-bottom: 10px;
    }
    
    /* Speziell für die Sidebar-Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stRadio label {
        background-color: #FFFFFF;
        border: 2px solid #000000;
        box-shadow: 3px 3px 0px 0px #000000;
        border-radius: 0px;
        font-family: 'Fira Code', monospace;
    }

    /* 4. Der "Analyse Starten" Button */
    div.stButton > button {
        background-color: #FF0055 !important; /* Neon Pink */
        color: #FFFFFF !important;
        border: 3px solid #000000 !important;
        box-shadow: 4px 4px 0px 0px #000000 !important;
        font-family: 'VT323', monospace !important;
        font-size: 24px !important;
        border-radius: 0px !important;
        transition: all 0.1s ease-in-out;
    }
    /* Klick-Effekt: Button drückt sich rein */
    div.stButton > button:active {
        box-shadow: 0px 0px 0px 0px #000000 !important;
        transform: translate(4px, 4px);
    }

    /* 5. Plotly Chart Anpassung */
    .js-plotly-plot .plotly {
        border: 3px solid #000000;
        box-shadow: 5px 5px 0px 0px #000000;
    }

    /* 6. Tab-Reiter */
    .stTabs [data-baseweb="tab"] {
        font-family: 'VT323', monospace;
        font-size: 1.2rem;
        border: 2px solid #000000;
        margin-right: 5px;
        background-color: #FFFF00; /* Gelbe Tabs */
        box-shadow: 3px 3px 0px 0px #000000;
        color: black;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00FFFF !important; /* Aktiver Tab Cyan */
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
    
    # API Key
    finnhub_api_key = st.text_input("Finnhub API Key", value="d5dgfp9r01qur4is43o0d5dgfp9r01qur4is43og", type="password")
    
    st.write("---")
    
    # SMART SEARCH
    search_query = st.text_input("Firma suchen", value="SAP", help="Gib den Firmennamen ein (z.B. Allianz, Tesla, BYD).")
    
    selected_ticker = None
    selected_name = None
    
    if finnhub_api_key and search_query:
        try:
            fh_client = finnhub.Client(api_key=finnhub_api_key)
            lookup = fh_client.symbol_lookup(search_query)
            results = lookup.get('result', [])
            
            if results:
                # Liste erstellen
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
    # UPDATED: Erweiterte Zeiträume
    view_option = st.selectbox(
        "Chart-Zeitraum", 
        ["1 Tag", "1 Woche", "1 Monat", "1 Jahr", "3 Jahre", "5 Jahre", "Max (All Time)"], 
        index=3 # Default auf 1 Jahr
    )
    st.write("---")
    strategy_type = st.radio("Strategie (Szenario):", ["Swing Trading (Kurz)", "Trendfolge (Mittel)", "Investment (Lang)"], index=0)

# --- Funktionen ---

def get_news_from_finnhub(api_key, ticker, company_name):
    """ 
    News-Suche V38.3: PRIORITY SWAP.
    """
    if not api_key: return [], ticker
    try:
        finnhub_client = finnhub.Client(api_key=api_key)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        
        final_news = []
        used_ticker = ticker
        
        has_suffix = "." in ticker
        
        if has_suffix:
            clean_ticker = ticker.split(".")[0] 
            try:
                news_us = finnhub_client.company_news(clean_ticker, _from=start, to=end)
            except:
                news_us = []
            
            if news_us:
                final_news = news_us
                used_ticker = clean_ticker
            else:
                time.sleep(0.2) 
                news_orig = finnhub_client.company_news(ticker, _from=start, to=end)
                if news_orig:
                    final_news = news_orig
                    used_ticker = ticker 
        else:
            final_news = finnhub_client.company_news(ticker, _from=start, to=end)
            used_ticker = ticker

        processed_news = []
        if final_news:
            for n in final_news[:5]: 
                headline = n.get('headline')
                url = n.get('url')
                summary = n.get('summary', '')
                if headline:
                    text_to_analyze = f"{headline}. {summary}"
                    blob = TextBlob(text_to_analyze)
                    score = blob.sentiment.polarity
                    processed_news.append({"title": headline, "link": url, "score": score})
                    
        return processed_news, used_ticker
    except Exception as e:
        return [], ticker

def get_data_and_indicators(ticker, api_key, company_name):
    try:
        stock = yf.Ticker(ticker)
        # UPDATED: Wir laden 'max', damit auch 5 Jahre/All Time funktionieren
        df = stock.history(period="max", interval="1d")
        
        if df.empty: return None, None, None, None

        # Indikatoren (berechnet auf der vollen Historie für Genauigkeit)
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
            df['BB_Lower'] = df.iloc[:, -5]
            df['BB_Upper'] = df.iloc[:, -3]

        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV_EMA'] = ta.ema(df['OBV'], length=20)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        news_list, used_ticker_for_news = get_news_from_finnhub(api_key, ticker, company_name)

        return df, stock.info, news_list, used_ticker_for_news
    except Exception as e: return None, None, None, None

def calculate_smart_score(df, news_list):
    score = 0
    cat_trend = 0; cat_momentum = 0; cat_volume = 0
    reasons = [] 
    
    if df.empty: return 0, 0, [], {}
    curr = df.iloc[-1]; close = curr['Close']
    
    # --- 1. TREND ---
    if not pd.isna(curr.get('SMA_200')):
        if close > curr['SMA_200']: 
            score += 2; cat_trend += 1
            reasons.append({"sig": "🟢 Trend: Über SMA 200", "desc": "Der Kurs liegt über dem Durchschnitt der letzten 200 Tage. Das ist das wichtigste Zeichen für einen langfristigen Aufwärtstrend.", "tech": "Close > SMA200"})
        else: 
            score -= 2; cat_trend -= 1
            reasons.append({"sig": "🔴 Trend: Unter SMA 200", "desc": "Der Kurs liegt unter dem 200-Tage-Schnitt. Langfristig dominieren die Verkäufer (Bärenmarkt).", "tech": "Close < SMA200"})
    
    if not pd.isna(curr.get('EMA_50')):
        if close > curr['EMA_50']: 
            score += 1; cat_trend += 1
            reasons.append({"sig": "🟢 Trend: Über EMA 50", "desc": "Mittelfristig (letzte 50 Tage) geht es bergauf. Ein gutes Zeichen für Stabilität.", "tech": "Close > EMA50"})
        else: 
            score -= 1; cat_trend -= 1
            reasons.append({"sig": "🔴 Trend: Unter EMA 50", "desc": "Mittelfristig ist der Schwung verloren gegangen. Der Kurs hängt durch.", "tech": "Close < EMA50"})

    if not pd.isna(curr.get('EMA_50')) and not pd.isna(curr.get('SMA_200')):
        if curr['EMA_50'] > curr['SMA_200']: 
            score += 1; cat_trend += 1
            reasons.append({"sig": "🌟 Trend: Golden Cross", "desc": "Sehr starkes Kaufsignal! Der kurzfristige Trend kreuzt den langfristigen von unten nach oben.", "tech": "EMA50 > SMA200 (Crossover)"})
        else: score -= 1

    if not pd.isna(curr.get('ADX')):
        if curr['ADX'] > 25: 
            score += 1; cat_trend += 1
            reasons.append({"sig": f"💪 Trend: Stark (ADX {curr['ADX']:.0f})", "desc": "Der aktuelle Trend ist stabil und kein Zufall.", "tech": "ADX > 25"})
        elif curr['ADX'] < 20: 
            reasons.append({"sig": f"🐌 Trend: Schwach (ADX {curr['ADX']:.0f})", "desc": "Der Markt hat keine klare Richtung (Seitwärtsphase).", "tech": "ADX < 20"})

    # --- 2. MOMENTUM ---
    if not pd.isna(curr.get('RSI')):
        rsi = curr['RSI']
        if rsi < 30: 
            score += 2; cat_momentum += 1
            reasons.append({"sig": f"🟢 Momentum: RSI Oversold ({rsi:.0f})", "desc": "Die Aktie wurde panikartig verkauft. Eine Gegenbewegung ist wahrscheinlich.", "tech": "RSI < 30"})
        elif rsi > 70: 
            score -= 2; cat_momentum -= 1
            reasons.append({"sig": f"🔴 Momentum: RSI Overbought ({rsi:.0f})", "desc": "Die Aktie ist zu schnell gestiegen. Gewinnmitnahmen drohen.", "tech": "RSI > 70"})
        else: 
            reasons.append({"sig": f"⚪ Momentum: RSI Neutral ({rsi:.0f})", "desc": "Der Preis ist in einer gesunden Balance.", "tech": "30 < RSI < 70"})

    if 'MACD' in df.columns and not pd.isna(curr.get('MACD')):
        if curr['MACD'] > curr['MACD_Signal']: 
            score += 1; cat_momentum += 1
            reasons.append({"sig": "🟢 Momentum: MACD Bullish", "desc": "Der kurzfristige Schwung dreht gerade ins Positive.", "tech": "MACD > Signal"})
        else: 
            score -= 1; cat_momentum -= 1
            reasons.append({"sig": "🔴 Momentum: MACD Bearish", "desc": "Der kurzfristige Schwung kippt nach unten weg.", "tech": "MACD < Signal"})

    if 'BB_Lower' in df.columns and not pd.isna(curr.get('BB_Lower')):
        if close < curr['BB_Lower']: 
            score += 2; cat_momentum += 1
            reasons.append({"sig": "⚡ Momentum: Bollinger Breakout Low", "desc": "Statistischer Ausbruch nach unten. Rebound möglich.", "tech": "Close < Lower Band"})
        elif close > curr['BB_Upper']: 
            score -= 1; cat_momentum -= 1
            reasons.append({"sig": "⚠️ Momentum: Bollinger Breakout High", "desc": "Statistischer Ausbruch nach oben. Abkühlung wahrscheinlich.", "tech": "Close > Upper Band"})

    # --- 3. VOLUMEN / GELDFLUSS ---
    if not pd.isna(curr.get('MFI')):
        mfi = curr['MFI']
        if mfi > 80: 
            score -= 1; cat_volume -= 1
            reasons.append({"sig": f"⚠️ Geldfluss: MFI Heiß ({mfi:.0f})", "desc": "Markt ist überfüllt. Zu viel Geld zu schnell.", "tech": "MFI > 80"})
        elif mfi < 20: 
            score += 2; cat_volume += 1
            reasons.append({"sig": f"🟢 Geldfluss: MFI Panik ({mfi:.0f})", "desc": "Kapitulation der Anleger ('Washout').", "tech": "MFI < 20"})
        elif mfi > 50: 
            score += 1; cat_volume += 1
            reasons.append({"sig": f"🟢 Geldfluss: MFI Zufluss ({mfi:.0f})", "desc": "Es fließt gesundes Kapital in die Aktie.", "tech": "MFI > 50"})
    
    if 'OBV' in df.columns and not pd.isna(curr.get('OBV_EMA')):
        if curr['OBV'] > curr['OBV_EMA']: 
            score += 1; cat_volume += 1
            reasons.append({"sig": "📢 Geldfluss: OBV Trend", "desc": "Volumen bestätigt den Preistrend.", "tech": "OBV > OBV_EMA"})

    # --- 4. NEWS SENTIMENT ---
    if news_list:
        avg_sent = sum([n['score'] for n in news_list]) / len(news_list)
        if avg_sent > 0.15: 
            score += 1
            reasons.append({"sig": "📰 News: Positiv", "desc": "KI bewertet aktuelle Schlagzeilen (Finnhub) optimistisch.", "tech": f"Avg Sentiment {avg_sent:.2f} > 0.15"})
        elif avg_sent < -0.15: 
            score -= 1
            reasons.append({"sig": "📰 News: Negativ", "desc": "KI bewertet aktuelle Schlagzeilen (Finnhub) pessimistisch.", "tech": f"Avg Sentiment {avg_sent:.2f} < -0.15"})

    normalized_score = int(((score + 8) / 20) * 100)
    normalized_score = max(0, min(100, normalized_score))
    
    cats = {"Trend": cat_trend, "Momentum": cat_momentum, "Volumen": cat_volume}
    return score, normalized_score, reasons, cats

def find_swing_points(df, window=5):
    # Braucht genug Daten
    if len(df) < window * 2: return df 
    df['Swing_High'] = df['High'][(df['High'].shift(window) < df['High']) & (df['High'].shift(-window) < df['High'])]
    df['Swing_Low'] = df['Low'][(df['Low'].shift(window) > df['Low']) & (df['Low'].shift(-window) > df['Low'])]
    return df

# --- Hauptprogramm ---

# Init Session State für Persistenz beim Umschalten
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

# UPDATED: Die "Motorhaube" als reine Text-Liste ohne Scroll-Balken
motorhaube_text = """
DATENBASIS & INDIKATOREN (MOTORHAUBE)

TREND
• SMA 200: Langfristig (~10 Monate). Das "Gesetz" der Wall Street.
• EMA 50: Mittelfristig (~2,5 Monate). Der Quartalstrend.

MOMENTUM
• RSI (14): Kurzfristig (3 Wochen). Standard nach J. Welles Wilder.
• MACD (12/26/9): Mix aus schnellem & langsamem Trend.
• Bollinger (20): Monatsbasis.

SMART MONEY & NEWS
• MFI (14): Geldfluss der letzten 3 Wochen.
• OBV (20): Volumen-Trend vs. Preis.
• Sentiment: KI-Analyse der letzten 14 Tage.
• Short: Aktuellster Börsen-Bericht.
"""

# Layout Split: Button und direkt daneben nur das Fragezeichen (Space trick)
c_btn, c_help, c_space = st.columns([0.12, 0.02, 0.86]) 

with c_btn:
    if st.button("Analyse starten 🚀", type="primary"):
        st.session_state.analysis_active = True

with c_help:
    # UPDATED: Nur ein Leerzeichen als Inhalt -> Streamlit rendert nur das Fragezeichen
    st.markdown(" ", help=motorhaube_text)

# Analyse läuft, wenn Button gedrückt wurde ODER Session State aktiv ist
if selected_ticker and st.session_state.analysis_active:
    with st.spinner(f"Analysiere {selected_ticker} auf Herz und Nieren..."):
        
        df, info, news_data, used_news_ticker = get_data_and_indicators(selected_ticker, finnhub_api_key, selected_name)
        
        if df is not None and not df.empty:
            raw_score, confidence, pattern_list, cats = calculate_smart_score(df, news_data)
            current_price = df['Close'].iloc[-1]
            currency = info.get('currency', '')
            long_name = info.get('longName', selected_ticker)

            # --- 1. CHART VIEW LOGIC ---
            view_df = df.copy()
            
            if view_option == "1 Tag": 
                view_df = df.tail(1) 
            elif view_option == "1 Woche":
                view_df = df.tail(5)
            elif view_option == "1 Monat":
                view_df = df.tail(22) 
            elif view_option == "1 Jahr":
                view_df = df.tail(252)
            elif view_option == "3 Jahre":
                view_df = df.tail(252*3)
            elif view_option == "5 Jahre":
                view_df = df.tail(252*5)
            
            view_df = find_swing_points(view_df)
            
            tab1, tab2, tab3 = st.tabs(["📊 Chart (Elliott & Trend)", "📈 MFI & RSI", "📰 News (Finnhub)"])
            
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=view_df.index, open=view_df['Open'], high=view_df['High'], low=view_df['Low'], close=view_df['Close'], name='Kurs'))
                
                if len(view_df) > 20:
                    fig.add_trace(go.Scatter(x=view_df.index, y=view_df['SMA_200'], line=dict(color='blue'), name='SMA 200'))
                
                sh = view_df[view_df['Swing_High'].notna()]; sl = view_df[view_df['Swing_Low'].notna()]
                fig.add_trace(go.Scatter(x=sh.index, y=sh['Swing_High'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=8), name='High'))
                fig.add_trace(go.Scatter(x=sl.index, y=sl['Swing_Low'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=8), name='Low'))
                
                title_suffix = f" ({view_option})"
                fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"Chart: {long_name}{title_suffix}")
                st.plotly_chart(fig, width="stretch")
            
            with tab2: st.line_chart(view_df[['MFI', 'RSI']])
            with tab3:
                 if used_news_ticker != selected_ticker:
                     st.info(f"ℹ️ **Smart-Switch:** Suche '{selected_ticker}', aber News geladen von US-Ticker '**{used_news_ticker}**' (bessere Datenverfügbarkeit).")
                 else:
                     st.caption(f"News geladen für Symbol: **{used_news_ticker}**")
                 
                 if news_data:
                    for n in news_data: 
                        score_icon = "⚪"
                        if n['score'] > 0.15: score_icon = "🟢"
                        elif n['score'] < -0.15: score_icon = "🔴"
                        st.markdown(f"{score_icon} [{n['title']}]({n['link']})")
                 else: 
                     st.info(f"Keine News für {used_news_ticker} in den letzten 14 Tagen gefunden.")

            # --- 2. FACTS ---
            st.divider()
            c1, c2, c3 = st.columns([1, 2, 1.5])
            
            with c1:
                st.metric("Aktueller Kurs", f"{current_price:.2f} {currency}")
                st.write("---")
                
                vol_curr = df['Volume'].iloc[-1]; vol_avg = df['Volume'].tail(20).mean()
                pct_vol = ((vol_curr - vol_avg)/vol_avg)*100 if vol_avg > 0 else 0
                
                st.markdown("**Handelsaktivität**", help="QUANTITÄT (Der 'Lärm-Pegel'): Misst die reine Menge an gehandelten Aktien.")
                if vol_curr > vol_avg * 1.5: st.markdown(f"🔥 **Hoch** (+{pct_vol:.0f}%)")
                else: st.markdown(f"💤 **Normal** ({pct_vol:+.0f}%)")
                
                st.write("") 
                short_float = info.get('shortPercentOfFloat', None)
                short_ratio = info.get('shortRatio', None)
                
                if short_float is not None and isinstance(short_float, (int, float)):
                    st.markdown("**Short Interest**", help="Quote der leerverkauften Aktien im Verhältnis zum Free Float.")
                    sf_val = short_float * 100
                    
                    if sf_val > 20: 
                        st.error(f"🔴 **Extrem: {sf_val:.2f}%**")
                        st.caption("Kampfzone! Squeeze möglich.")
                    elif sf_val > 10: 
                        st.warning(f"🟠 **Hoch: {sf_val:.2f}%**")
                        st.caption("Squeeze-Chance / Warnsignal.")
                    elif sf_val > 5: 
                        st.markdown(f"🟡 **Moderat: {sf_val:.2f}%**")
                        st.caption("Erhöhte Skepsis.")
                    else: 
                        st.success(f"🟢 **Niedrig: {sf_val:.2f}%**")
                        st.caption("Kein Verkaufsdruck.")
                        
                    if short_ratio:
                        st.caption(f"Days to Cover: {short_ratio:.1f}")
                else:
                    st.markdown("**Short Interest**")
                    st.caption("Keine Daten (oft nur US).")

                st.write("") 
                adx_val = df['ADX'].iloc[-1]
                st.markdown("**Trendstärke (ADX)**")
                if adx_val > 25: st.success(f"Stark ({adx_val:.0f})")
                else: st.warning(f"Schwach ({adx_val:.0f})")

            with c2:
                st.subheader("Analyse-Ergebnis")
                st.markdown(f"### Gesamt-Score: {confidence}/100")
                st.progress(confidence, text=f"{confidence}% Bullishness")
                
                if confidence >= 75: st.success("🚀 **STARKER KAUF**")
                elif confidence >= 55: st.success("🟢 **KAUFEN**")
                elif confidence >= 40: st.warning("🟡 **BEOBACHTEN**")
                else: st.error("🔴 **MEIDEN / VERKAUFEN**")
                
                st.write("---")
                k1, k2, k3 = st.columns(3)
                k1.metric("Trend", f"{cats['Trend']}")
                k2.metric("Momentum", f"{cats['Momentum']}")
                k3.metric("Geldfluss", f"{cats['Volumen']}", help="QUALITÄT (Die Richtung): Misst, ob das Volumen den Kurs treibt. Positiv = Smart Money kauft.")

            with c3:
                st.markdown("##### ℹ️ So liest du den Score")
                st.markdown("""
                | Score | Signal | Handlung |
                | :--- | :--- | :--- |
                | **≥ 75%** | 🚀 **Stark** | Top-Chance. |
                | **≥ 55%** | 🟢 **Kaufen** | Gutes Setup. |
                | **≥ 40%** | 🟡 **Beobachten** | Abwarten. |
                | **< 40%** | 🔴 **Meiden** | Zu riskant. |
                """)

            # --- 3. DETAILWERTE ---
            st.divider()
            st.subheader("📖 Bedeutung der Detail-Werte")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("### 📈 Trend")
                st.markdown("""
                | Wert | Bedeutung |
                | :--- | :--- |
                | 3 bis 4 | Starker Aufwärtstrend |
                | 1 bis 2 | Positiv (Wackelig) |
                | 0 | Seitwärts |
                | < 0 | Abwärtstrend |
                """)
            with d2:
                st.markdown("### 🚀 Momentum")
                st.markdown("""
                | Wert | Bedeutung |
                | :--- | :--- |
                | > 0 | Hohe Dynamik (Speed) |
                | 0 | Pause (Luft holen) |
                | < 0 | Dynamik nimmt ab |
                """)
            with d3:
                st.markdown("### 💰 Geldfluss")
                st.markdown("""
                | Wert | Bedeutung |
                | :--- | :--- |
                | > 0 | Investoren kaufen (Zufluss) |
                | 0 | Neutral (Gleichgewicht) |
                | < 0 | Investoren verkaufen (Abfluss) |
                """)

            # --- 4. SIGNALE ---
            with st.expander("🔍 Detaillierte Signale (Warum dieser Score?)"):
                if not pattern_list: st.write("Keine Auffälligkeiten.")
                else:
                    for p in pattern_list:
                        st.markdown(f"**{p['sig']}**")
                        st.write(f"{p['desc']}")
                        st.markdown(f"<small style='color:grey'>⚙️ *Technik: {p['tech']}*</small>", unsafe_allow_html=True)
                        st.write("---")

            # --- 5. SZENARIO ---
            st.divider()
            st.subheader(f"Szenario-Planung ({strategy_type})")
            
            if "Swing" in strategy_type:
                st.info("ℹ️ **Kurzfristig (Swing):** Wir nutzen enge Grenzen (1,5x ATR).")
                stop_m = 1.5; targ_m = 2.5
            elif "Trend" in strategy_type:
                st.info("ℹ️ **Mittelfristig (Trend):** Wir geben dem Kurs mehr Luft (3,0x ATR).")
                stop_m = 3.0; targ_m = 5.0
            else:
                st.info("ℹ️ **Langfristig (Invest):** Maximale Toleranz (5,0x ATR).")
                stop_m = 5.0; targ_m = 10.0
            
            atr = df['ATR'].iloc[-1]
            if pd.isna(atr): atr = current_price * 0.02
            stop = current_price - (stop_m*atr); target = current_price + (targ_m*atr)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Eintritt", f"{current_price:.2f}")
            k2.metric("Stop-Loss", f"{stop:.2f}")
            k3.metric("Ziel", f"{target:.2f}")

            # --- 6. ELLIOTT WELLEN (Fix & Toggle) ---
            st.divider()
            
            ew_col1, ew_col2 = st.columns([3, 1])
            with ew_col1:
                st.subheader("🌊 Elliott-Wellen Prognose")
            with ew_col2:
                # FEATURE TOGGLE: Standard vs Experimental
                ew_method = st.selectbox("Berechnungsmethode:", ["Standard (Macro Range)", "Experimentell (Wave Hunter)"])

            is_bullish = current_price > df['SMA_200'].iloc[-1]
            
            # --- METHODE A: STANDARD (ROBUST) ---
            if "Standard" in ew_method:
                elliott_df = df.tail(500) 
                last_low = elliott_df['Low'].min()
                last_high = elliott_df['High'].max()
                range_h_l = last_high - last_low

                if is_bullish:
                    st.success(f"**Status (Standard):** Langfristiger Aufwärtstrend (Basis: 2 Jahre).")
                    t1 = last_low + (range_h_l * 1.0)
                    t2 = last_low + (range_h_l * 1.618)
                    t3 = last_low + (range_h_l * 2.618)
                    inval = last_low
                else:
                    st.error(f"**Status (Standard):** Langfristige Korrektur (Basis: 2 Jahre).")
                    t1 = last_high - (range_h_l * 1.0)
                    t2 = last_high - (range_h_l * 1.618)
                    t3 = last_high - (range_h_l * 2.618)
                    inval = last_high

            # --- METHODE B: EXPERIMENTELL (WAVE HUNTER) ---
            else:
                scan_df = find_swing_points(df.tail(252).copy(), window=5) 
                last_highs = scan_df[scan_df['Swing_High'].notna()]['Swing_High'].tail(3).values
                last_lows = scan_df[scan_df['Swing_Low'].notna()]['Swing_Low'].tail(3).values

                st.warning("⚠️ **Experimenteller Modus:** Diese Methode sucht nervös nach dem letzten lokalen Muster (Micro-Structure).")

                if len(last_highs) > 0 and len(last_lows) > 0:
                    local_high = last_highs[-1]
                    local_low = last_lows[-1]
                    
                    if is_bullish:
                        calc_base = local_low
                        calc_range = (local_high - local_low) if local_high > local_low else (current_price - local_low)
                        
                        st.markdown(f"**Erkannte Basis:** Letztes lokales Tief bei {local_low:.2f}")
                        
                        t1 = calc_base + (calc_range * 1.0)
                        t2 = calc_base + (calc_range * 1.618)
                        t3 = calc_base + (calc_range * 2.618)
                        inval = local_low
                    else:
                        calc_base = local_high
                        calc_range = (local_high - local_low) if local_high > local_low else (local_high - current_price)
                        
                        st.markdown(f"**Erkannte Basis:** Letztes lokales Hoch bei {local_high:.2f}")

                        t1 = calc_base - (calc_range * 1.0)
                        t2 = calc_base - (calc_range * 1.618)
                        t3 = calc_base - (calc_range * 2.618)
                        inval = local_high
                else:
                    st.info("Zu wenig Volatilität für Wave-Hunter. Nutze Standard.")
                    t1=0; t2=0; t3=0; inval=0

            # --- AUSGABE DER ZIELE ---
            t1 = max(0.01, t1); t2 = max(0.01, t2); t3 = max(0.01, t3)

            z1_col, z2_col, z3_col = st.columns(3)
            
            if is_bullish:
                z1_col.metric("🏁 Ziel 1", f"{t1:.2f}", help="Konservativ (1.0 Ext)")
                z2_col.metric("🏆 Ziel 2", f"{t2:.2f}", help="Standard (1.618 Ext)")
                z3_col.metric("🚀 Ziel 3", f"{t3:.2f}", help="Maximal (2.618 Ext)")
                st.caption(f"⛔ **Invalidierung:** Fällt Kurs unter **{inval:.2f}**")
            else:
                z1_col.metric("🏁 Ziel 1", f"{t1:.2f}", help="Konservativ Down")
                z2_col.metric("🏆 Ziel 2", f"{t2:.2f}", help="Standard Down")
                
                if t3 <= 0.1: z3_col.metric("📉 Ziel 3", "Crash", help="Rechnerisch <= 0")
                else: z3_col.metric("📉 Ziel 3", f"{t3:.2f}", help="Maximal Down")
                
                st.caption(f"⛔ **Invalidierung:** Steigt Kurs über **{inval:.2f}**")

            with st.expander("🧠 Wellen-Psychologie"):
                st.markdown("""
                * **Standard:** Nutzt die Spanne der letzten 2 Jahre für große Ziele.
                * **Experimentell:** Nutzt nur die letzten erkannten Spitzen/Tiefs für schnelle Ziele.
                """)

        else:
            st.error(f"Konnte {selected_ticker} nicht finden. Bitte Suche prüfen.")
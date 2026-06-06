import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. AYARLAR ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
FETCH_LIMIT = 400       
VISIBLE_LIMIT = 150     
CHART_HEIGHT = 1400     
ATR_MULTIPLIER = 0.1  # Sapma hassasiyeti
DONCHIAN_PERIODS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
trade_log = []

exchange = ccxt.binance({'options': {'defaultType': 'future'}})

# --- 2. HESAPLAMA MOTORU ---
def calculate_hurst(series, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def calculate_half_kelly(win_rate=0.55, profit_factor=1.6):
    b, p = profit_factor, win_rate
    q = 1 - p
    kelly_f = (b * p - q) / b
    return max(0, kelly_f / 2) 

# --- 3. DASHBOARD ---
app = dash.Dash(__name__)
app.layout = html.Div(style={'backgroundColor': '#0b0e11', 'color': '#eaecef', 'fontFamily': 'Segoe UI'}, children=[
    html.Div([
        html.H2(f"🚀 QUANTIFINE LADDER-STOP: {SYMBOL}", style={'margin': '0', 'color': '#f3ba2f'}),
        html.Div(id='live-status-bar', style={'fontSize': '18px', 'marginTop': '10px'})
    ], style={'padding': '20px', 'borderBottom': '1px solid #2b3139'}),
    dcc.Graph(id='main-chart', config={'displayModeBar': False}),
    html.Div([
        html.H4("Strateji Karar Merkezi (Ladder Stop Aktif):"),
        html.Div(id='trade-logs', style={'height': '300px', 'overflowY': 'scroll', 'backgroundColor': '#161a1e', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #2b3139', 'fontSize': '13px', 'fontFamily': 'monospace'})
    ], style={'padding': '20px'}),
    dcc.Interval(id='update-interval', interval=15*1000, n_intervals=0)
])

@app.callback(
    [Output('main-chart', 'figure'), Output('live-status-bar', 'children'), Output('trade-logs', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def run_engine(n):
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)
        df_all = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df_all['time'] = pd.to_datetime(df_all['time'], unit='ms')
        
        df_all['atr'] = ta.atr(df_all['high'], df_all['low'], df_all['close'], length=14)
        df_all['rsi'] = ta.rsi(df_all['close'], length=14)
        df_all['cci'] = ta.cci(df_all['high'], df_all['low'], df_all['close'], length=20)
        df_all['adx'] = ta.adx(df_all['high'], df_all['low'], df_all['close'], length=14)['ADX_14']
        
        # 10x Donchian + Sapma
        dev = df_all['atr'] * ATR_MULTIPLIER 
        for p in DONCHIAN_PERIODS:
            df_all[f'd_high_{p}'] = df_all['high'].rolling(window=p).max() + dev
            df_all[f'd_low_{p}'] = df_all['low'].rolling(window=p).min() - dev
        
        df_all['sma'] = ta.sma(df_all['close'], length=20)
        df_all['z_score'] = (df_all['close'] - df_all['sma']) / ta.stdev(df_all['close'], length=20)
        
        df = df_all.iloc[-VISIBLE_LIMIT:].copy()
        h_val = calculate_hurst(df_all['close'].values)
        current = df.iloc[-1]
        
        # Kırılan Kanalları Say (Ladder Mantığı)
        broken_highs = sum(1 for p in DONCHIAN_PERIODS if current['close'] > current[f'd_high_{p}'])
        broken_lows = sum(1 for p in DONCHIAN_PERIODS if current['close'] < current[f'd_low_{p}'])
        
        # STOP LOSS AYARI: "2 kanal kırsın, ilkinde stop olsun"
        sl_level = None
        if broken_highs >= 2:
            # Örn: 3 kanal kırıldıysa (10,20,30), stop 20 periyotluk olandadır (index: broken-2)
            sl_index = broken_highs - 2
            sl_level = current[f'd_high_{DONCHIAN_PERIODS[sl_index]}']
        elif broken_lows >= 2:
            sl_index = broken_lows - 2
            sl_level = current[f'd_low_{DONCHIAN_PERIODS[sl_index]}']

        decision, color = "BEKLEMEDE", "#eaecef"
        if h_val > 0.52: # TREND
            if broken_highs >= 2:
                decision, color = f"🟢 TREND LONG (Stop: {sl_level:.2f})", "#00ff00"
            elif broken_lows >= 2:
                decision, color = f"🟠 TREND SHORT (Stop: {sl_level:.2f})", "#ff4444"

        k_perc = calculate_half_kelly() * 100
        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] H:{h_val:.2f} | Kırılan:{broken_highs} | Stop:{f'{sl_level:.1f}' if sl_level else 'N/A'} | %{k_perc:.1f} | {decision}"
        trade_log.append(html.Div(log_msg, style={'color': color, 'marginBottom': '5px'}))
        if len(trade_log) > 30: trade_log.pop(0)

        # GÖRSELLEŞTİRME
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.4, 0.12, 0.12, 0.12, 0.12])
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='BTC'), row=1, col=1)
        
        # Stop Seviyesini Çiz
        if sl_level:
            fig.add_hline(y=sl_level, line_dash="dash", line_color="yellow", annotation_text="AKTIF STOP", row=1, col=1)

        for p in DONCHIAN_PERIODS:
            fig.add_trace(go.Scatter(x=df['time'], y=df[f'd_high_{p}'], line=dict(color='rgba(0, 255, 204, 0.1)', width=1), hoverinfo='skip'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['time'], y=df['adx'], name='ADX', line=dict(color='#f3ba2f')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], name='RSI', line=dict(color='white')), row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['cci'], name='CCI', line=dict(color='cyan')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['time'], y=df['z_score'], name='Z-Score', fill='tozeroy', line=dict(color='#32a8ff')), row=5, col=1)

        fig.update_layout(template='plotly_dark', height=CHART_HEIGHT, xaxis_rangeslider_visible=False, showlegend=False)
        return fig, html.Span(f"Ladder Stop Aktif | Kırılan Seviye: {broken_highs}"), trade_log[::-1]

    except Exception as e:
        return go.Figure(), f"Hata: {e}", [html.P(f"Hata: {e}")]

if __name__ == '__main__':
    app.run(debug=True)
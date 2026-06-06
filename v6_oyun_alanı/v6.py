import os
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
from dash import Dash, html, dcc, Output, Input
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# --- KONFİGÜRASYON ---
API_KEY = 'BINANCE_API_KEY'
API_SECRET = 'BINANCE_API_SECRET'
SYMBOL = 'BTCUSDT'
TIMEFRAME = Client.KLINE_INTERVAL_1_HOUR

client = Client(API_KEY, API_SECRET)

class QuantiFineAlgo:
    def __init__(self):
        self.position = None # 'LONG', 'SHORT' veya None
        self.entry_price = 0
        self.stop_loss = 0

    def kelly_position_size(self, win_rate=0.55, avg_win=1.5, avg_loss=1.0):
        """Kelly Criterion ile pozisyon boyutlandırması"""
        # Kelly Formül: K = W - [(1-W) / R]
        # W = Kazanma oranı, R = Win/Loss oranı
        if avg_loss == 0:
            return 0
        r = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / r)
        # Güvenlik için Kelly'nin yarısını kullan (Half Kelly)
        return max(0, min(kelly * 0.5, 0.25))  # Max %25 pozisyon

    def get_data(self):
        # Mum verilerini çek
        bars = client.get_historical_klines(SYMBOL, TIMEFRAME, "200 hours ago UTC")
        df = pd.DataFrame(bars, columns=['time','open','high','low','close','vol','ct','qv','nt','tb','tq','i'])
        df[['high','low','close']] = df[['high','low','close']].astype(float)
        
        # --- ŞEMADAKİ HESAPLAMALAR ---
        # 1. ATR ve ADX
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        # 2. RSI ve Z-Score
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['Z_Score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # 3. Hurst Üsteli (Pencere: 5)
        def hurst(s):
            lags = range(2, 5)
            tau = [np.sqrt(np.std(np.subtract(s[l:], s[:-l]))) for l in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        df['Hurst'] = df['close'].rolling(window=5).apply(hurst)
        
        # 4. Donchian x10 ve 0.005 ATR Sapması
        df['Donchian_U'] = ta.donchian(df['high'], df['low'], length=10).iloc[:, 0]
        df['Entry_Point'] = df['Donchian_U'] + (df['ATR'] * 0.005)
        
        # 5. S/R Sınırları (Donchian 50 - Straight Lines)
        df['RES'] = df['high'].rolling(50).max().shift(1)
        df['SUP'] = df['low'].rolling(50).min().shift(1)
            
        return df

    def check_logic(self, df):
        last = df.iloc[-1]
        atr_avg = df['ATR'].rolling(50).mean().iloc[-1]
        
        # ATR Filtresi: Ölü Bölge
        if last['ATR'] < atr_avg * 0.5:
            return None, "Ölü Bölge (Düşük Volatilite)"

        # S/R Referansları
        res = last['RES']
        sup = last['SUP']
        
        if pd.isna(res) or pd.isna(sup):
            return None, "Yetersiz Veri (S/R)"

        # REJİM TESPİTİ (Hurst)
        # --- TREND (Hurst > 0.52) ---
        if last['Hurst'] > 0.52:
            if last['RSI'] > 50 and last['ADX'] > 20:
                if last['close'] > res: # Direnç Kırılımı
                    return 'LONG', "Trend Long (Breakout)"
            elif last['RSI'] < 50 and last['ADX'] > 20:
                if last['close'] < sup: # Destek Kırılımı
                    return 'SHORT', "Trend Short (Breakdown)"

        # --- REVERSION (Hurst < 0.48) ---
        elif last['Hurst'] < 0.48:
            if last['Z_Score'] > 2 and last['ADX'] > 20: 
                if last['close'] > res: # Dirençten Dönüş (Fade)
                    return 'SHORT', f"Rev Short (Fade Z:{last['Z_Score']:.2f})"
            elif last['Z_Score'] < -2 and last['ADX'] > 20: 
                 if last['close'] < sup: # Destekten Dönüş (Fade)
                    return 'LONG', f"Rev Long (Fade Z:{last['Z_Score']:.2f})"
        
        return None, "Beklemede"

    def stop_update(self, df):
        if not self.position: return 0
        last = df.iloc[-1]
        
        # Donchian Trailing Stop
        if self.position == 'LONG':
            # Stop: Destek Çizgisi (Asla düştüğü yerden aşağı inmez)
            if last['SUP'] > self.stop_loss:
                self.stop_loss = last['SUP']
        elif self.position == 'SHORT':
            # Stop: Direnç Çizgisi (Asla yükseldiği yerden yukarı çıkmaz)
            if last['RES'] < self.stop_loss:
                self.stop_loss = last['RES']
                
        return self.stop_loss

# --- DASHBOARD ARAYÜZÜ ---
bot = QuantiFineAlgo()
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    html.H2("QuantiFine Canlı İşlem Terminali", className="text-center my-4 text-info"),
    dcc.Interval(id='update', interval=10000), # 10 saniyede bir güncelle
    dbc.Row([
        dbc.Col(dcc.Graph(id='main-chart'), width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bot Durumu"),
                dbc.CardBody(id='status-panel')
            ], color="dark", outline=True)
        ], width=4)
    ])
])

@app.callback(
    [Output('main-chart', 'figure'), Output('status-panel', 'children')],
    [Input('update', 'n_intervals')]
)
def refresh(n):
    df = bot.get_data()
    decision, reason = bot.check_logic(df)
    
    # Grafik oluşturma
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='yellow', width=1), name='SMA 20'))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
    
    # Dashboard bilgileri
    panel = [
        html.H5(f"Karar: {decision if decision else 'NÖTR'}", className="text-warning"),
        html.P(f"Neden: {reason}"),
        html.P(f"Hurst: {df['Hurst'].iloc[-1]:.3f}"),
        html.P(f"ADX: {df['ADX'].iloc[-1]:.1f}"),
        html.P(f"Z-Score: {df['Z_Score'].iloc[-1]:.2f}")
    ]
    
    return fig, panel

if __name__ == '__main__':
    app.run_server(debug=True)
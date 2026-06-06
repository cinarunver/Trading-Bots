import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from colorama import Fore, Style, init

init(autoreset=True)

# --- 1. AYARLAR (v6.py Yeni Algoritma Sync) ---
CONFIG = {
    "SYMBOL_LIST": ["SOL/USDT"],
    "TIMEFRAME": "1h",
    "LIMIT": 10000,  # Daha fazla veri
    "STARTING_BALANCE": 1000, 
    "FEE_RATE": 0.001,     
    "RISK_PER_TRADE_PERCENT": 0.02,
    
    # v6.py Ayarları
    "KER_WINDOW": 14,  # Kaufman Efficiency Ratio Window
    "ADX_THRESHOLD": 20,
    "SMA_PERIODS": list(range(35, 10, -1)),  # 35→11 (25 adet SMA)
    
    # Debug Mode
    "DEBUG": True,
}

# --- 2. HESAPLAMA MOTORU ---


def calculate_pivots(high, low, close):
    """Pivot Point seviyeleri hesapla (Standard, Fibonacci, Camarilla)"""
    # Classic Pivot Point
    pp = (high + low + close) / 3
    
    # Standard Pivot Seviyeleri
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    
    # Fibonacci Pivot Seviyeleri
    fib_r1 = pp + 0.382 * (high - low)
    fib_s1 = pp - 0.382 * (high - low)
    fib_r2 = pp + 0.618 * (high - low)
    fib_s2 = pp - 0.618 * (high - low)
    fib_r3 = pp + 1.0 * (high - low)
    fib_s3 = pp - 1.0 * (high - low)
    
    # Camarilla Pivot Seviyeleri
    cam_r1 = close + (high - low) * 1.1 / 12
    cam_s1 = close - (high - low) * 1.1 / 12
    cam_r2 = close + (high - low) * 1.1 / 6
    cam_s2 = close - (high - low) * 1.1 / 6
    cam_r3 = close + (high - low) * 1.1 / 4
    cam_s3 = close - (high - low) * 1.1 / 4
    cam_r4 = close + (high - low) * 1.1 / 2
    cam_s4 = close - (high - low) * 1.1 / 2
    
    return {
        'pp': pp,
        'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3,
        'fib_r1': fib_r1, 'fib_s1': fib_s1, 'fib_r2': fib_r2, 'fib_s2': fib_s2, 
        'fib_r3': fib_r3, 'fib_s3': fib_s3,
        'cam_r1': cam_r1, 'cam_s1': cam_s1, 'cam_r2': cam_r2, 'cam_s2': cam_s2,
        'cam_r3': cam_r3, 'cam_s3': cam_s3, 'cam_r4': cam_r4, 'cam_s4': cam_s4
    }

def kelly_position_size(win_rate=0.55, avg_win=1.5, avg_loss=1.0):
    """Kelly Criterion ile pozisyon boyutlandırması"""
    if avg_loss == 0:
        return 0.02  # Fallback to default
    r = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / r)
    # Half Kelly for safety, max 25%
    return max(0.01, min(kelly * 0.5, 0.25))

def fetch_data(exchange, symbol):
    print(f"{Fore.CYAN}Veri: {symbol} ({CONFIG['LIMIT']})...{Style.RESET_ALL} ", end='')
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # İndikatörler
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['rsi'] = ta.rsi(df['close'], length=14) 
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        # Kaufman Efficiency Ratio (KER)
        # KER = Abs(Close - Close[n]) / Sum(Abs(Close - Close[1])) over n
        print(f"-> KER... ", end='')
        ker_win = CONFIG["KER_WINDOW"]
        change = (df['close'] - df['close'].shift(ker_win)).abs()
        volatility = (df['close'] - df['close'].shift(1)).abs().rolling(window=ker_win).sum()
        df['ker'] = change / volatility
        df['ker'] = df['ker'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Z-Score
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # S/R Donchian Lines (Düz Hesaplama)
        print(f"-> S/R Lines... ", end='')
        win = CONFIG.get('SR_WINDOW', 50)
        # Shift 1 to avoid lookahead bias
        df['RES'] = df['high'].rolling(win).max().shift(1)
        df['SUP'] = df['low'].rolling(win).min().shift(1)
        df['MID'] = (df['RES'] + df['SUP']) / 2
        

        # ATR Average (Required for Dead Zone)
        df['atr_avg'] = df['atr'].rolling(window=50).mean()

        print("Tamam.")
        df = df.dropna().reset_index(drop=True)
        
        # Debug stats
        if CONFIG["DEBUG"]:
            print(f"\n{Fore.YELLOW}DEBUG - Veri İstatistikleri:{Style.RESET_ALL}")
            print(f"  Satır sayısı: {len(df)}")
            print(f"  KER ortalama: {df['ker'].mean():.3f}")
            print(f"  KER min/max: {df['ker'].min():.3f} / {df['ker'].max():.3f}")
            print(f"  Z-Score min/max: {df['z_score'].min():.2f} / {df['z_score'].max():.2f}")
            print(f"  ADX ortalama: {df['adx'].mean():.1f}")
            
            # Count potential signals
            ker_above = (df['ker'] > 0.7).sum()
            z_above_2 = (df['z_score'] > 2).sum()
            z_below_m2 = (df['z_score'] < -2).sum()
            adx_above = (df['adx'] > 20).sum()
            
            print(f"\n  KER > 0.7: {ker_above} ({ker_above/len(df)*100:.1f}%)")
            print(f"  Z > 2: {z_above_2} ({z_above_2/len(df)*100:.1f}%)")
            print(f"  Z < -2: {z_below_m2} ({z_below_m2/len(df)*100:.1f}%)")
            print(f"  ADX > 20: {adx_above} ({adx_above/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"\n{Fore.RED}Hata: {e}{Style.RESET_ALL}")
        return None

# --- 3. BACKTEST İŞLEYİCİSİ ---
def run_backtest(df):
    balance = CONFIG["STARTING_BALANCE"]
    position = None
    trades = []
    equity_curve = [balance]
    
    debug = CONFIG["DEBUG"]
    signal_counts = {"TREND_LONG": 0, "TREND_SHORT": 0, "REV_LONG": 0, "REV_SHORT": 0, "SKIPPED_ATR": 0, "SKIPPED_KER": 0}
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        ts = row['timestamp']
        
        # S/R Referansları
        res = row['RES']
        sup = row['SUP']
        
        # --- POZİSYON YÖNETİMİ (Donchian Trailing Stop) ---
        if position:
            close = False
            close_reason = ""
            
            if position["type"] == "LONG":
                # Stop seviyesi yukarı güncellenir (Asla aşağı inmez)
                # Donchian Alt Bandı (SUP) takip eden stop olarak kullanılır
                if sup > position['stop_price']:
                    position['stop_price'] = sup
                
                if price < position['stop_price']:
                    close = True
                    close_reason = f"STOP HIT (Price {price:.2f} < Stop {position['stop_price']:.2f})"
                        
            else: # SHORT
                # Stop seviyesi aşağı güncellenir
                # Donchian Üst Bandı (RES) takip eden stop olarak kullanılır
                if res < position['stop_price']:
                    position['stop_price'] = res
                
                if price > position['stop_price']:
                    close = True
                    close_reason = f"STOP HIT (Price {price:.2f} > Stop {position['stop_price']:.2f})"

            if close:
                size = position['size']
                entry = position['entry']
                
                if position["type"] == "LONG":
                    gross_pnl = (price - entry) * size
                else:
                    gross_pnl = (entry - price) * size
                
                exit_fee = (size * price) * CONFIG["FEE_RATE"]
                net_pnl = gross_pnl - position["fee"] - exit_fee
                
                balance += position["inv"] + net_pnl
                
                if debug:
                    color = Fore.GREEN if net_pnl > 0 else Fore.RED
                    print(f"{color}EXIT {position['type']}: {entry:.2f} -> {price:.2f} PnL: ${net_pnl:.2f}{Style.RESET_ALL}")
                
                trades.append({
                    'type': position['type'],
                    'strategy': position.get('strategy', 'UNKNOWN'),
                    'entry_date': position['date'], 'exit_date': ts,
                    'entry_price': entry, 'exit_price': price,
                    'pnl': net_pnl, 'balance': balance,
                    'pnl_pct': (net_pnl / position["inv"]) * 100,
                    'reason': close_reason
                })
                position = None
        
        # --- SİNYAL ÜRETİMİ ---
        elif position is None:
            # Dead Zone Check
            atr_filter = row['atr'] >= (row['atr_avg'] * 0.5)
            if not atr_filter:
                signal_counts["SKIPPED_ATR"] += 1
                equity_curve.append(balance)
                continue
            
            # KER Check
            ker_val = row['ker']
            if np.isnan(ker_val) or np.isnan(res) or np.isnan(sup): 
                signal_counts["SKIPPED_KER"] += 1
                equity_curve.append(balance)
                continue
            
            sig_type = None
            initial_stop = None
            strategy = None
            
            # --- TREND (KER > 0.7) ---
            # Yüksek Efficiency = Güçlü Trend
            if ker_val > 0.7:
                if row['rsi'] > 50 and row['adx'] > CONFIG['ADX_THRESHOLD']:
                    if price > res:  # Direnç Kırılımı = Long
                        sig_type = "LONG"
                        strategy = "TREND"
                        initial_stop = sup # Stop: Destek
                        signal_counts["TREND_LONG"] += 1
                elif row['rsi'] < 50 and row['adx'] > CONFIG['ADX_THRESHOLD']:
                    if price < sup:  # Destek Kırılımı = Short
                        sig_type = "SHORT"
                        strategy = "TREND"
                        initial_stop = res # Stop: Direnç
                        signal_counts["TREND_SHORT"] += 1
            
            # --- REVERSION ---
            # User request: "Hurst'u komple çıkar, KER > 0.7 girsin"
            # Bu durumda Reversion mantığı devre dışı bırakılıyor.
            
            if sig_type and initial_stop:
                # Kelly Criterion pozisyon boyutlandırması
                kelly_fraction = kelly_position_size()
                risk_usd = balance * kelly_fraction
                dist = abs(price - initial_stop)
                if dist == 0: dist = price * 0.01
                
                calc_size = risk_usd / dist
                max_size = (balance * 0.98) / price
                final_size = min(calc_size, max_size)
                
                cost = final_size * price
                fee = cost * CONFIG["FEE_RATE"]
                
                if cost > 10:
                    balance -= (cost + fee)
                    position = {
                        "type": sig_type, "entry": price,
                        "size": final_size, "inv": cost, "fee": fee,
                        "date": ts,
                        "stop_price": initial_stop,
                        "strategy": strategy
                    }
                    
                    if debug:
                        print(f"{Fore.CYAN}ENTRY {sig_type} ({strategy}): {price:.2f} Stop: {initial_stop:.2f} KER={ker_val:.2f}{Style.RESET_ALL}")

        equity_curve.append(balance)
    
    # Debug Summary
    if debug:
        print(f"\n{Fore.YELLOW}SİNYAL SAYILARI:{Style.RESET_ALL}")
        for k, v in signal_counts.items():
            print(f"  {k}: {v}")
        
    return trades, equity_curve

# --- 4. ANALİZ ---
class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(trades, equity_curve):
        if not trades: return None
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        wins = df_trades[df_trades['pnl'] > 0]
        
        win_rate = len(wins) / total_trades
        gross_profit = wins['pnl'].sum() if not wins.empty else 0
        losses = df_trades[df_trades['pnl'] <= 0]
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            "Total Trades": total_trades,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Max Drawdown": max_drawdown,
            "Net PnL": equity_curve[-1] - equity_curve[0],
            "Avg Win $": wins['pnl'].mean() if not wins.empty else 0
        }

# --- 5. GÖRSELLEŞTİRME (Plotly HTML) ---
def plot_results(symbol, df, trades, equity_curve):
    print(f"\n{Fore.CYAN}Grafik hazırlanıyor (Plotly HTML Detaylı)...{Style.RESET_ALL}")
    
    if not trades:
        print("İşlem yok.")
        return

    # Metrics hesapla (Grafik başlığı için)
    metrics = PerformanceAnalyzer.calculate_metrics(trades, equity_curve)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            f"{symbol} - Net PnL: ${metrics['Net PnL']:.2f} (WR: {metrics['Win Rate']:.0%})", 
            "RSI & ADX", 
            "Kaufman Efficiency Ratio (KER)", 
            "Bakiye Eğrisi"
        )
    )
    
    # ----------------------------------------------------
    # PANEL 1: FİYAT ve S/R KANALLARI
    # ----------------------------------------------------
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='OHLC',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ), row=1, col=1)
    
    # Donchian S/R Lines (Res/Sup)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RES'], name='Resistance', line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SUP'], name='Support', line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MID'], name='Mid', line=dict(color='gray', width=1, dash='dot'), opacity=0.5), row=1, col=1)
    
    # İŞLEMLER VE DETAYLI HOVER
    for i, t in enumerate(trades):
        is_win = t['pnl'] > 0
        color = '#00ff88' if is_win else '#ff4444'
        
        # Giriş İndeksi
        entry_rows = df[df['timestamp'] == t['entry_date']]
        if not entry_rows.empty:
            row = entry_rows.iloc[0]
            
            # Detaylı Giriş Hover Metni
            entry_hover = (
                f"<b>🔵 İŞLEM #{i+1} GİRİŞ</b><br>"
                f"<b>Yön:</b> {t['type']} ({t['strategy']})<br>"
                f"<b>Fiyat:</b> ${t['entry_price']:.2f}<br>"
                f"<b>Stop:</b> ${t.get('stop_price', 0):.2f}<br>"
                f"<b>Tarih:</b> {t['entry_date']}<br>"
                f"<br><b>📊 İNDİKATÖRLER:</b><br>"
                f"KER: {row['ker']:.2f}<br>"
                f"RSI: {row['rsi']:.1f}<br>"
                f"Z-Score: {row['z_score']:.2f}<br>"
                f"ADX: {row['adx']:.1f}<br>"
                f"RES: {row['RES']:.2f} / SUP: {row['SUP']:.2f}"
            )
            
            # Giriş Marker
            fig.add_trace(go.Scatter(
                x=[t['entry_date']], y=[t['entry_price']],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if t['type'] == 'LONG' else 'triangle-down',
                    size=14, color='yellow', line=dict(width=2, color='black')
                ),
                name=f'#{i+1} Giriş',
                hovertemplate=entry_hover + '<extra></extra>',
                showlegend=False
            ), row=1, col=1)

        # Çıkış İndeksi
        exit_rows = df[df['timestamp'] == t['exit_date']]
        if not exit_rows.empty:
            row_exit = exit_rows.iloc[0]
            
            # Detaylı Çıkış Hover Metni
            pnl_emoji = "✅" if is_win else "❌"
            exit_hover = (
                f"<b>{pnl_emoji} İŞLEM #{i+1} ÇIKIŞ</b><br>"
                f"<b>PnL:</b> ${t['pnl']:.2f} ({t['pnl_pct']:.2f}%)<br>"
                f"<b>Fiyat:</b> ${t['exit_price']:.2f}<br>"
                f"<b>Tarih:</b> {t['exit_date']}<br>"
                f"<b>Neden:</b> {t['reason']}<br>"
                f"<br><b>📊 ÇIKIŞ ANI:</b><br>"
                f"KER: {row_exit['ker']:.2f}<br>"
                f"RSI: {row_exit['rsi']:.1f}<br>"
                f"Z-Score: {row_exit['z_score']:.2f}"
            )
            
            # Çıkış Marker
            fig.add_trace(go.Scatter(
                x=[t['exit_date']], y=[t['exit_price']],
                mode='markers',
                marker=dict(symbol='circle', size=10, color=color, line=dict(width=1, color='white')),
                name=f'#{i+1} Çıkış',
                hovertemplate=exit_hover + '<extra></extra>',
                showlegend=False
            ), row=1, col=1)
            
            # İşlem Çizgisi (Ok)
            fig.add_trace(go.Scatter(
                x=[t['entry_date'], t['exit_date']], 
                y=[t['entry_price'], t['exit_price']],
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

    # ----------------------------------------------------
    # PANEL 2: RSI & ADX
    # ----------------------------------------------------
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], name='ADX', line=dict(color='yellow', width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=CONFIG['ADX_THRESHOLD'], line_dash="dash", line_color="orange", row=2, col=1) # ADX Eşik
    
    # ----------------------------------------------------
    # PANEL 3: KER (Hurst Replacement)
    # ----------------------------------------------------
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ker'], name='KER', line=dict(color='cyan', width=1.5)), row=3, col=1)
    # Kritik Seviye
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Trend (>0.7)")
    
    # ----------------------------------------------------
    # PANEL 4: BAKİYE
    # ----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=df['timestamp'].iloc[:len(equity_curve)], 
        y=equity_curve, 
        name='Bakiye', 
        line=dict(color='gold', width=2), 
        fill='tozeroy'
    ), row=4, col=1)
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        height=1000,
        title=f"QuantiFine v6 Backtest - {symbol}",
        hovermode='closest', # Closest mod, belirli markera gelince detay gösterir
        xaxis_rangeslider_visible=False
    )
    
    # HTML Kaydet
    html_file = f"Backtest_v6_{symbol.replace('/','_')}.html"
    fig.write_html(html_file)
    print(f"{Fore.GREEN}📊 Grafik güncellendi: {html_file}{Style.RESET_ALL}")
    
    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath(html_file))

def main():
    exchange = ccxt.binance()
    for sym in CONFIG["SYMBOL_LIST"]:
        df = fetch_data(exchange, sym)
        if df is not None:
            t, e = run_backtest(df)
            if t:
                plot_results(sym, df, t, e)
            else:
                print(f"\n{Fore.RED}İşlem Sinyali Oluşmadı.{Style.RESET_ALL}")
        time.sleep(1)

if __name__ == "__main__":
    main()
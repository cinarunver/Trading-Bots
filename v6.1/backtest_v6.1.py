import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import sys
import logging
from datetime import datetime
from colorama import Fore, Back, Style, init
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

init(autoreset=True)

# --- 1. AYARLAR ---
CONFIG = {
    "SYMBOL_LIST": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"],
    "TIMEFRAME": "1h",
    "LIMIT": 5000,
    "STARTING_BALANCE": 1000, 
    "FEE_RATE": 0.0004,     
    
    "RISK_PER_TRADE_PERCENT": 0.02,
    
    # v6.1 Updated Settings
    "ATR_MULTIPLIER_DEV": 0.1,  
    "HURST_WINDOW": 20,       
    "HURST_TREND": 0.52,      
    
    # Donchian Periyotları
    "PERIODS": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}

# --- 2. HESAPLAMA MOTORU ---
def calculate_hurst(series, max_lag=20):
    if len(series) < max_lag: return 0.5
    lags = range(2, max_lag)
    try:
        vals = np.array(series)
        tau = [np.sqrt(np.std(np.subtract(vals[lag:], vals[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: 
        return 0.5

def fetch_data(exchange, symbol):
    print(f"{Fore.CYAN}Veri: {symbol} ({CONFIG['LIMIT']})...{Style.RESET_ALL} ", end='')
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # --- İndikatörler ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['rsi'] = ta.rsi(df['close'], length=14) # Visual only?
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        # Hurst
        print(f"-> Hurst... ", end='')
        df['hurst'] = df['close'].rolling(window=100).apply(lambda x: calculate_hurst(x, max_lag=20), raw=True)
        
        # Donchian + Sapma
        dev = df['atr'] * CONFIG['ATR_MULTIPLIER_DEV']
        
        for p in CONFIG["PERIODS"]:
            # Backtest için shift(1) kuralı (Previous bar's channel)
            roll_high = df['high'].rolling(window=p).max().shift(1)
            roll_low = df['low'].rolling(window=p).min().shift(1)
            
            df[f'd_high_{p}'] = roll_high + dev
            df[f'd_low_{p}'] = roll_low - dev

        print("Tamam.")
        return df.dropna().reset_index(drop=True)
        
    except Exception as e:
        print(f"\n{Fore.RED}Hata: {e}{Style.RESET_ALL}")
        return None

# --- 3. BACKTEST İŞLEYİCİSİ (LADDER STOP LOGIC) ---
def run_backtest(df):
    balance = CONFIG["STARTING_BALANCE"]
    position = None
    trades = []
    equity_curve = [balance]
    
    periods = CONFIG["PERIODS"]
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        ts = row['timestamp']
        
        # --- Current Break Calculation ---
        # Kaç kanalın dışındayız?
        broken_highs = 0
        broken_lows = 0
        for p in periods:
            if price > row[f'd_high_{p}']: broken_highs += 1
            if price < row[f'd_low_{p}']: broken_lows += 1
            
        # --- POZİSYON YÖNETİMİ ---
        if position:
            # Dynamic Ladder Stop Calculation
            current_stop = position['stop_price']
            new_stop = current_stop
            
            if position["type"] == "LONG":
                # Kırılan high sayısı arttıkça stop yukarı taşınır
                # Logic: broken >= 2 -> stop index = broken - 2
                if broken_highs >= 2:
                    sl_index = broken_highs - 2
                    potential_new_stop = row[f'd_high_{periods[sl_index]}']
                    # Stop sadece yukarı hareket eder (Trailing)
                    if potential_new_stop > new_stop:
                        new_stop = potential_new_stop
                
                # Exit Check
                if price < new_stop:
                    close = True
                else:
                    close = False
                    position['stop_price'] = new_stop # Update trailed stop
                    
            else: # SHORT
                if broken_lows >= 2:
                    sl_index = broken_lows - 2
                    potential_new_stop = row[f'd_low_{periods[sl_index]}']
                    # Stop sadece aşağı hareket eder
                    if potential_new_stop < new_stop:
                        new_stop = potential_new_stop
                
                if price > new_stop:
                    close = True
                else:
                    close = False
                    position['stop_price'] = new_stop

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
                
                trades.append({
                    'type': position['type'],
                    'entry_date': position['date'], 'exit_date': ts,
                    'entry_price': entry, 'exit_price': price,
                    'pnl': net_pnl, 'balance': balance,
                    'pnl_pct': (net_pnl / position["inv"]) * 100,
                    'reason': 'LADDER_STOP'
                })
                position = None
        
        # --- SİNYAL ÜRETİMİ (v6.1 Updated) ---
        elif position is None:
            # Entry Logic: Hurst > 0.52 AND Broken >= 2
            h_val = row['hurst']
            
            if h_val > CONFIG["HURST_TREND"]:
                sig_type = None
                initial_stop = None
                
                if broken_highs >= 2:
                    sig_type = "LONG"
                    # Initial Stop: Index broken_highs - 2
                    sl_idx = broken_highs - 2
                    initial_stop = row[f'd_high_{periods[sl_idx]}']
                    
                elif broken_lows >= 2:
                    sig_type = "SHORT"
                    sl_idx = broken_lows - 2
                    initial_stop = row[f'd_low_{periods[sl_idx]}']
            
                if sig_type:
                    risk_usd = balance * CONFIG["RISK_PER_TRADE_PERCENT"]
                    
                    # Stop mesafesine göre size hesapla
                    dist_to_stop = abs(price - initial_stop)
                    if dist_to_stop == 0: dist_to_stop = price * 0.01 # Fallback
                    
                    calc_size = risk_usd / dist_to_stop
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
                            "stop_price": initial_stop
                        }

        equity_curve.append(balance)
        
    return trades, equity_curve

# --- 4. ANALİZ VE GÖRSELLEŞTİRME ---
class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(trades, equity_curve):
        if not trades: return None
        df_trades = pd.DataFrame(trades)
        returns = df_trades['pnl_pct'] / 100.0
        
        total_trades = len(trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades
        gross_profit = wins['pnl'].sum() if not wins.empty else 0
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
            "Avg Win $": wins['pnl'].mean() if not wins.empty else 0,
            "Avg Loss $": losses['pnl'].mean() if not losses.empty else 0
        }

def plot_results(symbol, df, trades, equity_curve):
    if not trades:
        print(f"{symbol}: İşlem yok.")
        return
        
    metrics = PerformanceAnalyzer.calculate_metrics(trades, equity_curve)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    fig.canvas.manager.set_window_title(f"{symbol} - v6.1 LADDER STOP")
    
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], figure=fig)
    
    # --- PANEL 1 ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161b22')
    ax1.plot(df['timestamp'], df['close'], color='#58a6ff', linewidth=1, label='Fiyat', zorder=1)
    
    # Kanal 10 ve 100 göster
    ax1.plot(df['timestamp'], df['d_high_100'], color='green', linestyle='--', alpha=0.3)
    ax1.plot(df['timestamp'], df['d_low_100'], color='red', linestyle='--', alpha=0.3)
    
    for t in trades:
        c = '#2ea043' if t['pnl'] > 0 else '#da3633'
        ax1.plot([t['entry_date'], t['exit_date']], [t['entry_price'], t['exit_price']], color=c, linewidth=2)
        ax1.scatter(t['entry_date'], t['entry_price'], marker='^' if t['type']=='LONG' else 'v', color='white', s=100, zorder=10)

    ax1.set_title(f"{symbol} - LADDER STOP STRATEGY", color='white', fontsize=14, fontweight='bold')
    
    # --- CHART INFO BOX (REQUESTED) ---
    stats_text = (
        f"PNL: ${metrics['Net PnL']:.2f}\n"
        f"Trades: {metrics['Total Trades']}\n"
        f"WR: {metrics['Win Rate']:.1%}\n"
        f"PF: {metrics['Profit Factor']:.2f}\n"
        f"Max DD: {metrics['Max Drawdown']:.1%}\n"
        f"Avg Win: ${metrics['Avg Win $']:.2f}"
    )
    # Box properties
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='#f0883e')
    # Place text in upper left or right
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props, color='#f0883e', fontfamily='monospace')

    ax1.grid(True, alpha=0.2)

    # --- PANEL 2 (Hurst) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#161b22')
    ax2.plot(df['timestamp'], df['hurst'], color='purple')
    ax2.axhline(CONFIG['HURST_TREND'], color='green', linestyle='--')
    ax2.set_ylabel("Hurst")
    
    # --- PANEL 3 (Equity) ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#161b22')
    dates = [df['timestamp'].iloc[0]] + [t['exit_date'] for t in trades]
    bals = [CONFIG["STARTING_BALANCE"]]
    for t in trades: bals.append(t['balance'])
    ax3.step(dates, bals, where='post', color='#f0883e')
    ax3.set_ylabel("Balance")

    plt.tight_layout()
    fname = f"Backtest_v6.1_Ladder_{symbol.replace('/','_')}.png"
    plt.savefig(fname, facecolor='#0d1117')
    
    print(f"\n{Fore.YELLOW}>>> {symbol} RESULTS <<<{Style.RESET_ALL}")
    print(stats_text) # Print consistency
    plt.show()

def main():
    exchange = ccxt.binance()
    for sym in CONFIG["SYMBOL_LIST"]:
        df = fetch_data(exchange, sym)
        if df is not None:
            t, e = run_backtest(df)
            if t:
                plot_results(sym, df, t, e)
            else:
                print("İşlem Sinyali Oluşmadı.")
        time.sleep(1)

if __name__ == "__main__":
    main()

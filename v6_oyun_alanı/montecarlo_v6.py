import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

init(autoreset=True)

# --- AYARLAR (v6.py Yeni Algoritma Sync) ---
CONFIG = {
    "SYMBOL_LIST": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],  # 3 coin yeterli
    "TIMEFRAME": "1h",
    "LIMIT": 2000,  # Backtest ile aynı
    "STARTING_BALANCE": 1000,
    
    # MC Ayarları
    "MC_SIMULATIONS": 2000, 
    "MC_FORECAST_LEN": 200,
    
    # Yeni v6 Ayarları
    "HURST_WINDOW": 5,
    "ADX_THRESHOLD": 20,
    "SMA_PERIODS": list(range(20, 10, -1)),
    
    # Debug Mode
    "DEBUG": True,
}

def calculate_hurst(series, max_lag=5):
    if len(series) < max_lag: return 0.5
    lags = range(2, max_lag)
    try:
        vals = np.array(series)
        tau = [np.sqrt(np.std(np.subtract(vals[lag:], vals[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

def fetch_data(exchange, symbol):
    """Fetch and prepare data"""
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        if not bars:
            return None
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['rsi'] = ta.rsi(df['close'], length=14) 
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        
        df['hurst'] = df['close'].rolling(window=CONFIG["HURST_WINDOW"]).apply(
            lambda x: calculate_hurst(x, max_lag=5), raw=True
        )
        
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        for p in CONFIG["SMA_PERIODS"]:
            df[f'sma_{p}'] = ta.sma(df['close'], length=p)
            
        df['atr_avg'] = df['atr'].rolling(window=50).mean()
        
        cleaned_df = df.dropna().reset_index(drop=True)
        print(f"  DEBUG: Raw rows={len(df)}, After dropna={len(cleaned_df)}")
        
        return cleaned_df
        
    except Exception as e:
        print(f"{Fore.RED}  ERROR in fetch_data: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return None

def generate_trades(df, symbol=""):
    """Run backtest loop and return trade returns with debug logging"""
    trade_returns = []
    position = None
    debug = CONFIG["DEBUG"]
    trade_count = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        
        # Position Management (SMA Ladder Stop)
        if position:
            close = False
            sma_values = [row[f'sma_{p}'] for p in CONFIG["SMA_PERIODS"]]
            
            if position["type"] == "LONG":
                valid_stops = [s for s in sma_values if price > s]
                if valid_stops:
                    new_stop = max(valid_stops)
                    if new_stop > position['stop_price']:
                        position['stop_price'] = new_stop
                if price < position['stop_price']: 
                    close = True
                    
            else: # SHORT
                valid_stops = [s for s in sma_values if price < s]
                if valid_stops:
                    new_stop = min(valid_stops)
                    if new_stop < position['stop_price']:
                        position['stop_price'] = new_stop
                if price > position['stop_price']: 
                    close = True

            if close:
                entry = position['entry']
                if position['type'] == 'LONG': 
                    ret = (price - entry) / entry
                else: 
                    ret = (entry - price) / entry
                
                ret -= 0.0008 # Fee
                trade_returns.append(ret)
                
                if debug and trade_count < 5:  # İlk 5 işlemi göster
                    color = Fore.GREEN if ret > 0 else Fore.RED
                    print(f"  {color}EXIT {position['type']}: Entry={entry:.2f} Exit={price:.2f} Ret={ret*100:.2f}%{Style.RESET_ALL}")
                
                position = None
                trade_count += 1
        
        # Signal Generation
        elif position is None:
            # ATR Filter
            if row['atr'] < (row['atr_avg'] * 0.5): 
                continue
                
            h_val = row['hurst']
            if np.isnan(h_val): 
                continue
            
            sig_type = None
            initial_stop = None
            
            # REVERSION (H < 0.5)
            if h_val < 0.5:
                if row['z_score'] > 2 and row['adx'] > CONFIG['ADX_THRESHOLD']:
                    if price > row['sma_20']:
                        sig_type = "LONG"
                        initial_stop = row['sma_20']
                elif row['z_score'] < -2 and row['adx'] > CONFIG['ADX_THRESHOLD']:
                    if price < row['sma_20']:
                        sig_type = "SHORT"
                        initial_stop = row['sma_20']
            
            # TREND (H >= 0.5)
            else:
                if row['rsi'] > 50 and row['adx'] > CONFIG['ADX_THRESHOLD']:
                    if price > row['sma_20']:
                        sig_type = "LONG"
                        initial_stop = row['sma_20']
                    elif price < row['sma_20']:
                        sig_type = "SHORT"
                        initial_stop = row['sma_20']
            
            if sig_type and initial_stop:
                position = {"type": sig_type, "entry": price, "stop_price": initial_stop}
                
                if debug and trade_count < 5:
                    regime = "REVERSION" if h_val < 0.5 else "TREND"
                    print(f"  {Fore.CYAN}ENTRY {sig_type} ({regime}): Price={price:.2f} H={h_val:.2f} Z={row['z_score']:.2f} ADX={row['adx']:.1f}{Style.RESET_ALL}")
    
    return trade_returns

def run_monte_carlo(trade_returns):
    if not trade_returns: return None, None
    print(f"\n{Fore.YELLOW}Monte Carlo Simülasyonu Başlıyor... ({len(trade_returns)} işlem verisi ile){Style.RESET_ALL}")
    
    # Stats
    avg_ret = np.mean(trade_returns) * 100
    std_ret = np.std(trade_returns) * 100
    print(f"  Ort. Getiri: {avg_ret:.2f}%, Std: {std_ret:.2f}%")
    
    simulations = []
    final_balances = []
    
    for _ in range(CONFIG["MC_SIMULATIONS"]):
        balance = CONFIG["STARTING_BALANCE"]
        curve = [balance]
        sim_trades = random.choices(trade_returns, k=CONFIG["MC_FORECAST_LEN"])
        
        for ret in sim_trades:
            balance = balance * (1 + ret)
            if balance < 0: balance = 0
            curve.append(balance)
            
        simulations.append(curve)
        final_balances.append(balance)
        
    return simulations, final_balances

def plot_mc(simulations, finals):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.canvas.manager.set_window_title(f"Monte Carlo v6 ({CONFIG['MC_SIMULATIONS']} Sims)")
    
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#161b22')
    
    for sim in simulations[:500]:
        ax1.plot(sim, color='#58a6ff', alpha=0.03, linewidth=1)
        
    sim_array = np.array(simulations)
    median_path = np.median(sim_array, axis=0)
    p05_path = np.percentile(sim_array, 5, axis=0)
    p95_path = np.percentile(sim_array, 95, axis=0)
    
    ax1.plot(median_path, color='#f2cc60', linewidth=2.5, label='Medyan')
    ax1.plot(p05_path, color='#da3633', linewidth=2, linestyle='--', label='%5 Alt')
    ax1.plot(p95_path, color='#2ea043', linewidth=2, linestyle='--', label='%95 Üst')
    
    ax1.axhline(CONFIG["STARTING_BALANCE"], color='gray', linestyle=':', alpha=0.5)
    ax1.set_title(f"Gelecek {CONFIG['MC_FORECAST_LEN']} İşlem Projeksiyonu", color='white', fontsize=14)
    ax1.set_ylabel("Bakiye ($)", color='white')
    ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='gray')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#161b22')
    ax2.hist(finals, bins=50, color='#238636', alpha=0.8, edgecolor='#0d1117')
    ax2.set_title("Olası Son Bakiyeler", color='white')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#0d1117')
    ax3.axis('off')
    
    avg_end = np.mean(finals)
    med_end = np.median(finals)
    min_end = np.min(finals)
    max_end = np.max(finals)
    ruin_count = len([x for x in finals if x < CONFIG["STARTING_BALANCE"]*0.5])
    ruin_prob = (ruin_count / len(finals)) * 100
    prob_profit = (len([x for x in finals if x > CONFIG["STARTING_BALANCE"]]) / len(finals)) * 100
    
    stats_text = (
        f"--- SONUÇLAR ---\n"
        f"Ort: ${avg_end:,.2f}\n"
        f"Med: ${med_end:,.2f}\n"
        f"Min/Max: ${min_end:.0f}/${max_end:.0f}\n"
        f"Kâr Olasılığı: %{prob_profit:.1f}\n"
        f"Batış Riski: %{ruin_prob:.1f}"
    )
    
    ax3.text(0.1, 0.5, stats_text, color='white', fontsize=12,
             bbox=dict(facecolor='#161b22', edgecolor='#30363d', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig("MonteCarlo_v6_Result.png", facecolor='#0d1117')
    print(f"{Fore.GREEN}Grafik Kaydedildi.{Style.RESET_ALL}")
    plt.show()

def main():
    print(f"{Fore.CYAN}{'='*60}")
    print(f"MONTE CARLO DEBUG MODE")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    exchange = ccxt.binance()
    all_returns = []
    
    for sym in CONFIG["SYMBOL_LIST"]:
        print(f"\n{Fore.YELLOW}>>> {sym} <<<{Style.RESET_ALL}")
        df = fetch_data(exchange, sym)
        if df is not None:
            print(f"  Veri: {len(df)} satır")
            trades = generate_trades(df, sym)
            all_returns.extend(trades)
            print(f"  {Fore.GREEN}Toplam: {len(trades)} işlem{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}Veri Hatası!{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"TOPLAM İŞLEM: {len(all_returns)}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    if len(all_returns) < 5:
        print(f"{Fore.RED}Yetersiz veri!{Style.RESET_ALL}")
    else:
        sims, finals = run_monte_carlo(all_returns)
        plot_mc(sims, finals)

if __name__ == "__main__":
    main()

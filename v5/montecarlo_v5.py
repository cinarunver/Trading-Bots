import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)

# --- AYARLAR (v5.py ile Uyumlu) ---
CONFIG = {
    "SYMBOL_LIST": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"],
    "TIMEFRAME": "1h",
    "LIMIT": 1500,          
    "STARTING_BALANCE": 1000,
    
    # MC Ayarları
    "MC_SIMULATIONS": 10000, 
    "MC_FORECAST_LEN": 2000,
    
    # v5 Strateji Parametreleri
    "HURST_WINDOW": 50,     # v5.py: 50
    "HURST_TREND_MIN": 0.55,# v5.py: 0.55
    "HURST_REVERT_MAX": 0.45,# v5.py: 0.45
    "ADX_THRESHOLD": 25,
    "ATR_MULT": 2.0,        # v5.py: 2.0
    "LOOKBACK": 15          # v5.py: 15
}

def calculate_hurst(series):
    if len(series) < 30: return 0.5
    values = series.values
    lags = range(2, 20)
    try:
        tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except: return 0.5

def fetch_and_generate_trades(exchange):
    all_trade_returns = []
    
    print(f"{Fore.MAGENTA}--- v5 Monte Carlo: Veri Analizi ---{Style.RESET_ALL}")
    
    for sym in CONFIG["SYMBOL_LIST"]:
        print(f"Veri İndiriliyor: {sym}...", end='')
        try:
            bars = exchange.fetch_ohlcv(sym, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
            if not bars:
                print(f" {Fore.RED}Veri Yok!{Style.RESET_ALL}")
                continue
                
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # İndikatörler
            df['ATR'] = df.ta.atr(length=14)
            df['ADX'] = df.ta.adx(length=14)['ADX_14']
            df['SMA_20'] = df['close'].rolling(20).mean()
            
            # Donchian (v5 Lookback: 15)
            df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
            df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)
            
            # Hurst
            df['HURST'] = df['close'].rolling(CONFIG["HURST_WINDOW"]).apply(calculate_hurst)
            
            print(f" Taranıyor...", end='')
            
            # v5 Backtest Logic
            position = None
            for i in range(100, len(df)):
                row = df.iloc[i]
                price = row['close']
                
                if position:
                    close = False
                    if position['type'] == 'LONG' and price < row['SUP']: close = True
                    if position['type'] == 'SHORT' and price > row['RES']: close = True
                    
                    if close:
                        entry = position['entry']
                        if position['type'] == 'LONG': ret = (price - entry) / entry
                        else: ret = (entry - price) / entry
                        
                        ret -= 0.0008 # Fee
                        all_trade_returns.append(ret)
                        position = None
                
                elif position is None:
                    h = row['HURST']
                    if np.isnan(h): continue
                    
                    # v5 Entry Logic
                    long_cond = (h > CONFIG['HURST_TREND_MIN']) and (price > row['RES']) and (row['ADX'] > CONFIG["ADX_THRESHOLD"])
                    short_cond = (h < CONFIG['HURST_REVERT_MAX']) and (price > row['SMA_20'])
                    
                    if long_cond: position = {'type': 'LONG', 'entry': price}
                    elif short_cond: position = {'type': 'SHORT', 'entry': price}
            
            print(f" {Fore.GREEN}Tamam.{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}Hata ({sym}): {e}{Style.RESET_ALL}")
            
    return all_trade_returns

def run_monte_carlo(trade_returns):
    if not trade_returns: return None, None
    
    print(f"\n{Fore.YELLOW}v5 Simülasyonu Başlıyor... ({len(trade_returns)} işlem verisi ile){Style.RESET_ALL}")
    
    simulations = []
    final_balances = []
    
    for _ in range(CONFIG["MC_SIMULATIONS"]):
        balance = CONFIG["STARTING_BALANCE"]
        curve = [balance]
        
        sim_trades = random.choices(trade_returns, k=CONFIG["MC_FORECAST_LEN"])
        
        for ret in sim_trades:
            # v5 Risk: %2 Risk Per Turn or Fixed Fraction?
            # v5.py uses dynamic sizing strictly. 
            # MC için basitlik adına %20 sabit pozisyon varsayımı (Yüksek Riske Karşı Test)
            risk_size = balance * 0.20 
            pnl = risk_size * ret
            balance += pnl
            if balance < 0: balance = 0
            curve.append(balance)
            
        simulations.append(curve)
        final_balances.append(balance)
        
    return simulations, final_balances

def plot_mc(simulations, finals):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.canvas.manager.set_window_title(f"Monte Carlo v5 Analizi")
    
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
    
    # 1. Simülasyon Yolları
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#161b22')
    
    for sim in simulations[:300]:
        ax1.plot(sim, color='#d2a8ff', alpha=0.05, linewidth=1) # v5 için Mor tema
        
    sim_array = np.array(simulations)
    median_path = np.median(sim_array, axis=0)
    p05_path = np.percentile(sim_array, 5, axis=0)
    p95_path = np.percentile(sim_array, 95, axis=0)
    
    ax1.plot(median_path, color='#f2cc60', linewidth=2.5, label='Medyan')
    ax1.plot(p05_path, color='#da3633', linewidth=2, linestyle='--', label='%5 Kötü')
    ax1.plot(p95_path, color='#2ea043', linewidth=2, linestyle='--', label='%95 İyi')
    
    ax1.axhline(CONFIG["STARTING_BALANCE"], color='gray', linestyle=':', alpha=0.5)
    ax1.set_title(f"v5 Stratejisi - Gelecek {CONFIG['MC_FORECAST_LEN']} İşlem Projeksiyonu", color='white', fontsize=14)
    ax1.set_ylabel("Bakiye ($)", color='white')
    ax1.legend(loc='upper left', facecolor='#161b22')
    ax1.grid(True, color='#30363d', linestyle=':', alpha=0.3)
    
    # 2. Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#161b22')
    ax2.hist(finals, bins=50, color='#8957e5', alpha=0.8, edgecolor='#0d1117')
    ax2.set_title("Olası Son Bakiyeler", color='white')
    ax2.set_xlabel("Bakiye ($)", color='white')
    
    # 3. İstatistikler
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#0d1117')
    ax3.axis('off')
    
    avg_end = np.mean(finals)
    med_end = np.median(finals)
    ruin_prob = (len([x for x in finals if x < CONFIG["STARTING_BALANCE"]*0.5]) / len(finals)) * 100
    prob_profit = (len([x for x in finals if x > CONFIG["STARTING_BALANCE"]]) / len(finals)) * 100
    
    stats_text = (
        f"--- v5 İSTATİSTİKLERİ ---\n\n"
        f"Ortalama Bakiye : ${avg_end:,.2f}\n"
        f"Medyan Bakiye   : ${med_end:,.2f}\n"
        f"Kâr Olasılığı   : %{prob_profit:.1f}\n"
        f"Batma Riski     : %{ruin_prob:.1f}"
    )
    
    ax3.text(0.1, 0.5, stats_text, color='white', fontsize=12,
             bbox=dict(facecolor='#161b22', edgecolor='#8957e5', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig("MonteCarlo_v5_Result.png", facecolor='#0d1117')
    print(f"{Fore.GREEN}Grafik Kaydedildi: MonteCarlo_v5_Result.png{Style.RESET_ALL}")
    plt.show()

def main():
    ex = ccxt.binance()
    trades = fetch_and_generate_trades(ex)
    
    if len(trades) < 5:
        print(f"{Fore.RED}Yetersiz veri! ({len(trades)} işlem){Style.RESET_ALL}")
    else:
        sims, finals = run_monte_carlo(trades)
        plot_mc(sims, finals)

if __name__ == "__main__":
    main()

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import time
import os
import logging
from datetime import datetime, timedelta
from colorama import Fore, Back, Style, init
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Mac/Linux Renk Uyumu
init(autoreset=True)

# --- 1. AYARLAR (Quantifine Sniper v3.2 ile uyumlu) ---
CONFIG = {
    "SYMBOL_LIST": ["STABLE/USDT", "ZIL/USDT", "RIVER/USDT", "XRP/USDT", "DOGE/USDT","BNB/USDT"],
    "TIMEFRAME": "15m",
    "LIMIT": 10000,              
    "STARTING_BALANCE": 50, 
    "FEE_RATE": 0.0005,         
    
    # --- STRATEJİ ---
    "HURST_WINDOW": 50,
    "HURST_TREND_MIN": 0.55,
    "HURST_REVERT_MAX": 0.45,
    "ADX_THRESHOLD": 25,
    "LOOKBACK": 15,
    
    # --- RİSK ---
    "RISK_PER_TRADE": 0.02,
    "ATR_MULTIPLIER": 2.0,
    
    # MC
    "MC_SIMULATIONS": 2000
}

# --- 2. YARDIMCI FONSİYONLAR ---

def calculate_hurst(series):
    """Hurst Exponent: H < 0.5 Mean Reverting, H > 0.5 Trending"""
    # Rolling işleminde Series gelecektir
    if len(series) < 30: return 0.5
    
    # Pandas Series to Numpy array handling
    vals = series.values if hasattr(series, 'values') else np.array(series)
    
    lags = range(2, 20)
    # Hurst hesaplaması (standart deviation of differences)
    # Performans için loop minimize edilebilir ama 5000 bar için acceptable
    try:
        tau = [np.sqrt(np.std(np.subtract(vals[lag:], vals[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return 0.5

# --- 3. VERİ ÇEKME ---
def fetch_historical_data(exchange, symbol):
    print(f"{Fore.CYAN}Veri indiriliyor: {symbol}...{Style.RESET_ALL}")
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # İndikatörler
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        adx_df = df.ta.adx(length=14)
        df['ADX'] = adx_df['ADX_14']
        
        # Hurst Exponent (Trend Gücü / Karakteri)
        # Performans Notu: Rolling apply yavaş olabilir, 5000 satır için kabul edilebilir.
        print(f"   {symbol} için Hurst hesaplanıyor...", end='\r')
        df['HURST'] = df['close'].rolling(window=CONFIG["HURST_WINDOW"]).apply(calculate_hurst)
        
        # Donchian (Kanal)
        df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
        df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)
        
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"{Fore.RED}Hata ({symbol}): {e}{Style.RESET_ALL}")
        return None

# --- 4. BACKTEST MOTORU (v5 Hybrid) ---
def run_backtest(df):
    balance = CONFIG["STARTING_BALANCE"]
    position = None
    trades = []
    equity_curve = [balance]
    
    # İstatistikler (Sadece bu coin özelinde backtest süresince tutulan)
    stats = {
        "WINS": 1, "LOSSES": 1, 
        "TOTAL_PROFIT": 0.0, "TOTAL_LOSS": 0.0
    }
    
    # ADF hesaplaması yavaş olabilir, her mumda hesaplamak yerine
    # Backtest olduğu için loop içinde hesaplayacağız.
    # Optimizasyon: Sadece işlem yoksa ve potansiyel sinyal varsa hesapla?
    # Doğruluk için her mumda hesaplamak en iyisi (Bot öyle yapıyor).
    
    # İlerleme çubuğu için
    print("Simülasyon Koşuyor...")
    
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        price = current_row['close']
        timestamp = current_row['timestamp']
        
        # --- POZİSYON YÖNETİMİ ---
        if position:
            close_trade = False
            
            # Çıkış: Donchian Kırılımı
            if position['type'] == 'LONG':
                if price < current_row['SUP']: close_trade = True
            elif position['type'] == 'SHORT':
                if price > current_row['RES']: close_trade = True
                
            if close_trade:
                # PnL Hesapla
                size = position['size']
                entry_price = position['entry']
                inv = position['inv']
                
                # Değerler
                if position['type'] == 'LONG':
                    pnl = (price - entry_price) * size
                else:
                    pnl = (entry_price - price) * size
                
                exit_fee = (size * price) * CONFIG["FEE_RATE"]
                entry_fee = position['fee']
                net_pnl = pnl - entry_fee - exit_fee
                
                balance += (inv if position['type'] == 'LONG' else 0) + pnl - exit_fee 
                # Not: Short için balance yönetimi basitleştirilmiştir (Spot Only varsayımı veya PnL ekleme)
                # Aslında doğrusu: Balance += Net PnL. (Margin koyduk varsayalım)
                # Üstteki kodda 'inv' + pnl diyerek bakiyeyi güncelliyoruz (Spot mantığına yakın).
                # Ancak short ise 'inv' zaten bizde duruyor muydu? Short açınca Balance azalır mı?
                # Basitlik için: Balance += Net PnL (Margin hesabı gibi)
                # Düzeltme:
                # balance += net_pnl
                # Ancak, girişte balance'dan düşmüş müydük?
                # Eski koda göre: wallet["USDT"] -= fee (sadece komisyon). 
                # Ama bir de teminat (inv) bloke edilir mi? Simülasyonda bakiye takibini net pnl üzerinden yapalım.
                # v5.py kodu: wallet["USDT"] += (inv if Long else 0) + pnl - exit_fee
                # Demek ki Long girişte inv düşülmeliydi.
                
                if net_pnl > 0:
                    stats["WINS"] += 1
                    stats["TOTAL_PROFIT"] += net_pnl
                else:
                    stats["LOSSES"] += 1
                    stats["TOTAL_LOSS"] += abs(net_pnl)
                
                trades.append({
                    'type': position['type'],
                    'entry_date': position['date'],
                    'exit_date': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': net_pnl,
                    'pnl_pct': (net_pnl / inv) * 100,
                    'balance': balance,
                    'risk_used': position['risk_pct']
                })
                position = None
        
        # --- GİRİŞ SİNYALLERİ ---
        elif position is None:
            # Parametreler
            h_val = current_row['HURST']
            atr_val = current_row['ATR']
            sma = current_row['SMA_20']
            
            # Strateji A: Trend Takip (Hurst > 0.55 + ADX + Donchian Breakout)
            is_trending = h_val > CONFIG["HURST_TREND_MIN"]
            long_trend = is_trending and (price > current_row['RES']) and (current_row['ADX'] > CONFIG["ADX_THRESHOLD"])
            
            # Strateji B: Mean Reversion (Hurst < 0.45 + Price/SMA Gap)
            is_reverting = h_val < CONFIG["HURST_REVERT_MAX"]
            short_revert = is_reverting and (price > sma)
            
            if long_trend or short_revert:
                sig_type = "LONG" if long_trend else "SHORT"
                
                # SIZING
                risk_usd = balance * CONFIG["RISK_PER_TRADE"]
                stop_dist = atr_val * CONFIG["ATR_MULTIPLIER"]
                
                if stop_dist > 0:
                    position_size = risk_usd / stop_dist 
                else:
                    position_size = 0 
                
                # Max Bakiye Kontrolü (%90)
                max_size = (balance * 0.90) / price
                final_size = min(position_size, max_size)
                
                cost = final_size * price
                entry_fee = cost * CONFIG["FEE_RATE"]
                
                if cost > 5: # Min işlem limiti
                    # Bakiyeden düş (Long ise)
                    # Simülasyon olduğu için: Long ise 'cost' kadar düşeriz. Short ise teminat marjini.
                    # Basitlik: balance -= cost (Long)
                    if sig_type == "LONG":
                        balance -= cost 
                    
                    balance -= entry_fee # Komisyon her türlü düşer
                    
                    position = {
                        'type': sig_type,
                        'entry': price,
                        'size': final_size,
                        'date': timestamp,
                        'risk_usd': risk_usd,
                        'inv': cost, 
                        'fee': entry_fee,
                        'risk_pct': CONFIG["RISK_PER_TRADE"] * 100
                    }
        
        equity_curve.append(balance)
        
    return trades, equity_curve

# --- 5. MONTE CARLO ---
def run_monte_carlo(trades):
    if not trades: return None
    returns = [t['pnl_pct'] / 100.0 for t in trades]
    sim_results = []
    start_bal = CONFIG["STARTING_BALANCE"]
    
    for _ in range(CONFIG["MC_SIMULATIONS"]):
        sim_bal = start_bal
        r_rets = random.choices(returns, k=len(returns))
        path = [start_bal]
        for r in r_rets:
            sim_bal *= (1 + r)
            path.append(sim_bal)
        sim_results.append(path)
    return sim_results

# --- 6. GÖRSELLEŞTİRME (Tek Pencere) ---
def plot_results(symbol, df, trades, equity_curve, mc_results):
    try:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('#0d1117')
        fig.canvas.manager.set_window_title(f"{symbol} - Trend Master Analysis")
        
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Panel 1: Fiyat
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor('#161b22')
        ax1.plot(df['timestamp'], df['close'], color='#58a6ff', linewidth=1, label='Close')
        ax1.set_title(f"{symbol} Fiyat ve İşlemler (Hybrid v8)", color='#c9d1d9')
        
        for t in trades:
            col = '#3fb950' if t['type']=='LONG' else '#f85149'
            ax1.scatter(t['entry_date'], t['entry_price'], color=col, marker='^' if t['type']=='LONG' else 'v', s=80, zorder=5)
            ax1.scatter(t['exit_date'], t['exit_price'], color=col, marker='o', s=50)
            ax1.plot([t['entry_date'], t['exit_date']], [t['entry_price'], t['exit_price']], color=col, linestyle='--', alpha=0.5)

        # Panel 2: Equity
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.set_facecolor('#161b22')
        ax2.plot(equity_curve, color='#f0883e')
        ax2.axhline(CONFIG["STARTING_BALANCE"], color='gray', linestyle='--')
        ax2.set_title("Bakiye Eğrisi", color='#c9d1d9')
        
        # Panel 3: Pie
        ax3 = fig.add_subplot(gs[1, 2])
        wins = len([t for t in trades if t['pnl']>0])
        ax3.pie([wins, len(trades)-wins], labels=['Win', 'Loss'], colors=['#3fb950', '#f85149'], autopct='%1.1f%%')
        ax3.set_title("Win/Loss", color='#c9d1d9')
        
        # Panel 4: Risk Stats
        ax4 = fig.add_subplot(gs[1, 3])
        ax4.set_facecolor('#161b22')
        ax4.axis('off')
        avg_risk = np.mean([t['risk_used'] for t in trades]) if trades else 0
        txt = f"Toplam İşlem: {len(trades)}\n"
        txt += f"Ort. Kelly Riski: %{avg_risk:.2f}\n"
        txt += f"Son Bakiye: ${equity_curve[-1]:.2f}\n"
        net_profit = equity_curve[-1] - CONFIG["STARTING_BALANCE"]
        txt += f"Net Kâr: ${net_profit:.2f}"
        ax4.text(0.1, 0.5, txt, color='white', fontsize=12)
        
        # Panel 5: MC Histogram
        if mc_results:
            finals = [s[-1] for s in mc_results]
            ax5 = fig.add_subplot(gs[2, :2])
            ax5.set_facecolor('#161b22')
            ax5.hist(finals, bins=40, color='#3fb950', edgecolor='black')
            ax5.set_title("MC Final Dağılımı", color='#c9d1d9')
            
            # Panel 6: MC Paths
            ax6 = fig.add_subplot(gs[2, 2:])
            ax6.set_facecolor('#161b22')
            for s in mc_results[:100]: ax6.plot(s, alpha=0.1)
            ax6.set_title("MC Yolları", color='#c9d1d9')
        
        plt.tight_layout()
        
        safe_sym = symbol.replace('/', '_')
        fname = f"Analiz_v5_{safe_sym}.png"
        plt.savefig(fname, facecolor='#0d1117')
        print(f"{Fore.GREEN}Kaydedildi: {fname}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}--- PENCEREYİ KAPATINCA DEVAM EDER ---{Style.RESET_ALL}")
        plt.show()
        plt.close('all')
        
    except ImportError: print("Matplotlib yok.")

def main():
    exchange = ccxt.binance()
    for sym in CONFIG["SYMBOL_LIST"]:
        df = fetch_historical_data(exchange, sym)
        if df is not None:
            t, e = run_backtest(df)
            mc = run_monte_carlo(t)
            
            # Konsol Raporu
            if t:
                win_rate = len([x for x in t if x['pnl']>0])/len(t)*100
                print(f"Bitti: {sym} | İşlem: {len(t)} | WR: %{win_rate:.1f} | Son: ${e[-1]:.2f}")
                plot_results(sym, df, t, e, mc)
            else:
                print(f"{sym}: İşlem sinyali yok.")
                
        time.sleep(1)

if __name__ == "__main__":
    main()

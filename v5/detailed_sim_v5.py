import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from colorama import Fore, Style, init

init(autoreset=True)

# --- AYARLAR ---
CONFIG = {
    # Daha fazla coin ve veri
    "SYMBOL_LIST": ["STABLE/USDT", "ZIL/USDT", "RIVER/USDT", "AVAX/USDT", "DOGE/USDT"],
    "TIMEFRAME": "15m",
    "LIMIT": 10000,  # "Uzun süreli" dediği için limiti artırdık
    
    # v5 Parametreleri
    "HURST_WINDOW": 50,
    "HURST_TREND_MIN": 0.55,
    "HURST_REVERT_MAX": 0.45,
    "ADX_THRESHOLD": 25,
    "LOOKBACK": 15,
    
    "RISK_PER_TRADE": 0.02,
    "ATR_MULTIPLIER": 2.0,
    "STARTING_BALANCE": 50
}

# --- YARDIMCI FONSİYONLAR ---

def calculate_hurst(series):
    """Standard Hurst Exponent"""
    values = series.values
    if len(values) < 20: return 0.5
    lags = range(2, 20)
    try:
        tau = [np.sqrt(np.std(values[lag:] - values[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except: return 0.5

def fetch_data(exchange, symbol):
    print(f"{Fore.CYAN}Veri Çekiliyor: {symbol} ({CONFIG['LIMIT']} bar)...{Style.RESET_ALL}")
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG['LIMIT'])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # İndikatörler
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA_20'] = df['close'].rolling(20).mean()
        adx = df.ta.adx(length=14); df['ADX'] = adx['ADX_14']
        
        # Hurst
        print(f"   Hurst Hesaplanıyor ({CONFIG['HURST_WINDOW']} rolling)...")
        df['HURST'] = df['close'].rolling(CONFIG['HURST_WINDOW']).apply(calculate_hurst)
        
        # Donchian
        df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
        df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)
        
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"{Fore.RED}Hata: {e}{Style.RESET_ALL}")
        return None

def run_simulation(df):
    balance = CONFIG["STARTING_BALANCE"]
    position = None
    trades = []
    
    # Context verilerini saklamak için
    trade_logs = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        ts = row['timestamp']
        
        # --- POZİSYON ---
        if position:
            close = False
            if position['type'] == 'LONG' and price < row['SUP']: close = True
            if position['type'] == 'SHORT' and price > row['RES']: close = True
            
            if close:
                entry = position['entry']
                size = position['size']
                if position['type'] == 'LONG': pnl = (price - entry) * size
                else: pnl = (entry - price) * size
                
                balance += pnl
                trades.append({
                    'type': position['type'], 'entry_date': position['date'], 'exit_date': ts,
                    'entry_price': entry, 'exit_price': price, 'pnl': pnl, 
                    'context': position['context']
                })
                position = None
                
        # --- SİNYAL ---
        elif position is None:
            h = row['HURST']
            adx = row['ADX']
            sma = row['SMA_20']
            res = row['RES']
            
            # v5 Mantığı
            long_cond = (h > CONFIG['HURST_TREND_MIN']) and (price > res) and (adx > CONFIG['ADX_THRESHOLD'])
            short_cond = (h < CONFIG['HURST_REVERT_MAX']) and (price > sma) # Mean Reversion Short
            
            sig = None
            if long_cond: sig = "LONG"
            elif short_cond: sig = "SHORT"
            
            if sig:
                risk = balance * CONFIG["RISK_PER_TRADE"]
                stop = row['ATR'] * CONFIG["ATR_MULTIPLIER"]
                if stop > 0:
                    size = risk / stop
                    # Max Bakiye Koruma
                    max_s = (balance * 0.95) / price
                    size = min(size, max_s)
                    
                    context_str = f"H:{h:.2f} ADX:{adx:.1f}"
                    if sig == "LONG": context_str += f"\nBrk>{res:.2f}"
                    else: context_str += f"\nGap>{sma:.2f}"
                    
                    position = {
                        'type': sig, 'entry': price, 'size': size, 'date': ts,
                        'context': context_str
                    }

    return trades, balance

def plot_detailed(symbol, df, trades, final_bal):
    if not trades:
        print(f"{symbol}: İşlem yok.")
        return

    # Stil Ayarları
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12)) # Daha yüksek görsel
    fig.patch.set_facecolor('#0d1117')
    fig.canvas.manager.set_window_title(f"{symbol} DETAYLI ANALİZ")
    
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], figure=fig)
    
    # --- PANEL 1: FİYAT VE İŞLEMLER ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161b22')
    ax1.plot(df['timestamp'], df['close'], color='#58a6ff', linewidth=1.2, label='Fiyat', zorder=1)
    
    # Kanalları daha belirgin ama sade yap
    ax1.plot(df['timestamp'], df['RES'], color='#238636', linestyle='--', linewidth=1, alpha=0.5, label='Üst Bant')
    ax1.plot(df['timestamp'], df['SUP'], color='#da3633', linestyle='--', linewidth=1, alpha=0.5, label='Alt Bant')
    
    # İşlemleri Çiz
    for t in trades:
        # Renk: Kazanç Yeşil, Kayıp Kırmızı
        color = '#3fb950' if t['pnl'] > 0 else '#f85149'
        
        # Giriş/Çıkış Noktaları
        ax1.scatter(t['entry_date'], t['entry_price'], color='white', marker='^' if t['type']=='LONG' else 'v', s=100, zorder=10, edgecolors=color, linewidth=2)
        ax1.scatter(t['exit_date'], t['exit_price'], color=color, marker='X' if t['pnl']<0 else 'o', s=80, zorder=10)
        
        # Bağlantı Çizgisi (Kesik çizgi ile daha temiz)
        ax1.plot([t['entry_date'], t['exit_date']], [t['entry_price'], t['exit_price']], color=color, linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Arka planı boya (İşlem süresince)
        ax1.axvspan(t['entry_date'], t['exit_date'], color=color, alpha=0.1)

    ax1.set_ylabel("Fiyat ($)", color='white')
    ax1.legend(loc='upper left', facecolor='#161b22', edgecolor='gray')
    ax1.set_title(f"{symbol} - Trend & Hurst Stratejisi (Sniper v5)", color='#c9d1d9', fontsize=14, fontweight='bold')
    ax1.grid(True, color='#30363d', linestyle=':', alpha=0.5)

    # --- PANEL 2: HURST EXPONENT (Sinyal Kaynağı) ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('#161b22')
    
    h_line = df['HURST']
    ax2.plot(df['timestamp'], h_line, color='#d2a8ff', linewidth=1, label='Hurst Exponent')
    
    # Kritik Bölgeler
    ax2.axhline(CONFIG['HURST_TREND_MIN'], color='#3fb950', linestyle='--', alpha=0.5)
    ax2.axhline(CONFIG['HURST_REVERT_MAX'], color='#f85149', linestyle='--', alpha=0.5)
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    
    # Bölgeleri Boya (Trend vs Reversion)
    ax2.fill_between(df['timestamp'], h_line, CONFIG['HURST_TREND_MIN'], where=(h_line > CONFIG['HURST_TREND_MIN']), color='#3fb950', alpha=0.2, label='Trend Bölgesi')
    ax2.fill_between(df['timestamp'], h_line, CONFIG['HURST_REVERT_MAX'], where=(h_line < CONFIG['HURST_REVERT_MAX']), color='#f85149', alpha=0.2, label='Reversion Bölgesi')
    
    ax2.set_ylabel("Hurst", color='white')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, color='#30363d', linestyle=':', alpha=0.5)

    # --- PANEL 3: BAKİYE EĞRİSİ ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('#161b22')
    
    # Bakiye eğrisini oluştur (Her işlem bitiminde)
    dates = [df['timestamp'].iloc[0]] + [t['exit_date'] for t in trades]
    balance_curve = [CONFIG["STARTING_BALANCE"]]
    current_bal = CONFIG["STARTING_BALANCE"]
    for t in trades:
        current_bal += t['pnl']
        balance_curve.append(current_bal)
        
    ax3.step(dates, balance_curve, where='post', color='#f0883e', linewidth=2)
    ax3.axhline(CONFIG["STARTING_BALANCE"], color='gray', linestyle='--', alpha=0.5)
    
    # Sonuç Metni
    wins = len([x for x in trades if x['pnl']>0])
    wr = (wins/len(trades))*100 if trades else 0
    total_profit = current_bal - CONFIG["STARTING_BALANCE"]
    
    info_text = f"Toplam İşlem: {len(trades)}\nWin Rate: %{wr:.1f}\nNet Kâr: ${total_profit:.2f}"
    ax3.text(0.02, 0.7, info_text, transform=ax3.transAxes, color='white', fontsize=11, bbox=dict(facecolor='#161b22', edgecolor='gray', alpha=0.8))
    
    ax3.set_ylabel("Bakiye ($)", color='white')
    ax3.grid(True, color='#30363d', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    fname = f"Sim_v5_{symbol.replace('/','_')}.png"
    plt.savefig(fname, facecolor='#0d1117')
    print(f"{Fore.GREEN}Grafik Güncellendi: {fname}{Style.RESET_ALL}")
    plt.show() # Blocking
    plt.close('all')

def main():
    exchange = ccxt.binance()
    for sym in CONFIG["SYMBOL_LIST"]:
        df = fetch_data(exchange, sym)
        if df is not None:
            trades, end_bal = run_simulation(df)
            plot_detailed(sym, df, trades, end_bal)

if __name__ == "__main__":
    main()

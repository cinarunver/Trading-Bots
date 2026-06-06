import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import time
from datetime import datetime, timedelta
from colorama import Fore, Back, Style, init

# --- 1. AYARLAR (v7.1 TREND MASTER ile uyumlu) ---
CONFIG = {
    "SYMBOL_LIST": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", 
        "XRP/USDT", "DOGE/USDT", "PEPE/USDT"
    ],
    "TIMEFRAME": "15m",
    "LIMIT": 5000,          
    "STARTING_BALANCE": 1000, 
    "FEE_RATE": 0.0005,    
    
    # Trend Master v7.1 Strateji
    "ADX_THRESHOLD": 20,
    "LOOKBACK": 12,           # Giriş Tetikleyicisi
    
    # ÖLÜ ALAN & TREND FİLTRESİ
    "EMA_PERIOD": 50,         # Trend Yönü için (Eskisi 20 idi)
    "DEAD_ZONE_MULTIPLIER": 1.0, 
    
    # ÇIKIŞ (ADAPTİF STOP):
    "BASE_EXIT": 10,
    "MAX_EXIT": 24,
    "EXIT_INCREMENT": 3,
    
    # Monte Carlo Ayarları
    "MC_SIMULATIONS": 1000, 
}

init(autoreset=True)

# --- 2. VERİ ÇEKME ---
def fetch_historical_data(exchange, symbol):
    print(f"{Fore.CYAN}Veri indiriliyor: {symbol}...{Style.RESET_ALL}")
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # --- İNDİKATÖRLER ---
        # ADX ve RSI
        adx = df.ta.adx(length=14)
        df = pd.concat([df, adx], axis=1)
        df.rename(columns={'ADX_14': 'ADX'}, inplace=True)
        df['RSI'] = df.ta.rsi(length=14)
        
        # 2. TREND YÖNÜ (EMA EĞİMİ)
        df['EMA'] = df.ta.ema(length=CONFIG["EMA_PERIOD"])
        df['EMA_PREV'] = df['EMA'].shift(1) # Eğim için önceki mum
        
        atr = df.ta.atr(length=14)
        df['DZ_TOP'] = df['EMA'] + (atr * CONFIG["DEAD_ZONE_MULTIPLIER"])
        df['DZ_BOT'] = df['EMA'] - (atr * CONFIG["DEAD_ZONE_MULTIPLIER"])
        
        # 3. GİRİŞ/ÇIKIŞ KANALLARI (Donchian)
        df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
        df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)
        
        # Donchian Channels (Dinamik Çıkışlar için ÖN HESAPLAMA)
        # Olası tüm çıkış periyotlarını (5, 7, 9, ... 15) önceden hesaplıyoruz
        # Böylece döngü içinde hızlıca erişebiliriz.
        current_lb = CONFIG["BASE_EXIT"]
        while current_lb <= CONFIG["MAX_EXIT"]:
            df[f'EXIT_RES_{current_lb}'] = df['high'].rolling(current_lb).max().shift(1)
            df[f'EXIT_SUP_{current_lb}'] = df['low'].rolling(current_lb).min().shift(1)
            current_lb += CONFIG["EXIT_INCREMENT"]
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"{Fore.RED}Hata ({symbol}): {e}{Style.RESET_ALL}")
        return None

# --- 3. BACKTEST MOTORU (v5.0 ADAPTIVE) ---
def run_backtest(df):
    balance = CONFIG["STARTING_BALANCE"]
    position = None
    trades = []
    equity_curve = [balance]
    
    # Adaptif Ayar Başlangıcı
    current_exit_lookback = CONFIG["BASE_EXIT"]
    
    for i, row in df.iterrows():
        price = row['close']
        
        # --- POZİSYON VARSA (YÖNET) ---
        if position:
            close_trade = False
            exit_price = 0
            
            # Dinamik Stop Seviyelerini Çek
            # O anki mod neyse (5, 7, 9...) ona ait kolona bak
            try:
                exit_res = row[f'EXIT_RES_{current_exit_lookback}']
                exit_sup = row[f'EXIT_SUP_{current_exit_lookback}']
            except KeyError:
                # Kolon bulunamazsa varsayılana dön (Güvenlik)
                exit_res = row[f'EXIT_RES_{CONFIG["BASE_EXIT"]}']
                exit_sup = row[f'EXIT_SUP_{CONFIG["BASE_EXIT"]}']

            
            # Trailing Stop Kontrolü
            if position['type'] == 'LONG':
                if price < exit_sup:
                    close_trade = True
                    exit_price = price
            
            elif position['type'] == 'SHORT':
                if price > exit_res:
                    close_trade = True
                    exit_price = price
            
            if close_trade:
                # Kapanış Hesapla
                entry_val = position['size'] * position['entry']
                exit_val = position['size'] * exit_price
                fee = (entry_val + exit_val) * CONFIG["FEE_RATE"]
                
                if position['type'] == "LONG":
                    gross_pnl = exit_val - entry_val
                else: # SHORT
                    gross_pnl = entry_val - exit_val
                
                net_pnl = gross_pnl - fee
                balance += net_pnl
                
                # --- ADAPTİF MANTIK (DERS ÇIKARMA) ---
                if net_pnl <= 0:
                    # Zarar -> Gevşet (Daha fazla alan ver)
                    current_exit_lookback = min(current_exit_lookback + CONFIG["EXIT_INCREMENT"], CONFIG["MAX_EXIT"])
                else:
                    # Kar -> Sıkılaştır (Fabrika Ayarlarına Dön)
                    current_exit_lookback = CONFIG["BASE_EXIT"]
                
                trades.append({
                    'type': position['type'],
                    'entry_date': position['date'],
                    'exit_date': row['timestamp'],
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (net_pnl / position['temp_balance']) * 100, 
                    'balance': balance,
                    'exit_mode': current_exit_lookback # Kayıt (Analiz için)
                })
                
                position = None
        
        # --- POZİSYON YOKSA (ARA) ---
        else:
            # Ölü Alan Kontrolü
            in_dead_zone = (price > row['DZ_BOT']) and (price < row['DZ_TOP'])
            
            # Trend Yönü Kontrolü (EMA Eğimi)
            is_trend_up = row['EMA'] > row['EMA_PREV']
            is_trend_down = row['EMA'] < row['EMA_PREV']
            
            if not in_dead_zone:
                # Sinyal Kontrolleri (Trend Master v7.1 Logic)
                # LONG: EMA Yukarı Baksın + Ölü Alan Üstü Kırılsın + Donchian Tepesi Kırılsın
                long_sig = is_trend_up and (price > row['DZ_TOP']) and (price > row['RES']) and (row['ADX'] > CONFIG["ADX_THRESHOLD"])
                
                # SHORT: EMA Aşağı Baksın + Ölü Alan Altı Kırılsın + Donchian Dibi Kırılsın
                short_sig = is_trend_down and (price < row['DZ_BOT']) and (price < row['SUP']) and (row['ADX'] > CONFIG["ADX_THRESHOLD"])
                
                if long_sig:
                    size = (balance * (1 - CONFIG["FEE_RATE"])) / price
                    position = {
                        'type': 'LONG', 'entry': price, 'size': size, 
                        'date': row['timestamp'], 'temp_balance': balance
                    }
                    
                elif short_sig:
                    size = (balance * (1 - CONFIG["FEE_RATE"])) / price
                    position = {
                        'type': 'SHORT', 'entry': price, 'size': size, 
                        'date': row['timestamp'], 'temp_balance': balance
                    }
        
        equity_curve.append(balance)
        
    return trades, equity_curve

# --- 4. MONTE CARLO SİMÜLASYONU ---
def run_monte_carlo(trades):
    if not trades: return None
    
    returns = [t['pnl_pct'] / 100.0 for t in trades] # % PnL'i ondalığa çevir
    start_bal = CONFIG["STARTING_BALANCE"]
    
    sim_results = []
    
    print(f"\n{Fore.MAGENTA}Monte Carlo Simülasyonu Başlıyor ({CONFIG['MC_SIMULATIONS']} adet)...{Style.RESET_ALL}")
    
    for _ in range(CONFIG["MC_SIMULATIONS"]):
        sim_balance = start_bal
        # İşlem geçmişinden rastgele seçim yap (Resampling / Bootstrap)
        # Orijinal işlem sayısı kadar rastgele işlem seçelim
        random_returns = random.choices(returns, k=len(returns))
        
        path = [start_bal]
        for ret in random_returns:
            sim_balance = sim_balance * (1 + ret)
            path.append(sim_balance)
            
        sim_results.append(path)
        
    return sim_results

# --- 5. RAPORLAMA ---
def report_results(symbol, trades, equity, mc_results):
    print(f"\n{Back.BLUE}{Fore.WHITE} --- {symbol} BACKTEST SONUÇLARI --- {Style.RESET_ALL}")
    
    if not trades:
        print(f"{Fore.RED}Hiç işlem açılmadı.{Style.RESET_ALL}")
        return

    # Temel Metrikler
    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    win_rate = (wins / total_trades) * 100
    total_pnl = equity[-1] - CONFIG["STARTING_BALANCE"]
    roi = (total_pnl / CONFIG["STARTING_BALANCE"]) * 100
    
    print(f"Toplam İşlem: {total_trades}")
    print(f"Kazanma Oranı (Win Rate): {Fore.GREEN if win_rate > 50 else Fore.RED}%{win_rate:.2f}{Style.RESET_ALL}")
    print(f"Net Kâr/Zarar: {Fore.GREEN if total_pnl > 0 else Fore.RED}${total_pnl:.2f}{Style.RESET_ALL} (ROİ: %{roi:.2f})")
    print(f"Son Bakiye: ${equity[-1]:.2f}")
    
    # Monte Carlo Analizi
    if mc_results:
        final_balances = [sim[-1] for sim in mc_results]
        avg_final = np.mean(final_balances)
        median_final = np.median(final_balances)
        min_final = np.min(final_balances)
        max_final = np.max(final_balances)
        
        # Risk of Ruin (Bakiye başlangıcın %50 altına düşme ihtimali olarak basitleştirelim)
        ruin_count = 0
        for sim in mc_results:
            if min(sim) < CONFIG["STARTING_BALANCE"] * 0.5:
                ruin_count += 1
        prob_ruin = (ruin_count / len(mc_results)) * 100
        
        print(f"\n{Fore.YELLOW}--- Monte Carlo Analizi ---{Style.RESET_ALL}")
        print(f"Ortalama Beklenti (Bakiye): ${avg_final:.2f}")
        print(f"Medyan Beklenti: ${median_final:.2f}")
        print(f"En Kötü Senaryo: ${min_final:.2f}")
        print(f"En İyi Senaryo: ${max_final:.2f}")
        print(f"Risk of Ruin (Batar %50): %{prob_ruin:.2f}")


# --- 6. GÖRSELLEŞTİRME (v3.1 Stili) ---
def plot_results(symbol, df, trades, equity_curve, mc_results):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mticker
    except ImportError:
        print(f"{Fore.RED}Matplotlib yüklü değil.{Style.RESET_ALL}")
        return

    # 1. TEK PENCERE AYARI (Full Screen Modu için büyük boy)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12)) 
    fig.patch.set_facecolor('#0d1117')
    fig.canvas.manager.set_window_title(f"{symbol} - Trend Master v7.1 Analizi")

    # Grid Düzeni: 3 Satır, 4 Sütun
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # --- PANEL 1: FİYAT VE SİNYALLER (En Üst, Tam Genişlik) ---
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.set_facecolor('#161b22')
    ax_price.plot(df['timestamp'], df['close'], color='#58a6ff', linewidth=1, label='Fiyat')
    
    # İşlemleri İşaretle
    for t in trades:
        entry_time = t['entry_date']
        exit_time = t['exit_date']
        entry_price = t['entry_price']
        exit_price = t['exit_price']
        is_win = t['pnl'] > 0
        
        # Giriş Marker
        marker = '^' if t['type'] == 'LONG' else 'v'
        color = '#3fb950' if t['type'] == 'LONG' else '#f85149'
        ax_price.scatter(entry_time, entry_price, marker=marker, color=color, s=100, zorder=5, edgecolors='white')
        
        # Çıkış Marker
        exit_marker = 'v' if t['type'] == 'LONG' else '^'
        exit_col = '#3fb950' if is_win else '#f85149'
        ax_price.scatter(exit_time, exit_price, marker=exit_marker, color=exit_col, s=80, zorder=5, edgecolors='white')
        
        # İşlem Çizgisi
        ax_price.plot([entry_time, exit_time], [entry_price, exit_price], color=exit_col, linestyle='--', alpha=0.6)

    ax_price.set_title(f'{symbol} Fiyat ve Sinyaller', color='#c9d1d9', fontweight='bold')
    ax_price.grid(True, alpha=0.15)
    
    # --- PANEL 2: EQUITY CURVE (Orta Sol - 2 birim) ---
    ax_equity = fig.add_subplot(gs[1, :2])
    ax_equity.set_facecolor('#161b22')
    ax_equity.plot(equity_curve, color='#f0883e', linewidth=2)
    ax_equity.axhline(CONFIG["STARTING_BALANCE"], color='gray', linestyle='--', alpha=0.5)
    ax_equity.set_title('Bakiye Gelişimi (Equity)', color='#c9d1d9')
    ax_equity.grid(True, alpha=0.15)
    
    # --- PANEL 3: İŞLEM BAŞARISI PASTA (Orta Sağ - 1 birim) ---
    ax_pie = fig.add_subplot(gs[1, 2])
    
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = len(trades) - wins
    if trades:
        ax_pie.pie([wins, losses], labels=['Kazanç', 'Kayıp'], colors=['#3fb950', '#f85149'], autopct='%1.1f%%', textprops={'color': 'white'})
        ax_pie.set_title(f'Win Rate ({len(trades)} İşlem)', color='#c9d1d9')

    # --- PANEL 4: MONTE CARLO (Orta Sağ Köşe - 1 birim) ---
    # Percentiles Bar
    if mc_results:
        final_balances = [sim[-1] for sim in mc_results]
        
        ax_bar = fig.add_subplot(gs[1, 3])
        ax_bar.set_facecolor('#161b22')
        percentiles = [5, 25, 50, 75, 95]
        vals = [np.percentile(final_balances, p) for p in percentiles]
        colors_bar = ['#f85149' if v < CONFIG["STARTING_BALANCE"] else '#3fb950' for v in vals]
        
        ax_bar.barh([f"%{p}" for p in percentiles], vals, color=colors_bar)
        ax_bar.axvline(CONFIG["STARTING_BALANCE"], color='orange', linestyle='--')
        ax_bar.set_title('Risk Analizi (Percentiles)', color='#c9d1d9', fontsize=9)
        ax_bar.tick_params(axis='y', labelsize=8)
        
        # --- PANEL 5: MC HİSTOGRAM (Alt Sol - 2 birim) ---
        ax_hist = fig.add_subplot(gs[2, :2])
        ax_hist.set_facecolor('#161b22')
        n, bins, patches = ax_hist.hist(final_balances, bins=40, edgecolor='#30363d')
        for i, p in enumerate(patches):
            center = (bins[i] + bins[i+1])/2
            p.set_facecolor('#3fb950' if center >= CONFIG["STARTING_BALANCE"] else '#f85149')
        ax_hist.axvline(CONFIG["STARTING_BALANCE"], color='orange', linestyle='--')
        ax_hist.set_title('Monte Carlo Dağılımı', color='#c9d1d9')
        
        # --- PANEL 6: MC YOLLARI (Alt Sağ - 2 birim) ---
        ax_paths = fig.add_subplot(gs[2, 2:])
        ax_paths.set_facecolor('#161b22')
        for sim in mc_results[:100]:
            col = '#3fb950' if sim[-1] > CONFIG["STARTING_BALANCE"] else '#f85149'
            ax_paths.plot(sim, color=col, alpha=0.1)
        ax_paths.set_title('Simülasyon Yolları (İlk 100)', color='#c9d1d9')

    plt.tight_layout()
    
    # KAYDETME VE GÖSTERME
    safe_sym = symbol.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"Analiz_{safe_sym}_{timestamp}.png"
    plt.savefig(filename, facecolor='#0d1117', edgecolor='none')
    print(f"{Fore.GREEN}Grafik Kaydedildi: {filename}{Style.RESET_ALL}")
    
    # Pencerenin maksimize edilmesi (Backend'e göre değişir ama figsize büyük zaten)
    try:
        mng = plt.get_current_fig_manager()
        # MacOS için tam ekran komutu (genellikle backend'e bağlıdır)
        # mng.resize(*mng.window.maxsize()) # TkAgg vs için
        # Mac'te 'macosx' backend kullanılıyorsa full screen zordur, figsize yeterli.
        pass
    except: pass
    
    print(f"{Fore.CYAN}--- PENCEREYİ KAPATINCA SIRADAKİ COIN GELECEK ---{Style.RESET_ALL}")
    plt.show() 
    plt.close('all') # KESİN TEMİZLİK

def main():
    try:
        # Binance Public
        exchange = ccxt.binance()
        
        for sym in CONFIG["SYMBOL_LIST"]:
            df = fetch_historical_data(exchange, sym)
            if df is not None and not df.empty:
                t, e = run_backtest(df)
                if t:
                    mc = run_monte_carlo(t)
                    report_results(sym, t, e, mc)
                    # Yeni Görselleştirme Fonksiyonu
                    # df'i de gönderiyoruz ki fiyat grafiği çizilebilsin
                    plot_results(sym, df, t, e, mc) 
                else:
                    print(f"{sym}: İşlem sinyali oluşmadı.")
            time.sleep(1)

        print(f"\n{Fore.GREEN}Tüm Testler Tamamlandı.{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print("\nDurduruldu.")

if __name__ == "__main__":
    main()

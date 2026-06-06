"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SNIPER v3.2 BACKTEST & MONTE CARLO SIMULASYONU                  ║
║                                                                              ║
║  Bu kod iki analiz sunar:                                                    ║
║  1. BACKTEST: Son 1 aylik gercek veri uzerinde strateji testi                ║
║  2. MONTE CARLO: 2000 farkli senaryoda 1000 mumlu simulasyon                 ║
║                                                                              ║
║  Strateji: Sniper v3.2 (ATR Enhanced)                                        ║
║  - Donchian Breakout (15 Bar)                                                ║
║  - ADX > 25 Filtresi                                                         ║
║  - RSI 30-70 Araligi                                                         ║
║  - ATR Bazli Dinamik Stop Loss (2x ATR)                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
#                              AYARLAR
# ══════════════════════════════════════════════════════════════════════════════

# Veri Ayarlari
SYMBOL = "SOL-USD"           # Test edilecek coin (Yahoo formatı)
PERIOD = "1y"                # 1 yillik veri cek
TIMEFRAME = "1h"             # 1 saatlik mum (yfinance 15m sinirli veri veriyor)

# Monte Carlo Ayarlari
NUM_SIMULATIONS = 2000       # 2000 farkli senaryo
SIM_LENGTH_BARS = 1000       # Her simulasyon 1000 mum

# Backtest Ayarlari (1 ay = ~720 bar @ 1h)
BACKTEST_BARS = 720          # Son 1 aylik veri (1h)

# ═══════════════════════════════════════════════════════════════
# SNIPER v3.2 STRATEJI PARAMETRELERI (Bot ile birebir ayni!)
# ═══════════════════════════════════════════════════════════════
LOOKBACK = 15                # Donchian kanali suresi
ADX_THRESHOLD = 25           # Minimum ADX degeri
RSI_MAX_LONG = 70            # Long icin max RSI
RSI_MIN_SHORT = 30           # Short icin min RSI
ATR_MULTIPLIER = 2.0         # Dinamik stop loss carpani
COMMISSION = 0.0005          # %0.05 komisyon

# Para Yonetimi
STARTING_CAPITAL = 50

# ══════════════════════════════════════════════════════════════════════════════
#                           VERI HAZIRLAMA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_and_prepare_data():
    """yfinance'den veri cek ve indikatorleri hesapla"""
    print(f"[*] {SYMBOL} verileri indiriliyor ({PERIOD}, {TIMEFRAME})...")
    
    df = yf.download(SYMBOL, period=PERIOD, interval=TIMEFRAME, progress=False)
    
    # MultiIndex duzeltmesi (yfinance 1.1.0+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if len(df) < SIM_LENGTH_BARS + LOOKBACK + 100:
        print(f"[!] Yeterli veri yok! {len(df)} bar mevcut, {SIM_LENGTH_BARS + LOOKBACK + 100} gerekli.")
        return None
        
    print(f"[OK] {len(df)} bar veri alindi.")
    
    # --- INDIKATORLER (Sniper v3.2 ile ayni) ---
    adx_df = df.ta.adx(length=14)
    df['ADX'] = adx_df['ADX_14']
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)  # v3.2 eklentisi
    
    # Donchian Kanallari (Shift(1) ile onceki mumlar baz alinir)
    df['RES'] = df['High'].rolling(LOOKBACK).max().shift(1)  # Direnc (Long giris)
    df['SUP'] = df['Low'].rolling(LOOKBACK).min().shift(1)   # Destek (Short giris)
    
    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
#                     SIMULASYON MOTORU (SNIPER v3.2)
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(df, start_idx, sim_length, record_trades=False):
    """
    Sniper v3.2 stratejisi ile tek bir simulasyon
    
    Args:
        df: Fiyat verisi DataFrame
        start_idx: Baslangic index'i
        sim_length: Simulasyon uzunlugu (bar)
        record_trades: True ise islem detaylarini kaydet
    
    Returns:
        dict: Simulasyon sonuclari
    """
    capital = STARTING_CAPITAL
    position = None  # {'type': 'LONG'/'SHORT', 'entry': price, 'size': qty, 'entry_idx': idx}
    
    trades = []
    trade_details = []
    equity_curve = []
    
    end_idx = min(start_idx + sim_length, len(df))
    
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        curr_price = float(row['Close'])
        res = float(row['RES'])
        sup = float(row['SUP'])
        adx = float(row['ADX'])
        rsi = float(row['RSI'])
        atr = float(row['ATR'])
        
        # ═══════════════════════════════════════════════════════════
        # POZISYON VARSA: CIKIS KONTROLU (v3.2 ATR Stop dahil)
        # ═══════════════════════════════════════════════════════════
        if position:
            pos_type = position['type']
            entry = position['entry']
            size = position['size']
            entry_idx = position['entry_idx']
            
            # Anlik deger hesapla
            if pos_type == "LONG":
                curr_value = size * curr_price
                # v3.2: Dinamik ATR Stop
                dynamic_stop = entry - (atr * ATR_MULTIPLIER)
                should_exit = (curr_price < sup) or (curr_price < dynamic_stop)
            else:  # SHORT
                pnl_raw = (entry - curr_price) * size
                curr_value = (size * entry) + pnl_raw
                # v3.2: Dinamik ATR Stop
                dynamic_stop = entry + (atr * ATR_MULTIPLIER)
                should_exit = (curr_price > res) or (curr_price > dynamic_stop)
            
            if should_exit:
                # Komisyon dus ve kapat
                final_value = curr_value * (1 - COMMISSION)
                pnl_pct = (final_value / (size * entry) - 1) * 100
                trades.append(pnl_pct)
                
                if record_trades:
                    trade_details.append({
                        'type': pos_type,
                        'entry_idx': entry_idx - start_idx,
                        'entry_price': entry,
                        'exit_idx': i - start_idx,
                        'exit_price': curr_price,
                        'pnl_pct': pnl_pct
                    })
                
                capital = final_value
                position = None
            
            equity_curve.append(curr_value if position else capital)
        
        # ═══════════════════════════════════════════════════════════
        # POZISYON YOKSA: GIRIS SINYALI ARA
        # ═══════════════════════════════════════════════════════════
        else:
            # Long Sinyal: Fiyat > Direnc AND ADX > 25 AND RSI < 70
            long_signal = (curr_price > res) and (adx > ADX_THRESHOLD) and (rsi < RSI_MAX_LONG)
            # Short Sinyal: Fiyat < Destek AND ADX > 25 AND RSI > 30
            short_signal = (curr_price < sup) and (adx > ADX_THRESHOLD) and (rsi > RSI_MIN_SHORT)
            
            if long_signal and capital > 10:
                size = (capital * (1 - COMMISSION)) / curr_price
                position = {'type': 'LONG', 'entry': curr_price, 'size': size, 'entry_idx': i}
                capital = 0
                
            elif short_signal and capital > 10:
                size = (capital * (1 - COMMISSION)) / curr_price
                position = {'type': 'SHORT', 'entry': curr_price, 'size': size, 'entry_idx': i}
                capital = 0
            
            equity_curve.append(capital)
    
    # Simulasyon sonu: Acik pozisyon varsa kapat
    if position:
        final_price = float(df.iloc[end_idx - 1]['Close'])
        if position['type'] == 'LONG':
            final_value = position['size'] * final_price * (1 - COMMISSION)
        else:
            pnl = (position['entry'] - final_price) * position['size']
            final_value = ((position['size'] * position['entry']) + pnl) * (1 - COMMISSION)
        
        capital = final_value
        pnl_pct = (final_value / (position['size'] * position['entry']) - 1) * 100
        trades.append(pnl_pct)
        
        if record_trades:
            trade_details.append({
                'type': position['type'],
                'entry_idx': position['entry_idx'] - start_idx,
                'entry_price': position['entry'],
                'exit_idx': end_idx - start_idx - 1,
                'exit_price': final_price,
                'pnl_pct': pnl_pct
            })
    
    return {
        'final_capital': capital,
        'trades': trades,
        'trade_details': trade_details,
        'equity_curve': equity_curve,
        'num_trades': len(trades)
    }

# ══════════════════════════════════════════════════════════════════════════════
#                         GRAFIK FONKSIYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def plot_backtest(df, start_idx, result, title_suffix=""):
    """Backtest sonuclarini gorsel olarak goster"""
    
    end_idx = min(start_idx + BACKTEST_BARS, len(df))
    price_data = df.iloc[start_idx:end_idx]['Close'].values
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#0d1117')
    
    # --- 1. FIYAT GRAFIGI + ISLEM NOKTALARI ---
    ax1 = axes[0]
    ax1.set_facecolor('#161b22')
    
    ax1.plot(range(len(price_data)), price_data, color='#58a6ff', linewidth=1.2, label='Fiyat', alpha=0.9)
    
    # Islem isaretleri
    for trade in result['trade_details']:
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        is_profit = trade['pnl_pct'] > 0
        
        if trade['type'] == 'LONG':
            ax1.scatter(entry_idx, trade['entry_price'], marker='^', s=120, c='#3fb950', 
                       edgecolors='white', linewidths=1, zorder=5, label='_nolegend_')
            exit_color = '#3fb950' if is_profit else '#f85149'
            ax1.scatter(exit_idx, trade['exit_price'], marker='v', s=120, c=exit_color, 
                       edgecolors='white', linewidths=1, zorder=5)
            ax1.plot([entry_idx, exit_idx], [trade['entry_price'], trade['exit_price']], 
                    color=exit_color, linestyle='--', alpha=0.4, linewidth=1)
        else:
            ax1.scatter(entry_idx, trade['entry_price'], marker='v', s=120, c='#f85149', 
                       edgecolors='white', linewidths=1, zorder=5)
            exit_color = '#3fb950' if is_profit else '#f85149'
            ax1.scatter(exit_idx, trade['exit_price'], marker='^', s=120, c=exit_color, 
                       edgecolors='white', linewidths=1, zorder=5)
            ax1.plot([entry_idx, exit_idx], [trade['entry_price'], trade['exit_price']], 
                    color=exit_color, linestyle='--', alpha=0.4, linewidth=1)
    
    ax1.set_title(f'SNIPER v3.2 BACKTEST - {SYMBOL} (1 Aylik){title_suffix}', 
                  fontsize=14, fontweight='bold', color='#c9d1d9', pad=15)
    ax1.set_ylabel('Fiyat ($)', fontsize=11, color='#c9d1d9')
    ax1.tick_params(colors='#8b949e')
    ax1.grid(True, alpha=0.15, color='#30363d')
    
    legend_elements = [
        Patch(facecolor='#3fb950', label='LONG Giris / Karli Cikis'),
        Patch(facecolor='#f85149', label='SHORT Giris / Zararli Cikis'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', facecolor='#161b22', 
               edgecolor='#30363d', labelcolor='#c9d1d9')
    
    # --- 2. EQUITY CURVE ---
    ax2 = axes[1]
    ax2.set_facecolor('#161b22')
    
    equity = result['equity_curve']
    
    ax2.fill_between(range(len(equity)), equity, STARTING_CAPITAL, 
                     where=[e >= STARTING_CAPITAL for e in equity],
                     color='#3fb950', alpha=0.2)
    ax2.fill_between(range(len(equity)), equity, STARTING_CAPITAL, 
                     where=[e < STARTING_CAPITAL for e in equity],
                     color='#f85149', alpha=0.2)
    ax2.plot(equity, color='#f0883e', linewidth=2, label='Bakiye')
    ax2.axhline(STARTING_CAPITAL, color='#8b949e', linestyle='--', linewidth=1, 
                label=f'Baslangic (${STARTING_CAPITAL})')
    
    final_val = equity[-1] if equity else STARTING_CAPITAL
    pnl = final_val - STARTING_CAPITAL
    pnl_pct = (pnl / STARTING_CAPITAL) * 100
    pnl_color = '#3fb950' if pnl >= 0 else '#f85149'
    
    ax2.set_title(f'BAKIYE | Final: ${final_val:.2f} ({pnl_pct:+.1f}%) | Islem: {len(result["trades"])}', 
                  fontsize=12, fontweight='bold', color=pnl_color, pad=10)
    ax2.set_xlabel('Bar (15 dakika)', fontsize=11, color='#c9d1d9')
    ax2.set_ylabel('Bakiye ($)', fontsize=11, color='#c9d1d9')
    ax2.tick_params(colors='#8b949e')
    ax2.grid(True, alpha=0.15, color='#30363d')
    ax2.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    
    plt.tight_layout()
    plt.savefig('/Users/cinarunver/Desktop/main/QuantiFine/backtest_1month.png', 
                dpi=150, facecolor='#0d1117', edgecolor='none')
    print("[SAVED] backtest_1month.png")
    plt.show()


def plot_monte_carlo(results, all_trades, all_curves, num_sims):
    """Monte Carlo sonuclarini 4 panelde goster"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    
    results = np.array(results)
    
    # --- 1. SONUC DAGILIMI (Histogram) ---
    ax1 = axes[0, 0]
    ax1.set_facecolor('#161b22')
    
    n, bins, patches = ax1.hist(results, bins=50, edgecolor='#30363d', linewidth=0.5)
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        patch.set_facecolor('#3fb950' if bin_center >= STARTING_CAPITAL else '#f85149')
        patch.set_alpha(0.7)
    
    ax1.axvline(STARTING_CAPITAL, color='#f0883e', linestyle='--', linewidth=2.5, 
                label=f'Maliyet (${STARTING_CAPITAL})')
    ax1.axvline(np.median(results), color='#58a6ff', linestyle='-', linewidth=2.5, 
                label=f'Medyan (${np.median(results):,.1f})')
    
    ax1.set_xlabel('Final Bakiye ($)', fontsize=11, color='#c9d1d9')
    ax1.set_ylabel('Frekans', fontsize=11, color='#c9d1d9')
    ax1.set_title('MONTE CARLO - Sonuc Dagilimi', fontsize=13, fontweight='bold', color='#c9d1d9')
    ax1.tick_params(colors='#8b949e')
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax1.grid(True, alpha=0.15, color='#30363d')
    
    # --- 2. EQUITY EGRILERI (100 ornek) ---
    ax2 = axes[0, 1]
    ax2.set_facecolor('#161b22')
    
    for curve in all_curves[:100]:
        if not curve: continue
        final = curve[-1]
        color = '#3fb950' if final >= STARTING_CAPITAL else '#f85149'
        alpha = 0.12 if color == '#3fb950' else 0.2
        ax2.plot(curve, color=color, alpha=alpha, linewidth=0.7)
    
    ax2.axhline(STARTING_CAPITAL, color='#f0883e', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Bar', fontsize=11, color='#c9d1d9')
    ax2.set_ylabel('Bakiye ($)', fontsize=11, color='#c9d1d9')
    ax2.set_title(f'MONTE CARLO - 100 Senaryo Ornegi', fontsize=12, fontweight='bold', color='#c9d1d9')
    ax2.tick_params(colors='#8b949e')
    ax2.grid(True, alpha=0.15, color='#30363d')
    
    # --- 3. ISLEM KAZANMA ORANI ---
    ax3 = axes[1, 0]
    ax3.set_facecolor('#161b22')
    
    if all_trades:
        wins = len([t for t in all_trades if t > 0])
        losses = len([t for t in all_trades if t <= 0])
        total = len(all_trades)
        
        bars = ax3.bar(['Kazanan', 'Kaybeden'], [wins, losses], 
                       color=['#3fb950', '#f85149'], edgecolor='white', linewidth=1.5)
        
        for bar, count in zip(bars, [wins, losses]):
            pct = (count / total) * 100 if total > 0 else 0
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                    f'{count}\n(%{pct:.1f})', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='#c9d1d9')
        
        ax3.set_ylabel('Islem Sayisi', fontsize=11, color='#c9d1d9')
        ax3.set_title(f'ISLEM ISTATISTIKLERI | Toplam: {total}', 
                      fontsize=12, fontweight='bold', color='#c9d1d9')
    ax3.tick_params(colors='#8b949e')
    ax3.grid(True, alpha=0.15, color='#30363d', axis='y')
    
    # --- 4. PERCENTILE ANALIZI ---
    ax4 = axes[1, 1]
    ax4.set_facecolor('#161b22')
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    values = [np.percentile(results, p) for p in percentiles]
    labels = [f'%{p}' for p in percentiles]
    colors = ['#f85149' if v < STARTING_CAPITAL else '#3fb950' for v in values]
    
    bars = ax4.barh(labels, values, color=colors, edgecolor='white', linewidth=1)
    ax4.axvline(STARTING_CAPITAL, color='#f0883e', linestyle='--', linewidth=2)
    
    for bar, val in zip(bars, values):
        ax4.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2, 
                f'${val:,.1f}', va='center', fontsize=10, fontweight='bold', color='#c9d1d9')
    
    ax4.set_xlabel('Final Bakiye ($)', fontsize=11, color='#c9d1d9')
    ax4.set_title('PERCENTILE ANALIZI', fontsize=12, fontweight='bold', color='#c9d1d9')
    ax4.tick_params(colors='#8b949e')
    ax4.grid(True, alpha=0.15, color='#30363d', axis='x')
    
    # Ana baslik
    profitable = np.sum(results > STARTING_CAPITAL)
    profit_rate = (profitable / num_sims) * 100
    
    fig.suptitle(f'SNIPER v3.2 MONTE CARLO ANALIZI - {SYMBOL}\n'
                 f'{num_sims} Senaryo | {SIM_LENGTH_BARS} Mum | Kar Orani: %{profit_rate:.1f}', 
                 fontsize=15, fontweight='bold', color='#f0883e', y=1.02)
    
    plt.tight_layout()
    plt.savefig('/Users/cinarunver/Desktop/main/QuantiFine/monte_carlo_2000.png', 
                dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    print("[SAVED] monte_carlo_2000.png")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#                              ANA FONKSIYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Ana calistirma fonksiyonu"""
    
    df = fetch_and_prepare_data()
    if df is None:
        return
    
    # ══════════════════════════════════════════════════════════════════════════
    #                    BOLUM 1: 1 AYLIK BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("   BOLUM 1: 1 AYLIK BACKTEST (Son ~30 gun)")
    print("="*70)
    
    # Son 1 aylik veriyi al
    backtest_start = max(0, len(df) - BACKTEST_BARS)
    backtest_result = run_simulation(df, backtest_start, BACKTEST_BARS, record_trades=True)
    
    final = backtest_result['final_capital']
    pnl = final - STARTING_CAPITAL
    pnl_pct = (pnl / STARTING_CAPITAL) * 100
    
    print(f"\n[BACKTEST SONUCU]")
    print(f"   Baslangic:        ${STARTING_CAPITAL}")
    print(f"   Final:            ${final:.2f}")
    print(f"   Kar/Zarar:        ${pnl:.2f} ({pnl_pct:+.1f}%)")
    print(f"   Toplam Islem:     {len(backtest_result['trades'])}")
    
    if backtest_result['trades']:
        wins = [t for t in backtest_result['trades'] if t > 0]
        losses = [t for t in backtest_result['trades'] if t <= 0]
        win_rate = len(wins) / len(backtest_result['trades']) * 100
        print(f"   Kazanma Orani:    %{win_rate:.1f}")
        if wins:
            print(f"   Ort. Kazanc:      %{np.mean(wins):.2f}")
        if losses:
            print(f"   Ort. Kayip:       %{np.mean(losses):.2f}")
    
    if backtest_result['trade_details']:
        print(f"\n[ISLEM DETAYLARI]")
        for i, t in enumerate(backtest_result['trade_details'][:15], 1):  # ilk 15 islem
            status = "+" if t['pnl_pct'] > 0 else "-"
            print(f"   {i:2}. {t['type']:5} | ${t['entry_price']:8.2f} -> ${t['exit_price']:8.2f} | {t['pnl_pct']:+6.2f}% {status}")
        if len(backtest_result['trade_details']) > 15:
            print(f"   ... ve {len(backtest_result['trade_details'])-15} islem daha")
    
    plot_backtest(df, backtest_start, backtest_result)
    
    # ══════════════════════════════════════════════════════════════════════════
    #                    BOLUM 2: MONTE CARLO (2000 Senaryo)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"   BOLUM 2: MONTE CARLO SIMULASYONU ({NUM_SIMULATIONS} Senaryo, {SIM_LENGTH_BARS} Mum)")
    print("="*70)
    
    max_start = len(df) - SIM_LENGTH_BARS - 1
    if max_start < NUM_SIMULATIONS:
        print(f"[!] Veri yetersiz, simulasyon sayisi {max_start}'e dusuruldu.")
        num_sims = max_start
    else:
        num_sims = NUM_SIMULATIONS
    
    # Rastgele baslangic noktalari sec (tekrarsiz)
    if num_sims > max_start:
        start_indices = np.random.choice(range(0, max_start), size=num_sims, replace=True)
    else:
        start_indices = np.random.choice(range(0, max_start), size=num_sims, replace=False)
    
    results = []
    all_trades = []
    all_curves = []
    
    print(f"\n[*] {num_sims} senaryo hesaplaniyor...")
    
    for idx, start in enumerate(start_indices):
        res = run_simulation(df, start, SIM_LENGTH_BARS, record_trades=False)
        results.append(res['final_capital'])
        all_trades.extend(res['trades'])
        if idx < 100:
            all_curves.append(res['equity_curve'])
        
        if (idx + 1) % 500 == 0:
            print(f"   [{idx + 1}/{num_sims}] tamamlandi...")
    
    results = np.array(results)
    
    # --- MONTE CARLO RAPORU ---
    win_trades = [t for t in all_trades if t > 0]
    loss_trades = [t for t in all_trades if t <= 0]
    profitable = np.sum(results > STARTING_CAPITAL)
    
    print("\n" + "="*70)
    print("   MONTE CARLO SONUC RAPORU")
    print("="*70)
    print(f"   Toplam Simulasyon:     {num_sims}")
    print(f"   Her Sim Uzunlugu:      {SIM_LENGTH_BARS} mum")
    print(f"   Baslangic Sermaye:     ${STARTING_CAPITAL}")
    print("-"*70)
    print(f"   En Kotu %5:            ${np.percentile(results, 5):,.2f}")
    print(f"   En Kotu %10:           ${np.percentile(results, 10):,.2f}")
    print(f"   Medyan Sonuc:          ${np.median(results):,.2f}")
    print(f"   Ortalama Sonuc:        ${np.mean(results):,.2f}")
    print(f"   En Iyi %10:            ${np.percentile(results, 90):,.2f}")
    print(f"   En Iyi %5:             ${np.percentile(results, 95):,.2f}")
    print("-"*70)
    print(f"   Karli Sim Orani:       %{(profitable/num_sims)*100:.1f}")
    print(f"   Agir Kayip (<%50):     %{(np.sum(results < STARTING_CAPITAL*0.5)/num_sims)*100:.1f}")
    
    if all_trades:
        print("-"*70)
        print(f"   Toplam Islem:          {len(all_trades)}")
        print(f"   Kazanma Orani:         %{(len(win_trades)/len(all_trades))*100:.1f}")
        if win_trades:
            print(f"   Ort. Kazanc:           %{np.mean(win_trades):.2f}")
        if loss_trades:
            print(f"   Ort. Kayip:            %{np.mean(loss_trades):.2f}")
        if win_trades and loss_trades:
            profit_factor = abs(sum(win_trades) / sum(loss_trades)) if sum(loss_trades) != 0 else float('inf')
            print(f"   Profit Factor:         {profit_factor:.2f}")
    print("="*70)
    
    plot_monte_carlo(results, all_trades, all_curves, num_sims)
    
    print("\n[TAMAMLANDI] Kaydedilen dosyalar:")
    print("   - backtest_1month.png (1 Aylik Backtest)")
    print("   - monte_carlo_2000.png (Monte Carlo Analizi)")

if __name__ == "__main__":
    main()
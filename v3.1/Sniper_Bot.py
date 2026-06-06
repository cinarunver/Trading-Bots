import ccxt
import pandas as pd
import pandas_ta as ta
import time
import os
import logging
from datetime import datetime
from colorama import Fore, Back, Style, init

# Mac/Linux Renk Uyumu
init(autoreset=True)

# --- ⚙️ KAPTAN KÖŞKÜ (AYARLAR) ---
CONFIG = {
    "SYMBOL_LIST": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", 
        "XRP/USDT", "DOGE/USDT", "PEPE/USDT"
    ],
    "TIMEFRAME": "15m",
    "LIMIT": 100,
    "PAPER_TRADING": True,   # Gerçek para için False yap
    "STARTING_BALANCE": 50,  # Başlangıç Sermayesi ($)
    "REFRESH_RATE": 5,       # Ekran yenileme hızı (Saniye)
    
    # Sniper v3.1 Strateji (15 Bar Hızlandırılmış)
    "ADX_THRESHOLD": 25,
    "RSI_MAX_LONG": 70,
    "RSI_MIN_SHORT": 30,
    "LOOKBACK": 15
}

# --- 📝 LOGLAMA ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler("sniper_master.log")] # Ekrana log basmıyoruz, ekranı tabloya ayırdık
)

# --- 💰 CÜZDAN ---
wallet = {
    "USDT": CONFIG["STARTING_BALANCE"],
    "REALIZED_PNL": 0.0
}
position = None 
trade_history = []

# --- 🌐 BAĞLANTI ---
try:
    if CONFIG["PAPER_TRADING"]:
        exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})
    else:
        exchange = ccxt.binance({
            'apiKey': 'YOUR_BINANCE_API_KEY',
            'secret': 'YOUR_BINANCE_SECRET_KEY',
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
except Exception as e:
    logging.error(f"Bağlantı Hatası: {e}")

def clear_screen(): os.system('clear')

def fetch_data(symbol):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # İndikatörler
        adx = df.ta.adx(length=14); df = pd.concat([df, adx], axis=1)
        df.rename(columns={'ADX_14': 'ADX'}, inplace=True)
        df['RSI'] = df.ta.rsi(length=14)
        
        # Donchian 15
        df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
        df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)
        return df.iloc[-1]
    except: return None

def execute_trade(action, symbol, price):
    global wallet, position
    fee = 0.0005
    
    if action == "OPEN_LONG":
        size = (wallet["USDT"] / price) * (1 - fee)
        position = {"sym": symbol, "type": "LONG", "entry": price, "size": size, "inv": wallet["USDT"]}
        wallet["USDT"] = 0
        logging.info(f"LONG AÇILDI: {symbol} @ {price}")
        record_trade(symbol, "LONG AÇ", price, 0)
        
    elif action == "OPEN_SHORT":
        size = (wallet["USDT"] / price) * (1 - fee)
        position = {"sym": symbol, "type": "SHORT", "entry": price, "size": size, "inv": wallet["USDT"]}
        # Short teminatı
        logging.info(f"SHORT AÇILDI: {symbol} @ {price}")
        record_trade(symbol, "SHORT AÇ", price, 0)
        
    elif action == "CLOSE":
        pnl = 0; val = 0
        if position["type"] == "LONG":
            val = (position["size"] * price) * (1 - fee)
            pnl = val - position["inv"]
            wallet["USDT"] = val
        elif position["type"] == "SHORT":
            raw_pnl = (position["entry"] - price) * position["size"]
            val = position["inv"] + raw_pnl - (position["inv"] * fee)
            pnl = raw_pnl
            wallet["USDT"] = val
        
        wallet["REALIZED_PNL"] += pnl
        logging.info(f"KAPATILDI: {symbol} | PnL: {pnl}")
        record_trade(symbol, f"KAPAT ({position['type']})", price, pnl)
        position = None

def record_trade(symbol, action, price, pnl):
    now = datetime.now().strftime("%H:%M")
    trade_history.append({
        "Zaman": now, "Coin": symbol, "İşlem": action, "Fiyat": f"${price:.4f}", 
        "PnL": f"${pnl:.2f}" if "KAPAT" in action else "-"
    })
    if len(trade_history) > 6: trade_history.pop(0)

def main():
    print("Master Bot Başlatılıyor... Veriler Toplanıyor...")
    
    while True:
        try:
            dashboard_data = [] # Ekrana basılacak veriler
            
            # --- 1. TÜM COINLERİ TARA VE İŞLEM YAP ---
            for sym in CONFIG["SYMBOL_LIST"]:
                row = fetch_data(sym)
                if row is None: continue
                
                curr = row['close']
                adx = row['ADX']
                rsi = row['RSI']
                res = row['RES']
                sup = row['SUP']
                
                # Tablo için Durum Belirle
                status_msg = "BEKLİYOR"
                status_col = Fore.WHITE
                
                # A) POZİSYON VARSA (YÖNET)
                if position and position["sym"] == sym:
                    status_msg = f"{position['type']} SÜRÜYOR"
                    status_col = Fore.CYAN
                    
                    # Çıkış Kontrolü
                    exit_l = (curr < sup)
                    exit_s = (curr > res)
                    
                    if (position["type"] == "LONG" and exit_l) or (position["type"] == "SHORT" and exit_s):
                        execute_trade("CLOSE", sym, curr)
                        status_msg = "KAPATILDI"
                        status_col = Fore.YELLOW
                
                # B) POZİSYON YOKSA (FIRSAT ARA)
                elif position is None:
                    # Sinyal Koşulları
                    long_sig = (curr > res) and (adx > CONFIG["ADX_THRESHOLD"]) and (rsi < CONFIG["RSI_MAX_LONG"])
                    short_sig = (curr < sup) and (adx > CONFIG["ADX_THRESHOLD"]) and (rsi > CONFIG["RSI_MIN_SHORT"])
                    
                    if long_sig:
                        execute_trade("OPEN_LONG", sym, curr)
                        status_msg = "GİRİŞ (LONG)"
                        status_col = Fore.GREEN
                    elif short_sig:
                        execute_trade("OPEN_SHORT", sym, curr)
                        status_msg = "GİRİŞ (SHORT)"
                        status_col = Fore.RED
                
                # C) BAŞKA COINDE POZİSYON VARSA (KİLİTLİ)
                elif position and position["sym"] != sym:
                    status_msg = "LOCKED"
                    status_col = Fore.LIGHTBLACK_EX # Sönük renk

                # Veriyi listeye ekle
                dashboard_data.append({
                    "Symbol": sym, "Price": curr, "Sup": sup, "Res": res,
                    "ADX": adx, "RSI": rsi, "Status": status_msg, "Color": status_col
                })
                
                time.sleep(0.1) # API nezaketi

            # --- 2. EKRANI GÜNCELLE (DASHBOARD) ---
            clear_screen()
            now = datetime.now().strftime("%H:%M:%S")
            
            # Bakiye Hesabı (Canlı Equity)
            equity = wallet["USDT"]
            if position:
                # Listedeki güncel fiyattan hesapla
                curr_p = next((x['Price'] for x in dashboard_data if x['Symbol'] == position['sym']), position['entry'])
                unrealized = (curr_p - position['entry']) * position['size'] if position['type'] == "LONG" else (position['entry'] - curr_p) * position['size']
                equity = (wallet["USDT"] if position['type']=='SHORT' else 0) + position['inv'] + unrealized

            pnl_col = Fore.GREEN if wallet["REALIZED_PNL"] >= 0 else Fore.RED
            
            print(f"{Back.BLUE}{Fore.WHITE}   QUANTIFINE MASTER SNIPER v3.1   {Style.RESET_ALL}")
            print(f"   ⏰ {now} | ⚡ Hız: 15 Bar | 🔄 Yenileme: {CONFIG['REFRESH_RATE']}sn")
            print("-" * 75)
            print(f"   💰 Bakiye (Equity): {Fore.YELLOW}${equity:.2f}{Style.RESET_ALL}")
            print(f"   📊 Toplam Kâr/Zarar: {pnl_col}${wallet['REALIZED_PNL']:.2f}{Style.RESET_ALL}")
            
            if position:
                pos_col = Fore.GREEN if position['type']=="LONG" else Fore.RED
                entry_p = position['entry']
                # Anlık coin fiyatını bul
                curr_p = next((x['Price'] for x in dashboard_data if x['Symbol'] == position['sym']), 0)
                u_pnl = (curr_p - entry_p)*position['size'] if position['type']=="LONG" else (entry_p - curr_p)*position['size']
                u_col = Fore.GREEN if u_pnl >= 0 else Fore.RED
                print(f"   🎯 AKTİF: {pos_col}{position['sym']} ({position['type']}){Style.RESET_ALL} | PnL: {u_col}${u_pnl:.2f}{Style.RESET_ALL}")
            else:
                print(f"   🔭 DURUM: {Fore.WHITE}FIRSAT TARANIYOR (NAKİTTE){Style.RESET_ALL}")

            print("\n" + "-" * 88)
            print(f"   {'COIN':<10} {'FİYAT':<10} {'DESTEK (S)':<12} {'DİRENÇ (L)':<12} {'ADX':<6} {'RSI':<6} {'DURUM':<15}")
            print("-" * 88)
            
            for d in dashboard_data:
                # ADX Renk
                adx_c = Fore.GREEN if d['ADX'] > CONFIG["ADX_THRESHOLD"] else Fore.RED
                # RSI Renk
                rsi_c = Fore.GREEN if 30 < d['RSI'] < 70 else Fore.YELLOW
                
                print(f"   {d['Color']}{d['Symbol']:<10} ${d['Price']:<9.4f} ${d['Sup']:<11.4f} ${d['Res']:<11.4f} {adx_c}{d['ADX']:.1f}{d['Color']:<2} {rsi_c}{d['RSI']:.1f}{d['Color']:<2} {d['Status']:<15}{Style.RESET_ALL}")
            
            print("-" * 88)
            
            # Geçmiş İşlemler (Kısa Liste)
            if trade_history:
                print(f"{Fore.YELLOW}   SON İŞLEMLER:{Style.RESET_ALL}")
                for t in trade_history:
                    col = Fore.GREEN if "$" in t['PnL'] and float(t['PnL'].replace('$',''))>0 else Fore.WHITE
                    print(f"   {t['Zaman']} {t['Coin']} {t['İşlem']} {col}{t['PnL']}{Style.RESET_ALL}")
            
            print(f"\n{Fore.BLUE}[i] Çıkmak için CTRL+C (Caffeinate ile çalıştırın){Style.RESET_ALL}")
            
            time.sleep(CONFIG["REFRESH_RATE"])

        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Hata: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
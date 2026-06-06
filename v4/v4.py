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

# --- ⚙️ KAPTAN KÖŞKÜ (TREND MASTER v7.1 AYARLARI) ---
CONFIG = {
    "SYMBOL_LIST": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", 
        "XRP/USDT", "DOGE/USDT", "PEPE/USDT"
    ],
    "TIMEFRAME": "15m",
    "LIMIT": 200,
    "PAPER_TRADING": True,
    "STARTING_BALANCE": 50,
    "REFRESH_RATE": 5,
    
    # --- STRATEJİ PARAMETRELERİ ---
    "ADX_THRESHOLD": 20,
    "LOOKBACK": 12,           # Giriş Tetikleyicisi
    
    # ÖLÜ ALAN & TREND FİLTRESİ
    "EMA_PERIOD": 50,         # Trend Yönü için daha uzun vade (20 -> 50 yaptık)
    "DEAD_ZONE_MULTIPLIER": 1.0, 
    
    # ÇIKIŞ (ADAPTİF)
    "BASE_EXIT": 10,
    "MAX_EXIT": 24,
    "EXIT_INCREMENT": 3
}

dynamic_settings = {} 

# --- 📝 LOGLAMA ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler("trend_master.log")]
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
            'apiKey': 'SENIN_API_KEYIN',
            'secret': 'SENIN_SECRET_KEYIN',
            'options': {'defaultType': 'future'},
            'enableRateLimit': True
        })
except Exception as e:
    logging.error(f"Bağlantı Hatası: {e}")

def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')

def fetch_data(symbol, dynamic_exit_lookback):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 1. İndikatörler
        adx = df.ta.adx(length=14); df = pd.concat([df, adx], axis=1)
        df.rename(columns={'ADX_14': 'ADX'}, inplace=True)
        df['RSI'] = df.ta.rsi(length=14)
        
        # 2. TREND YÖNÜ (EMA EĞİMİ)
        # EMA Periodunu 50 yaptık ki ana yönü görsün, gürültüye kanmasın
        df['EMA'] = df.ta.ema(length=CONFIG["EMA_PERIOD"])
        
        # EMA'nın bir önceki mumdaki değeri (Yön tespiti için)
        df['EMA_PREV'] = df['EMA'].shift(1)
        
        # 3. ÖLÜ ALAN
        atr = df.ta.atr(length=14)
        df['DZ_TOP'] = df['EMA'] + (atr * CONFIG["DEAD_ZONE_MULTIPLIER"])
        df['DZ_BOT'] = df['EMA'] - (atr * CONFIG["DEAD_ZONE_MULTIPLIER"])
        
        # 4. GİRİŞ/ÇIKIŞ KANALLARI
        df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
        df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)

        df['EXIT_RES'] = df['high'].rolling(dynamic_exit_lookback).max().shift(1)
        df['EXIT_SUP'] = df['low'].rolling(dynamic_exit_lookback).min().shift(1)

        return df.iloc[-1]
    except: return None

def execute_trade(action, symbol, price):
    global wallet, position, dynamic_settings
    fee = 0.0005
    current_setting = dynamic_settings.get(symbol, CONFIG["BASE_EXIT"])

    if action == "OPEN_LONG":
        size = (wallet["USDT"] / price) * (1 - fee)
        position = {"sym": symbol, "type": "LONG", "entry": price, "size": size, "inv": wallet["USDT"]}
        wallet["USDT"] = 0
        logging.info(f"LONG AÇILDI: {symbol} @ {price}")
        record_trade(symbol, "LONG AÇ", price, 0)
        
    elif action == "OPEN_SHORT":
        size = (wallet["USDT"] / price) * (1 - fee)
        position = {"sym": symbol, "type": "SHORT", "entry": price, "size": size, "inv": wallet["USDT"]}
        wallet["USDT"] = 0
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
        
        if pnl <= 0:
            new_setting = min(current_setting + CONFIG["EXIT_INCREMENT"], CONFIG["MAX_EXIT"])
        else:
            new_setting = CONFIG["BASE_EXIT"]
            
        dynamic_settings[symbol] = new_setting
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
    print("💎 QUANTIFINE TREND MASTER v7.1 BAŞLATILIYOR...")
    print(f"🛡️ Filtre: EMA({CONFIG['EMA_PERIOD']}) Eğim Kontrolü + DeadZone")
    time.sleep(2)
    
    while True:
        try:
            dashboard_data = [] 
            active_stop_price = 0 
            
            for sym in CONFIG["SYMBOL_LIST"]:
                if sym not in dynamic_settings: dynamic_settings[sym] = CONFIG["BASE_EXIT"] 
                
                row = fetch_data(sym, dynamic_settings[sym])
                if row is None: continue
                
                curr = row['close']
                adx = row['ADX']
                ema_curr = row['EMA']
                ema_prev = row['EMA_PREV']
                
                # --- TREND YÖNÜ TESPİTİ (EN ÖNEMLİ KISIM) ---
                is_trend_up = ema_curr > ema_prev
                is_trend_down = ema_curr < ema_prev
                
                # Ölü Alan Sınırları
                dz_top = row['DZ_TOP']
                dz_bot = row['DZ_BOT']
                in_dead_zone = (curr > dz_bot) and (curr < dz_top)
                
                status_msg = "BEKLİYOR"
                status_col = Fore.WHITE
                
                # --- POZİSYON YÖNETİMİ ---
                if position and position["sym"] == sym:
                    status_msg = f"{position['type']} SÜRÜYOR"
                    status_col = Fore.CYAN
                    
                    if position["type"] == "LONG": active_stop_price = row['EXIT_SUP']
                    else: active_stop_price = row['EXIT_RES']
                    
                    should_close_long = (position["type"] == "LONG" and curr < row['EXIT_SUP'])
                    should_close_short = (position["type"] == "SHORT" and curr > row['EXIT_RES'])
                    
                    if should_close_long or should_close_short:
                        execute_trade("CLOSE", sym, curr)
                        status_msg = "STOP OUT"
                        status_col = Fore.YELLOW
                
                # --- FIRSAT ARAMA (YÖN KORUMALI) ---
                elif position is None:
                    
                    if in_dead_zone:
                        status_msg = "⛔ ÖLÜ ALAN"
                        status_col = Fore.LIGHTBLACK_EX
                    else:
                        # LONG ŞARTLARI:
                        # 1. EMA YUKARI bakıyor olmalı (ZORUNLU)
                        # 2. Ölü Alanın Üstünü kırmalı
                        # 3. Donchian Tepesini kırmalı
                        long_sig = is_trend_up and (curr > dz_top) and (curr > row['RES']) and (adx > CONFIG["ADX_THRESHOLD"])
                        
                        # SHORT ŞARTLARI:
                        # 1. EMA AŞAĞI bakıyor olmalı (ZORUNLU)
                        # 2. Ölü Alanın Altını kırmalı
                        # 3. Donchian Dibini kırmalı
                        short_sig = is_trend_down and (curr < dz_bot) and (curr < row['SUP']) and (adx > CONFIG["ADX_THRESHOLD"])
                        
                        if long_sig:
                            execute_trade("OPEN_LONG", sym, curr)
                            status_msg = "GİRİŞ (LONG)"
                            status_col = Fore.GREEN
                        elif short_sig:
                            execute_trade("OPEN_SHORT", sym, curr)
                            status_msg = "GİRİŞ (SHORT)"
                            status_col = Fore.RED
                        else:
                            # Sinyal yok ama neden? Yön uyuşmazlığı olabilir.
                            if not in_dead_zone:
                                status_msg = "YÖN UYMUYOR"
                                status_col = Fore.MAGENTA

                dashboard_data.append({
                    "Symbol": sym, "Price": curr, 
                    "Trend": "YUKARI ↗" if is_trend_up else "AŞAĞI ↘",
                    "InZone": in_dead_zone,
                    "ADX": adx, "Status": status_msg, "Color": status_col
                })
                time.sleep(0.1)

            clear_screen()
            now = datetime.now().strftime("%H:%M:%S")
            
            equity = wallet["USDT"]
            if position:
                curr_p = next((x['Price'] for x in dashboard_data if x['Symbol'] == position['sym']), position['entry'])
                unrealized = (curr_p - position['entry']) * position['size'] if position['type'] == "LONG" else (position['entry'] - curr_p) * position['size']
                equity = (wallet["USDT"] if position['type']=='SHORT' else 0) + position['inv'] + unrealized

            print(f"{Back.MAGENTA}{Fore.WHITE}   💎 QUANTIFINE TREND MASTER v7.1   {Style.RESET_ALL}")
            print(f"   ⏰ {now} | 🛡️ Trend Filtresi: EMA 50")
            print("-" * 85)
            print(f"   💰 Bakiye: {Fore.YELLOW}${equity:.2f}{Style.RESET_ALL} | Kâr/Zarar: ${wallet['REALIZED_PNL']:.2f}")
            
            if position:
                print(f"   🎯 AKTİF: {position['sym']} ({position['type']}) | Fiyat: ${curr_p:.4f}")

            print("\n" + "-" * 95)
            print(f"   {'COIN':<10} {'FİYAT':<10} {'ANA TREND':<12} {'ÖLÜ ALAN?':<12} {'ADX':<6} {'DURUM':<15}")
            print("-" * 95)
            
            for d in dashboard_data:
                zone_icon = "EVET" if d['InZone'] else "HAYIR"
                # Trend Rengi
                trend_col = Fore.GREEN if "YUKARI" in d['Trend'] else Fore.RED
                
                print(f"   {d['Color']}{d['Symbol']:<10} ${d['Price']:<9.4f} {trend_col}{d['Trend']:<12} {Fore.WHITE}{zone_icon:<12} {d['ADX']:.1f}{d['Color']:<2} {d['Status']:<15}{Style.RESET_ALL}")
            
            print("-" * 95)
            if trade_history:
                print(f"{Fore.YELLOW}   SON İŞLEMLER:{Style.RESET_ALL}")
                for t in trade_history:
                    print(f"   {t['Zaman']} {t['Coin']} {t['İşlem']} {t['PnL']}")
            
            time.sleep(CONFIG["REFRESH_RATE"])

        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Hata: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
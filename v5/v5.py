import ccxt
import pandas as pd
import pandas_ta as ta
import time
import os
import logging
import numpy as np
from datetime import datetime
from colorama import Fore, Back, Style, init

# Mac/Linux/Windows Renk Uyumu
init(autoreset=True)

# --- ⚙️ QUANTIFINE MASTER AYARLAR ---
CONFIG = {
    "SYMBOL_LIST": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"],
    "TIMEFRAME": "15m",
    "LIMIT": 150,
    "PAPER_TRADING": True,   
    "STARTING_BALANCE": 1000, 
    "REFRESH_RATE": 10,
    
    # --- Risk & Boyutlandırma ---
    "RISK_PER_TRADE": 0.02,    # Her işlemde kasanın %2'sini riske et
    "ATR_MULTIPLIER": 2.0,     # Stop mesafesi için volatilite katsayısı
    
    # --- Strateji Eşikleri ---
    "HURST_WINDOW": 50,
    "HURST_TREND_MIN": 0.55,   # Bu değerin üstü trend (Persistence)
    "HURST_REVERT_MAX": 0.45,  # Bu değerin altı ortalamaya dönüş (Anti-persistence)
    "ADX_THRESHOLD": 25,
    "LOOKBACK": 15
}

# --- 📝 LOGLAMA ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler("quantifine_pro.log")]
)

# --- 💰 CÜZDAN & POZİSYON ---
wallet = {"USDT": CONFIG["STARTING_BALANCE"], "REALIZED_PNL": 0.0}
position = None 
trade_history = []

# --- 🌐 BAĞLANTI ---
exchange = ccxt.binance({'options': {'defaultType': 'future'}, 'enableRateLimit': True})

def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')

def calculate_hurst(series):
    """Hurst Exponent: H < 0.5 Mean Reverting, H > 0.5 Trending"""
    if len(series) < 30: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def fetch_data(symbol):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["TIMEFRAME"], limit=CONFIG["LIMIT"])
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # İndikatörler
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(df['high'], df['low'], df['close'], length=14)
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        adx_df = df.ta.adx(length=14)
        df['ADX'] = adx_df['ADX_14']
        
        # Hurst Exponent (Trend Gücü / Karakteri)
        df['HURST'] = df['close'].rolling(window=CONFIG["HURST_WINDOW"]).apply(calculate_hurst)
        
        # Donchian (Kanal)
        df['RES'] = df['high'].rolling(CONFIG["LOOKBACK"]).max().shift(1)
        df['SUP'] = df['low'].rolling(CONFIG["LOOKBACK"]).min().shift(1)
        
        return df.iloc[-1]
    except Exception as e:
        logging.error(f"Veri Hatası ({symbol}): {e}")
        return None

def execute_trade(action, symbol, price, size=0):
    global wallet, position
    fee_rate = 0.0005
    
    if "OPEN" in action:
        cost = size * price
        fee = cost * fee_rate
        pos_type = "LONG" if "LONG" in action else "SHORT"
        
        position = {
            "sym": symbol, "type": pos_type, "entry": price, 
            "size": size, "inv": cost, "fee": fee
        }
        wallet["USDT"] -= fee # Giriş komisyonu
        logging.info(f"{pos_type} AÇILDI: {symbol} | Boyut: {size:.4f} | Maliyet: {cost:.2f}")
        record_trade(symbol, f"{pos_type} AÇ", price, 0)
        
    elif action == "CLOSE":
        # PnL Hesaplama
        if position["type"] == "LONG":
            pnl = (price - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - price) * position["size"]
            
        exit_fee = (position["size"] * price) * fee_rate
        net_pnl = pnl - position["fee"] - exit_fee
        
        wallet["USDT"] += (position["inv"] if position["type"] == "LONG" else 0) + pnl - exit_fee
        wallet["REALIZED_PNL"] += net_pnl
        
        logging.info(f"KAPATILDI: {symbol} | Net PnL: {net_pnl:.2f}")
        record_trade(symbol, f"KAPAT ({position['type']})", price, net_pnl)
        position = None

def record_trade(symbol, action, price, pnl):
    now = datetime.now().strftime("%H:%M")
    trade_history.append({"Zaman": now, "Coin": symbol, "İşlem": action, "Fiyat": f"${price:.2f}", "PnL": f"${pnl:.2f}" if "KAPAT" in action else "-"})
    if len(trade_history) > 8: trade_history.pop(0)

def main():
    print(f"{Fore.CYAN}QuantiFine Master Bot Başlatılıyor...")
    
    while True:
        try:
            dashboard_data = []
            
            for sym in CONFIG["SYMBOL_LIST"]:
                row = fetch_data(sym)
                if row is None or np.isnan(row['HURST']): continue
                
                curr, h_val, atr, sma = row['close'], row['HURST'], row['ATR'], row['SMA_20']
                status_msg, status_col = "TARANIYOR", Fore.WHITE
                
                # --- 1. POZİSYON YÖNETİMİ ---
                if position and position["sym"] == sym:
                    status_msg = f"{position['type']} AKTİF"
                    status_col = Fore.CYAN
                    
                    # Çıkış Koşulu: Fiyat Donchian kanalının dışına taşarsa (Trend sonu)
                    exit_long = (curr < row['SUP'])
                    exit_short = (curr > row['RES'])
                    
                    if (position["type"] == "LONG" and exit_long) or (position["type"] == "SHORT" and exit_short):
                        execute_trade("CLOSE", sym, curr)
                
                # --- 2. SİNYAL ÜRETİMİ (Sadece pozisyon yoksa) ---
                elif position is None:
                    # Strateji A: Trend Takip (Hurst > 0.55 + ADX + Donchian Breakout)
                    is_trending = h_val > CONFIG["HURST_TREND_MIN"]
                    long_trend = is_trending and (curr > row['RES']) and (row['ADX'] > CONFIG["ADX_THRESHOLD"])
                    
                    # Strateji B: Mean Reversion (Hurst < 0.45 + Price/SMA Gap)
                    is_reverting = h_val < CONFIG["HURST_REVERT_MAX"]
                    short_revert = is_reverting and (curr > sma) # Fiyat ortalamanın üstündeyse aşağı dönecek
                    
                    if long_trend or short_revert:
                        # POZİSYON BOYUTLANDIRMA (Volatility Adjusted)
                        risk_usd = wallet["USDT"] * CONFIG["RISK_PER_TRADE"]
                        stop_dist = atr * CONFIG["ATR_MULTIPLIER"]
                        
                        if stop_dist > 0:
                            calc_size = risk_usd / stop_dist
                            # Bakiye kontrolü
                            max_size = (wallet["USDT"] * 0.9) / curr
                            final_size = min(calc_size, max_size)
                            
                            act = "OPEN_LONG" if long_trend else "OPEN_SHORT"
                            execute_trade(act, sym, curr, final_size)
                            status_msg = "GİRİŞ YAPILDI"
                            status_col = Fore.GREEN

                dashboard_data.append({
                    "Symbol": sym, "Price": curr, "Hurst": h_val, "ATR": atr,
                    "Status": status_msg, "Color": status_col
                })
                time.sleep(0.2)

            # --- 3. DASHBOARD ÇIKTISI ---
            clear_screen()
            now = datetime.now().strftime("%H:%M:%S")
            print(f"{Back.MAGENTA}{Fore.WHITE}   QUANTIFINE PRO SNIPER v3.2   {Style.RESET_ALL} | {now}")
            print(f"   Bakiye: ${wallet['USDT']:.2f} | Toplam PnL: {Fore.GREEN if wallet['REALIZED_PNL'] >=0 else Fore.RED}${wallet['REALIZED_PNL']:.2f}{Style.RESET_ALL}")
            print("-" * 85)
            print(f"   {'COIN':<10} {'FİYAT':<12} {'HURST':<8} {'ATR':<8} {'DURUM':<15}")
            
            for d in dashboard_data:
                h_col = Fore.GREEN if d['Hurst'] > 0.5 else Fore.YELLOW
                print(f"   {d['Symbol']:<10} {d['Price']:<12.4f} {h_col}{d['Hurst']:<8.2f} {Fore.WHITE}{d['ATR']:<8.4f} {d['Color']}{d['Status']}")
            
            print("-" * 85)
            if trade_history:
                print(f"   SON İŞLEMLER:")
                for t in trade_history:
                    print(f"   {t['Zaman']} | {t['Coin']} | {t['İşlem']} | {t['Fiyat']} | {t['PnL']}")

            time.sleep(CONFIG["REFRESH_RATE"])

        except KeyboardInterrupt: break
        except Exception as e:
            print(f"Sistem Hatası: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
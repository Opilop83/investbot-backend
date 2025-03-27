import time
import threading
import pandas as pd
import numpy as np  
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from news_analyzer import get_sentiment_for_symbols
from xtb_client import XTBClient
from analytics import calculate_indicators
from ai_trading_enhancements import (
    generate_trading_signal_1m,  # ✅ DODAJ TO
    generate_trading_signal_5m,
    generate_trading_signal_15m,
    generate_trading_signal_1h,

)



def restart_thread(target_func, args):
    """🔄 Restartuje wątek, jeśli został zakończony"""
    thread = threading.Thread(target=target_func, args=args, daemon=True)
    thread.start()
    return thread
    
def decide_trade_action_crypto(
    signals,
    adx_1m, adx_5m, adx_15m, adx_1h,
    sma50_15m, sma200_15m, sma50_1h, sma200_1h,
    last_close, bb_upper, bb_lower,
    rsi_5m, rsi_15m,
    momentum_5m, momentum_15m,
    spread, volume, symbol, avg_volume
):
    """
    📊 Decyduje, czy otworzyć pozycję na kryptowalutach, bazując na wskaźnikach technicznych.
    Zwraca "buy", "sell" lub "hold".
    """

    # --- 1. Debug-Logi ---
    print("\n🔍 **Decyzja handlowa - Debug wskaźników** 🔍")
    print(f"📈 Sygnały: {signals}")
    print(f"ADX 1m: {adx_1m}, ADX 5m: {adx_5m}, ADX 15m: {adx_15m}, ADX 1h: {adx_1h}")
    print(f"SMA50 15m: {sma50_15m}, SMA200 15m: {sma200_15m}")
    print(f"SMA50 1h: {sma50_1h}, SMA200 1h: {sma200_1h}")
    print(f"📊 Cena zamknięcia: {last_close}, BB Upper: {bb_upper}, BB Lower: {bb_lower}")
    print(f"RSI 5m: {rsi_5m}, RSI 15m: {rsi_15m}")
    print(f"Momentum 5m: {momentum_5m}, Momentum 15m: {momentum_15m}")
    print(f"⚠️ Spread: {spread}")
    print(f"📊 Wolumen: {volume}, Średni wolumen: {avg_volume}")
    print("--------------------------------------------------\n")

    # --- 2. Policz liczbę sygnałów buy i sell ---
    buy_count = signals.count("buy")
    sell_count = signals.count("sell")

    # --- 3. Jeśli wszystkie sygnały to HOLD => natychmiast "hold" ---
    if buy_count == 0 and sell_count == 0:
        print("⚠️ Wszystkie sygnały to HOLD. Pomijam transakcję.")
        return "hold"

    # --- 4. Sprawdź warunki trendu (SMA) ---
    trend_up = (sma50_15m > sma200_15m) and (sma50_1h > sma200_1h)
    trend_down = (sma50_15m < sma200_15m) and (sma50_1h < sma200_1h)

    # --- 5. Decyzja wstępna na podstawie ADX i liczby sygnałów ---
    #    - Warunek 1: ADX_15m > 20 => min. 2 sygnały
    #    - Warunek 2: ADX_15m > 15 => min. 3 sygnały
    final_decision = "hold"
    if adx_15m > 20 and (buy_count >= 2 or sell_count >= 2):
        final_decision = "buy" if buy_count > sell_count else "sell"
    elif adx_15m > 15 and (buy_count >= 3 or sell_count >= 3):
        final_decision = "buy" if buy_count > sell_count else "sell"

    # --- 6. Jeśli nadal "hold", kończymy ---
    if final_decision == "hold":
        print("⚠️ Brak jednoznacznego sygnału, pomijam transakcję.")
        return "hold"

    # --- 7. Sprawdzenie warunków Bollingera i Momentum ---
    #    - BUY: jeśli zbyt mocno wybiło powyżej BB albo momentum jest ujemne przy niskim ADX
    if final_decision == "buy":
        if last_close > bb_upper * 1.01:
            print("⚠️ BUY: Cena wybiła powyżej BB – możliwa korekta. HOLD.")
            return "hold"
        if momentum_15m < 0 and adx_15m < 30:
            print("⚠️ BUY: Momentum nie potwierdza trendu i ADX dość niski. HOLD.")
            return "hold"

    #    - SELL: jeśli zbyt mocno wybiło poniżej BB albo momentum jest dodatnie przy niskim ADX
    if final_decision == "sell":
        if last_close < bb_lower * 0.99:
            print("⚠️ SELL: Cena wybiła poniżej BB – możliwe odbicie. HOLD.")
            return "hold"
        if momentum_15m > 0 and adx_15m < 30:
            print("⚠️ SELL: Momentum nie potwierdza trendu spadkowego i ADX dość niski. HOLD.")
            return "hold"

    # --- 8. Sprawdzenie spreadu w % (dynamiczny próg) ---
    spread_percent = (spread / last_close) * 100
    spread_limit = 3.5  # domyślne 3%
    # jeśli ADX > 20 i RSI > 60 lub spora różnica adx_1m - adx_5m => spread_limit = 5
    if adx_15m > 20 and (rsi_15m > 60 or (adx_1m - adx_5m) > 5):
        spread_limit = 6

    if spread_percent > spread_limit:
        print(f"❌ Spread = {spread_percent:.2f}% > {spread_limit}%. HOLD.")
        return "hold"

    # --- 9. Ostateczna decyzja ---
    print(f"✅ Decyzja: Otwieram pozycję {final_decision.upper()} dla {symbol}")
    return final_decision






def analyze_and_trade(xtb_client):
    """🔍 Analizuje symbole i podejmuje decyzje handlowe cyklicznie."""
    
    while True:
        print("\n🔄🔄🔄🔄 *** Nowa iteracja analizy ***")
        print("📊 Pobieram listę otwartych pozycji...")
        open_positions = xtb_client.get_open_positions()

        print("📊 Filtruję symbole według zmienności i trendu...")
        filtered_symbols = xtb_client.filter_symbols_by_volatility_and_trend()

        print(f"✅ Wyfiltrowano {len(filtered_symbols)} symboli do analizy.")

        if not filtered_symbols:
            print("⚠️ Brak symboli do analizy, czekam 60 sekund.")
            time.sleep(60)  
            continue

        open_positions_symbols = set(open_positions["symbol"].unique()) if not open_positions.empty else set()

        for symbol in filtered_symbols:
            try:
                print(f"\n📈📈📈📈📈📈 Rozpoczynam analizę dla {symbol}...")
                
                if symbol in open_positions_symbols:
                    print(f"⚠️ Pomijam {symbol}, ponieważ istnieje już otwarta pozycja.")
                    continue  

                print("📊 Pobieram dane świecowe dla różnych interwałów...")
                signal_1m, df_1m = analyze_interval(xtb_client, symbol, interval=1, history_days=1)
                signal_5m, df_5m = analyze_interval(xtb_client, symbol, interval=5, history_days=2)
                signal_15m, df_15m = analyze_interval(xtb_client, symbol, interval=15, history_days=5)
                signal_1h, df_1h = analyze_interval(xtb_client, symbol, interval=60, history_days=30)

                print(f"📊📊📊📊 Sygnały dla {symbol}: 1M={signal_1m}, 5M={signal_5m}, 15M={signal_15m}, 1H={signal_1h}")

                print("📊 Generuję sygnał trendu dla 15M...")
                trend_signal = generate_trading_signal_15m(df_15m, xtb_client, symbol)

                print("📊 Pobieram ceny ask/bid...")
                ask_price = xtb_client.get_current_price_with_type(symbol, "buy")
                bid_price = xtb_client.get_current_price_with_type(symbol, "sell")

                if ask_price is None or bid_price is None:
                    print(f"❌ Błąd: Brak poprawnych cen dla {symbol}. Pomijam.")
                    continue

                spread = ask_price - bid_price
                last_close = ask_price  

                print(f"📊 Cena ASK: {ask_price}, BID: {bid_price}, Spread: {spread}")

                print("📊 Pobieram wskaźniki techniczne...")
                adx_1m = df_1m["ADX"].iloc[-1] if not df_1m.empty else 0
                adx_5m = df_5m["ADX"].iloc[-1] if not df_5m.empty else 0
                adx_15m = df_15m["ADX"].iloc[-1] if not df_15m.empty else 0
                adx_1h = df_1h["ADX"].iloc[-1] if not df_1h.empty else 0

                sma50_15m = df_15m["SMA50"].iloc[-1] if not df_15m.empty else 0
                sma200_15m = df_15m["SMA200"].iloc[-1] if not df_15m.empty else 0
                sma50_1h = df_1h["SMA50"].iloc[-1] if not df_1h.empty else 0
                sma200_1h = df_1h["SMA200"].iloc[-1] if not df_1h.empty else 0


                bb_upper = df_15m["BB_Upper"].iloc[-1]
                bb_lower = df_15m["BB_Lower"].iloc[-1]

                rsi_5m = df_5m["RSI"].iloc[-1]
                rsi_15m = df_15m["RSI"].iloc[-1]

                momentum_5m = df_5m["Momentum"].iloc[-1]
                momentum_15m = df_15m["Momentum"].iloc[-1]

                volume = df_1m["Volume"].iloc[-1]

 
                df_15m = calculate_indicators(df_15m, 15)  # Obliczamy wskaźniki
                if not df_15m.empty:
                    avg_volume = df_15m["Avg_Volume"].iloc[-1] if "Avg_Volume" in df_15m.columns else 0
                else:
                    avg_volume = 0

                
                
                print("📊 Decyzja tradingowa...")
                grouped_signals = [signal_1m, signal_5m, signal_15m, signal_1h]
                final_decision = decide_trade_action_crypto(
                    grouped_signals, adx_1m, adx_5m, adx_15m, adx_1h,
                    sma50_15m, sma200_15m, sma50_1h, sma200_1h, last_close,
                    bb_upper, bb_lower, rsi_5m, rsi_15m, momentum_5m, momentum_15m,
                    spread, volume, symbol, avg_volume
                )

                if final_decision == "hold":
                    print(f"⚠️⚠️⚠️ Decyzja HOLD. Pomijam {symbol}.")
                    continue
                else:
                    print(f"✅✅✅ Decyzja: Otwieram pozycję {final_decision.upper()} dla {symbol}")

                
                

                if trend_signal == "hold":
                    print(f"⚠️ Brak potwierdzenia trendu dla {symbol}. Pomijam.")
                    continue

                print(f"✅ Trend potwierdzony: {trend_signal.upper()} dla {symbol}")

                open_positions = xtb_client.get_open_positions()
                open_positions_symbols = set(open_positions["symbol"].unique()) if not open_positions.empty else set()

                if symbol in open_positions_symbols:
                    print(f"⚠️ Pozycja dla {symbol} już otwarta, pomijam.")
                    continue  

                print("📊 Obliczam poziomy SL i TP...")
                stop_loss, take_profit = xtb_client.calculate_sl_tp_custom(symbol, final_decision)

                print(f"🔍 SL/TP dla {symbol}: SL={stop_loss}, TP={take_profit}")

                if stop_loss is None or take_profit is None or stop_loss == 0 or take_profit == 0:
                    print(f"❌ Niepoprawne wartości SL/TP dla {symbol}. Pomijam.")
                    continue

                print("📊 Pobieram minimalny wolumen...")
                min_volume, step_lot_size, max_volume = xtb_client.get_minimum_volume(symbol)

                print("📊 Obliczam wielkość pozycji...")
                volume = xtb_client.calculate_volume(last_close, min_volume, step_lot_size, max_volume)

                if volume is None or volume == 0:
                    print(f"❌ Objętość transakcji dla {symbol} wynosi 0. Pomijam.")
                    continue

                print(f"📊 Finalny wolumen transakcji: {volume}")
                if not xtb_client.is_connection_active():
                    print("⚠️ WebSocket zerwany. Ponawiam połączenie...")
                    xtb_client.reconnect()
                    xtb_client.ensure_connection()


                print("🚀 Otwieram pozycję...")
                success = xtb_client.open_trade(symbol, volume, final_decision, stop_loss, take_profit)

                if success:
                    print(f"✅ Zlecenie otwarte dla {symbol}!")
                else:
                    print(f"❌ Błąd otwierania zlecenia dla {symbol}.")

            except Exception as e:
                print(f"❌ Błąd podczas analizy dla {symbol}: {e}")

        print("⏳ Czekam 120 sekund przed kolejną analizą...")
        time.sleep(120)






def analyze_interval(xtb_client, symbol, interval, history_days):
    """Analizuje określony interwał czasowy i generuje sygnał oraz zwraca DataFrame."""

    # 📥 Pobieramy dane świecowe dla wybranego interwału
    df = xtb_client.get_candlestick_data(symbol, interval, history_days)

    # 📥 Pobieramy dane 1H tylko dla dłuższych interwałów (15M i 60M)
    df_1h = None
    if interval in [15, 60]:
        df_1h = xtb_client.get_candlestick_data(symbol, interval=60, history_days=history_days)

    # ❌ Sprawdzamy, czy pobrane dane nie są puste
    if df is None or df.empty:
        print(f"❌ Brak danych dla {symbol} na interwale {interval} min. Pomijam analizę.")
        return "no data", None

    if interval in [15, 60] and (df_1h is None or df_1h.empty):
        print(f"⚠️ Brak danych 1H dla {symbol}. Analiza może być mniej dokładna.")

    # 🔍 Obliczamy wskaźniki techniczne
    df = calculate_indicators(df, interval)
    if df_1h is not None:
        df_1h = calculate_indicators(df_1h, 60)  # 1H zawsze ma ten sam interwał

    # 📌 Pobieramy wsparcie i opór (zabezpieczenie przed brakiem wartości)
    support = df.get('Support', pd.Series([df['Close'].iloc[-1] * 0.99])).iloc[-1]
    resistance = df.get('Resistance', pd.Series([df['Close'].iloc[-1] * 1.01])).iloc[-1]
    last_close = df['Close'].iloc[-1]

    print(f"🔍 {symbol} | Last Close: {last_close:.2f}, Support: {support:.2f}, Resistance: {resistance:.2f}")

    # 🔥 Wybór odpowiedniego generatora sygnału w zależności od interwału
    if interval == 1:
        signal = generate_trading_signal_1m(df, xtb_client, symbol)
    elif interval == 5:
        signal = generate_trading_signal_5m(df, xtb_client, symbol)
    elif interval == 15:
        signal = generate_trading_signal_15m(df, xtb_client, symbol)
    elif interval == 60:
        signal = generate_trading_signal_1h(df, xtb_client, symbol)
    else:
        signal = "hold"

    return signal, df

def determine_trend(df):
    """
    Analizuje trend na podstawie SMA50, SMA200 i ADX.
    """
    sma50 = df["Close"].rolling(window=50).mean().iloc[-1]
    sma200 = df["Close"].rolling(window=200).mean().iloc[-1]

    if "ADX" in df.columns:  # ✅ Poprawione (w kodzie ADX jest zapisany z dużej litery)
        adx = df["ADX"].iloc[-1]
    else:
        print("⚠️ Brak ADX w danych. Używam domyślnej wartości 20.")
        adx = 20  # Domyślna wartość dla braku danych ADX

    if sma50 > sma200 and adx > 20:
        return "up"
    elif sma50 < sma200 and adx > 20:
        return "down"
    else:
        return "sideways"



def calculate_atr(df, period=14):
    """
    Oblicza wskaźnik Average True Range (ATR) na podstawie danych świecowych.
    """
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift(1))
    df["low_close"] = abs(df["low"] - df["close"].shift(1))

    df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    return df["true_range"].rolling(window=period).mean().iloc[-1]








def detect_and_open_trade(xtb_client):
    """
    📈 Wykrywa dynamiczne wybicia i otwiera pozycje z potwierdzeniem wolumenu.
    """

    print("\n🔄 *** Nowa iteracja analizy wybić ***")
      


    print("📊 Pobieram listę dostępnych symboli...")
    symbols = xtb_client.SELECTED_SYMBOLS
    if not symbols:
        print("⚠️ Brak symboli do analizy. Sprawdź assets_list.xlsx.")
        time.sleep(10)
        return

    print("📊 Pobieram otwarte pozycje...")
    open_positions = xtb_client.get_open_positions()
    open_positions_dict = {pos["symbol"]: pos for _, pos in open_positions.iterrows()}

    for symbol in symbols:
        print(f"\n📊📊📊📊 Analiza wybicia dla {symbol}")

       

        try:
            print("📊 Pobieram dane świecowe...")
            df_1m = xtb_client.get_candlestick_data(symbol, interval=1, history_days=1)
            df_5m = xtb_client.get_candlestick_data(symbol, interval=5, history_days=2)

            if df_1m is None or df_5m is None or df_1m.empty or df_5m.empty:
                print(f"⚠️ Brak wymaganych danych historycznych dla {symbol}. Pomijam.")
                continue

            print("📊 Obliczam wskaźniki techniczne...")
            df_1m = calculate_indicators(df_1m, 1)
            df_5m = calculate_indicators(df_5m, 5)

            print("📊 Analizuję wybicie...")
            momentum_1m = df_1m["Momentum"].iloc[-1]
            rsi_1m = df_1m["RSI"].iloc[-1]
            adx_1m = df_1m["ADX"].iloc[-1]  # Obecna wartość ADX
            adx_1m_prev = df_1m["ADX"].iloc[-2]  # Poprzednia wartość ADX


            momentum_5m = df_5m["Momentum"].iloc[-1]
            rsi_5m = df_5m["RSI"].iloc[-1]
            adx_5m = df_5m["ADX"].iloc[-1]

            last_close = df_1m["Close"].iloc[-1]
            prev_close = df_1m["Close"].iloc[-2]
            price_change = (last_close - prev_close) / prev_close * 100  # Procentowa zmiana ceny

            volume = df_1m["Volume"].iloc[-1]
            avg_volume_5m = df_5m["Volume"].rolling(window=5).mean().iloc[-1]

            breakout_detected = False
            
            adx_threshold       = 20
            rsi_upper_threshold = 60
            rsi_lower_threshold = 40
            volume_factor       = 0.6
            price_change_up     = 0.02
            price_change_down   = -0.02

            # ---------------------------
            # Debug – wypisujemy każdy warunek z oceną True/False
            print("\n🔍 **Debug parametry wykrycia wybicia**")
            print(f"    ADX (1m)         = {adx_1m:.2f}, próg > {adx_threshold} -> {adx_1m > adx_threshold}")
            print(f"    RSI (1m)         = {rsi_1m:.2f}")
            print(f"       RSI>70?       -> {rsi_1m > rsi_upper_threshold}")
            print(f"       RSI<30?       -> {rsi_1m < rsi_lower_threshold}")
            print(f"    Momentum (1m)    = {momentum_1m:.2f}")
            print(f"    Volume (1m)      = {volume:.2f}, avgVol(5m)={avg_volume_5m:.2f}, factor={volume_factor}")
            print(f"       vol>avgVol*1.2? -> {volume > (avg_volume_5m * volume_factor)}")
            print(f"    PriceChange(1m)  = {price_change:.2f} %")
            print(f"       > {price_change_up}%? -> {price_change > price_change_up}")
            print(f"       < {price_change_down}%? -> {price_change < price_change_down}")
            print("--------------------------------------------------\n")

            final_decision = None  # Inicjalizacja na początku, aby uniknąć błędu dostępu
            breakout_detected = False

            # ✅ **BUY - silne wybicie w górę**
            if adx_1m > 20 and adx_1m > adx_1m_prev and rsi_1m > 60 and momentum_1m > 0 and volume > avg_volume_5m * 0.6 and price_change > 0.02:
                final_decision = "buy"
                breakout_detected = True

            # ✅ **SELL - silne wybicie w dół**
            elif adx_1m > 20 and adx_1m > adx_1m_prev and rsi_1m < 40 and momentum_1m < 0 and volume > avg_volume_5m * 0.6 and price_change < -0.02:
                final_decision = "sell"
                breakout_detected = True


            if not breakout_detected:
                print(f"⚠️ Brak dynamicznego wybicia dla {symbol}. Pomijam.")
                continue

            if final_decision is None:
                print(f"⚠️ Błąd: final_decision nadal None dla {symbol}, sprawdź logikę!")
                continue

            print(f"✅ Wykryto wybicie! Otwieram pozycję {final_decision.upper()} dla {symbol}")



            print("📊 Pobieram ceny bid/ask...")
            ask_price = xtb_client.get_current_price_with_type(symbol, "buy")
            bid_price = xtb_client.get_current_price_with_type(symbol, "sell")
            entry_price = ask_price if final_decision == "buy" else bid_price

            print("📊 Obliczam poziomy SL i TP...")
            stop_loss, take_profit = xtb_client.calculate_sl_tp_custom(symbol, final_decision)

            print("📊 Obliczam wielkość pozycji...")
            min_volume, step_lot_size, max_volume = xtb_client.get_minimum_volume(symbol)
            volume = xtb_client.calculate_volume(entry_price, min_volume, step_lot_size, max_volume)

            print("🚀 Otwieram pozycję...")
            success = xtb_client.open_trade(symbol, volume, final_decision, stop_loss, take_profit)

            if success:
                print(f"✅ Zlecenie otwarte dla {symbol}!")
            else:
                print(f"❌ Błąd otwierania zlecenia dla {symbol}.")

        except Exception as e:
            print(f"❌ Błąd podczas analizy wybicia dla {symbol}: {e}")

    print("⏳ Czekam 60 sekund przed kolejną analizą...")
    time.sleep(30)

def adjust_sl_tp(xtb_client, symbol, entry_price, trade_type, last_price, atr_multiplier=1, digits=2):
    """
    Dostosowuje poziomy SL i TP w oparciu o ATR i aktualną cenę.
    """
    try:
        entry_price = float(entry_price)
        last_price = float(last_price)

        print(f"🔍 Debug adjust_sl_tp: entry_price={entry_price}, last_price={last_price}, atr_multiplier={atr_multiplier}")

        atr = abs(last_price - entry_price) * atr_multiplier
        atr = min(atr, entry_price * 0.1)  # ATR nie może być większy niż 10% ceny

        # Pobranie aktualnych SL/TP z XTB API
        current_sl, current_tp = xtb_client.get_current_sl_tp(symbol)

        if current_sl is None or current_tp is None:
            print(f"⚠️ Brak wcześniejszych SL/TP, ustawiam domyślne wartości.")
            current_sl = entry_price * 0.95  # SL 5% poniżej ceny wejścia
            current_tp = entry_price * 1.10  # TP 10% powyżej ceny wejścia

        # **Trailing Stop Loss - zabezpieczanie zysków**
        if trade_type == "buy":
            new_sl = max(current_sl, last_price - atr)  # SL przesuwa się w górę
            new_tp = max(current_tp, last_price + atr * 2)  # TP dostosowuje się do trendu
        else:
            new_sl = min(current_sl, last_price + atr)  # SL przesuwa się w dół
            new_tp = min(current_tp, last_price - atr * 2)  # TP dostosowuje się do trendu

        # **Zaokrąglanie SL/TP do precyzji aktywa**
        new_sl = round(new_sl, digits)
        new_tp = round(new_tp, digits)

        print(f"📊 Nowe SL: {new_sl}, Nowe TP: {new_tp} (zaokrąglone do {digits} miejsc po przecinku)")
        return new_sl, new_tp

    except Exception as e:
        print(f"❌ Błąd w adjust_sl_tp: {e}")
        return None, None


def monitor_open_positions(xtb_client):
    """
    Monitoruje otwarte pozycje i dynamicznie zarządza SL oraz TP w oparciu o trend.
    """
    while True:
        print("\n🔄 *** Nowa iteracja monitorowania otwartych pozycji ***")

        open_positions = xtb_client.get_open_positions()
        if open_positions is None or open_positions.empty:
            print("✅ Brak otwartych pozycji. Monitorowanie zakończone.")
            time.sleep(120)
            continue

        for _, position in open_positions.iterrows():
            symbol = position["symbol"]
            trade_type = "buy" if position["cmd"] == 0 else "sell"
            entry_price = float(position["open_price"])
            order_id = position["order"]
            volume = float(position["volume"])
            profit = float(position["profit"])

            # ✅ Pobranie aktualnej ceny rynkowej
            close_price = xtb_client.get_current_price_with_type(symbol, trade_type)
            if close_price is None or close_price == 0:
                print(f"⚠️ {symbol} - Brak aktualnej ceny rynkowej. Pomijam analizę.")
                continue

            print(f"\n🔍 Pozycja {order_id} ({symbol}) | Debug danych pozycji:")
            print(position.to_dict())
            
            
            
            # ✅ Pobranie informacji o aktywie (leverage i lot_size)
            symbol_info = xtb_client.get_symbol_info(symbol)
            leverage = symbol_info.get("leverage", 1)  # Domyślnie 1x, jeśli brak danych
            lot_size = symbol_info.get("lot_size", 1)  # Domyślnie 1 jednostka

            # ✅ Obliczenie marży
            margin = (entry_price * volume * lot_size) / leverage
            margin = round(margin, 2)  # Zaokrąglamy do 2 miejsc po przecinku

            print(f"\n🔍 Pozycja {order_id} ({symbol}) | Debug danych pozycji:")
            print(position.to_dict())
            print(f"💰 Marża: {margin} USD (Dźwignia: {leverage}x, Lot: {lot_size})")
            
            

            # ✅ Pobranie danych świec do analizy trendu
            df = xtb_client.get_candlestick_data(symbol, interval=60, history_days=30)
            if df is None or df.empty:
                print(f"⚠️ Brak danych świecowych dla {symbol}. Pomijam analizę trendu.")
                continue

            trend_direction = determine_trend(df)  # Analiza trendu
            atr = calculate_atr(df)  # Pobranie wartości ATR

            if atr is None or atr <= 0:
                print(f"⚠️ {symbol} - Nie można obliczyć ATR. Pomijam analizę SL/TP.")
                continue

            # ✅ Pobranie liczby miejsc po przecinku (`digits`) dla danego aktywa
            symbol_info = xtb_client.get_symbol_info(symbol)
            digits = symbol_info.get("precision", 2)  # Pobranie dokładności ceny z API

            # ✅ Wywołujemy `adjust_sl_tp`
            new_sl, new_tp = adjust_sl_tp(xtb_client, symbol, entry_price, trade_type, close_price, atr_multiplier=1.5, digits=digits)

            if new_sl is None or new_tp is None:
                print(f"❌ Błąd przy aktualizacji SL/TP dla {symbol}, pomijam...")
                continue

            # ✅ Aktualizujemy SL/TP w XTB
            xtb_client.update_position_sl_tp(order_id, new_sl, new_tp)

        time.sleep(60)  # Monitorowanie co 2 minuty







import threading
import time

def start_bot():
    """🚀 Uruchamia bota tradingowego z automatycznym restartem wątków"""
    print("🚀 Uruchamiam bota tradingowego...")

    # **1️⃣ Inicjalizacja klienta XTB**
    try:
        xtb_client = XTBClient()
    except Exception as e:
        print(f"❌ Błąd podczas inicjalizacji XTBClient: {e}")
        return

    # **2️⃣ Debug: Sprawdzanie połączenia**
    if not xtb_client.is_connection_active():
        print("❌ Błąd połączenia z XTB API. Restartuję bota...")
        xtb_client.reconnect()
        if not xtb_client.is_connection_active():
            print("❌ Połączenie z XTB nie powiodło się! Zamykanie bota.")
            return

    # **3️⃣ Pobranie i filtrowanie symboli**
    print("📊 Pobieram listę symboli do analizy...")
    try:
        filtered_symbols = xtb_client.filter_symbols_by_volatility_and_trend()
        print(f"✅ Wyfiltrowano {len(filtered_symbols)} aktywów do analizy.")
    except Exception as e:
        print(f"❌ Błąd podczas filtrowania symboli: {e}")
        return

    if not filtered_symbols:
        print("⚠️ Brak symboli spełniających kryteria. Bot nie otworzy transakcji.")
        return

    # **4️⃣ Uruchamianie wątków**
    print("🚀 Uruchamianie wątków bota...")
    threads = {
        "detect_trade": restart_thread(detect_and_open_trade, (xtb_client,)),  
        "monitor_positions": restart_thread(monitor_open_positions, (xtb_client,)),  
        "analyze_trade": restart_thread(analyze_and_trade, (xtb_client,))
    }

    try:
        while True:
            time.sleep(10)  # ✅ Krótka pauza, by nie obciążać CPU

            # **🔄 Restartowanie wątków, jeśli się zakończyły**
            for name, thread in threads.items():
                if not thread.is_alive():
                    print(f"⚠️ Wątek `{name}` zakończył działanie! Restartuję...")
                    try:
                        if name == "detect_trade":
                            threads[name] = restart_thread(detect_and_open_trade, (xtb_client,))
                        elif name == "monitor_positions":
                            threads[name] = restart_thread(monitor_open_positions, (xtb_client,))
                        elif name == "analyze_trade":
                            threads[name] = restart_thread(analyze_and_trade, (xtb_client,))
                    except Exception as e:
                        print(f"❌ Błąd podczas restartu wątku `{name}`: {e}")

    except KeyboardInterrupt:
        print("🛑 Zatrzymano bota, zamykanie wątków...")
        xtb_client.close()
        print("✅ Bot zakończył działanie.")

import logging

logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("InvestBot AI uruchomiony.")

if __name__ == "__main__":
    start_bot()








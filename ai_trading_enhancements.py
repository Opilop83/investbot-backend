import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import Input
from analytics import calculate_indicators
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from news_analyzer import get_sentiment  # Import funkcji analizy sentymentu

def generate_trading_signal_1m(df, xtb_client, symbol):
    """Generuje sygnał kupna/sprzedaży dla interwału 1 minuta."""
    try:
        macd, macd_prev = df["MACD"].iloc[-1], df["MACD"].iloc[-2]
        macd_signal, macd_signal_prev = df["Signal"].iloc[-1], df["Signal"].iloc[-2]
        rsi = df["RSI"].iloc[-1]
        vwap = df["VWAP"].iloc[-1]
        last_close = df["Close"].iloc[-1]
        volume = df["Volume"].iloc[-1]
        sma50 = df["SMA50"].iloc[-1]
        sma200 = df["SMA200"].iloc[-1]
        adx = df["ADX"].iloc[-1]
        adx_prev = df["ADX"].iloc[-2]

        # Trendy SMA (poluzowane warunki)
        sma50_prev = df["SMA50"].iloc[-2]
        trend_up = sma50 > sma200 * 0.995 and sma50 > sma50_prev  # Tolerancja ±0.5%
        trend_down = sma50 < sma200 * 1.005 and sma50 < sma50_prev

        # ADX - delikatnie obniżamy próg
        adx_rising = adx > adx_prev
        adx_min = 10  # Było 12

        # Przecięcie MACD (poluzowane)
        macd_cross_up = (macd > macd_signal and macd_prev < macd_signal_prev) or (macd > 0 and macd_prev < 0)
        macd_cross_down = (macd < macd_signal and macd_prev > macd_signal_prev) or (macd < 0 and macd_prev > 0)

        # Filtr wolumenu
        avg_volume_5 = df["Volume"].rolling(window=5).mean().iloc[-1]

        # 📌 **BUY**
        if macd_cross_up and rsi > 45 and last_close > vwap * 0.995 and adx > adx_min and adx_rising and trend_up:
            return "buy"

        # 📌 **SELL**
        elif macd_cross_down and rsi < 55 and last_close < vwap * 1.005 and adx > adx_min and adx_rising and trend_down:
            return "sell"

    except Exception as e:
        print(f"⚠️ Błąd w generowaniu sygnału 1M dla {symbol}: {e}")

    return "hold"



def generate_trading_signal_5m(df, xtb_client, symbol):
    """Generuje sygnał kupna/sprzedaży dla interwału 5 minut."""
    try:
        macd, macd_prev = df["MACD"].iloc[-1], df["MACD"].iloc[-2]
        rsi = df["RSI"].iloc[-1]
        adx = df["ADX"].iloc[-1]
        last_close = df["Close"].iloc[-1]
        volume = df["Volume"].iloc[-1]
        sma50 = df["SMA50"].iloc[-1]
        sma200 = df["SMA200"].iloc[-1]

        # Filtr wolumenu
        avg_volume_5 = df["Volume"].rolling(window=5).mean().iloc[-1]

        # Trendy SMA (poluzowane)
        trend_up = sma50 > sma200 * 0.99
        trend_down = sma50 < sma200 * 1.01

        # 📌 **BUY**
        if macd > macd_prev and rsi > 45 and adx > 15 and trend_up:
            return "buy"

        # 📌 **SELL**
        elif macd < macd_prev and rsi < 55 and adx > 15 and trend_down:
            return "sell"

    except Exception as e:
        print(f"⚠️ Błąd w generowaniu sygnału 5M dla {symbol}: {e}")

    return "hold"



def generate_trading_signal_15m(df, xtb_client, symbol):
    """Generuje sygnał kupna/sprzedaży dla interwału 15 minut."""
    try:
        last_close = df["Close"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        macd_signal = df["Signal"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        stochastic_k = df["%K"].iloc[-1]
        stochastic_d = df["%D"].iloc[-1]
        adx = df["ADX"].iloc[-1]
        trend_up = df["SMA50"].iloc[-1] > df["SMA200"].iloc[-1]
        trend_down = df["SMA50"].iloc[-1] < df["SMA200"].iloc[-1]

        # 📌 **BUY**
        if macd > macd_signal and rsi > 45 and stochastic_k > 15 and stochastic_d > 15 and trend_up and adx > 20:
            return "buy"

        # 📌 **SELL**
        elif macd < macd_signal and rsi < 55 and stochastic_k < 85 and stochastic_d < 85 and trend_down and adx > 20:
            return "sell"

    except Exception as e:
        print(f"⚠️ Błąd w generowaniu sygnału 15M dla {symbol}: {e}")

    return "hold"



def generate_trading_signal_1h(df, xtb_client, symbol):
    """Generuje sygnał kupna/sprzedaży dla interwału 1H."""
    try:
        adx = df["ADX"].iloc[-1]
        mfi = df["MFI"].iloc[-1]
        last_close = df["Close"].iloc[-1]
        stochastic_k = df["%K"].iloc[-1]
        stochastic_d = df["%D"].iloc[-1]
        trend_up = df["SMA50"].iloc[-1] > df["SMA200"].iloc[-1]
        trend_down = df["SMA50"].iloc[-1] < df["SMA200"].iloc[-1]

        if adx > 20 and mfi < 80 and stochastic_k > 15 and trend_up:
            return "buy"

        elif adx > 20 and mfi > 20 and stochastic_k < 85 and trend_down:
            return "sell"

    except Exception as e:
        print(f"⚠️ Błąd w generowaniu sygnału 1H dla {symbol}: {e}")

    return "hold"
















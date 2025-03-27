import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from scipy.special import expit  # Sigmoid dla normalizacji
from scipy.signal import argrelextrema


def calculate_indicators(df, interval):
    """Oblicza wskaźniki techniczne w zależności od interwału."""

    if interval in [1, 5]:
        df = calculate_sma(df, 50)
        df = calculate_sma(df, 200)
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_vwap(df)
        df = calculate_atr(df)
        df = calculate_stochastic_oscillator(df)
        df = calculate_bollinger_bands(df)
        df = calculate_ema(df, 5)
        df = calculate_ema(df, 10)
        df = calculate_adx(df)
        df = calculate_mfi(df)
        df = calculate_ichimoku(df)
        df = calculate_obv(df)
        df = calculate_pivot_points(df)
        df = calculate_support_resistance(df)
        df = calculate_volume_profile(df)
        df = calculate_momentum(df, period=10)

    elif interval in [15, 60]:
        df = calculate_sma(df, 50)
        df = calculate_sma(df, 200)
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_atr(df)
        df = calculate_bollinger_bands(df)
        df = calculate_stochastic_oscillator(df)
        df = calculate_adx(df)
        df = calculate_ichimoku(df)
        df = calculate_obv(df)
        df = calculate_pivot_points(df)
        df = calculate_support_resistance(df)
        df = calculate_ema(df, 5)
        df = calculate_ema(df, 10)
        df = calculate_mfi(df)
        df = calculate_volume_profile(df)
        df = calculate_vwap(df)
        df = calculate_momentum(df, period=10)

    # ✅ Obliczamy średni wolumen
    df["Avg_Volume"] = df["Volume"].rolling(window=20, min_periods=1).mean()

    return df




def calculate_obv(df):
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

def calculate_pivot_points(df):
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    return df


def calculate_sma(df, period):
    """Oblicza Simple Moving Average (SMA) dla danego okresu."""
    if "Close" not in df.columns:
        print("⚠️ Brak kolumny 'Close' w danych! Pomijam obliczenia SMA.")
        return df

    df[f"SMA{period}"] = df["Close"].rolling(window=period, min_periods=1).mean().astype("float64")
    return df


def calculate_macd(df):
    """Oblicza MACD oraz linię sygnału MACD z normalizacją wartości."""
    
    if "Close" not in df.columns:
        print("⚠️ Brak kolumny 'Close' w danych! Pomijam obliczenia MACD.")
        return df

    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    
    df["MACD"] = (short_ema - long_ema) / long_ema  # ✅ Normalizacja jako procent ceny
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ✅ Filtrujemy ekstremalne wartości MACD (błędy w obliczeniach)
    if abs(df["MACD"].iloc[-1]) > 5:  # MACD powinien być w zakresie -5 do +5
        print(f"⚠️ Podejrzana wartość MACD: {df['MACD'].iloc[-1]}. Normalizuję...")
        df["MACD"] = df["MACD"] / abs(df["MACD"].iloc[-1]) * 2  # Skaluje wartości do maks 2
    
    return df


def calculate_rsi(df, period=14):
    """Oblicza Relative Strength Index (RSI)."""
    delta = df["Close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_atr(df, period=14):
    """
    Oblicza Average True Range (ATR) na podstawie cen High, Low i Close.
    
    :param df: DataFrame zawierający kolumny 'High', 'Low', 'Close'
    :param period: Okres ATR (domyślnie 14)
    :return: DataFrame z dodaną kolumną 'ATR'
    """

    # Sprawdzenie, czy wymagane kolumny są dostępne
    required_columns = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required_columns):
        print("⚠️ Brak wymaganych kolumn w danych do obliczenia ATR:", required_columns)
        df["ATR"] = np.nan  # Dodanie pustej kolumny ATR
        return df

    # Obliczenie True Range
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))

    # Prawdziwy zasięg (True Range)
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Obliczenie ATR jako średniej kroczącej z True Range
    df["ATR"] = true_range.rolling(window=period, min_periods=1).mean()

    return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Oblicza g贸rne i dolne pasma Bollingera, obs艂uguj膮c NaN."""
    sma = df["Close"].rolling(window=period, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=period, min_periods=1).std()
    
    df["BB_Upper"] = sma + (rolling_std * std_dev)
    df["BB_Lower"] = sma - (rolling_std * std_dev)

    return df

def calculate_stochastic_oscillator(df, period=20):
    """Oblicza Stochastic Oscillator (%K i %D), obsługując przypadki dzielenia przez zero."""
    df['L14'] = df['Low'].rolling(window=period, min_periods=1).min()
    df['H14'] = df['High'].rolling(window=period, min_periods=1).max()

    # Unikamy dzielenia przez zero
    denominator = df['H14'] - df['L14']
    df['%K'] = np.where(denominator != 0, 100 * ((df['Close'] - df['L14']) / denominator), np.nan)
    
    df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()

    # Usuwamy zbędne kolumny pomocnicze
    df.drop(columns=['L14', 'H14'], inplace=True)

    return df



def calculate_adx(df, period=14):
    """Oblicza Average Directional Index (ADX)."""
    df['TR'] = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    df['DM_plus'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']), df['High'] - df['High'].shift(), 0)
    df['DM_minus'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()), df['Low'].shift() - df['Low'], 0)
    df['TR14'] = df['TR'].rolling(window=period).sum()
    df['DM_plus14'] = df['DM_plus'].rolling(window=period).sum()
    df['DM_minus14'] = df['DM_minus'].rolling(window=period).sum()
    df['DI_plus'] = 100 * (df['DM_plus14'] / df['TR14'])
    df['DI_minus'] = 100 * (df['DM_minus14'] / df['TR14'])
    df['DX'] = (abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()
    return df

def calculate_ichimoku(df):
    """Oblicza wska藕nik Ichimoku Cloud."""
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high9 + low9) / 2
    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high26 + low26) / 2
    df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    high52 = df['High'].rolling(window=52).max()
    low52 = df['Low'].rolling(window=52).min()
    df['Senkou_span_B'] = ((high52 + low52) / 2).shift(26)
    return df



def calculate_support_resistance(df, order=30, min_distance_ratio=0.02):
    """
    Oblicza poziomy wsparcia i oporu na podstawie historycznych maksim贸w i minim贸w oraz wolumenu.
    
    - `order`: liczba 艣wiec do analizy dla lokalnych ekstrem贸w (domy艣lnie: 50 艣wiec)
    - `min_distance_ratio`: minimalna r贸偶nica mi臋dzy wsparciem/oporem a aktualn膮 cen膮 (np. 2%)
    """
    if df.empty or 'Low' not in df.columns or 'High' not in df.columns:
        df['Support'] = np.nan
        df['Resistance'] = np.nan
        return df

    # 馃搶 Pobieramy ekstremalne warto艣ci cenowe (min/max)
    local_min = df['Low'].iloc[argrelextrema(df['Low'].values, np.less_equal, order=order)[0]]
    local_max = df['High'].iloc[argrelextrema(df['High'].values, np.greater_equal, order=order)[0]]

    # 馃搶 Pobieramy zmienno艣膰 rynku (ATR) jako filtr dla wsparcia i oporu
    atr = df['High'] - df['Low']  
    avg_atr = atr.rolling(window=order).mean().iloc[-1] if len(atr) >= order else atr.mean()

    # 馃搶 Ostatnia cena zamkni臋cia
    last_close = df['Close'].iloc[-1]

    # 馃搶 Filtrowanie poziom贸w wsparcia/oporu
    min_distance = last_close * min_distance_ratio  # Minimalna odleg艂o艣膰 od ceny

    valid_supports = local_min[(abs(local_min - last_close) > min_distance) & (local_min < last_close)]
    valid_resistances = local_max[(abs(local_max - last_close) > min_distance) & (local_max > last_close)]

    # 馃搶 Dodajemy analiz臋 wolumenu
    if 'Volume' in df.columns:
        avg_volume = df['Volume'].rolling(window=order).mean().iloc[-1]
        strong_supports = valid_supports[df['Volume'].iloc[valid_supports.index] > avg_volume]
        strong_resistances = valid_resistances[df['Volume'].iloc[valid_resistances.index] > avg_volume]

        support = strong_supports.iloc[-1] if not strong_supports.empty else valid_supports.iloc[-1] if not valid_supports.empty else last_close - avg_atr
        resistance = strong_resistances.iloc[-1] if not strong_resistances.empty else valid_resistances.iloc[-1] if not valid_resistances.empty else last_close + avg_atr
    else:
        support = valid_supports.iloc[-1] if not valid_supports.empty else last_close - avg_atr
        resistance = valid_resistances.iloc[-1] if not valid_resistances.empty else last_close + avg_atr

    # 馃搶 Zaokr膮glamy warto艣ci do 5 miejsc po przecinku
    support = round(float(support), 5) if not np.isnan(support) else np.nan
    resistance = round(float(resistance), 5) if not np.isnan(resistance) else np.nan

    # 馃搶 Eksportujemy warto艣ci do DataFrame
    df.loc[df.index[-1], 'Support'] = support
    df.loc[df.index[-1], 'Resistance'] = resistance

        
    return df
    
    



def calculate_vwap(df):
    """Oblicza VWAP (Volume Weighted Average Price) dla 15M i 1H."""
    
    if "Volume" not in df.columns or "Close" not in df.columns:
        print("⚠️ Brak wymaganych kolumn do obliczenia VWAP.")
        df["VWAP"] = np.nan
        return df

    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    
    # ✅ Korekcja, jeśli VWAP jest poza zakresem cen rynkowych
    if df["VWAP"].iloc[-1] > df["High"].max() * 1.1 or df["VWAP"].iloc[-1] < df["Low"].min() * 0.9:
        print(f"⚠️ Podejrzana wartość VWAP ({df['VWAP'].iloc[-1]}). Koryguję...")
        df["VWAP"] = df["Close"].rolling(window=20, min_periods=1).mean()  # Alternatywne obliczenie

    return df




def calculate_ema(df, period):
    """Oblicza Exponential Moving Average (EMA) dla danego okresu."""
    df[f"EMA{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df



def calculate_mfi(df, period=14):
    """Oblicza Money Flow Index (MFI)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_flow_sum = positive_flow.rolling(window=period).sum()
    negative_flow_sum = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    df["MFI"] = mfi
    return df


def calculate_volume_profile(df, bins=20):
    """Oblicza poziomy wsparcia i oporu na podstawie Volume Profile."""
    price_range = np.linspace(df["Low"].min(), df["High"].max(), bins)
    volume_profile = []

    for i in range(1, len(price_range)):
        volume = df[(df["Close"] >= price_range[i-1]) & (df["Close"] < price_range[i])]["Volume"].sum()
        volume_profile.append((price_range[i-1], volume))

    volume_profile = sorted(volume_profile, key=lambda x: x[1], reverse=True)
    
    support = volume_profile[0][0]
    resistance = volume_profile[1][0]
    
    df["Volume_Profile_Support"] = support
    df["Volume_Profile_Resistance"] = resistance
    return df


def calculate_roc(df, period=10):
    """?? Oblicza Rate of Change (ROC) - miernik momentum."""
    df["ROC"] = df["Close"].pct_change(periods=period) * 100
    return df

def calculate_momentum(df, period=10):
    """Oblicza wskaźnik Momentum i normalizuje wartości."""
    
    if "Close" not in df.columns:
        print("⚠️ Brak kolumny 'Close' w danych! Pomijam obliczenia Momentum.")
        return df

    df["Momentum"] = df["Close"] - df["Close"].shift(period)
    
    # ✅ Normalizacja - przekształcamy Momentum do wartości procentowej
    df["Momentum"] = (df["Momentum"] / df["Close"].shift(period)) * 100
    
    # ✅ Zabezpieczenie przed wartościami ekstremalnymi
    if abs(df["Momentum"].iloc[-1]) > 20:  # Momentum powinno być w zakresie -20% do +20%
        print(f"⚠️ Podejrzana wartość Momentum: {df['Momentum'].iloc[-1]}. Normalizuję...")
        df["Momentum"] = df["Momentum"] / abs(df["Momentum"].iloc[-1]) * 10  # Skaluje wartości do maks 10
    
    return df


def calculate_williams_r(df, period=14):
    """?? Oblicza Williams %R (overbought/oversold indicator)."""
    highest_high = df["High"].rolling(window=period).max()
    lowest_low = df["Low"].rolling(window=period).min()
    df["Williams_R"] = -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low))
    return df

def calculate_cci(df, period=20, constant=0.015):
    """?? Oblicza Commodity Channel Index (CCI)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = (tp - sma_tp).abs().rolling(window=period).mean()
    df["CCI"] = (tp - sma_tp) / (constant * mad)
    return df
    
    

   
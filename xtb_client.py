import json
import os
import csv
import time
import pandas as pd
import websocket
from dotenv import load_dotenv
import atexit
from analytics import calculate_indicators
from analytics import calculate_support_resistance
from analytics import calculate_atr  # Jeśli funkcja jest w analytics.py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from ai_trading_enhancements import generate_trading_signal_15m, generate_trading_signal_1h
from datetime import datetime
import ssl
import random
import requests
import traceback
load_dotenv()
XTB_USER = os.getenv("XTB_USER")
XTB_PASSWORD = os.getenv("XTB_PASSWORD")
XTB_API_URL = os.getenv("XTB_API_URL")

class XTBClient:
    def __init__(self, capital=10000):
        self.ws = None
        self.session_id = None
        self.last_request_time = 0
        self.request_interval = 1
        self.assets_file = os.path.join(os.getcwd(), "assets_list.xlsx")
        self.SELECTED_SYMBOLS = self.load_assets_from_excel()
        self.capital = capital
        self.log_file = "trade_log.csv"

        # Pobranie danych logowania z .env
        self.user = os.getenv("XTB_USER")
        self.password = os.getenv("XTB_PASSWORD")
        self.api_url = os.getenv("XTB_API_URL")

        if not all([self.user, self.password, self.api_url]):
            raise ValueError("Brak wymaganych danych logowania do XTB API!")

        # Poprawiona inicjalizacja klienta XTB - bez parametru URL
        self.client = APIClient()

        # Logowanie do API XTB
        self.client.login(XTB_USER, XTB_PASSWORD)


    def rate_limited_request(self, function, *args, **kwargs):
        """📊 Ogranicza liczbę zapytań do API."""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)

        self.last_request_time = time.time()
        return function(*args, **kwargs)

    def connect(self, retries=3, wait_time=5):
        """🔄 Nawiązuje połączenie z API XTB i sprawdza `streamSessionId`, ponawiając w razie błędu."""
        
        for attempt in range(1, retries + 1):
            try:
                # ✅ Jeśli WebSocket już istnieje, zamykamy je przed otwarciem nowego
                if self.ws:
                    try:
                        print("⚠️ Zamykam aktywne połączenie WebSocket...")
                        self.ws.close()
                        time.sleep(2)
                    except Exception as e:
                        print(f"⚠️ Brak aktywnego WebSocket do zamknięcia. ({e})")

                self.ws = None  # Resetujemy WebSocket, aby uniknąć błędów

                # 🔹 Tworzymy nowe połączenie WebSocket
                print(f"🔄 Próba {attempt}/{retries}: Logowanie do XTB API...")
                self.ws = websocket.create_connection(XTB_API_URL)

                login_payload = json.dumps({
                    "command": "login",
                    "arguments": {
                        "userId": XTB_USER,
                        "password": XTB_PASSWORD
                    }
                })
                self.ws.send(login_payload)
                response = json.loads(self.ws.recv())

                 

                if response.get("status"):
                    self.session_id = response.get("streamSessionId")

                    if self.session_id:
                        print(f"✅ Zalogowano do XTB API! `streamSessionId`: {self.session_id}")
                        return True  # Sukces, kończymy pętlę
                    else:
                        print(f"⚠️ Brak `streamSessionId`! Próba {attempt}/{retries}. Czekam {wait_time}s...")
                        time.sleep(wait_time)
                        continue  # Kolejna próba

                else:
                    print(f"❌ Błąd logowania: {response}")
                    time.sleep(wait_time)

            except Exception as e:
                if "SocketError" in str(e) or "LOGIN_TIMEOUT" in str(e):
                    print(f"⚠️ XTB API zerwało połączenie. Próba {attempt}/{retries}. Czekam {wait_time}s...")
                    time.sleep(wait_time)
                    continue  # Kolejna próba logowania
                else:
                    print(f"❌ Błąd podczas logowania do API XTB: {e}")
                    time.sleep(wait_time)

            # Po błędzie zamykamy WebSocket
            if self.ws:
                self.ws.close()
                self.ws = None  # Resetujemy WebSocket
        
            self.session_id = None  # Resetujemy session_id
            time.sleep(wait_time)  # Czekamy przed kolejną próbą

        print("❌ Nie udało się połączyć z XTB API po kilku próbach.")
        return False  # Zwrot False jeśli po `retries` próbach nie udało się połączyć



    def api_command(self, command, params={}):
        """Wysyła zapytanie do API XTB"""
        return self.client.execute(command, params)


    def ensure_connection(self):
        """Sprawdza połączenie i loguje ponownie, jeśli WebSocket jest rozłączony."""
        if not self.is_connection_active():
            print("🔄 Połączenie z XTB API zerwane. Ponawiam logowanie...")
            self.reconnect()

    def load_assets_from_excel(self):
        """Ładuje symbole z pliku Excel."""
        if not os.path.exists(self.assets_file):
            print(f"❌ Plik {self.assets_file} nie istnieje!")
            return []
        try:
            df = pd.read_excel(self.assets_file)
            return df["symbol"].dropna().tolist() if "symbol" in df.columns else []
        except Exception as e:
            print(f"❌ Błąd wczytywania pliku Excel: {e}")
            return []

    def is_connection_active(self):
        """Sprawdza, czy WebSocket jest nadal połączony i działa poprawnie."""
        try:
            if self.ws and self.ws.connected:
                # Testowe wysłanie ping-ponga, aby sprawdzić czy połączenie nie jest "martwe"
                self.ws.ping()
                return True
            return False
        except Exception as e:
            print(f"⚠️ WebSocket prawdopodobnie nie działa: {e}")
            return False


    def reconnect(self, max_attempts=5, wait_time=5):
        """🔄 Ponawia połączenie z XTB API, zamykając stare sesje i obsługując błędy WebSocket."""

        for attempt in range(1, max_attempts + 1):
            try:
                # ✅ Jeśli WebSocket jest już aktywny, nie rób nic
                if self.is_connection_active():
                    print("✅ WebSocket już działa. Nie trzeba ponownie łączyć.")
                    return True

                print(f"🔄 Próba {attempt}/{max_attempts}: Ponawiam logowanie do XTB API...")

                # ✅ Bezpieczne zamknięcie WebSocket, jeśli istnieje
                if self.ws:
                    try:
                        if self.ws.connected:
                            print("⚠️ Zamykam aktywne połączenie WebSocket...")
                            self.ws.close()
                        else:
                            print("⚠️ WebSocket już był zamknięty.")
                    except Exception as e:
                        print(f"⚠️ Błąd przy zamykaniu WebSocket: {e}")

                    self.ws = None  # Resetuj instancję WebSocket

                # ✅ Spróbuj wylogować się, jeśli sesja istnieje
                if self.session_id:
                    try:
                        print("🔄 Wylogowuję aktywną sesję XTB...")
                        self.logout()  # Jeśli masz metodę `logout()`, użyj jej
                        self.session_id = None
                    except Exception as e:
                        print(f"⚠️ Błąd przy wylogowywaniu: {e}")

                # ✅ Spróbuj ponownie połączyć się z API
                print("🔄 Logowanie do XTB API...")
                if not self.connect():
                    print("❌ Błąd połączenia z API XTB.")
                    raise ConnectionError("Nie udało się połączyć z API XTB.")

                if self.session_id:
                    print(f"✅ Ponownie połączono z XTB API! `streamSessionId`: {self.session_id}")
                    return True

            except Exception as e:
                print(f"❌ Błąd reconnecta: {e}")

            # ⏳ Stopniowe zwiększanie czasu oczekiwania między próbami (np. 5s, 7s, 10s...)
            wait_time += random.uniform(2, 5)
            time.sleep(wait_time)

        print("❌ Nie udało się ponownie połączyć z XTB API.")
        return False





    def logout(self):
        """Wysyła żądanie wylogowania z XTB API i zamyka WebSocket."""
        try:
            # ✅ Sprawdzamy, czy w ogóle jest sesja do zamknięcia
            if not self.session_id:
                print("⚠️ Brak aktywnej sesji `session_id`. Nie ma potrzeby wylogowania.")
                return True

            if self.ws:
                logout_payload = json.dumps({"command": "logout"})

                try:
                    self.ws.send(logout_payload)
                    response = json.loads(self.ws.recv())

                    if response.get("status"):
                        print("✅ Poprawnie wylogowano z XTB API.")
                    else:
                        print(f"⚠️ Błąd wylogowania: {response}")

                except Exception as e:
                    print(f"⚠️ Błąd podczas wysyłania żądania `logout`: {e}")

                # ✅ Bezpieczne zamykanie WebSocket
                try:
                    self.ws.close()
                    print("🔌 WebSocket zamknięty.")
                except Exception as e:
                    print(f"⚠️ Błąd podczas zamykania WebSocket: {e}")

                self.ws = None  # Reset WebSocket
                self.session_id = None  # Reset session_id

            else:
                print("⚠️ Brak aktywnego WebSocket do zamknięcia.")

        except Exception as e:
            print(f"❌ Błąd podczas wylogowania: {e}")
            self.ws = None  # Czyszczenie obiektu WebSocket po błędzie
            self.session_id = None  # Reset sesji po błędzie

        return True  # ✅ Zwracamy True, aby inne funkcje wiedziały, że sesja została zakończona


    # Funkcja do zarządzania WebSocket
    def reconnect_websocket(self):
        try:
            self.disconnect_websocket()  # Spróbuj zamknąć poprzednie połączenie
            time.sleep(3)  # Krótka pauza
            self.connect_websocket()  # Ponowne połączenie
            print("✅ Ponownie połączono WebSocket.")
        except Exception as e:
            print(f"❌ Błąd ponownego łączenia WebSocket: {e}")


    def get_tick_size(self, symbol):
        """Pobiera wielkość ticka dla danego symbolu."""
        symbol_data = self.get_symbol(symbol)
        if symbol_data and 'tickSize' in symbol_data:
            return float(symbol_data['tickSize'])
        else:
            print(f"❌ Błąd: tick_size dla {symbol} to None!")
            return 0.01  # Domyślna wartość dla bezpieczeństwa



    def filter_symbols_by_volatility_and_trend(self, min_atr=0.001, min_adx=10, max_range_factor=0.035):
        """
        Filtruje symbole na podstawie:
        ✅ Zmienności (ATR) – unika niskiej zmienności.
        ✅ Siły trendu (ADX) – unika słabych trendów.
        ✅ Konsolidacji (wsparcie vs opór, wstęgi Bollingera) – unika rynków bocznych.
        """
        filtered_symbols = []

        for symbol in self.SELECTED_SYMBOLS:
            print(f"📥 Pobieram dane świecowe dla {symbol} (1H)...")

            # Pobranie danych 1H (60 min) – 30 dni historii
            df = self.get_candlestick_data(symbol, 60, 60)

            if df is None or df.empty:
                print(f"⚠️ Brak danych świecowych dla {symbol}, pomijam.")
                continue

            # **Obliczenie wskaźników dla interwału 1H**
            df = calculate_indicators(df, 60)

            # Sprawdzenie, czy wskaźniki zostały poprawnie dodane
            required_indicators = ["ATR", "ADX", "Support", "Resistance", "BB_Upper", "BB_Lower", "Close"]
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

            if missing_indicators:
                print(f"⚠️ Brak wymaganych wskaźników dla {symbol}: {', '.join(missing_indicators)}. Pomijam.")
                continue

            # Pobranie ostatnich wartości wskaźników
            atr = df["ATR"].iloc[-1]
            adx = df["ADX"].iloc[-1]
            support = df["Support"].iloc[-1]
            resistance = df["Resistance"].iloc[-1]
            bb_upper = df["BB_Upper"].iloc[-1]
            bb_lower = df["BB_Lower"].iloc[-1]
            last_price = df["Close"].iloc[-1]

            # **1️⃣ Filtr ATR i ADX (unikamy niskiej zmienności i słabych trendów)**
            if atr < min_atr or adx < min_adx:
                print(f"⚠️ Symbol {symbol} nie spełnia kryteriów zmienności/trendu (ATR={atr:.2f}, ADX={adx:.2f}). Pomijam.")
                continue

            # **2️⃣ Filtr konsolidacji - zbyt wąski zakres wsparcia/oporu**
            if (resistance - support) < last_price * max_range_factor:
                print(f"⚠️ Konsolidacja: wsparcie ({support:.5f}) i opór ({resistance:.5f}) są zbyt blisko siebie dla {symbol}. Pomijam.")
                continue

      
            # ✅ Jeśli symbol przeszedł wszystkie testy, dodajemy go do listy
            print(f"✅ Symbol {symbol} spełnia kryteria (ATR={atr:.2f}, ADX={adx:.2f}) i nie jest w konsolidacji.")
            filtered_symbols.append(symbol)

        print(f"✅ Wyfiltrowano {len(filtered_symbols)} aktywów spełniających kryteria.")
        return filtered_symbols




    def log_trade_decision(self, action, symbol, order_id=None, price=None, sl=None, tp=None, volume=None, status="PENDING"):
        """Loguje decyzje tradingowe bota do pliku CSV."""

        log_headers = ["timestamp", "action", "symbol", "order_id", "price", "sl", "tp", "volume", "status"]
        file_exists = os.path.exists(self.log_file)

        try:
            with open(self.log_file, mode="a", newline="") as file:
                writer = csv.writer(file)

                # Jeśli plik nie istnieje, dodaj nagłówki
                if not file_exists:
                    writer.writerow(log_headers)

                # Zapisujemy dane, konwertując `None` na "N/A"
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                    action,
                    symbol if symbol else "UNKNOWN",
                    order_id if order_id else "N/A",
                    round(price, 5) if isinstance(price, (int, float)) else "N/A",
                    round(sl, 5) if isinstance(sl, (int, float)) else "N/A",
                    round(tp, 5) if isinstance(tp, (int, float)) else "N/A",
                    round(volume, 5) if isinstance(volume, (int, float)) else "N/A",
                    status
                ])

            print(f"📜 Zalogowano decyzję: {action} dla {symbol}, order: {order_id}")

        except Exception as e:
            print(f"❌ Błąd zapisu do logu: {e}")




    def detect_price_anomaly(self, symbol, window_size=5, threshold=2.0):
        """Wykrywa anomalie w skokach cen bid/ask."""
        log_file = "price_log.csv"

        if not os.path.isfile(log_file):
            return False  # Brak danych

        df = pd.read_csv(log_file)

        # Filtrujemy dane dla danego symbolu
        df = df[df["symbol"] == symbol]
    
        if len(df) < window_size:
            return False  # Za mało danych do analizy
    
        # Konwersja na liczby
        df["bid_price"] = df["bid_price"].astype(float)
        df["ask_price"] = df["ask_price"].astype(float)

        # Obliczamy zmiany cen w ostatnich `window_size` rekordach
        df["bid_change"] = df["bid_price"].diff()
        df["ask_change"] = df["ask_price"].diff()

        # Obliczamy odchylenie standardowe zmian cen
        bid_std = np.std(df["bid_change"].dropna())
        ask_std = np.std(df["ask_change"].dropna())

        # Pobieramy najnowszą zmianę ceny
        latest_bid_change = df["bid_change"].iloc[-1]
        latest_ask_change = df["ask_change"].iloc[-1]

        # Sprawdzamy, czy zmiana przekracza próg anomalii
        if abs(latest_bid_change) > threshold * bid_std or abs(latest_ask_change) > threshold * ask_std:
            print(f"⚠️ ANOMALIA CENOWA dla {symbol}! Skok ceny: Bid {latest_bid_change:.2f}, Ask {latest_ask_change:.2f}")
            return True  # Wykryto anomalię

        return False  # Brak anomalii    



    def get_current_price_with_type(self, symbol, trade_type, retries=3):
        """📊 Pobiera aktualną cenę ask/bid z API XTB z obsługą błędów."""

        self.ensure_connection()

        # 🛑 **Walidacja symbolu**
        if not symbol or not isinstance(symbol, str):
            print(f"❌ Nieprawidłowy symbol: {symbol} ({type(symbol)})")
            return None

        request_payload = json.dumps({"command": "getSymbol", "arguments": {"symbol": symbol}})

        for attempt in range(1, retries + 1):
            try:
                # 🔄 **Sprawdzenie połączenia przed wysłaniem żądania**
                if not self.is_connection_active():
                    print(f"⚠️ WebSocket nieaktywny. Ponawiam połączenie... (Próba {attempt}/{retries})")
                    self.reconnect()
                    self.ensure_connection()

                # 🛑 **Sprawdzenie, czy WebSocket jest gotowy**
                if not self.ws:
                    print(f"❌ WebSocket nie został poprawnie zainicjalizowany. Przerywam próbę {attempt}.")
                    time.sleep(attempt * random.uniform(1, 2))
                    continue

                self.ws.send(request_payload)
                response = json.loads(self.ws.recv())

                # 🛑 **Sprawdzenie poprawności odpowiedzi**
                if not response or not isinstance(response, dict):
                    print(f"❌ Niepoprawna odpowiedź API dla {symbol}. Próba {attempt}/{retries}")
                    time.sleep(attempt * random.uniform(1, 2))
                    continue

                if not response.get("status"):
                    print(f"⚠️ API XTB zwróciło błąd dla {symbol}: {response}. Próba {attempt}/{retries}")
                    time.sleep(attempt * random.uniform(1, 2))
                    continue

                return_data = response.get("returnData")

                # 🔄 **Obsługa pustego returnData (brak danych)**
                if not return_data or not isinstance(return_data, dict):
                    print(f"⚠️ Brak `returnData` w odpowiedzi API dla {symbol}. Czekam 3s i ponawiam pobieranie...")
                    time.sleep(3)
                    if attempt < retries:
                        continue  # Spróbuj ponownie pobrać cenę
                    else:
                        print(f"❌ Nie udało się pobrać danych dla {symbol} po {retries} próbach. Pomijam.")
                        return None

                ask = return_data.get("ask")
                bid = return_data.get("bid")

                # 🛑 **Sprawdzenie wartości ask/bid**
                if ask is None or bid is None:
                    print(f"⚠️ Brak danych ask/bid dla {symbol}. Próba {attempt}/{retries}")
                    time.sleep(attempt * random.uniform(1, 2))
                    continue

                price = ask if trade_type == "buy" else bid

                # 🛑 **Sprawdzenie poprawności ceny**
                if not isinstance(price, (int, float)) or price <= 0:
                    print(f"⚠️ Niepoprawna cena ({price}) dla {symbol}. Próba {attempt}/{retries}")
                    time.sleep(attempt * random.uniform(1, 2))
                    continue

                print(f"✅ Cena dla {symbol} (próba {attempt}): ask={ask}, bid={bid}, trade_type={trade_type}")
                return price

            except websocket.WebSocketConnectionClosedException:
                print(f"⚠️ WebSocket rozłączony. Ponawiam połączenie... (Próba {attempt}/{retries})")
                self.reconnect()
                self.ensure_connection()

            except json.JSONDecodeError:
                print(f"❌ Błąd dekodowania JSON w odpowiedzi API dla {symbol}. Próba {attempt}/{retries}")

            except Exception as e:
                print(f"❌ Błąd pobierania ceny dla {symbol} (próba {attempt}): {e}")

            # ⏳ Stopniowe wydłużanie czasu przed kolejną próbą
            time.sleep(attempt * random.uniform(1, 2))

        print(f"❌ Nie udało się pobrać ceny dla {symbol} po {retries} próbach.")
        return None




    def get_position_details(self, order_id):
        """Pobiera szczegóły pozycji dla podanego ID."""
        self.ensure_connection()
        payload = json.dumps({"command": "getTrades", "arguments": {"openedOnly": True}})
    
        try:
            self.ws.send(payload)
            response = json.loads(self.ws.recv())
            print(f"🔍 Debug get_position_details response: {response}")  # Debugowanie odpowiedzi

            if response.get("status"):
                positions = response["returnData"]
                for position in positions:
                    if position["order"] == order_id:
                        return position  # Zwracamy szczegóły pozycji, jeśli znajdziemy pasujący order_id

            print(f"⚠️ Nie znaleziono pozycji o ID {order_id}.")
            return None
        except Exception as e:
            print(f"❌ Błąd pobierania szczegółów pozycji {order_id}: {e}")
            return None




    
    def get_minimum_volume(self, symbol):
        """Zwraca minimalny, maksymalny wolumen oraz krok lotu dla symbolu z API XTB."""
        self.ensure_connection()
        request_payload = json.dumps({"command": "getSymbol", "arguments": {"symbol": symbol}})
        try:
            self.ws.send(request_payload)
            response = json.loads(self.ws.recv())
            print(f"🔍 Debug response: {response}")  # Pełna odpowiedź API

            if response.get("status"):
                min_lot_size = response["returnData"].get("lotMin", 0.01)
                step_lot_size = response["returnData"].get("lotStep", 0.01)
                max_lot_size = response["returnData"].get("lotMax", 100.0)  # Domyślnie 100, jeśli brak danych
                print(f"🔍 lotMin: {min_lot_size}, lotStep: {step_lot_size}, lotMax: {max_lot_size}")
                return min_lot_size, step_lot_size, max_lot_size

        except Exception as e:
            print(f"❌ Błąd pobierania minimalnego wolumenu dla {symbol}: {e}")

        return 0.01, 0.01, 100.0  # Domyślne wartości w razie błędu



    def validate_volume(self, volume, min_lot_size, step_lot_size):
        """Waliduje i zaokrągla wolumen zgodnie z min_lot_size i step_lot_size."""
        if volume < min_lot_size:
            volume = min_lot_size
        rounded_volume = round(volume / step_lot_size) * step_lot_size
        return max(min_lot_size, round(rounded_volume, 3))
    


    def calculate_volume(self, entry_price, min_volume, step_lot_size, max_volume):
        """Oblicza optymalny wolumen transakcji, uwzględniając min/max lot oraz krok lotu."""
        if entry_price <= 0:
            print("❌ Błędna wartość entry_price.")
            return min_volume

        target_min_value = 100  # Minimalna wartość transakcji w PLN
        target_max_value = 400  # Maksymalna wartość transakcji w PLN

        print(f"🔍 Debug: entry_price={entry_price}, min_volume={min_volume}, step_lot_size={step_lot_size}, max_volume={max_volume}")

        # **Krok 1: Obliczamy początkowy wolumen na podstawie target_min_value**
        volume = target_min_value / entry_price

        # **Krok 2: Zaokrąglamy do `step_lot_size`**
        volume = round(volume / step_lot_size) * step_lot_size

        # **Krok 3: Sprawdzamy, czy wolumen mieści się w przedziale min/max lotu**
        volume = max(min_volume, min(volume, max_volume))

        # **Krok 4: Finalna walidacja pod kątem wartości transakcji**
        transaction_value = volume * entry_price

        if transaction_value < target_min_value:
            volume = round((target_min_value / entry_price) / step_lot_size) * step_lot_size
            volume = max(min_volume, min(volume, max_volume))  # Jeszcze raz sprawdzamy min/max
        elif transaction_value > target_max_value:
            volume = round((target_max_value / entry_price) / step_lot_size) * step_lot_size
            volume = max(min_volume, min(volume, max_volume))  # Jeszcze raz sprawdzamy min/max

        print(f"✅ Finalny wolumen: {volume} (dostosowany do kroków lotu: {step_lot_size})")
        return volume




    def get_open_positions(self, retry_attempts=3):
        """Pobiera listę otwartych pozycji z XTB API z obsługą reconnect."""
    
        self.ensure_connection()  # Upewnij się, że sesja jest aktywna
        request_payload = json.dumps({"command": "getTrades", "arguments": {"openedOnly": True}})
    
        for attempt in range(1, retry_attempts + 1):
            try:
                self.ws.send(request_payload)
                response = json.loads(self.ws.recv())

                if response.get("status"):
                    trades = response["returnData"]
                    if not trades:
                        print("✅ Brak otwartych pozycji.")
                        return pd.DataFrame()

                    df = pd.DataFrame(trades)

                    # Sprawdź, czy są poprawne kolumny
                    if "symbol" not in df.columns:
                        if "instrument" in df.columns:
                            df.rename(columns={"instrument": "symbol"}, inplace=True)
                        else:
                            print(f"⚠️ Brak kolumny 'symbol'. Oto dostępne kolumny: {df.columns}")
                            return pd.DataFrame()

                    df['type'] = df['cmd'].map({0: 'buy', 1: 'sell'})
                    df['profit_margin'] = ((df['close_price'] - df['open_price']) / df['open_price']) * 100
                    df['ai_generated'] = df['customComment'].str.contains('AI Trading Bot', na=False)

                    return df

                else:
                    print(f"❌ Błąd pobierania otwartych pozycji (próba {attempt}/{retry_attempts}): {response}")
        
            except Exception as e:
                print(f"❌ Błąd pobierania otwartych pozycji (próba {attempt}/{retry_attempts}): {e}")
            
                if attempt < retry_attempts:
                    print("🔄 Ponawiam połączenie z XTB API...")
                    self.reconnect()
                    time.sleep(2)

        print("⚠️ Nie udało się pobrać otwartych pozycji po kilku próbach.")
        return pd.DataFrame()  # Zwróć pusty DataFrame, jeśli nie uda się pobrać danych.





    def get_trades(self, openedOnly=True):
        """
        Pobiera listę otwartych lub zamkniętych pozycji z XTB API.
        :param openedOnly: Jeśli True, pobiera tylko otwarte pozycje.
        :return: DataFrame zawierający otwarte pozycje lub None, jeśli wystąpił błąd.
        """
        self.ensure_connection()  # Sprawdzamy połączenie

        request_payload = {
            "command": "getTrades",
            "arguments": {"openedOnly": openedOnly}
        }

        try:
            self.ws.send(json.dumps(request_payload))
            response = json.loads(self.ws.recv())

            if response.get("status") and "returnData" in response:
                trades = response["returnData"]
                if trades:
                    return pd.DataFrame(trades)  # Konwersja na DataFrame dla łatwiejszej analizy
                else:
                    print("✅ Brak otwartych pozycji.")
                    return pd.DataFrame()  # Zwracamy pusty DataFrame
            else:
                print(f"❌ Błąd pobierania pozycji: {response}")
                return None

        except Exception as e:
            print(f"❌ Błąd podczas pobierania pozycji: {e}")
            return None

    def requires_price_in_close(self):
        """
        Sprawdza, czy API XTB wymaga podania ceny przy zamykaniu pozycji.
        Niektóre konta mogą wymagać ceny, a inne nie – zależy od warunków brokera.
        """
        symbol_test = "EURUSD"  # Można użyć dowolnego popularnego symbolu
        test_order = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "cmd": 2,  # Zamknięcie pozycji
                    "order": 123456789,  # Losowy numer orderu, by sprawdzić odpowiedź API
                    "symbol": symbol_test,
                    "volume": 0.1,  # Minimalny wolumen testowy
                }
            }
        }

        try:
            self.ws.send(json.dumps(test_order))
            response = json.loads(self.ws.recv())
            error_code = response.get("errorCode", "")

            if error_code in ["BE001", "BE002"]:
                print("⚠️ API wymaga podania ceny przy zamykaniu pozycji.")
                return True
            else:
                print("✅ API nie wymaga podania ceny przy zamykaniu pozycji.")
                return False

        except Exception as e:
            print(f"❌ Błąd testowania wymaganej ceny w zamykaniu pozycji: {e}")
            return False  # Domyślnie zakładamy, że nie jest wymagana



    def is_order_active(self, order_id, retry_attempts=3):
        """Sprawdza, czy dany order_id nadal istnieje w otwartych pozycjach"""
        for attempt in range(1, retry_attempts + 1):
            open_positions = self.get_trades(openedOnly=True)

            if open_positions is None:
                print(f"⚠️ Próba {attempt}/{retry_attempts}: `get_trades()` zwróciło None. Ponawiam...")
                time.sleep(2)
                continue  # Ponów próbę pobrania

            if open_positions.empty:
                print(f"✅ Brak otwartych pozycji. Order {order_id} nie istnieje.")
                return False

            if order_id in open_positions["order"].values:
                print(f"✅ Order {order_id} jest nadal aktywny.")
                return True

            print(f"⚠️ Order {order_id} nie został znaleziony w otwartych pozycjach. Ponawiam próbę...")
            time.sleep(2)

        print(f"❌ Order {order_id} nadal nie został znaleziony po {retry_attempts} próbach.")
        return False



    def close_position(self, order_id, retry_attempts=3):
        """Zamyka otwartą pozycję z obsługą reconnect, sprawdzeniem statusu i pełnym debugowaniem."""

        self.ensure_connection()  # Sprawdzenie połączenia

        # ✅ Pobranie danych pozycji
        open_positions = self.get_trades(openedOnly=True)  # Pobieranie dokładniejsze niż get_open_positions()

        if open_positions.empty or order_id not in open_positions["order"].values:
            print(f"⚠️ Pozycja {order_id} nie istnieje w aktualnie otwartych pozycjach! Pomijam zamknięcie.")
            return False  

        position = open_positions[open_positions["order"] == order_id].iloc[0]
        symbol = position["symbol"]
        volume = float(position["volume"])
        # ✅ Pobranie ceny otwarcia
        open_price = position["open_price"]  # 📌 DODAJ TĘ LINJKĘ

        # ✅ Pobranie aktualnych danych rynkowych
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            print(f"❌ Nie udało się pobrać informacji o symbolu {symbol}. Anulowanie zamknięcia.")
            return False

        bid = symbol_info.get("bid", 0)
        ask = symbol_info.get("ask", 0)
        close_price = bid if position["cmd"] == 1 else ask  # Dla SELL zamykasz na BID, dla BUY na ASK
        # ✅ Debugowanie marży przed zamknięciem pozycji
        margin_calculated = self.calculate_margin(symbol, volume, open_price)
        print(f"📊 Przed zamknięciem: {symbol} | Marża: {margin_calculated} | Open Price: {open_price} | Close Price: {close_price}")


        if close_price == 0:
            print(f"❌ Nie udało się pobrać poprawnej ceny zamknięcia dla {symbol}.")
            return False

        print(f"🔍 Próba zamknięcia {symbol} | Cena: {close_price} | Wolumen: {volume}")

        # ✅ Pobranie aktualnej ceny zamknięcia, jeśli nie istnieje w pozycji
        close_price = position.get("close_price", None)
        if close_price is None or close_price < 0:
            close_price = self.get_current_price_with_type(symbol, "sell" if position["cmd"] == 1 else "buy")

        # Dostosowanie ceny do aktualnego bid/ask
        if position["cmd"] == 1:  # SELL
            close_price = max(close_price, symbol_info["bid"])
        else:  # BUY
            close_price = min(close_price, symbol_info["ask"])

        print(f"🔍 Poprawiona cena zamknięcia dla {symbol}: {close_price}")


        # ✅ Przygotowanie payloadu
        trade_info = {
            "cmd": 2,  # Zamknięcie pozycji
            "order": order_id,
            "symbol": symbol,
            "volume": volume,
            "customComment": "AI Trading Bot - Close Position"
        }

        # ✅ Sprawdzenie, czy API wymaga `price`
        if self.requires_price_in_close():
            trade_info["price"] = round(close_price, symbol_info.get("digits", 2))

        request_payload = json.dumps({"command": "tradeTransaction", "arguments": {"tradeTransInfo": trade_info}})

        print(f"📩 Payload zamknięcia: {request_payload}")


        for attempt in range(1, retry_attempts + 1):
            try:
                if not self.is_connection_active():
                    print(f"⚠️ WebSocket rozłączony. Ponawiam połączenie... (Próba {attempt}/{retry_attempts})")
                    self.reconnect()
                    self.ensure_connection()

                print(f"📤 Wysyłam żądanie do API... (Próba {attempt}/{retry_attempts})")
                self.ws.send(request_payload)
                response = json.loads(self.ws.recv())

                print(f"🔍 Debug - pełna odpowiedź API XTB: {response}")

                if response.get("status"):
                    print(f"✅ Pozycja {order_id} zamknięta.")
                    return True  # Zakończ funkcję

                elif response.get("errorCode") == "BE9":
                    print(f"⚠️ Pozycja {order_id} nie istnieje w API. Sprawdzam jeszcze raz otwarte pozycje...")
                    if not self.is_order_active(order_id):
                        print(f"✅ Order {order_id} faktycznie został zamknięty.")
                        return True

                else:
                    print(f"❌ API zwróciło błąd: {response.get('errorDescr', 'Nieznany błąd')}")

            except Exception as e:
                print(f"❌ Błąd zamykania pozycji: {e}")

            time.sleep(attempt * 2)  # Opóźnienie dla kolejnej próby

        print(f"❌ Nie udało się zamknąć pozycji po {retry_attempts} próbach.")
        return False





    def get_candlestick_data(self, symbol, interval, history_days, retries=3):
        """📥 Pobiera dane świecowe z XTB API, obsługując błędy SSL, WebSocket oraz sprawdzając poprawność danych."""

        self.ensure_connection()  # Sprawdza, czy WebSocket jest aktywny
        start_time = int(time.time() * 1000) - (history_days * 24 * 60 * 60 * 1000)

        for attempt in range(1, retries + 1):  # Maksymalnie 3 próby pobrania danych
            try:
                request_payload = json.dumps({
                    "command": "getChartRangeRequest",
                    "arguments": {
                        "info": {
                            "start": start_time,
                            "end": int(time.time() * 1000),
                            "period": interval,
                            "symbol": str(symbol)  # 🔄 Konwersja na string
                        }
                    }
                })
                self.ws.send(request_payload)
                response = self.ws.recv()

                if not response:
                    print(f"❌ Pusta odpowiedź API XTB dla {symbol} ({interval}M), próba {attempt}/{retries}. Ponawiam...")
                    time.sleep(2)
                    continue

                response = json.loads(response)

                if not response.get("status"):
                    print(f"❌ Błąd API XTB: {response}. Próba {attempt}/{retries}.")
                    time.sleep(2)
                    continue

                if "returnData" not in response or "rateInfos" not in response["returnData"]:
                    print(f"❌ Brak klucza 'returnData' w odpowiedzi API dla {symbol}: {response}")
                    time.sleep(2)
                    continue

                prices = response["returnData"]["rateInfos"]
                digits = response["returnData"]["digits"]

                if not prices:
                    print(f"❌ Brak danych świecowych dla {symbol} ({interval}M). Próba {attempt}/{retries}.")
                    time.sleep(2)
                    continue

                df = self._process_candlestick_data(prices, digits, interval)

                if df is None or df.empty:
                    print(f"⚠️ {symbol} [interval {interval}]: Otrzymano pusty DataFrame. Ponawiam próbę...")
                    time.sleep(2)
                    continue

                # ✅ **Walidacja danych świecowych (sprawdzamy błędne wartości)**
                invalid_rows = df[
                    (df["Open"] <= 0) | (df["High"] <= 0) | (df["Low"] <= 0) | (df["Close"] <= 0) |
                    (df["High"] < df["Low"]) | (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
                ]

                if not invalid_rows.empty:
                    print(f"❌ {symbol} [{interval}M]: BŁĘDNE ŚWIECE! Odrzucam {len(invalid_rows)} świec.")
                    print(invalid_rows)  # Debug: wyświetlamy problematyczne świece
                    time.sleep(2)
                    continue  # Spróbujmy pobrać nowe dane

                # ✅ **Sprawdzenie opóźnienia świec**
                try:
                    last_candle_time = pd.to_datetime(df["timestamp"].iloc[-1], utc=True).tz_convert(None)  # 🔄 `tz-naive`
                except Exception as e:
                    print(f"❌ Błąd konwersji timestampu dla {symbol}: {e}")
                    continue  # Spróbuj pobrać dane jeszcze raz

                current_time = datetime.utcnow()  # `tz-naive`
                delay = (current_time - last_candle_time).total_seconds() / 60  # Opóźnienie w minutach
                max_delay = interval * 4  # Maksymalne opóźnienie = 4x interwał (np. 60 min dla 15M)

                print(f"💾 Pobranie świec dla {symbol} [interval {interval}]: {len(df)} świec.")
                print(f"🔍 Ostatnia świeca: {last_candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC | Opóźnienie: {delay:.1f} min.")

                if delay > max_delay:
                    print(f"⚠️ {symbol} [interval {interval}]: Opóźnienie {delay:.1f} min. Pobieram nowe świece...")
                    time.sleep(2)
                    continue  # Jeśli świeca jest za stara, próbujemy pobrać nowsze dane

                return df  # ✅ Jeśli wszystko jest OK, zwracamy dane

            except ssl.SSLError as ssl_error:
                print(f"⚠️ Błąd SSL podczas pobierania świec {symbol} ({interval} min), próba {attempt}/{retries}: {ssl_error}")
                time.sleep(3)  # Poczekaj dłużej, aby uniknąć natychmiastowego ponowienia

            except websocket.WebSocketConnectionClosedException:
                print(f"⚠️ Połączenie WebSocket zostało zamknięte. Ponawiam połączenie... (Próba {attempt})")
                self.reconnect()
                self.ensure_connection()

            except Exception as e:
                print(f"❌ Błąd pobierania świec {symbol} ({interval} min), próba {attempt}/{retries}: {e}")
                time.sleep(2)

        # 🔄 **Jeśli po 3 próbach nadal nie mamy danych, wymuszamy reconnect**
        print(f"❌ Nie udało się pobrać świec dla {symbol} po {retries} próbach. Resetuję połączenie z XTB API...")
        self.reconnect()
        return None



    def _process_candlestick_data(self, prices, digits, interval):
        """📊 Przetwarza dane świecowe na DataFrame i oblicza wskaźniki techniczne."""

        if not prices:
            print("⚠️ Otrzymano pustą listę świec.")
            return None

        factor = 10 ** digits
        df = pd.DataFrame(prices)

        if df.empty or "ctm" not in df.columns:
            print("⚠️ Otrzymano pusty lub nieprawidłowy DataFrame.")
            return None

        # ✅ Przetwarzanie timestampów i cen
        df["timestamp"] = pd.to_datetime(df["ctm"], unit="ms", utc=True)
        df["Open"] = df["open"] / factor  # Open to wartość absolutna
        df["Close"] = df["Open"] + df["close"] / factor  # Close to wartość względna względem Open
        df["High"] = df["Open"] + df["high"] / factor  # High to wartość względna względem Open
        df["Low"] = df["Open"] + df["low"] / factor  # Low to wartość względna względem Open
        df["Volume"] = df["vol"]

        # ✅ **Usunięcie błędnych wartości**
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        df.dropna(inplace=True)

        df = df[
            (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0) &
            (df["High"] >= df["Low"]) & (df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])
        ]

        if df.empty:
            print(f"❌ {interval}M: Po oczyszczeniu danych nie pozostały żadne świece!")
            return None  # Brak poprawnych świec do analizy

        print(f"✅ {interval}M: Oczyszczone dane, przekazuję do `calculate_indicators()`")
        return calculate_indicators(df, interval)

    def determine_trend(self, df):
        """
        Określa trend na podstawie SMA50 i SMA200.
        """
        if "SMA50" not in df.columns or "SMA200" not in df.columns:
            return "unknown"

        sma50 = df["SMA50"].iloc[-1]
        sma200 = df["SMA200"].iloc[-1]

        if sma50 > sma200:
            return "uptrend"
        elif sma50 < sma200:
            return "downtrend"
        else:
            return "sideways"


    def get_current_sl_tp(self, symbol):
        """
        Pobiera aktualny Stop Loss (SL) i Take Profit (TP) dla podanej pozycji.
        """
        try:
            # Pobranie wszystkich otwartych pozycji
            positions = self.get_open_positions()
            
            # ✅ Sprawdzamy czy `positions` nie jest pusty
            if positions is None or positions.empty:
                print(f"⚠️ Brak otwartych pozycji w `get_current_sl_tp` dla {symbol}.")
                return None, None  

            # ✅ Iterujemy po pozycjach i wyszukujemy symbol
            for _, position in positions.iterrows():
                if position["symbol"] == symbol:
                    try:
                        current_sl = float(position.get("sl", 0))  # Bezpieczny dostęp do SL
                        current_tp = float(position.get("tp", 0))  # Bezpieczny dostęp do TP

                        print(f"🔍 Aktualne SL: {current_sl}, TP: {current_tp} dla {symbol}")
                        return current_sl, current_tp
                    except Exception as e:
                        print(f"❌ Błąd konwersji SL/TP dla {symbol}: {e}")
                        return None, None

            print(f"⚠️ Pozycja dla {symbol} nie została znaleziona. Ustawiam domyślne wartości.")
            return None, None  # Jeśli nie ma pozycji dla symbolu

        except Exception as e:
            print(f"❌ Błąd pobierania SL/TP dla {symbol}: {e}")
            return None, None





    def calculate_sl_tp_custom(self, symbol, trade_type):
        """
        Oblicza dynamiczne poziomy SL i TP na podstawie:
        - wsparć i oporów z interwału 1H (ostatnie 7 dni)
        - aktualnego trendu
        - wartości ATR jako bufor
        - zaokrąglenia do tick_size
        """

        print("\n🔎 --- Obliczanie dynamicznych SL i TP ---")

        ask_price = self.get_current_price_with_type(symbol, "buy")
        bid_price = self.get_current_price_with_type(symbol, "sell")
        
        try:
            ask_price = float(ask_price)
            bid_price = float(bid_price)
        except ValueError:
            print(f"❌ BŁĄD: Cena ASK/BID dla {symbol} nie jest liczbą! ASK={ask_price}, BID={bid_price}")
            return None, None

        if ask_price <= 0 or bid_price <= 0:
            print(f"❌ BŁĄD: Nieprawidłowe ceny dla {symbol} -> ASK: {ask_price}, BID: {bid_price}")
            return None, None

        # ✅ Pobranie tick_size dla dokładności zaokrąglenia
        symbol_info = self.get_symbol_info(symbol)
        tick_size = symbol_info.get("tickSize", 0.01)  # Domyślnie 0.01
        stops_level = symbol_info.get("stopsLevel", 0) * tick_size  # Ustawienie domyślnej wartości
        
        # Obsługa przypadku, gdy `stopsLevel` jest None lub błędne
        if stops_level is None or stops_level <= 0:
            print(f"⚠️ Brak wartości stopsLevel dla {symbol}, ustawiam domyślnie na {tick_size * 2}")
            stops_level = tick_size * 2  # Minimalna wartość zapobiegająca błędom
        
        # ✅ Pobranie danych 1H z ostatnich 7 dni
        df = self.get_candlestick_data(symbol, interval=60, history_days=7*24)

        if df is None or df.empty or len(df) < 20:  # Minimum 20 świec dla poprawnej analizy
            print(f"⚠️ Brak wystarczających danych świecowych dla {symbol}. Używam standardowych poziomów (-4% SL, +8% TP).")
            return self.calculate_standard_sl_tp(symbol, trade_type)

        # ✅ Obliczenie wsparć i oporów
        df = calculate_support_resistance(df, order=20)  # 20 ostatnich świec 1H (~20h)

        # 📌 Obsługa błędu braku wsparcia/oporu
        support = df["Support"].iloc[-1] if "Support" in df.columns and not df["Support"].isna().all() else None
        resistance = df["Resistance"].iloc[-1] if "Resistance" in df.columns and not df["Resistance"].isna().all() else None

        # ✅ Pobranie wartości ATR jako bufor bezpieczeństwa
        atr = df["ATR"].iloc[-1] if "ATR" in df.columns and not df["ATR"].isna().all() else None

        # 📌 **Obsługa pustego ATR przed konwersją**
        if atr is None or atr == "" or pd.isna(atr):
            print(f"⚠️ Brak poprawnego ATR dla {symbol}. Używam standardowych wartości SL/TP.")
            return self.calculate_standard_sl_tp(symbol, trade_type)

        # 📌 **Bezpieczna konwersja wsparcia, oporu i ATR**
        try:
            support = float(support) if support is not None else None
            resistance = float(resistance) if resistance is not None else None
            atr = float(atr) if atr is not None else None
        except (ValueError, TypeError) as e:
            print(f"❌ BŁĄD: Niepoprawna wartość Support, Resistance lub ATR dla {symbol}: "
                  f"Support={support}, Resistance={resistance}, ATR={atr}, Błąd: {e}")
            return self.calculate_standard_sl_tp(symbol, trade_type)

        # ✅ Pobranie trendu z 1H interwału
        trend = self.determine_trend(df)
        min_distance = max(stops_level, atr * 2)  # ATR jako bufor
        # ✅ Ustalanie ceny wejścia
        entry_price = ask_price if trade_type == "buy" else bid_price

        # **Obliczanie SL i TP**
        if trade_type == "buy":
            take_profit = resistance if trend == "uptrend" and resistance else entry_price + min_distance * 2
            stop_loss = max(support, entry_price - min_distance) if support else entry_price - min_distance
        else:
            take_profit = support if trend == "downtrend" and support else entry_price - min_distance * 2
            stop_loss = min(resistance, entry_price + min_distance) if resistance else entry_price + min_distance

        # **Korekta SL i TP, jeśli są zbyt blisko entry_price**
        if trade_type == "buy":
            stop_loss = min(stop_loss, entry_price - stops_level)
            take_profit = max(take_profit, entry_price + stops_level * 2)
        else:
            stop_loss = max(stop_loss, entry_price + stops_level)
            take_profit = min(take_profit, entry_price - stops_level * 2)



        # ✅ Zaokrąglanie do tick_size
        precision = int(abs(round(np.log10(tick_size))))
        stop_loss = round(stop_loss, precision)
        take_profit = round(take_profit, precision)

        print(f"✅ Dynamiczne SL/TP dla {symbol}: SL={stop_loss}, TP={take_profit} (Trend: {trend}, ATR: {atr})")

        return stop_loss, take_profit










    def get_min_stop_distance(self, symbol):
        """ Pobiera minimalny dystans SL/TP dla danego symbolu i zapewnia poprawną obsługę błędów """
        self.ensure_connection()
        request_payload = json.dumps({"command": "getSymbol", "arguments": {"symbol": symbol}})

        try:
            self.ws.send(request_payload)
            response = json.loads(self.ws.recv())
        
            # Sprawdzamy, czy API zwróciło poprawną odpowiedź
            if not response.get("status") or "returnData" not in response:
                print(f"⚠️ Brak poprawnych danych dla {symbol}. Używam domyślnej wartości.")
                return 0.01  # Domyślna wartość minimalnego SL/TP
        
            return_data = response["returnData"]
            stops_level = return_data.get("stopsLevel", 0)  # Pobieramy stopsLevel (domyślnie 0)
            tick_size = return_data.get("tickSize", 0.01)  # Pobieramy wielkość ticka
        
            # Obliczamy minimalną odległość SL/TP
            min_distance = stops_level * tick_size
        
            # Zapewniamy, że SL/TP nigdy nie będą ustawione za blisko
            if min_distance == 0:
                min_distance = max(tick_size * 5, 0.01)  # Ustawiamy minimalny dystans jako 5 ticków

            print(f"✅ Minimalna odległość SL/TP dla {symbol}: {min_distance:.5f}")
            return min_distance

        except Exception as e:
            print(f"❌ Błąd pobierania minimalnej odległości SL/TP dla {symbol}: {e}")
        
        return 0.01  # Domyślna wartość minimalnego poziomu w przypadku błędu








    def get_symbol_info(self, symbol):
        """📥 Pobiera szczegółowe informacje o symbolu przez WebSocket."""
    
        self.ensure_connection()  # Upewnij się, że WebSocket jest aktywny

        request_payload = json.dumps({"command": "getSymbol", "arguments": {"symbol": symbol}})

        for attempt in range(2):  # ✅ Maksymalnie 2 próby, nie więcej
            try:
                print(f"📡 Wysyłanie zapytania o symbol {symbol} (próba {attempt+1}/2)...")
                self.ws.send(request_payload)
            
                response = self.ws.recv()
                response = json.loads(response)

                print(f"🔍 Debug - pełna odpowiedź API XTB dla {symbol}: {response}")  # 🔥 Debug API

                if response.get("status") and "returnData" in response and response["returnData"]:
                    return response["returnData"]

                print(f"⚠️ API zwróciło pustą odpowiedź dla {symbol}. Spróbuję ponownie..." if attempt == 0 else "❌ API nadal zwraca pustą odpowiedź.")
                time.sleep(1)  # Mała pauza przed kolejną próbą

            except Exception as e:
                print(f"❌ Błąd podczas pobierania informacji o symbolu {symbol}: {e}")
                time.sleep(2)  # Krótka pauza przed drugą próbą

        print(f"❌ Nie udało się pobrać danych dla {symbol}.")
        return {}







    def validate_price_step(self, price, step):
        """Zaokrągla cenę do najbliższego poprawnego kroku cenowego."""
        return round(price / step) * step


    def open_trade(self, symbol, volume, trade_type, stop_loss=None, take_profit=None, retries=3):
        """📊 Składa zlecenie otwarcia transakcji z dodatkowymi logami debugowania"""

        self.ensure_connection()

        # ✅ Pobranie informacji o symbolu
        symbol_info = self.get_symbol_info(symbol)
    
        if not symbol_info:
            print(f"❌ Nie udało się pobrać informacji o symbolu {symbol}. Anulowanie transakcji.")
            return False

        print(f"🔍 Debug - dane symbolu {symbol}: {symbol_info}")

        digits = symbol_info.get("digits", 2)
        min_distance = max(symbol_info.get("stopsLevel", 0) * (10 ** -digits), 0.001 * symbol_info.get("ask", 1))
        price_step = symbol_info.get("tickSize", 0.01)
        ask = symbol_info.get("ask", 0)
        bid = symbol_info.get("bid", 0)

        # ✅ Sprawdzenie czy API zwróciło poprawne ceny
        if ask == 0 or bid == 0:
            print(f"❌ Brak poprawnych cen dla {symbol}. Pomijam transakcję.")
            return False

        entry_price = ask if trade_type == "buy" else bid
        print(f"🔍 Entry Price dla {symbol}: {entry_price} | Trade Type: {trade_type}")

        # ✅ Pobranie minimalnego wolumenu
        min_volume, step_lot_size, max_volume = self.get_minimum_volume(symbol)
        volume = self.calculate_volume(entry_price, min_volume, step_lot_size, max_volume)

        # ✅ Walidacja wolumenu
        if volume is None or volume == 0:
            print(f"❌ Wolumen transakcji wynosi 0. Pomijam.")
            return False

        volume = max(min(volume, max_volume), min_volume)
        print(f"🔍 Finalny wolumen dla {symbol}: {volume}")

        # ✅ Sprawdzamy poprawność SL i TP
        stop_loss = self.validate_price_step(stop_loss, price_step) if stop_loss is not None else None
        take_profit = self.validate_price_step(take_profit, price_step) if take_profit is not None else None

        # ✅ Zapobiegamy ustawieniu SL/TP na cenę wejścia lub za blisko ceny wejścia
        if stop_loss is not None and abs(stop_loss - entry_price) < min_distance:
            print(f"⚠️ Stop Loss ({stop_loss}) za blisko ceny wejścia. Usuwam SL.")
            stop_loss = None
        if take_profit is not None and abs(take_profit - entry_price) < min_distance:
            print(f"⚠️ Take Profit ({take_profit}) za blisko ceny wejścia. Usuwam TP.")
            take_profit = None

        print(f"🎯 SL: {stop_loss}, TP: {take_profit}, Minimalna odległość: {min_distance}")

        trade_info = {
            "symbol": symbol,
            "volume": volume,
            "cmd": 0 if trade_type == "buy" else 1,
            "type": 0,
            "price": round(entry_price, digits),
            "customComment": "AI Trading Bot"
        }

        if stop_loss is not None:
            trade_info["sl"] = round(stop_loss, digits)
        if take_profit is not None:
            trade_info["tp"] = round(take_profit, digits)

        request_payload = json.dumps({"command": "tradeTransaction", "arguments": {"tradeTransInfo": trade_info}})
        print(f"📩 Payload zlecenia: {request_payload}")

        for attempt in range(1, retries + 1):
            try:
                # ✅ Sprawdzenie połączenia WebSocket
                if not self.is_connection_active():
                    print(f"⚠️ WebSocket rozłączony. Ponawiam połączenie... (Próba {attempt}/{retries})")
                    self.reconnect()
                    self.ensure_connection()

                print(f"📤 Wysyłam zlecenie do API XTB... (Próba {attempt}/{retries})")
                self.ws.send(request_payload)
                response = json.loads(self.ws.recv())

                print(f"🔍 Debug - pełna odpowiedź API XTB: {response}")

                # ✅ Obsługa odpowiedzi API
                if response.get("status") and response.get("returnData"):
                    order_data = response.get("returnData")
                    if order_data is None or "order" not in order_data:
                        print(f"⚠️ API zwróciło pustą odpowiedź dla {symbol}.")
                        self.log_trade_decision("OPEN", symbol, None, entry_price, stop_loss, take_profit, volume, status="ERROR - Empty returnData")
                        return False

                    order_id = order_data["order"]
                    print(f"✅ Zlecenie otwarte dla {symbol}. ID zamówienia: {order_id}")

                    # ✅ Logowanie transakcji
                    self.log_trade_decision("OPEN", symbol, order_id, entry_price, stop_loss, take_profit, volume, status="SUCCESS")

                    # ✅ Sprawdzenie czy SL/TP zostało ustawione
                    if "sl" not in trade_info or "tp" not in trade_info:
                        print("⚠️ Brak SL lub TP w pierwotnym zleceniu. Próbuję ustawić je osobno.")
                        self.set_sl_tp(order_id, stop_loss, take_profit)
                    return True

                # 🛑 **Jeśli API zwróciło błąd**
                error_message = response.get('errorDescr', 'Nieznany błąd')
                print(f"❌ API zwróciło błąd dla {symbol}: {error_message}")
                self.log_trade_decision("OPEN", symbol, None, entry_price, stop_loss, take_profit, volume, status=f"ERROR - {error_message}")

            except Exception as e:
                print(f"❌ Błąd otwierania pozycji dla {symbol}: {e}")
                self.log_trade_decision("OPEN", symbol, None, entry_price, stop_loss, take_profit, volume, status=f"EXCEPTION - {e}")
                return False

            time.sleep(attempt * 2)  # Opóźnienie dla ponownej próby

        print(f"❌ Nie udało się otworzyć pozycji dla {symbol} po {retries} próbach.")
        return False








    def set_sl_tp(self, order_id, stop_loss, take_profit):
        """Ustawia SL i TP dla istniejącego zlecenia, jeśli nie zostały ustawione przy otwieraniu."""
        if not stop_loss and not take_profit:
            return
        modify_payload = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": {
                    "order": order_id,
                    "type": 2,  # ORDER_MODIFY
                    "sl": stop_loss,
                    "tp": take_profit
                }
            }
        }
        try:
            self.ws.send(json.dumps(modify_payload))
            response = json.loads(self.ws.recv())
            if response.get("status"):
                print(f"✅ SL/TP zmodyfikowane dla zlecenia {order_id}.")
            else:
                print(f"❌ Błąd modyfikacji SL/TP dla zlecenia {order_id}: {response['errorDescr']}")
        except Exception as e:
            print(f"❌ Błąd podczas modyfikacji SL/TP dla zlecenia {order_id}: {e}")
            
            
    def update_position_sl_tp(self, order_id, sl, tp, retry_attempts=3):
        """
        🔄 Poprawiona funkcja aktualizacji SL/TP z uwzględnieniem błędów BE9 i poprawnego ID zlecenia.
        """
        self.ensure_connection()

        # ✅ Pobranie aktualnej listy pozycji przed próbą aktualizacji
        open_positions = self.get_open_positions()
        if open_positions is None or open_positions.empty:
            print("⚠️ Brak otwartych pozycji. Pomijam aktualizację SL/TP.")
            return False

        # ✅ Sprawdzamy, czy order_id nadal istnieje w API XTB
        if order_id not in open_positions["order"].values:
            print(f"⚠️ Order {order_id} nie istnieje w API XTB. Pobieram pełną listę ponownie...")
            time.sleep(1)
            open_positions = self.get_open_positions()
            if order_id not in open_positions["order"].values:
                print(f"❌ Order {order_id} nadal nie istnieje. Pomijam aktualizację SL/TP.")
                return False

        position = self.get_position_details(order_id)
        if not position:
            print(f"⚠️ Pozycja {order_id} nie istnieje. Anulowanie aktualizacji SL/TP.")
            return False

        print(f"🔍 Debug pozycji {order_id}: {position}")

        symbol = position["symbol"]
        trade_type = "buy" if position["cmd"] == 0 else "sell"
        close_price = position["close_price"]
        symbol_info = self.get_symbol_info(symbol)

        # ✅ Pobranie dokładności (precision) dla symbolu
        digits = symbol_info.get("precision", 2)

        # ✅ Korekta SL/TP dla SELL i BUY, aby uniknąć odrzucenia przez API
        if trade_type == "sell":
            sl = max(sl, close_price + 0.1)
            tp = min(tp, close_price - 0.2)
        elif trade_type == "buy":
            sl = min(sl, close_price - 0.1)
            tp = max(tp, close_price + 0.2)

        # ✅ Zaokrąglenie SL/TP zgodnie z precyzją
        sl = round(sl, digits)
        tp = round(tp, digits)

        print(f"📊 Finalne wartości SL: {sl}, TP: {tp} (zaokrąglone do {digits} miejsc po przecinku)")

        trade_info = {
            "cmd": position["cmd"],  # ✅ 0 dla BUY, 1 dla SELL
            "type": 3,  # ✅ Modyfikacja SL/TP
            "order": order_id,
            "sl": sl,
            "tp": tp,
            "price": close_price,  # ✅ Wymagane w niektórych przypadkach
            "symbol": symbol,
            "volume": position["volume"],  # ✅ XTB wymaga podania wolumenu
            "customComment": "AI Trading Bot - Update SL/TP"
        }

        request_payload = {
            "command": "tradeTransaction",
            "arguments": {"tradeTransInfo": trade_info}
        }

        print(f"🔍 Payload aktualizacji SL/TP: {request_payload}")

        for attempt in range(1, retry_attempts + 1):
            try:
                self.ws.send(json.dumps(request_payload))
                response = json.loads(self.ws.recv())

                print(f"🔍 Debug response (update SL/TP): {response}")

                if response.get("status", False):
                    print(f"✅ SL/TP dla {order_id} został zaktualizowany: SL={sl}, TP={tp}")
                    return True

                print(f"❌ Błąd aktualizacji SL/TP: {response}")
                time.sleep(2)
            except Exception as e:
                print(f"❌ Błąd podczas aktualizacji SL/TP dla {order_id}: {e}")

        print(f"⚠️ Nie udało się zaktualizować SL/TP dla {order_id} po {retry_attempts} próbach.")
        return False




































atexit.register(lambda: getattr(globals().get("xtb_client"), "logout", lambda: None)())










import requests
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# 🔹 Klucz API do NewsAPI (UWAGA: Wstaw swój klucz API)
NEWS_API_KEY = "936e6ced8eec4858868ff737a86e3dcf"
analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    """Usuwa linki i znaki specjalne, normalizuje tekst"""
    text = re.sub(r'http\S+', '', text)  # Usuń linki
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Usuń znaki specjalne
    return text.lower()

def fetch_news(news_topic):
    """Pobiera newsy dla danego tematu (np. 'crypto', 'forex', 'stocks')"""
    try:
        url = f"https://newsapi.org/v2/everything?q={news_topic}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=5)
        news_data = response.json()
        
        if response.status_code != 200 or "articles" not in news_data:
            print(f"⚠️ Błąd API: {news_data.get('message', 'Nieznany błąd')}")
            return []
        
        return news_data["articles"][:10]  # Pobieramy 10 najnowszych newsów

    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd pobierania newsów: {e}")
        return []

def analyze_sentiment(news_articles):
    """Analizuje średni sentyment pobranych newsów"""
    sentiments = []
    weights = []
    now = datetime.utcnow()
    
    for i, article in enumerate(news_articles):
        text = clean_text(article["title"] + " " + article.get("description", ""))
        sentiment_score = analyzer.polarity_scores(text)['compound']  # Lepsza analiza niż TextBlob

        # Pobranie daty publikacji newsa
        weight = 2.0 if i < 3 else 1.0  # Najnowsze newsy mają większy wpływ
        sentiments.append(sentiment_score * weight)
        weights.append(weight)

    if sentiments:
        avg_sentiment = sum(sentiments) / sum(weights)  # Normalizujemy względem wagi
        return round(avg_sentiment, 4)  # Zaokrąglamy do 4 miejsc po przecinku

    return 0  # Neutralny sentyment

def load_assets_from_excel(file_path="assets_list.xlsx"):
    """Wczytuje symbole i kategorie z pliku Excel"""
    try:
        df = pd.read_excel(file_path)
        if "symbol" not in df.columns or "Kategoria" not in df.columns:
            print("⚠️ Błąd: Excel musi zawierać kolumny 'symbol' i 'Kategoria'.")
            return {}
        
        return dict(zip(df["symbol"], df["Kategoria"]))  # Tworzymy słownik {Symbol: Kategoria}
    
    except Exception as e:
        print(f"❌ Błąd wczytywania pliku Excel: {e}")
        return {}

def get_sentiment_for_symbols():
    """Pobiera sentyment dla każdego symbolu z Excela"""
    symbols_data = load_assets_from_excel()
    sentiment_dict = {}

    for symbol, category in symbols_data.items():
        sentiment = get_sentiment(category)
        sentiment_dict[symbol] = sentiment
        print(f"📊 AI Sentyment dla {symbol} ({category}): {sentiment}")

    return sentiment_dict  # Zwraca słownik {Symbol: Sentyment}

def get_sentiment(news_topic="stocks"):
    """Główna funkcja pobierania i analizy sentymentu newsów"""
    news_articles = fetch_news(news_topic)
    sentiment_score = analyze_sentiment(news_articles)
    return sentiment_score  # Wartość między -1 a 1


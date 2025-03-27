import requests
from textblob import TextBlob

def get_news_sentiment(symbol):
    """Pobiera i analizuje newsy dla danego symbolu."""
    
    query = f"{symbol} stock news"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey=936e6ced8eec4858868ff737a86e3dcf"
    
    response = requests.get(url)
    data = response.json()
    
    if "articles" not in data:
        return 0  # Brak danych
    
    articles = data["articles"][:5]  # Bierzemy 5 najnowszych newsów
    
    sentiment_scores = []
    for article in articles:
        text = article["title"] + " " + article["description"]
        score = TextBlob(text).sentiment.polarity
        sentiment_scores.append(score)

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
    return avg_sentiment  # Wynik od -1 (negatywny) do 1 (pozytywny)

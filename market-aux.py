"""
scratch data from market-aux, and save to market-aux.csv
"""
# MarketAux: MarketAux is a financial news aggregator that collects and curates financial headlines from various sources, including news websites, blogs, and social media platforms. It provides users with a comprehensive and up-to-date overview of the latest financial news, allowing them to stay informed about market developments and make better investment decisions. Users can access MarketAux through its website or mobile app and can customize their news feed based on their interests and preferences.
# MarketAux: MarketAux is a free financial news aggregator, and there is no cost associated with using their website or mobile app to access financial headlines. However, it is important to note that MarketAux does not provide a direct API for extracting data, so you may need to use web scraping techniques to programmatically gather information from their platform.
import pandas
import requests
from bs4 import BeautifulSoup
import requests
from dateutil.parser import parse
api_token = 'YhSwKU8m7l7bMWbMBoozcjvWYw4orqgjFAtWmUsI'
api_token = "yowU8NdYPC2ckntzG0DETs7bLm63z4JMdG2hETtC"


def get_headlines(api_token):
    url = "https://api.marketaux.com/v1/news/all?symbols=^GSPC&filter_entities=true&language=en&api_token=" + api_token

    cleaned_data = []
    docs = []
    for _ in range(3):
        response = requests.get(url)
        news_headlines = response.json()
        for article in news_headlines['data']:
            date = article['published_at']
            title = article['title']
            entities = article['entities']
            for entity in entities:
                highlights = entity['highlights']
                for highlight in highlights:
                    text = highlight['highlight']
                    sentiment = highlight["sentiment"]
                    docs.append({'text': text, 'senitment': sentiment})
            cleaned_data.append({'date': date, 'headline': title})

    return cleaned_data, docs


def extract_headlines(data):
    headlines = {}  # This will store dates with a list of headlines

    # Ensure 'data' is a list of dictionaries, each containing an article with 'publishedAt' and 'title'
    for item in data:
        # Parse the publication date of the article and format it as a string
        date = parse(item['date']).date()
        date_str = date.strftime('%Y-%m-%d')

        # Ensure there is a list to append headlines for the given date
        if date_str not in headlines:
            headlines[date_str] = []

        # Append the headline to the list for the given date
        headlines[date_str].append(item['headline'])

    return headlines


def run():
    headlines_data = get_headlines(api_token)
    organized_headlines = extract_headlines(headlines_data)

    # To print the organized headlines:
    for date, titles in organized_headlines.items():
        print(f"Date: {date}")
        for title in titles:
            print(f" - {title}")

class ApiTokenException(Exception):
    def __init__(self, msg):
        self.msg = msg


class MarketAux:
    max_page = 6666
    limit = 3
    api_token = api_token
    symbol = "^GSPC"
    filter_entities = 'true'
    language = "en"

    COL_TEXT = "text"
    COL_SENTIMENT = "sentiment"

    save_path = "dataset/market-aux.csv"

    def __init__(self, published_on: str = ""):
        self.published_on = published_on

    def get_url(self, page: int):
        if self.published_on == "":
            return f"https://api.marketaux.com/v1/news/all?" + \
                f"symbols={self.symbol}" + \
                f"&filter_entities={self.filter_entities}" + \
                f"&language={self.language}" + \
                f"&api_token={self.api_token}" + \
                f"&page={page}"
        else:
            return f"https://api.marketaux.com/v1/news/all?" + \
                f"symbols={self.symbol}" + \
                f"&filter_entities={self.filter_entities}" + \
                f"&language={self.language}" + \
                f"&api_token={self.api_token}" + \
                f"&published_on={self.published_on}" + \
                f"&page={page}"

    def get_docs(self):
        docs = []
        max_page = 100
        print(f"start get on {self.published_on}")
        for i in range(max_page):
            print(f"page {i} on {self.published_on}")
            url = self.get_url(i + 1)
            response = requests.get(url)
            news_headlines = response.json()
            if 'data' not in news_headlines:
                return docs, ApiTokenException(news_headlines)
            if len(news_headlines['data']) == 0:
                break
            for article in news_headlines['data']:
                date = article['published_at']
                title = article['title']
                entities = article['entities']
                for entity in entities:
                    highlights = entity['highlights']
                    for highlight in highlights:
                        text = highlight['highlight']
                        sentiment = highlight["sentiment"]
                        docs.append({self.COL_TEXT: text, self.COL_SENTIMENT: sentiment})
        print(f"success get on {self.published_on}")
        return docs, None

    def save(self, path=save_path):
        doc, e = self.get_docs()

        df = pandas.DataFrame(doc)
        # df['text'] = df['text'].applymap(lambda x: x.encode('string_escape').decode("utf8"))
        df.to_csv(path, index=False, escapechar="\\", quotechar='"')
        if e is not None:
            print(f"token is over on date {self.published_on} {e.msg}")
            raise e


def get_start_end_date(year: int, month: int):
    start_date = pandas.to_datetime(f"{year}-{month:02d}-01")
    days_in_month = start_date.days_in_month
    end_date = pandas.to_datetime(f"{year}-{month:02d}-{days_in_month:02d}")
    return start_date, end_date


def get_date_strings(start_date: pandas.Timestamp, end_date: pandas.Timestamp):
    date_strings = pandas.date_range(start=start_date, end=end_date, freq='1D').strftime('%Y-%m-%d').tolist()
    return date_strings


def gen_one_day_date(date_string: str):
    ma = MarketAux(published_on=date_string)
    ma.save(f"dataset/market-aux/{date_string}.csv")


if __name__ == '__main__':
    s, e = get_start_end_date(2024, 3)
    ds = get_date_strings(s, e)
    try:
        for d in ds:
            gen_one_day_date(d)
    except ApiTokenException as e:
        print("api token")

    # MarketAux().save()
    print("Extracting financial headlines...")

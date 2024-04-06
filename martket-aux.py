from dateutil.parser import parse
import requests
from bs4 import BeautifulSoup
api_token = "yowU8NdYPC2ckntzG0DETs7bLm63z4JMdG2hETtC"

def get_headline():
    url = "https://api.marketaux.com/v1/news/all?symbols=^GSPC&filter_entities=true&language=en&api_token=" + api_token

    clieaned_data =[]
    for _ in range(3):

        resp = requests.get(url)
        news = resp.json()
        for article in news["data"]:
            article
        resp

if __name__ == '__main__':
    get_headline()

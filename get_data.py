import pandas as pd
import yfinance as yf
from datetime import timedelta
from my_functions import *
from bs4 import BeautifulSoup
import requests
import csv


def daily_spy_data():
    spy_fifteen_min = pd.DataFrame(yf.download('spy', period='max', interval='5m', rounding=2))
    spy_one_day = pd.DataFrame(yf.download('spy', period='max', interval='1d', rounding=2))

    spy_fifteen_min.to_csv('spy_fifteen_min.csv')
    spy_one_day.to_csv('spy_one_day.csv')


def parse_spy_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'html.parser')

    with open('sp500_stock_symbols.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stock Symbol'])
        for row in soup.select('table.wikitable tbody tr td a.external'):
            if row.text != 'reports':
                if '.' in row.text:
                    updated  = row.text.replace('.', '-')
                    writer.writerow([updated])
                else:
                    writer.writerow([row.text])

parse_spy_symbols()


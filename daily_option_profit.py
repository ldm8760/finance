import csv
import pandas as pd
import yfinance as yf
from my_functions import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
today = datetime.today()


def from_csv_to_list(filename: str,
                     value_type: type,
                     header: bool = True) -> list:
    """
    Read a CSV file and return a list of values with a specified data type.

    Parameters:
        - filename (str): The name of the CSV file to read from (should include .csv)

        - value_type (type): data type to convert the values to (e.g. int, float, str)

        - header (bool): whether the CSV file has a header row (default: True)

    :return: a list of values with the specified data type

    Example:
        print(from_csv('example', float))

        [1.2, 3.4, 5.6, 7.8]

    """
    with open(f'{filename}', 'r') as csv_file:
        reader = csv.reader(csv_file)
        if header:
            next(reader)
        values = [value_type(row[0]) for row in reader]
    return values


def load_option_data(stock: yf.Ticker,
                     last_price: float,
                     save_data: bool = False) -> pd.DataFrame | pd.Series:
    """
    Generates data for ticker equity based on the current price.

    Basic description of generates data test.

    Parameters:

        stock: (Ticker): Takes a yfinance ticker as an object.

        last_price: (float): Takes a dataframe as an object.

        save_data: (bool): Transfers the dataframe to a csv
            file named daily_option_profit.csv, default False

    Returns:

        DataFrame or None
   """
    raw_data = []
    for date in stock.options:
        calls = pd.DataFrame(stock.option_chain(date).calls)
        calls = calls.drop(labels=['lastTradeDate', 'contractSize', 'currency',
                                   'lastPrice', 'change', 'percentChange',
                                   'percentChange', 'inTheMoney',
                                   'contractSymbol'], axis=1)
        calls.columns = ['STRIKE', 'BID', 'ASK', 'VOLUME', 'OI', 'IV']
        labels = [i for i in calls.columns]
        calls[labels[3]] = fix_data(calls[labels[3]], int)
        calls[labels[4]] = fix_data(calls[labels[4]], int)
        calls[labels[5]] = list(map(lambda x: two_dec(x), calls[labels[5]]))
        sells = pd.DataFrame(calls)
        sells = sells.replace(np.nan, 0)
        sells = sells[(sells['BID'] <= last_price) & (sells['BID'] != 0) & (sells['ASK'] != 0) & (sells['IV'] > 0)]
        sells['TV'] = np.where(sells['STRIKE'] < last_price, sells['STRIKE'] + sells['BID'] - last_price, sells['BID'])
        sells = sells[sells['TV'] > 0]
        highest_tv = sells['TV'].nlargest(3)
        sells = sells.loc[highest_tv.index]
        sells['Date'] = date_parse(date)
        sells = sells.set_index('Date')
        sells['TV'] = sells['TV'].round(2)
        sells['TVD'] = two_dec(sells['TV'] / ((to_datetype(date) - today).total_seconds() / 86400))
        raw_data.append(sells)

    df = pd.concat(raw_data, axis=0)
    if save_data:
        df.to_csv('daily_option_profit.csv')
    else:
        return df


def log_ticker_data():
    prices = from_csv_to_list('current_prices.csv', float)
    for idx, ticker in enumerate(from_csv_to_list('sp500_stock_symbols.csv', str)):
        stock = yf.Ticker(ticker)
        current_price = prices[idx]
        data = pd.DataFrame(load_option_data(stock, current_price))
        data.to_csv('logged_ticker_data.csv', mode='a')


def current_prices():
    prices = []
    for ticker in from_csv_to_list('sp500_stock_symbols.csv', str):
        stock = yf.Ticker(ticker)
        current_price = two_dec(stock.history(period='1d', interval='1m')['Close'][-1])
        prices.append(current_price)
    return prices


def write_to_csv(data_list: list,
                 filename: str,
                 column_name: str = 'Data'):
    with open(f'{filename}', 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'{column_name}'])
        for i, data in enumerate(data_list):
            writer.writerow([data])
            print(f'COMPLETE')


def logger_to_print():
    df = pd.read_csv('logged_ticker_data.csv')
    dataframes = []
    for index, row in df.iterrows():
        data = row[['Date', 'STRIKE', 'BID', 'ASK', 'VOLUME', 'OI', 'IV', 'TV']]
        dataframe = pd.DataFrame(data)
        dataframes.append(dataframe)

    result = pd.concat(dataframes)
    print(result)


def main(mode: str):
    if mode == 'update':
        print('Start price update')
        write_to_csv(current_prices(), 'current_prices.csv', f'Current Prices as of {datetime.now(tz=None)}')
    elif mode == 'log':
        print('Start logger')
        tix = from_csv_to_list('sp500_stock_symbols.csv', str)
        prices = from_csv_to_list('current_prices.csv', float)
        for tic, m in zip(tix, prices):
            print(f'{tic} loading')
            print(load_option_data(yf.Ticker(tic), m))
            df = load_option_data(yf.Ticker(tic), m)
            df.to_csv('logged_ticker_data.csv', mode='a')
    elif mode == 'new':
        pass


if __name__ == '__main__':
    main(mode='log')

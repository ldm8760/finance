import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime
import talib as ta
from my_functions import to_datetype

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)


def y_len(iterable: any) -> any:
    """Shorthand form of range(len(obj))"""
    return np.arange(np.size(iterable))

def obtain_data(option: str = 'download',
                sub_folder: str = 'extra_files/'):
    if option == 'download':
        df1 = pd.DataFrame(yf.download('SPY', auto_adjust=True, ignore_tz=True, period='60d', interval='5m', rounding=2))
        df2 = pd.DataFrame(yf.download('SPY', auto_adjust=True, ignore_tz=True, period='60d', interval='1d', rounding=2))
        df3 = pd.DataFrame(yf.download('^VIX', auto_adjust=True, ignore_tz=True, period='60d', interval='5m', rounding=2))
        print(df1.head())
        print(df2.head())
        print(df3.head())

        dfs = [df1, df2, df3]
        for i, df in enumerate(dfs):
            df.to_csv(f'{sub_folder}data_{i+1}.csv', index=True)

    
    if option == 'read':
        spy1 = pd.read_csv(f'{sub_folder}data_{1}.csv')
        spy2 = pd.read_csv(f'{sub_folder}data_{2}.csv')
        vix = pd.read_csv(f'{sub_folder}data_{3}.csv')
        return [spy1, spy2, vix]



spy1, spy2, vix = obtain_data('read')


def get_percentage_diff(dataframe, dataframe2):
    # dataframe, dataframe2 = spy1, spy2
    percentage_diff = []
    print(dataframe2.index[0])
    dataframe['date'] = dataframe['Datetime'].apply(to_datetype)
    print(dataframe)
    # dataframe['date'] = list(map(lambda x: datetime.strptime(str(x), '%Y-%m-%d'), dataframe2['Date']))
    for i in range(len(dataframe)):
        date = dataframe.index[i].date()
        open_price = dataframe.loc[dataframe['date'] == date, 'Open'].values[0]
        close_price = dataframe['Close'][i]
        diff = abs(close_price - open_price) / open_price * 100
        percentage_diff.append(diff)
    return percentage_diff

print(vix)

def get_time(dataframe):
    dummy_list = []
    for i in dataframe.index:
        # t = datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').time()
        # total = (t.hour * 3600 + t.minute * 60) / 3600
        # dummy_list.append(total)
        dummy_list.append(0)
    return dummy_list

def process(dataframe):
    dataframe['Percentage Diff'] = get_percentage_diff(dataframe, spy2)
    dataframe['time'] = get_time(dataframe)
    dataframe['date'] = list(map(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').date(), dataframe.index))

process(spy1)

def ml(spy1, spy2, vix1):
    closes = []
    for i in range(len(spy1['date'].value_counts())):
        for j in range(spy1['date'].value_counts()[i]):
            closes.append(spy2['Close'][i])
    spy1['EOD Close'] = closes
    spy1['VIX'] = vix1['Close']
    spy1['RSI_9'] = ta.RSI(spy1['Close'], timeperiod=9)
    spy1['SMA_20'] = ta.SMA(spy1['Close'], timeperiod=20)
    spy1['SMA_50'] = ta.SMA(spy1['Close'], timeperiod=50)
    spy1['RSI_14'] = ta.RSI(spy1['Close'], timeperiod=14)

    spy1['MACD'], spy1['MACD_Signal'], spy1['MACD_Hist'] = ta.MACD(spy1['Close'])
    spy1['BB_Upper'], spy1['BB_Middle'], spy1['BB_Lower'] = ta.BBANDS(spy1['Close'])
    spy1['Stoch_K'], spy1['Stoch_D'] = ta.STOCH(spy1['High'], spy1['Low'], spy1['Close'])

    spy1['CCI'] = ta.CCI(spy1['High'], spy1['Low'], spy1['Close'])

    spy1.dropna(axis=0, inplace=True)

    spy1 = spy1[(spy1['time'] >= 9.5) & (spy1['time'] <= 11.5)]

    X = spy1[['Close', 'Volume', 'Percentage Diff', 'SMA_20', 'SMA_50', 'RSI_9', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'Stoch_K', 'Stoch_D', 'CCI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'VIX']]
    
    y = spy1['EOD Close']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.7, random_state=42)

    reg = LinearRegression()
    reg.fit(train_X, train_y)

    predictions = reg.predict(test_X)

    mse = np.mean((predictions - test_y) ** 2)
    print(f'Mean Squared Error: {mse}')
    y_pred = reg.predict(test_X)

    plt.scatter(test_y, y_pred)
    x = range(380, 416)
    y = [i for i in x]
    plt.plot(x, y)
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title("Model Predictions vs. Actual Values")
    plt.show()


ml(spy1, spy2, vix)

if __name__ == '__main__':
    print('Complete')

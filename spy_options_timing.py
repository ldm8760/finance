from my_functions import *
import pandas as pd
import yfinance as yf
from datetime import timedelta
import matplotlib.pyplot as plt


stock = yf.Ticker("spy")
pd.set_option('display.max_columns', None)
ds1 = pd.read_csv('spy_fifteen_min.txt', index_col='Datetime')
ds2 = pd.read_csv('spy_one_day.txt', index_col='Date')
# spy_fifteen_min = yf.download('spy', start=f'{(datetime.today() - timedelta(58)).date()}',
#                               end=f'{datetime.today().date()}', interval='5m')
# spy_one_day = yf.download('spy', start=f'{(datetime.today() - timedelta(58)).date()}',
#                           end=f'{datetime.today().date()}', interval='1d')
h15m = pd.DataFrame(ds1)
h1d = pd.DataFrame(ds2)

h15m = h15m.drop(['High', 'Low', 'Adj Close', 'Volume'], axis=1)
h1d = h1d.drop(['Volume', 'Close'], axis=1)

for i in [h15m, h1d]:
    labels = [i for i in i.columns]
    for j in labels:
        i[j] = list(map(lambda x: two_dec(float(x)), i[j]))

hours_to_close = []
for i in h15m.index:
    i = to_datetime_type(i)
    hours_to_close.append((datetime(i.year, i.month, i.day, 16, 0) - i).total_seconds() / 3600)

h15m['HTC'] = hours_to_close
label15m = [i for i in h15m.columns]
label1d = [i for i in h1d.columns]

cd2 = []
xiter = 0
for i in h1d.index:
    num_at_date_i = [to_datetype(i) for i in h15m.index].count(to_datetype(i))
    for j in range(num_at_date_i):
        if hours_to_close[xiter] > 0:
            cd2.append([i, hours_to_close[xiter], round(abs(h1d[label1d[3]][i] - h15m[label15m[0]][xiter]), 2),
                        round(abs(h1d[label1d[3]][i] - h15m[label15m[0]][xiter]) / hours_to_close[xiter], 4)])
            xiter += 1

pd.set_option('display.max_rows', None)
df = pd.DataFrame(cd2)
print(df)
# 11:12AM best time to sell to open - unconfirmed
# nm = []
# for i in h1d.index:
#     nm.append([to_datetype(i) for i in h15m.index].count(to_datetype(i)))
#
# rnditer = 0
# for i in nm:
#     xplot, yplot = [], []
#     for j in range(i):
#         xplot.append(df[1][rnditer])
#         yplot.append(df[2][rnditer])
#         rnditer += 1
#     plt.plot(xplot, yplot)

# for i in y_len(df):
#     x = df[1][i]
#     y = df[2][i]
#     plt.scatter(x, y)
# plt.show()

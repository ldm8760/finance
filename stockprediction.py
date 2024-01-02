import pandas as pd
import yfinance as yf
import time
from my_functions import *

# Value investing: Low PE, Oversold Daily RSI, High Market Gap?, High PEG ratio

# Order of data:
# EPS - Done
# RSI - Done
# current pe ratio / historical pe ratio! - Done
# market cap!
# price to book - Done
# ev / ebitda
# Dividend payout ratio / Retention ratio

# tick = input("Type the ticker: ")
# sector = stock.info.get("quoteType")
# a = ["{:,}".format(i) for i in b]
# bookValue / sharesOutstanding -> share price / price to book (bookValue per share)
# "%Y-%m-%d %H:%M:%S%z" new datetime format, requires beautifulSoup4
# ACCESSING DATA FROM PANDAS || index = df.index[i], values = df["ColumnName"][df.index[i]]
# return [self.df2["Close"][i] / (self.quarterly_eps()[i]) for i in range(len(self.quarterly_eps()))]

start_time = time.perf_counter()
stock = yf.Ticker("aapl")
i = stock.get_info()
for key, value in i.items():
    print(key, value)
hist1d = stock.history(period='max', interval='1d')
print(stock.get_shares_full(datetime(2020, 1, 1), datetime.today()))

class Mined_Ticker:
    def __init__(self) -> None:
        self.shares = 0

    def shares(self, start = None, end = None):
        pass


def shares_outstanding():
    """Returns shares outstanding of the given security"""

    dfShares = pd.DataFrame(data=stock.get_shares())
    dfShares.loc[2022] = stock.info['sharesOutstanding']
    return [[dfShares.index[i], dfShares['BasicShares'][dfShares.index[i]]] for i in y_len(dfShares)]


class Earnings:
    """Returns historical data based on earnings reports"""

    def __init__(self):
        self.balance_sheet = stock.balance_sheet
        self.window_length = 4
        self.dates = [dt_parse(str(i)) for i in stock.earnings_history["Earnings Date"]]

    def quarterly_eps(self):
        """Returns the historical reported EPS (Earnings per Share)"""

        reports = stock.earnings_dates["Reported EPS"]
        return [i for i in reports[::-1]]

    def historical_ttm_eps(self):
        """Returns the historical TTM (Trailing Twelve Months) EPS of the security"""

        return [two_dec(sum(self.quarterly_eps()[i:i + self.window_length]))
                for i in range(len(self.quarterly_eps()) - self.window_length + 1)]

    def pe_ratio(self):
        """Returns the historical daily P/E (Price to Earnings) ratio"""

        history = [dt_parse(str(i)) for i in hist1d.index]
        historical_eps = stock.earnings_history["Reported EPS"]
        j = len(self.dates) - 1
        dailyEPS = []
        for i, v in enumerate(hist1d["Close"]):
            if history[i] < self.dates[j - 1]:
                dailyEPS.append(two_dec(
                    v / sum(historical_eps[j:j + self.window_length])
                ))
            else:
                j -= 1
        return dailyEPS

    def book_value(self):
        """Returns the book value of the equity - Under construction needs share price / price to book"""

        totalLiability = [i for i in self.balance_sheet.values[0]]
        totalAssets = [i for i in self.balance_sheet.values[3]]
        bookValue = [av - bv for av, bv in zip(totalAssets, totalLiability)]

        bvps = list(map(lambda x: two_dec(x),
                        [(v / shares_outstanding()[i][1])
                         for i, v in enumerate(bookValue)]))[::-1]

        return {av[0]: f"P/B: {bv}" for av, bv in
                zip(shares_outstanding(), bvps)}


class MarketCap:

    def __init__(self):
        self.shares = shares_outstanding()

    def daily_market_cap(self):
        dailyHistory = [dt_parse(str(hist1d.index[i])) for i in y_len(hist1d)]
        dailyMarketCap = []
        for i in range(len(dailyHistory)):  # Needs j index to fix IndexError
            if dailyHistory[i] < datetime.strftime(datetime.strptime(str(self.shares[i][0]), "%Y"), "%Y-%m-%d"):
                dailyMarketCap.append(hist1d["Close"][i] * self.shares[i][1])

        return dailyMarketCap


class Indicators:
    """Returns technical indicators"""

    def __init__(self):
        self.close = [i for i in hist1d['Close']]
        self.change = [self.close[i] - self.close[i - 1] for i in range(len(self.close))]

    def rsi(self):
        """Returns the Relative Strength Index of the equity"""

        result = []
        windowLen = 14
        prev_avg_gain = 0
        prev_avg_loss = 0
        gains = [self.change[i] / self.close[i - 1] * 100 if self.change[i] > 0
                 else 0 if self.change[i] < 0 else 0
                 for i in range(1, len(self.close))]

        losses = [abs(self.change[i] / self.close[i - 1]) * 100 if self.change[i] < 0
                  else 0 if self.change[i] > 0 else 0
                  for i in range(1, len(self.close))]

        for i in range(len(self.close) - 1):
            if i < windowLen:
                result.append(0)
            if i == windowLen:
                avg_gain = sum(gains[:i]) / len(gains[:i])
                avg_loss = sum(losses[:i]) / len(losses[:i])
            else:
                avg_gain = (prev_avg_gain * (windowLen - 1) + gains[i]) / windowLen
                avg_loss = (prev_avg_loss * (windowLen - 1) + losses[i]) / windowLen

            prev_avg_gain = avg_gain
            prev_avg_loss = avg_loss

            rs = round(avg_gain / avg_loss, 2)
            result.append(round(100 - (100 / (1 + rs)), 2))

        return result

    def sma(self, window_length: int) -> list[float]:
        """Returns the Simple Moving Average of the equity"""

        result = []
        for i in range(len(self.close) - window_length):
            if i <= window_length:
                result.append(two_dec(sum(self.close[i:i + window_length]) / (i + 1)))
            else:
                result.append(two_dec(sum(self.close[i:i + window_length]) / window_length))

        return result

    def stochastic_oscillator(self, window_length):
        """Returns the historical slow stochastic oscillator - Under Construction"""

        window = []
        for i in range(window_length, len(self.close)):
            current = self.close[i - window_length:i]
            low = min(current)
            high = max(current)
            stoch = ((self.close[i - 1] - low) / (high - low)) * 100
            window.append(round(stoch, 2))
        return window


class Dividends:
    """Returns historical data based on dividends"""

    def __init__(self):
        self.divDF = pd.DataFrame(data=stock.dividends)

    def dividends(self):
        """Returns the dividend paid out to shareholders at the given date"""

        return [[dt_parse(str(self.divDF.index[i])), self.divDF['Dividends'][i]] for i in range(len(self.divDF))]

    def total_dividends(self):
        """Returns the total amount of dividends and the Dividend + Price value - Under Construction"""
        # return [sum(self.divDF["Dividends"][i]) for i in range(len(self.divDF))]
        return f"Total historical dividends paid: ${round(sum(self.divDF['Dividends']), 2)}"


if __name__ == '__main__':
    pass
    # print(f"Completed in {time.perf_counter() - start_time} second(s)")

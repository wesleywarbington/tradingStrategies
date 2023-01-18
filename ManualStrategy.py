import datetime as dt
import pandas as pd
from util import get_data
import indicators


class ManualStrategyLearner(object):

    def __init__(self, impact=0.005, commission=9.95):
        self.impact = impact
        self.commission = commission


    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):

        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates, colname = "Adj Close")  # automatically adds SPY
        prices_all.fillna(method="ffill", inplace=True)
        prices_all.fillna(method="bfill", inplace=True)
        prices = prices_all[[symbol]]

        extendedStartDate = pd.date_range(sd + dt.timedelta(days=-30), ed)
        pricesExtended = get_data([symbol], extendedStartDate, colname = "Adj Close")  # automatically adds SPY
        pricesExtended.fillna(method="ffill", inplace=True)
        pricesExtended.fillna(method="bfill", inplace=True)
        prices2 = pricesExtended[[symbol]]

        # ---------
        bbp = indicators.get_bbp(prices2)
        rsi = indicators.get_rsi(prices2)
        ema_cross = indicators.get_emaCross(prices2)

        indicator_df = prices.copy()
        indicator_df['bbp'] = bbp
        indicator_df['rsi'] = rsi
        indicator_df['ema_cross'] = ema_cross
        del indicator_df[symbol]

        signals = prices.copy()
        signals['decision'] = 0
        del signals[symbol]
        signals.values[:, :] = 0  # set them all to nothing

        for date, row in indicator_df.iterrows():
            a, b, c = row[0], row[1], row[2]
            signalCount = 0
            if a < 0.13:
                signalCount += 1
            elif a > 0.75:
                signalCount -= 1

            if b < 30:
                signalCount += 1
            elif b > 50:
                signalCount -= 1

            if c <= 0.967:
                signalCount += 1
            elif c >= 0.985:
                signalCount -= 1

            if signalCount >= 2:
                signals.at[date, 'decision'] = 1
            elif signalCount <= -2:
                signals.at[date, 'decision'] = -1
        # ---------

        trades = prices.copy()
        trades.values[:, :] = 0

        total = 0
        for date, signal in signals.iterrows():
            if list(signal)[0] == 1 and total == -1000:
                trades.at[date, symbol] = 2000
                total += 2000
            elif list(signal)[0] == 1 and total == 0:
                trades.at[date, symbol] = 1000
                total += 1000
            elif list(signal)[0] == -1 and total == 1000:
                trades.at[date, symbol] = -2000
                total -= 2000
            elif list(signal)[0] == -1 and total == 0:
                trades.at[date, symbol] = -1000
                total -= 1000

        return trades
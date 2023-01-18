import datetime as dt
import pandas as pd
from util import get_data
import indicators
import numpy as np
import RTLearner as rtl
import BagLearner as bl


class StrategyLearner(object):

    # constructor
    def __init__(self, impact=0.005, commission=9.95):
        self.impact = impact
        self.commission = commission
        self.bagLearner = bl.BagLearner(learner=rtl.RTLearner, kwargs={"leaf_size": 10}, bags=30, boost=False)


    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):
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
        indicator_df['return'] = (indicator_df[symbol].shift(-5)/indicator_df[symbol]) - 1
        indicator_df = indicator_df.dropna()
        indicator_df['Y'] = 0
        indicator_df.loc[indicator_df['return']<=(-0.03-self.impact),'Y'] = -1
        indicator_df.loc[indicator_df['return']>=(0.03+self.impact),'Y'] = 1
        del indicator_df['return']
        del indicator_df[symbol]

        xData = indicator_df.to_numpy()
        xData = np.delete(xData, 3, 1)

        yData = indicator_df.to_numpy()
        yData = np.delete(yData, 0, 1)
        yData = np.delete(yData, 0, 1)
        yData = np.delete(yData, 0, 1)

        self.bagLearner.add_evidence(xData, yData)


    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=10000):
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

        xData = indicator_df.to_numpy()
        signals = self.bagLearner.query(xData)

        trades = prices.copy()
        trades.values[:, :] = 0

        total = 0
        for index, signal in enumerate(signals):
            if signal == 1 and total == -1000:
                trades.iloc[index][symbol] = 2000
                total += 2000
            elif signal == 1 and total == 0:
                trades.iloc[index][symbol] = 1000
                total += 1000
            elif signal == -1 and total == 1000:
                trades.iloc[index][symbol] = -2000
                total -= 2000
            elif signal == -1 and total == 0:
                trades.iloc[index][symbol] = -1000
                total -= 1000

        return trades

if __name__ == "__main__":
    learner = StrategyLearner()
    learner.add_evidence()
    trades = learner.testPolicy()
    print(trades)

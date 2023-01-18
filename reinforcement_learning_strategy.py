import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import QLearner as ql
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import indicators


class StrategyLearner(object):

    # constructor
    def __init__(self, impact=0.005, commission=9.95):
        self.impact = impact
        self.commission = commission
        self.lookback = 20

    # this method should create a QLearner, and train it for trading
    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=100000):
        # get states
        states, daily_rets, prices = self.getStates(syms=[symbol], sd=sd, ed=ed)

        # initiate a QLearner
        self.learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9,
                                   rar=0.98, radr=0.999, dyna=0, verbose=False)
        # train the QLearner
        epochs = 200
        prev_port_val = 0  # final portfolio value in previous epoch
        for epoch in range(1, epochs + 1):
            port_val = sv
            # set the initial state
            action = self.learner.querysetstate(int(states.ix[0][0]))
            holding = self.actionToHolding(action)
            if holding != 0:
                reward = -1 * (self.commission + pd.np.abs(holding) * prices.ix[0].values[
                    0] * self.impact)
            else:
                reward = 0
            prev_holding = holding  # holding on the day before

            for day in range(1, states.shape[0]):
                reward += prev_holding * daily_rets.ix[day].values[0]
                port_val += reward
                action = self.learner.query(int(states.ix[day][0]), reward)
                holding = self.actionToHolding(action)
                # for next day: deduct transaction costs
                if holding != prev_holding:
                    reward = -1 * (self.commission + pd.np.abs(holding - prev_holding) * prices.ix[day].values[
                        0] * self.impact)
                else:
                    reward = 0
                prev_holding = holding

            if epoch >= 20 and prev_port_val == port_val: break
            prev_port_val = port_val

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 1, 1), sv=100000):
        # get states
        states, daily_rets, prices = self.getStates(syms=[symbol], sd=sd, ed=ed)

        prev_holding = 0
        self.flag = False
        self.total = 0
        holdings = states.copy()
        holdings[[symbol]] = 0
        for day in range(1, states.shape[0]):
            holdings.ix[day] = prev_holding
            action = self.learner.querysetstate(int(states.ix[day]))
            holding = self.actionToHolding_2(action)
            prev_holding = holding

        trades = holdings.copy()
        trades.values[:-1] = holdings.values[1:] - holdings.values[:-1]
        trades.values[-1] = holding - prev_holding  # values from the last iteration

        return trades

    def actionToHolding_2(self, action):
        if action == 0:
            holding = self.total
        elif self.flag == False:
            if action == 1:
                holding = 1000  # LONG
                self.total = 1000
            elif action == 2:
                holding = -1000  # SHORT
                self.total = -1000
            self.flag = True
        else:
            if action == 1:
                holding = 1000  # LONG
                self.total = 1000
            elif action == 2:
                holding = -1000  # SHORT
                self.total = -1000
        return holding

    def actionToHolding(self, action):
        if action == 0:
            holding = 0  # Zero shares
        elif action == 1:
            holding = 1000  # LONG
        elif action == 2:
            holding = -1000  # SHORT
        return holding

    def getStates(self, syms, sd, ed):
        dates = pd.date_range(sd - dt.timedelta(days=self.lookback * 2), ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        # Compute daily returns
        daily_rets = prices.copy()
        daily_rets.values[1:, :] = prices.values[1:, :] - prices.values[:-1, :]
        daily_rets.values[0, :] = np.nan

        # indicators
        bbp = indicators.get_bbp(price=prices, lookback=self.lookback)
        rsi = indicators.get_rsi(price=prices, lookback=self.lookback)
        ema_cross = indicators.get_emaCross(price=prices)

        # discretize three indicators and calculate states
        bbp_disc = pd.qcut(bbp[syms[0]], 10, labels=False)
        rsi_disc = pd.qcut(rsi[syms[0]], 10, labels=False)
        ema_cross_disc = pd.qcut(ema_cross[syms[0]], 10, labels=False)
        states = pd.DataFrame(100 * bbp_disc + 10 * ema_cross_disc + 1 * rsi_disc, index=bbp.index, columns=syms)

        # remove data before the start date
        prices = prices.ix[sd:]
        daily_rets = daily_rets.ix[sd:]
        states = states.ix[sd:]

        return states, daily_rets, prices
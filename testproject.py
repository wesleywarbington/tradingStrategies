import matplotlib.pyplot as plt
import datetime as dt
import ManualStrategy as ms
import decision_tree_strategy as dts
import reinforcement_learning_strategy as rls
from marketsimcode import compute_portvals
import numpy as np

# Computes cumulative return, standard deviation of daily returns, and mean of daily returns for a given portfolio (Used for analysis_1)
def getStats(portfolio):
    # Statistics
    daily_rets = portfolio.copy()
    daily_rets[1:] = (portfolio[1:] / portfolio[:-1].values) - 1
    daily_rets = daily_rets[1:]

    cumRet = (portfolio[-1]/portfolio[0])-1
    std_dailyRet = daily_rets.std()
    mean_dailyRet = daily_rets.mean()

    return cumRet, std_dailyRet, mean_dailyRet


# Turn a dataframe of trades into a dataframe representing the value of a portfolio (Used for analysis_1)
def trades_to_portfolio(trades):
    portVal = compute_portvals(orders_df=trades)
    portValNorm = portVal / portVal.iloc[0]
    return portValNorm


# Strategies evaluated for in and out of sample (returns portfolio dataframes) (Used for analysis_1)
def evalStrategies(symbol, sd, ed, sdTest, edTest, sv):
    # Decision Tree Strategy In-Sample
    SL = dts.StrategyLearner()
    SL.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades = SL.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portValNorm_1 = trades_to_portfolio(trades)

    # Reinforcement Learning Strategy In-Sample
    SL = rls.StrategyLearner()
    SL.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades = SL.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portValNorm_2 = trades_to_portfolio(trades)

    #Manual Strategy In-Sample
    SL = ms.ManualStrategyLearner()
    trades = SL.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portValNorm_3 = trades_to_portfolio(trades)

    # Baseline Strategy (Buy and hold 1000 shares) In-Sample
    trades.values[:, :] = 0  # set them all to nothing
    trades.values[0, :] = 1000  # add a BUY at the start
    trades.values[-1, :] = -1000  # exit on the last day
    portValNorm_4 = trades_to_portfolio(trades)

    # Decision Tree Strategy Out-of-Sample
    SL = dts.StrategyLearner()
    SL.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades = SL.testPolicy(symbol=symbol, sd=sdTest, ed=edTest, sv=sv)
    portValNorm_5 = trades_to_portfolio(trades)

    # Reinforcement Learning Strategy Out-of-Sample
    SL = rls.StrategyLearner()
    SL.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades = SL.testPolicy(symbol=symbol, sd=sdTest, ed=edTest, sv=sv)
    portValNorm_6 = trades_to_portfolio(trades)

    # Manual Strategy Out-of-Sample
    SL = ms.ManualStrategyLearner()
    trades = SL.testPolicy(symbol=symbol, sd=sdTest, ed=edTest, sv=sv)
    portValNorm_7 = trades_to_portfolio(trades)

    # Baseline Strategy (Buy and hold 1000 shares) Out-of-Sample
    trades.values[:, :] = 0  # set them all to nothing
    trades.values[0, :] = 1000  # add a BUY at the start
    trades.values[-1, :] = -1000  # exit on the last day
    portValNorm_8 = trades_to_portfolio(trades)

    return portValNorm_1, portValNorm_2, portValNorm_3, portValNorm_4, portValNorm_5, portValNorm_6, portValNorm_7, portValNorm_8

# Plotting and Statistics for strategies
def analysis_1(symbol, sd, ed, sdTest, edTest, sv):
    (DT_inSample, RL_inSample,
     ML_inSample, baseline_inSample,
     DT_outOfSample, RL_outOfSample,
     ML_outOfSample, baseline_outOfSample) = evalStrategies(symbol=symbol, sd=sd, ed=ed, sdTest=sdTest, edTest=edTest, sv=sv)


    # Plot portfolio values for all four strategies (In-Sample)
    plt.plot(DT_inSample.index, DT_inSample, c='red', label='Decision Tree')
    plt.plot(RL_inSample.index, RL_inSample, c='green', label='Reinforcement Learning')
    plt.plot(ML_inSample.index, ML_inSample, c='blue', label='Manual')
    plt.plot(baseline_inSample.index, baseline_inSample, c='purple', label='Buy & Hold 1000 Shares')

    plt.title('Daily Portfolio Value (In-Sample)')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/in_sample_normalized_portfolios.png")
    plt.clf()

    # Plot portfolio values for all four strategies (Out-of-Sample)
    plt.plot(DT_outOfSample.index, DT_outOfSample, c='red', label='Decision Tree')
    plt.plot(RL_outOfSample.index, RL_outOfSample, c='green', label='Reinforcement Learning')
    plt.plot(ML_outOfSample.index, ML_outOfSample, c='blue', label='Manual')
    plt.plot(baseline_outOfSample.index, baseline_outOfSample, c='purple', label='Buy & Hold 1000 Shares')

    plt.title('Daily Portfolio Value (Out-of-Sample)')
    plt.legend(loc='lower left')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/out_of_sample_normalized_portfolios.png")
    plt.clf()


    # Cumulative return, standard deviation of daily returns and mean of daily returns.
    benchmark_in_cumRet, benchmark_in_std, benchmark_in_mean = getStats(baseline_inSample)
    benchmark_out_cumRet, benchmark_out_std, benchmark_out_mean = getStats(baseline_outOfSample)
    ML_in_cumRet, ML_in_std, ML_in_mean = getStats(ML_inSample)
    ML_out_cumRet, ML_out_std, ML_out_mean = getStats(ML_outOfSample)
    DT_in_cumRet, DT_in_std, DT_in_mean = getStats(DT_inSample)
    DT_out_cumRet, DT_out_std, DT_out_mean = getStats(DT_outOfSample)
    RL_in_cumRet, RL_in_std, RL_in_mean = getStats(RL_inSample)
    RL_out_cumRet, RL_out_std, RL_out_mean = getStats(RL_outOfSample)

    # Write to statistics to text file
    lines = [
        f"Benchmark in-sample cumulative return: {benchmark_in_cumRet:.6f}",
        f"Benchmark in-sample standard deviation of daily returns: {benchmark_in_std:.6f}",
        f"Benchmark in-sample mean of daily returns: {benchmark_in_mean:.6f}",
        f"Benchmark out-of-sample cumulative return: {benchmark_out_cumRet:.6f}",
        f"Benchmark out-of-sample standard deviation of daily returns: {benchmark_out_std:.6f}",
        f"Benchmark out-of-sample mean of daily returns: {benchmark_out_mean:.6f}",
        f"----------------------------------------------------------",
        f"Manual Strategy portfolio Value in-sample cumulative return: {ML_in_cumRet:.6f}",
        f"Manual Strategy portfolio Value in-sample standard deviation of daily returns: {ML_in_std:.6f}",
        f"Manual Strategy portfolio Value in-sample mean of daily returns: {ML_in_mean:.6f}",
        f"Manual Strategy portfolio Value out-of-sample cumulative return: {ML_out_cumRet:.6f}",
        f"Manual Strategy portfolio Value out-of-sample standard deviation of daily returns: {ML_out_std:.6f}",
        f"Manual Strategy portfolio Value out-of-sample mean of daily returns: {ML_out_mean:.6f}",
        f"----------------------------------------------------------",
        f"Decision Tree Strategy portfolio Value in-sample cumulative return: {DT_in_cumRet:.6f}",
        f"Decision Tree Strategy portfolio Value in-sample standard deviation of daily returns: {DT_in_std:.6f}",
        f"Decision Tree Strategy portfolio Value in-sample mean of daily returns: {DT_in_mean:.6f}",
        f"Decision Tree Strategy portfolio Value out-of-sample cumulative return: {DT_out_cumRet:.6f}",
        f"Decision Tree Strategy portfolio Value out-of-sample standard deviation of daily returns: {DT_out_std:.6f}",
        f"Decision Tree Strategy portfolio Value out-of-sample mean of daily returns: {DT_out_mean:.6f}",
        f"----------------------------------------------------------",
        f"Reinforcement Learning Strategy portfolio Value in-sample cumulative return: {RL_in_cumRet:.6f}",
        f"Reinforcement Learning Strategy portfolio Value in-sample standard deviation of daily returns: {RL_in_std:.6f}",
        f"Reinforcement Learning Strategy portfolio Value in-sample mean of daily returns: {RL_in_mean:.6f}",
        f"Reinforcement Learning Strategy portfolio Value out-of-sample cumulative return: {RL_out_cumRet:.6f}",
        f"Reinforcement Learning Strategy portfolio Value out-of-sample standard deviation of daily returns: {RL_out_std:.6f}",
        f"Reinforcement Learning Strategy portfolio Value out-of-sample mean of daily returns: {RL_out_mean:.6f}"
             ]

    with open('outputs/statistics.txt', 'w') as f:
        f.write('\n'.join(lines))


# Calculate portfolio value based on variable impact (Used for analysis_2)
def getImpactPortfolio(impact, symbol, sd, ed, sv):
    SL1 = dts.StrategyLearner(impact=impact, commission=0.0)
    SL1.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades = SL1.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portVal = compute_portvals(orders_df=trades, start_val=sv, commission=0.00, impact=impact)
    normPortVal = portVal / portVal.iloc[0]

    SL2 = rls.StrategyLearner(impact=impact, commission=0.0)
    SL2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=10000)
    trades2 = SL2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portVal2 = compute_portvals(orders_df=trades2, start_val=sv, commission=0.00, impact=impact)
    normPortVal2 = portVal2 / portVal2.iloc[0]
    return normPortVal, normPortVal2


# Calculate number of trades based on variable impact (Used for analysis_2)
def getTradeCount(impact, symbol, sd, ed, sv):
    SL1 = dts.StrategyLearner(impact=impact, commission=0.0)
    SL1.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades = SL1.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    count = trades[symbol][trades[symbol]!=0].count()

    SL2 = rls.StrategyLearner(impact=impact, commission=0.0)
    SL2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades2 = SL2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    count2 = trades2[symbol][trades2[symbol]!=0].count()
    return count, count2


# Analysis on strategies when impact is varied.
def analysis_2(symbol, sd, ed, sv):
    dts_impact_005, rls_impact_005 = getImpactPortfolio(impact=0.005, symbol=symbol, sd=sd, ed=ed, sv=sv)
    dts_impact_01, rls_impact_01 = getImpactPortfolio(impact=0.01, symbol=symbol, sd=sd, ed=ed, sv=sv)
    dts_impact_025, rls_impact_025 = getImpactPortfolio(impact=0.025, symbol=symbol, sd=sd, ed=ed, sv=sv)

    # Portfolio Value decision tree
    plt.plot(dts_impact_005.index, dts_impact_005, c='red', label='Impact 0.005')
    plt.plot(dts_impact_01.index, dts_impact_01, c='blue', label='Impact 0.01')
    plt.plot(dts_impact_025.index, dts_impact_025, c='green', label='Impact 0.025')
    plt.title('Daily Portfolio Value (In-Sample) Decision Tree')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/impact_dts.png")
    plt.clf()

    # Portfolio Value reinforcement learner
    plt.plot(rls_impact_005.index, rls_impact_005, c='red', label='Impact 0.005')
    plt.plot(rls_impact_01.index, rls_impact_01, c='blue', label='Impact 0.01')
    plt.plot(rls_impact_025.index, rls_impact_025, c='green', label='Impact 0.025')
    plt.title('Daily Portfolio Value (In-Sample) Reinforcement Learner')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/impact_rls.png")
    plt.clf()

    # Number of trades
    impacts = np.linspace(0, 0.05, num=20)
    counts_1, counts_2 = [], []
    for impact in impacts:
        count1, count2 = getTradeCount(impact=impact, symbol=symbol, sd=sd, ed=ed, sv=sv)
        counts_1.append(count1)
        counts_2.append(count2)


    # Impact Vs Number of Trades
    plt.plot(impacts, counts_1, c='red', label="decision tree")
    plt.plot(impacts, counts_2, c='blue', label="reinforcement learner")
    plt.title('# of Trades dependent on impact (In-Sample)')
    plt.xlabel('Impact')
    plt.ylabel('Number of Trades')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("outputs/impact_vs_numTrades.png")
    plt.clf()


if __name__ == '__main__':
    analysis_1(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sdTest=dt.datetime(2010, 1, 1), edTest=dt.datetime(2011, 12, 31), sv=100000)
    analysis_2(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

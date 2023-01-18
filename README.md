# tradingStrategies
[testproject.py](testproject.py) is the main driver file. Here you can adjust the parameters for the analysis. The variables are the stock symbol, start date, end date (for training and test) and the starting value (in dollars) for the portfolio. For the analysis here, the values used were:

symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sdTest=dt.datetime(2010, 1, 1), edTest=dt.datetime(2011, 12, 31), sv=100000

The five charts as well as a .txt file with statistics are written to the [outputs](/outputs) directory.

The [data](/data) folder contains csv files with historic data for respective stock symbols.

# Plots
![plot](./outputs/in_sample_normalized_portfolios.png)

![plot](./outputs/out_of_sample_normalized_portfolios.png)

![plot](./outputs/impact_dts.png)

![plot](./outputs/impact_rls.png)

![plot](./outputs/impact_vs_numTrades.png)

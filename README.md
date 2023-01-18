# tradingStrategies
[testproject.py](testproject.py) is the main driver file. Here you can adjust the parameters for the analysis. The variables are the stock symbol, start date, end date (for training and test) and the starting value (in dollars) for the portfolio.

The five charts as well as a .txt file with statistics are written to the [outputs directory](/outputs).

The [data](/data) folder contains csv files with historic data for respective stock symbols.

# Plots
![plot](./outputs/in_sample_normalized_portfolios.png)

![plot](./outputs/out_of_sample_normalized_portfolios.png)

![plot](./outputs/impact_dts.png)

![plot](./outputs/impact_rls.png)

![plot](./outputs/impact_vs_numTrades.png)

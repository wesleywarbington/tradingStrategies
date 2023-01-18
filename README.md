# tradingStrategies
Development was done in a conda environment and the [environment](environment.yml) file defines all the dependencies.

[testproject.py](testproject.py) is the main driver file. Here you can adjust the parameters for the analysis. The variables are the stock symbol, start date, end date (for training and test) and the starting value (in dollars) for the portfolio. For the analysis here, the values used were:

symbol="JPM"  
sd=dt.datetime(2008, 1, 1)  
sdTest=dt.datetime(2010, 1, 1)  
ed=dt.datetime(2009, 12, 31)  
edTest=dt.datetime(2011, 12, 31)  
sv=100000

The five charts as well as a .txt file with statistics are written to the [outputs](/outputs) directory.

The [data](/data) folder contains csv files with historic data for respective stock symbols.

# Alternate Parameters
[decision_tree_strategy.py](decision_tree_strategy.py) has three parameters that will have varying effects on performance: leaf_size, bags, & boost. I defined them them as follows:

kwargs = {"leaf_size": 10}  
bags = 30  
boost = False

[reinforcement_learning_strategy.py](reinforcement_learning_strategy.py) has 6 parameters that can be changed for varying performance: self.lookback, alpha, gamma, rar, radr, and dyna. I defined them as follows:

self.lookback = 20  
alpha = 0.2  
gamma = 0.9  
rar = 0.98    
radr = 0.999  
dyna = 0

# Plots
![plot](./outputs/in_sample_normalized_portfolios.png)

![plot](./outputs/out_of_sample_normalized_portfolios.png)

![plot](./outputs/impact_dts.png)

![plot](./outputs/impact_rls.png)

![plot](./outputs/impact_vs_numTrades.png)

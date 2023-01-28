# tradingStrategies
[strategyEval_report.pdf](strategyEval_report.pdf) is a report that was written for a Georgia Tech masters course Fall of 2022. The code here extends upon the work in the original report and I have since added reinforcement learning to the mix of strategies.

An updated conceptual overview can be found here on Medium. Additonally, the contents of individual functions is briefly described at the bottom of this file.

Development was done in a conda environment and the [environment.yml](environment.yml) file defines all the dependencies.

[testproject.py](testproject.py) is the main driver file. Here you can adjust the parameters for the analysis. The variables are the stock symbol, start date, end date (for training and test) and the starting value (in dollars) for the portfolio. For the analysis here, the values used were:

```
symbol="JPM"  
sd=dt.datetime(2008, 1, 1)  
sdTest=dt.datetime(2010, 1, 1)  
ed=dt.datetime(2009, 12, 31)  
edTest=dt.datetime(2011, 12, 31)  
sv=100000
```

The five charts as well as a .txt file with statistics are written to the [outputs](/outputs) directory.

The [data](/data) directory contains csv files with historic data for respective stock symbols.

# Alternate Parameters
[decision_tree_strategy.py](decision_tree_strategy.py) has three parameters that will have varying effects on performance: leaf_size, bags, & boost. I defined them them as follows:

```
kwargs = {"leaf_size": 10}  
bags = 30  
boost = False
```

[reinforcement_learning_strategy.py](reinforcement_learning_strategy.py) has 6 parameters that can be changed for varying performance: self.lookback, alpha, gamma, rar, radr, and dyna. I defined them as follows:

```
self.lookback = 20  
alpha = 0.2  
gamma = 0.9  
rar = 0.98    
radr = 0.999  
dyna = 0
```

# Visualizations
![](./outputs/in_sample_normalized_portfolios.png) 

![](./outputs/out_of_sample_normalized_portfolios.png)

![](./outputs/impact_dts.png)

![](./outputs/impact_rls.png)

![](./outputs/impact_vs_numTrades.png)

# File Breakdowns

[util.py](util.py)  
This file contains some utility functions that the other files can use for processing of data. The "get_data" function is called multiple times from multiple separate files. This function builds and returns a data frame with daily stock data from the csv files in the data folder.

[marketsimcode.py](marketsimcode.py)  
This file takes in a dataframe that specifies buy or sell orders for each day and a respective stock symobl. It creates and returns a dataframe representing daily portfolio values based off the trades.

[indicators.py](indicators.py)  
This file contains the logic for calculating the three technical indicators data that the strategies use to make their trading decisions. The three indicators are the bollinger band percentage (BBP), relative strength index (RSI), and an exponential moving average cross (EMA Cross) which combines ema with a a 13 and 21 day lookback period into one indicator.

[ManualStrategy.py](ManualStrategy.py)  
This file's parameters were tuned with trial and error on the training data. Entering a long or short position is based on hardcoded thresholds for the three technical indicators.

[RTLearner.py](RTLearner.py)  
This file is the logic behind the decision tree strategy. It builds and queries the decision tree. Both functions utilize recursive logic. It randomly selects a feature to split on (those being the bbp value, rsi value, or ema_cross value).

[BagLearner.py](BagLearner.py)  
This file combines multiple instances of the RTLearner class and returns a more robust model.

[decision_tree_strategy.py](decision_tree_strategy.py)  
This file utilizes the BagLearner class which encompasses the RTLearner class. This is a supervised learning method where the "X" values are the daily technical indicator values. The "Y" value is created by dividing the closing price five days from present by the closing price for the present day. The intuition here is that if the value five days from present is greater than today, then buying today would yield more return than just holding cash. Then after looking into what these values could be, I arrived at setting a threshold of 0.03 +- impact to decide either to flag a long (+1) signal or a short (-1) signal. So, then we have three features (X) for today (techincal indicator values) and a single integer (+1 or -1) representing our "Y" value.
Then the x and y data is fed to the BagLearner and furthermore RTLearner to "add_evidence" and build the tree and then the "query" function is used to make buy or sell decisions on new or existing data.

[QLearner.py](QLearner.py)  
This file is a reinforcement learning class that is implemented with a Q-Learner algorithm. It requires an environment to interact with, which is defined in the reinforcement_learning_strategy. It also implements "dyna" which is an addition to the algorithm which can allow the model to bootstrap on already seen data when there is a lack of training data or it is expensive to obtain. 

[reinforcement_learning_strategy.py](reinforcement_learning_strategy.py)  
This file calls the QLearner class and then simulates the "environment" and feedback to the QLearner, which then in turn is able to return long or short signals after it has been trained. The data has to be transformed and received differently than the "X" "Y" form that a supervised learner would such as the decision tree. Instead, the QLearner works with three values; state, action and reward. The state is defined in this file by first calculating individual technical indicators (bbp, rsi, ema_cross). Then, these values are discretized into buckets (0-10) for each indicator. Lastly, the three bucketed values are combined into one single number (state) by multiplying an indicator by an increasing power of ten so that in the end we have 1000 potential states. BBP value is in the hundreds place, ema_cross is in the tens place adn RSI is in the ones place. So this defines the state. Their are three actions those being buuy, sell or do nothing. Lastly, we need to provide a reward to the QLearner. The reward is defined by multiplying the stock holding from the previous day by the daily return and then subtracting values related to impact and commission. With these three values defined, the add_evidence function trains for 200 epochs and or finsihes if the solution converges after 20 epochs.

[testproject.py](testproject.py)  
This is the main dirver file that brings everything together. It runs a few analysis functions that plot the data and write statistics to a .txt file. The functions "analysis_1" and "analysis_2" are the main functions while the others are helper functions for these two main ones.

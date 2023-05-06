####################################################################
# This file is for running the three different algorithms and 
# comparing their results. The algorithms are:
# 1. linear regression
# 2. lstm neural network
# 3. random forest
# The data is from tradingview.com and is the 4 hour chart of the:
# BTCUSDT 
# ETHUSDT
# SOLUSDT
# 
# Each crypto have nine test cases in three trend categories:
# 1. uptrend
# 2. downtrend
# 3. sideways
# 
# The project will make prediction on the price of the coin in the 
# next 4 hours given the previous 30 days of data. The project will
# then compare the prediction to the actual price and calculate the
# error. The error is then compared to the other algorithms. Each 
# trend category will be assigned a best algorithm that predicts
# price highest accuracy.
# 
# The data is split into 80% training and 20% testing.
####################################################################


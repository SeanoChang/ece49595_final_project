####################################################################
# This file is for running the linear regression algorithm and 
# make predictions on the price of the coin in the next 4 hours
# based on the previous 30 days of data. 
####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# linear regression
def linearRegression_model(X_train, y_train):    
    # train model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    return lm

# make predictions
def makePredictions(lm, X_test):
    # make predictions
    predictions = lm.predict(X_test)
    
    # return predictions
    return predictions

# evaluate model
def evaluateModel(y_test, predictions):
    # evaluate model
    print('Mean Absolute Error:', mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, predictions)))
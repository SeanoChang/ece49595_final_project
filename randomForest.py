####################################################################
# This file is for running the linear regression algorithm and 
# make predictions on the price of the coin in the next 4 hours
# based on the previous 30 days of data. 
####################################################################

# import libraries for random forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# random forest
def randomForest_model(X_train, y_train):
    # train model
    rf = RandomForestRegressor(n_estimators=100, random_state=101)
    rf.fit(X_train, y_train)

    return rf

# make predictions
def makePredictions_rf(rf, X_test):
    # make predictions
    predictions = rf.predict(X_test)
    
    # return predictions
    return predictions

# evaluate model
def evaluateModel_rf(y_test, predictions):
    # evaluate model
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


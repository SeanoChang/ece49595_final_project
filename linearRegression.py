####################################################################
# This file is for running the linear regression algorithm and 
# make predictions on the price of the coin in the next 4 hours
# based on the previous 30 days of data. 
####################################################################

# import libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# linear regression
def linearRegression_model(X_train, y_train):    
    # train model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    return lm

# make predictions
def lr_predictions(lm, X_test):
    # make predictions
    predictions = lm.predict(X_test)
    
    # return predictions
    return predictions

# evaluate model
def evaluate_lr(y_test, predictions):
    # return error
    return mean_squared_error(y_test, predictions)
####################################################################
# This file is for running the three different algorithms and 
# comparing their results. The algorithms are:
# 1. linear regression
# 2. lstm neural network
# 3. random forest
# The data is from tradingview.com and is the 4 hour chart of BTCUSDT 
# from OKX exchange. 
# 
# The dataset have nine test cases total for three trend categories:
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

# import libraries
import os
from lstm import *
from linearRegression import *
from randomForest import *
from data import *

# import data
data = getData()

# initialize lists for scores
scores = []

# algorithm list
algorithms = ['linear regression', 'lstm', 'random forest']

# for each dataset in data list 
for dataset in data:
    # preprocess data for linear regression and random forest
    preprocessed = dataPreprocessing(dataset)
    
    # split data into training and testing for linear regression and random forest
    X_train, X_test, y_train, y_test = splitData(preprocessed)
    
    ## linear regression ##
    lm = linearRegression_model(X_train, y_train)
    # make predictions
    predictions_lr = lr_predictions(lm, X_test)
    # evaluate model
    linear_score = evaluate_lr(y_test, predictions_lr)
    
    
    ## lstm ##
    model = lstm_model(X_train, n_neurons=150)
    # train model
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    # make predictions
    predictions_lstm = lstm_predictions(model, X_test)
    # evaluate model
    lstm_score = evaluate_lstm(model, y_test, predictions_lstm)
    
    
    ## random forest ##
    rf = randomForest_model(X_train, y_train, n_estimators=500)
    # make predictions
    predictions_rf = rf_predictions(rf, X_test)
    # evaluate model
    rf_score = evaluate_rf(y_test, predictions_rf)
    
    # add scores to list
    scores.append([linear_score, lstm_score, rf_score])
    
# get data file names
data_files = []
for coin in os.listdir('data'):
    for trend in os.listdir('data/' + coin):
        for file in os.listdir('data/' + coin + '/' + trend):
            data_files.append(file)
    
# print scores
for i in range(len(scores)):
    print()
    print(data_files[i], 'dataset scores:')
    for j in range(len(scores[i])):
        print(algorithms[j], 'score:', scores[i][j])
    print()
    
    # find best algorithm
    best_score = min(scores[i])
    best_algorithm = algorithms[scores[i].index(best_score)]
    print('Best algorithm:', best_algorithm)
    print()
    

        
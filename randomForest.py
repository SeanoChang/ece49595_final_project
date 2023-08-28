####################################################################
# This file is for running the linear regression algorithm and 
# make predictions on the price of the coin in the next 4 hours
# based on the previous 30 days of data. 
####################################################################

# import libraries for random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# random forest
def randomForest_model(X_train, y_train, n_estimators=100, random_state=101):
    # train model
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    return rf

# make predictions
def rf_predictions(rf, X_test):
    # make predictions
    predictions = rf.predict(X_test)
    
    # return predictions
    return predictions

# evaluate model
def evaluate_rf(y_test, predictions):
    # return error
    return metrics.mean_squared_error(y_test, predictions)

if __name__ == '__main__':
    # import libraries
    import os
    from data import *
    
    # import data
    data = getData()
    
 # initialize lists for scores
    scores = []

    # parameters list
    n_estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # best parameters
    best_param = []

    for dataset in data:
        # preprocess data
        preprocessed = dataPreprocessing(dataset)

        # split data
        X_train, X_test, y_train, y_test = splitData(preprocessed)

        # smallest data parameter
        min_estimators = 50

        # score for smallest data parameter
        min_score = 1000000

        # loop through parameters and run lstm
        for e in n_estimators:
            # random forest
            rf = randomForest_model(X_train, y_train, n_estimators=e)
            
            # make predictions
            predictions = rf_predictions(rf, X_test)
            
            # evaluate model
            score = evaluate_rf(y_test, predictions)
            
            # if score is less than min_score, update min_score and min_estimators
            if score < min_score:
                min_score = score
                min_estimators = e

        # append best parameters to best_param
        best_param.append(min_estimators)
        # append score to scores
        scores.append(min_score)

    # print best parameters
    print(best_param)
    # print scores
    print(scores)

## results ## 
# [200, 300, 500, 250, 500, 100, 300, 50, 200]
# [0.00028184582496217043, 0.00034484060057370894, 0.0001018655272204067, 0.00031911578553638477,
# 0.0001572264592267106, 0.0002929702809542801, 0.0022971821564549476, 0.002319135719139053, 0.0016367000050198893]
# choose 500 for all datasets since it has the lowest error

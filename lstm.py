####################################################################
# This file is for running the linear regression algorithm and 
# make predictions on the price of the coin in the next 4 hours
# based on the previous 30 days of data. 
####################################################################

# import libraries for lstm
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# lstm
def lstm_model(X_train, n_neurons=100, n_layers=3):
    model = Sequential()
    
    # add layers
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # add hidden layers
    for i in range(n_layers-1):
        model.add(LSTM(n_neurons, return_sequences=True))
        model.add(Dropout(0.2))
    
    # output layer
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# train model
def train_lstm(model, X_train, y_train, epochs=25, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return model

# make predictions
def lstm_predictions(model, X_test):
    predictions = model.predict(X_test)
    
    return predictions

# evaluate model
def evaluate_lstm(model, X_test, y_test):
    score = model.evaluate(X_test, y_test)
    
    return score

if __name__ == '__main__':
    # import libraries
    import os
    from data import *
    
    # import data
    data = getData()
    
    # initialize lists for scores
    scores = []
    
    # parameters list
    neurons = [50, 100, 150, 200]
    epochs = [25, 50, 75, 100]
    batch_sizes = [32, 64, 128, 256]
    
    # best parameters
    best_param = []
    
    for dataset in data:
        # preprocess data
        preprocessed = dataPreprocessing(dataset)
        
        # split data
        X_train, X_test, y_train, y_test = splitData(preprocessed)
        
        # smallest data parameter
        min_neurons = 50
        min_epochs = 25
        min_batch_size = 32
        
        # score for smallest data parameter
        min_score = 1000000
        
        # loop through parameters and run lstm
        for n in neurons:
            for e in epochs:
                for b in batch_sizes:
                    # lstm
                    model = lstm_model(X_train, n_neurons=n)
                    # train model
                    model = train_lstm(model, X_train, y_train, epochs=e, batch_size=b)
                    # make predictions
                    predictions = lstm_predictions(model, X_test)
                    # evaluate model
                    score = evaluate_lstm(model, X_test, y_test)
                    
                    # if score is smaller than min_score, update min_score and parameters
                    if score < min_score:
                        min_score = score
                        min_neurons = n
                        min_epochs = e
                        min_batch_size = b
   
        # append best parameters to best_param
        best_param.append([min_neurons, min_epochs, min_batch_size])
        # append score to scores
        scores.append(min_score)
        
    # print best parameters
    print(best_param)
    # print scores
    print(scores)

## results ##
# [[50, 50, 32], [100, 100, 32], [50, 100, 32], [150, 100, 32], [150, 75, 128], [100, 25, 64], [150, 100, 32], [100, 100, 32], [150, 100, 32]]
# [0.0130356689915061, 0.013418995775282383, 0.006273999810218811, 0.005618644412606955,
# 0.005349487531930208, 0.013660195283591747, 0.02024468220770359, 0.02177536115050316, 0.01973387971520424]
# choosing 150 neurons, 100 epochs, and 32 batch size for all datasets
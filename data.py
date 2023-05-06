####################################################################
# This file is for doing the data preprocessing. The data is from
# tradingview.com might have NaN values. This file will replace the
# NaN values with the mean of the column. The data is then scaled
# using the MinMaxScaler and StandardScaler. The data is then split
# into training and testing data. The training data is used to train
# the model and the testing data is used to test the model.
# 
# We will only use the close column for the linear regression 
# algorithm. 
####################################################################

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# data preprocessing
def dataPreprocessing_linear(df, coin, trend):
    # replace NaN values with mean of column
    df = df.fillna(df.mean())
    
    # scale data
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    
    # leave only close and time columns
    scaled_df = scaled_df.drop(['open', 'high', 'low', 'volume'], axis=1)

    # return scaled_df
    return scaled_df

# for LSTM model preprocessing
def dataPreprocessing_lstm(df, coin, trend):
    # replace NaN values with mean of column
    df = df.fillna(df.mean())
    
    # scale data
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    
    # convert date into 
    
    # return scaled_df
    return scaled_df

# split data into training and testing
def splitData(df, coin, trend):
    # split data into training and testing
    X = df.drop(['close'], axis=1)
    y = df['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    # return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

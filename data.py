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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# get data for uptrend, downtrend, and sideways
def getData():
    # get data from data/{coin}/{trend} directory
    data = []
    for coin in os.listdir('data'):
        for trend in os.listdir(f'data/{coin}'):
            for file in os.listdir(f'data/{coin}/{trend}'):
                df = pd.read_csv(f'data/{coin}/{trend}/{file}')
                data.append(df)
        
    # return data
    return data

# data preprocessing
def dataPreprocessing(df):
    # replace NaN values with mean of column
    df = df.fillna(df.mean())
    
    # scale data
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    
    # leave only close and time columns
    scaled_df = scaled_df.drop(['open', 'high', 'low'], axis=1)

    # return scaled_df
    return scaled_df

# split data into training and testing
def splitData(df):
    # split data into training and testing
    X = df.drop(['close'], axis=1)
    y = df['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    # return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

# inverse scale
def inverse_scale(df,predictions):
    # inverse scale
    scaler = MinMaxScaler()
    scaler.fit(df)
    predictions = scaler.inverse_transform(predictions)
    
    # return predictions
    return predictions

if __name__ == '__main__':
    data = getData()
    
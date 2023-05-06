####################################################################
# This file is for running the linear regression algorithm and 
# make predictions on the price of the coin in the next 4 hours
# based on the previous 30 days of data. 
####################################################################

# import libraries for lstm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# lstm
def lstm_model(X_train, y_train):

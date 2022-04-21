# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:38:18 2022

@author: jo_as
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras.backend as K
from ClassRNN import RNN
import matplotlib.pyplot as plt
import functions

#%%Preprossesing data, and ads y_prev to inputdata
df_train = pd.read_csv('no1_train.csv', sep=',')
df_validation = pd.read_csv('no1_validation.csv', sep=',')
x_train = np.asarray(df_train.values[:,1:-1]).astype(np.float32) #(225089, 8)
y_train = np.asarray(df_train.values[:,[-1]]).astype(np.float32) #(225089, 1)
x_val = np.asarray(df_validation.values[:,1:-1]).astype(np.float32) #(28136, 8)
y_val = np.asarray(df_validation.values[:,[-1]]).astype(np.float32) #(28136, 1)

test_data = df_validation.values
attributes = df_train.columns #[0,1,2,3,4,5,6,7]

print(attributes)
print(df_train.head())
x_train = np.concatenate((x_train[1:], y_train[:-1]), axis = 1) #Data in timestep t_i+1 uses y from t_i
y_train = y_train[:-1]
x_val = np.concatenate((x_val[1:], y_val[:-1]), axis = 1) #Data in timestep t_i+1 uses y from t_i
y_val = y_val[:-1]
print(x_train[0].shape)
print(y_train.shape)
print(x_val[0].shape)
print(y_val.shape)


#%%
N_seq = 20
x_train, y_train = functions.create_dataset(df_train, N_seq)
x_val, y_val = functions.create_dataset(df_validation, N_seq)

#%%
trained = True
model = RNN(n_seq = N_seq, n_dim = 9, filename = "./models/LSTM6420", trained = trained)#LSTMXXYY means X in lst parameter an Y seq_length
model.train(x_train, y_train, x_val, y_val, epochs = 20)


#%%    
print(x_val.shape)
print(y_val.shape)
forecasts = functions.n_in_1_out(x_val, model, 200)      

print(forecasts.shape)
print(y_val[-600:-200,-1].shape)
full_series = np.concatenate((y_val[-600:-200,-1], forecasts), axis = 0)
plt.plot(full_series, "r")
plt.plot(y_val[-600:,-1], "b")
















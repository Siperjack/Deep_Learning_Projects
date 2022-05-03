# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:32:04 2022

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


def n_in_1_out(data, model, pred_window): #data is assumed to be of the [batch, n_seq, n_feature] format
    start_ind = len(data) - pred_window
    model_input = data[[start_ind]]
    forecasts = []
    forecast = model.predict(model_input)[0,-1]
    forecasts.append(forecast)
    for i in range(pred_window - 1):
        model_input = data[[start_ind + 1 + i]]
        #print(model_input.shape)
        model_input[0, -1, -1] = forecasts[-1] #Last prev_y datapoint is replaced by last prediction
        forecast = model.predict(model_input)[0,-1]
        forecasts.append(forecast)
    return np.array(forecasts)

def create_dataset(data, n_seq):
    x_data = np.asarray(data.values[:,1:-1]).astype(np.float32) #(225089, 8)
    y_data = np.asarray(data.values[:,[-1]]).astype(np.float32) #(225089, 1)
    x_data = np.concatenate((x_data[1:], y_data[:-1]), axis = 1) #Data in timestep t_i+1 uses y from t_i
    y_data = y_data[:-1]
    x_data, x_norm = sklearn.preprocessing.normalize(x_data)
    y_data, y_norm = sklearn.preprocessing.normalize(y_data)
    
    x_list, y_list = [], []
    print(len(x_data))
    for i in range(len(x_data) - n_seq - 1):
        x_list.append(x_data[i:(i+n_seq)])
        y_list.append(y_data[i:(i+n_seq)])
        
    print(len(x_list))
    return np.array(x_list), np.array(y_list), x_norm, y_norm

def struct_imb(df_train, x0,x1):
    first_change = 0
    for i in range(20):
        df_train["total"][i]
        if df_train["total"][i+1] != df_train["total"][i]:
            first_change = i+1
    
    print(len(df_train["total"]))
    y = np.zeros(1 + len(df_train["total"][first_change + 6::12]))
    if first_change >= 6:
        y[0] = np.asarray(df_train["total"][first_change - 6])
        y[1:] = np.asarray(df_train["total"][first_change + 6::12])
    else:
        y[0] = np.asarray(df_train["total"][0])
        y[1:] = np.asarray(df_train["total"][first_change + 6::12])
    x = np.asarray([0] + [i for i in range(first_change + 6,len(df_train["total"]),12)])
    print(x)
    plt.plot(x[0:5],y[0:5])
    f = interp1d(x, y, kind = "cubic")
    x_new_end = x[-1]
    x_new = np.arange(0, len(df_train["total"]))[:x_new_end]
    print(x[-1], x_new[-1])
    print(f"x_new is {x_new}")
    fig, ax = plt.subplots((2))
    ax[0].plot(x_new[x0:x1],np.asarray(df_train["total"][x0:x1]),label = "real data")
    ax[0].plot(x_new[x0:x1],f(x_new)[x0:x1], label = "smooth data")
    ax[0].legend()
    ax[1].plot(x_new[x0:x1],np.asarray(df_train["total"][x0:x1])-f(x_new)[x0:x1],label = "error")
    ax[1].legend()
    
    fig, ax = plt.subplots((2))
    ax[0].plot(x_new[x0:x1],np.asarray(df_train["total"][x0:x1]),label = "real data")
    ax[0].plot(x_new[x0:x1],f(x_new)[x0:x1], label = "smooth data")
    ax[0].legend()
    ax[1].plot(x_new[x0:x1],np.asarray(df_train["total"][x0:x1])-f(x_new)[x0:x1],label = "error")
    ax[1].legend()

# struct_imb(df_train,-80,-1)

# def struct_imb2(data, x0,x1):
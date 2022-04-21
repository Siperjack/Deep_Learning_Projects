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
    
    
    x_list, y_list = [], []
    print(len(x_data))
    for i in range(len(x_data) - n_seq - 1):
        x_list.append(x_data[i:(i+n_seq)])
        y_list.append(y_data[i:(i+n_seq)])
        
    print(len(x_list))
    return np.array(x_list), np.array(y_list)
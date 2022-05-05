# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:06:13 2022

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

class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.history = {'loss':[],'val_loss':[]}

        def on_batch_end(self, batch, logs={}):
            self.history['loss'].append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.history['val_loss'].append(logs.get('val_loss'))
            
            
class RNN:
    def __init__(self, n_seq, n_dim, trained = False, filename = "Test", num_LSTM = 64, lr = 0.00001, dropout = 0, double = False, triple = False, reg_val = 0):
        self.n_seq  = n_seq
        self.n_dim = n_dim
        self.filename = filename
        self.num_LSTM = num_LSTM
        self.lr = lr
    
        
        Input = keras.Input(shape = (self.n_seq, self.n_dim), name='Input') # [n_batch, n_seq, n_dim]
        print("input shape is: ", Input.shape)
        #x = CuDNNLSTM(self.num_LSTM, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=False)(Input)
        x = layers.LSTM(self.num_LSTM, activation = "tanh", return_sequences = double, name = "LSTM_unit_1", kernel_regularizer=l2(reg_val))(Input)
        if dropout:
            x = layers.Dropout(dropout,name = "dropout_unit_1")(x)
        if double:
            x = layers.LSTM(self.num_LSTM, activation = "tanh", return_sequences = triple, name = "LSTM_unit_2", kernel_regularizer=l2(reg_val))(x)
            if triple:
                if dropout:
                    x = layers.Dropout(dropout,name = "dropout_unit_2")(x)
                x = layers.LSTM(self.num_LSTM, activation = "tanh", return_sequences = False, name = "LSTM_unit_3", kernel_regularizer=l2(reg_val))(x)
        Output = layers.Dense(1)(x) #output should be unbounded
        self.Model = models.Model(Input, Output, name='RNN')
        
        
        self.optim = keras.optimizers.Adam(learning_rate = self.lr)
        self.loss = keras.losses.MeanSquaredError()
        
        self.Model.compile(optimizer = self.optim, loss = self.loss)
        self.Model.summary()
        self.trained = trained
        if trained:
            self.Model.load_weights(filename)
            print("weights loaded")
        
    def train(self, x_train, y_train, x_val, y_val, batch_size = 32, epochs = 5):
        if not self.trained:
            history = LossHistory()
            self.Model.fit(
                x_train,
                y_train,
                epochs = epochs,
                shuffle = True,
                batch_size = batch_size,
                validation_data=(x_val, y_val),
                callbacks=[history]
                )
            
            self.Model.save_weights(self.filename)
            print("model is trained and weights saved")
            return history
        else:
            print("model is already trained")
            return None
    # def get_history(self):
    #     with open(file=self.history_filename, mode = "rb") as file:
    #         data = pickle.load(file)
    #     return data
            
    def predict(self, data):
        return self.Model.predict(data)
    
    def n_in_1_out(self, data, pred_window, feature_list, start_ind = False): #data is assumed to be of the [batch, n_seq, n_feature] format
        y_prev_loc = feature_list.get_loc("y_prev")
        print(f"u_prev_loc is : {y_prev_loc}")
        if not start_ind:
            start_ind = len(data) - pred_window
        else:
            if start_ind > len(data) - pred_window:
                print("start index to high")
                return np.ones(pred_window)
        forecasts = []
        print("datashape is: ", data.shape)
        model_input = data[[start_ind]]
        print("model input shape is:" ,model_input.shape)
        forecast = self.Model.predict(model_input)
        print("forecast shape is: ", forecast.shape)
        forecasts.append(forecast)
        for i in range(pred_window - 1):
            for j in range(1, i+2): #take i+1 last element of datapoint start index + i and replace y_prev with y_forecast
                data[start_ind + 1 + i, -j, y_prev_loc] = forecasts[-j] #Takes the last y_prev and replaces it with the forecast in place on data
            model_input = data[[start_ind + 1 + i]] #note that the other lag features are not changed, so predictions over 24 hours will not be valid
            #print(model_input.shape)
            #model_input[0, -1, -1] = forecasts[-1] #Last prev_y datapoint is replaced by last prediction
            forecast = self.Model.predict(model_input)
            forecasts.append(forecast)
        return np.array(forecasts).copy(), start_ind
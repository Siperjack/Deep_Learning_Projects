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

class RNN:
    def __init__(self, n_seq, n_dim, trained = False, filename = "Test"):
        self.n_seq  = n_seq
        self.n_dim = n_dim
        self.filename = filename
    
    
        Input = keras.Input(shape = (self.n_seq, self.n_dim), name='Input') # [n_batch, n_seq, n_dim]
        print("input shape is: ", Input.shape)
        
        x = layers.LSTM(32, activation = "relu", return_sequences=True, name = "LSTM_unit_1")(Input)
        Output = layers.Dense(1, activation = "linear")(x) #output should be unbounded
        
        self.Model = models.Model(Input, Output, name='RNN')
        
        
        self.optim = keras.optimizers.Adam(learning_rate = 0.0001)
        self.loss = keras.losses.MeanSquaredError()
        
        self.Model.compile(optimizer = self.optim, loss = self.loss)
        self.Model.summary()
        self.trained = trained
    
    
    def train(self, x_train, y_train, x_val, y_val, batch_size = 1024, epochs = 5):
        if not self.trained:
            
            self.Model.fit(
                x_train,
                y_train,
                epochs = epochs,
                shuffle = True,
                batch_size = batch_size,
                validation_data=(x_val, y_val)
                )
            
            self.Model.save_weights(self.filename)
            print("model is trained and weights saved")
        else:
            print("model is already trained")
            
    def predict(self, data):
        return self.Model.predict(data)
            
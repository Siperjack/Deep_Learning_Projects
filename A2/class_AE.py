import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, models
from stacked_mnist import StackedMNISTData, DataMode
import os
import matplotlib.pyplot as plt
from verification_net import VerificationNet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#Architecture
   
class AE:
    def __init__(self, latent_dim, filename, trained = True):
        self.n_dim = 28
        self.latent_dim = latent_dim
        self.filename = filename
        self.trained = trained
        
        """Encoder"""
        
        Encoder = models.Sequential(name='Encoder')
        Encoder.add(layers.Conv2D(32, (3,3),strides = 1, padding = "same", activation = "relu", input_shape = (self.n_dim, self.n_dim, 1)))
        #Encoder.add(layers.MaxPooling2D(pool_size = (2,2)))
        #Encoder.add(layers.Dropout(0.25))
        Encoder.add(layers.Conv2D(64, (3,3),strides = 2, padding = "same",activation = "relu"))
        #Encoder.add(layers.MaxPooling2D(pool_size = (2,2)))
        #Encoder.add(layers.Dropout(0.25))
        Encoder.add(layers.Conv2D(64, (3,3),strides = 2, padding = "same",activation = "relu"))
        #Encoder.add(layers.MaxPooling2D(pool_size = (2,2)))
        #Encoder.add(layers.Dropout(0.25))
        #Encoder.add(layers.Conv2D(32, (3,3),strides = 2, padding = "same",activation = "relu"))
        
        """Bottleneck of the autoencoder"""
        
        Encoder.add(layers.Flatten())
        #Encoder.add(layers.Dense(128, activation='relu'))
        #Encoder.add(layers.Dropout(0.5))
        Encoder.add(layers.Dense(self.latent_dim))
        #Encoder.add(layers.Normalization())
        
        
        """Decoder"""
        
        Decoder = models.Sequential(name='Decoder')
        Decoder.add(layers.Dense(1568, activation = "relu", input_shape = (self.latent_dim, 1)))
        Decoder.add(layers.Reshape((7, 7, 32*self.latent_dim)))
        Decoder.add(layers.Conv2DTranspose(32, (3,3), strides = 2,padding = "same", activation = "relu"))
        Decoder.add(layers.Conv2DTranspose(32, (3,3), strides = 2,padding = "same", activation = "relu"))
        Decoder.add(layers.Conv2DTranspose(1, (3,3), strides = 1,padding = "same", activation = "sigmoid"))
        
        """Compiling Encoder + Decoder with loss and optim"""
        
        AutoEncoder = models.Model(Encoder.input, 
                                   Decoder(Encoder.output), 
                                   name='AutoEncoder')
        
        def MSE(target, pred):
            error = tf.cast(target, tf.float32) - pred
            print((target))
            print((tf.cast(target, tf.float32)))
            return tf.reduce_mean(tf.square(error), axis = [1,2,3])
        #loss = MSE, does not work. Eery trained model returns blanks.
        loss = keras.losses.BinaryCrossentropy()
        optim = keras.optimizers.Adam(learning_rate = 0.001)
        
        AutoEncoder.compile(optimizer = optim, loss = loss, metrics=['accuracy'])
        
        """Setting class variables"""
        
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.AutoEncoder = AutoEncoder
        self.Encoder.summary()
        self.Decoder.summary()
        if trained:
            self.AutoEncoder.load_weights(filename)
            print("weights loaded")

    
    def train(self, train_images, val_images, batch_size = 1024, epochs = 5):
        if not self.trained:
            self.AutoEncoder.fit(
                train_images,
                train_images,#label
                epochs = epochs,
                shuffle = True,
                batch_size = batch_size,
                validation_data=(val_images, val_images)
                )
            
            self.AutoEncoder.save_weights(self.filename)
            print("model is trained and weights saved")
        else:
            print("model is already trained")
            
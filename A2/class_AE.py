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
        Encoder.add(layers.Conv2D(8, (3,3),strides = 1, padding = "same", activation = "relu", input_shape = (self.n_dim, self.n_dim, 1)))
        Encoder.add(layers.Conv2D(8, (3,3),strides = 2, padding = "same",activation = "relu"))
        Encoder.add(layers.Conv2D(8, (3,3),strides = 2, padding = "same",activation = "relu"))
        
        """Bottleneck of the autoencoder"""
        
        Encoder.add(layers.Flatten())
        Encoder.add(layers.Dense(self.latent_dim))
        
        """Decoder"""
        
        Decoder = models.Sequential(name='Decoder')
        Decoder.add(layers.Dense(7*7*8, activation = "relu", input_shape = (self.latent_dim, 1)))
        Decoder.add(layers.Reshape((7, 7, 8*self.latent_dim)))
        Decoder.add(layers.Conv2DTranspose(8, (3,3), strides = 2,padding = "same", activation = "relu"))
        Decoder.add(layers.Conv2DTranspose(8, (3,3), strides = 2,padding = "same", activation = "relu"))
        Decoder.add(layers.Conv2D(1, (3,3), strides = 1, padding = "same", activation = "sigmoid"))
        
        """Compiling Encoder + Decoder with loss and optim"""
        
        AutoEncoder = models.Model(Encoder.input, 
                                   Decoder(Encoder.output), 
                                   name='AutoEncoder')
        

        loss = keras.losses.BinaryCrossentropy()
        optim = keras.optimizers.Adam(learning_rate = 0.01)
        
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
            
    def predict(self, val_images):
        dataset_size = len(val_images)
        Nchannels = len(val_images[0,0,0]) #MxNxNxNchannel
        results_stacked = np.zeros((dataset_size, 28, 28, Nchannels))
        for i in range(Nchannels):
            results_stacked[:,:,:,[i]] = self.AutoEncoder.predict(val_images[:,:,:,[i]])
        return results_stacked
    
    def predict_from_latent(self, z_stacked):
        dataset_size = len(z_stacked)
        Nchannels = len(z_stacked[0,0]) #MxNxNxNchannel
        generated_stacked = np.zeros((dataset_size, 28, 28, Nchannels))
        for i in range(Nchannels):
            generated_stacked[:,:,:,[i]] = self.Decoder.predict(z_stacked[:,:,[i]])
        return generated_stacked
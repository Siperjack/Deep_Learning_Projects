import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, models, Input
from stacked_mnist import StackedMNISTData, DataMode
import os
import matplotlib.pyplot as plt
from verification_net import VerificationNet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.disable_eager_execution()


#Architecture
   
class VAE:
    def __init__(self, filename, trained = True):
        self.n_dim = 28
        self.latent_dim = 2
        self.filename = filename
        self.trained = trained
        self.combined_loss_weight = 1000
        
    
        """Encoder"""
        
        EncoderInput = Input(shape = (self.n_dim, self.n_dim, 1),name='EncoderInput')
        x = layers.Conv2D(32, (3,3),strides = 1, padding = "same", activation = "relu")(EncoderInput)#prevous layer output
        #x = layers.MaxPooling2D(pool_size = (2,2)))(x)
        #x = layers.Dropout(0.25))(x))
        x = layers.Conv2D(64, (3,3),strides = 2, padding = "same",activation = "relu")(x)
        x = layers.Conv2D(64, (3,3),strides = 2, padding = "same",activation = "relu")(x)
        
        """Bottleneck of Encoder"""
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        self.mu = layers.Dense(self.latent_dim, name = "mu")(x)
        self.log_sigma = layers.Dense(self.latent_dim, name = "log_sigma")(x)
        #z = Sampling()([mu, log_sigma])
        def sample_from_gaussian(inputs):
            mu, log_sigma = inputs
            epsilon = tf.keras.backend.random_normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1])) #beckend got loads of distributions
            return mu + tf.exp(0.5 * log_sigma) * epsilon
            
        EncoderOutput = layers.Lambda(sample_from_gaussian, name = "EncoderOutput")([self.mu, self.log_sigma])#Lambdalayer allows function input
        
        Encoder = models.Model(EncoderInput, EncoderOutput, name='Encoder')
        
        
        """Decoder"""
        
        DecoderInput = Input(shape = (self.latent_dim, 1), name='DecoderInput')
        x = layers.Dense(7*7*16*self.latent_dim, activation = "relu")(DecoderInput)
        x = layers.Reshape((7, 7, 32*self.latent_dim))(x)
        x = layers.Conv2DTranspose(32, (4,4), strides = 2,padding = "same", activation = "relu")(x)
        x = layers.Conv2DTranspose(32, (4,4), strides = 1,padding = "same", activation = "relu")(x)
        DecoderOutput = layers.Conv2DTranspose(1, (4,4), strides = 2,padding = "same", activation = "sigmoid")(x)
        
        Decoder = models.Model(DecoderInput,DecoderOutput, name='Decoder')
        
        """Compiling Encoder + Decoder"""
        
        Encoder_output = Encoder.output
        out = Decoder(Encoder_output)
        AutoEncoder = models.Model(Encoder.input, out, name='AutoEncoder')
        
        loss = keras.losses.BinaryCrossentropy()
        #loss = self.calc_comb_loss_dummy
        optim = keras.optimizers.Adam(learning_rate = 0.01)
        #AutoEncoder.add_loss(self.calc_comb_loss)#This allows losses without ytarget,ypred
        #.compile(optimizer = optim)
        AutoEncoder.compile(optimizer = optim, 
                            loss=loss, 
                            metrics=[
                                keras.losses.binary_crossentropy,
                                tf.keras.metrics.mean_squared_error
                                ])
        #metrics = [self.calc_MSE, self.calc_KL_divergence]
        #evaluate 
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.AutoEncoder = AutoEncoder
        self.Encoder.summary()
        self.Decoder.summary()
        if trained:
            self.AutoEncoder.load_weights(filename)
            print("weights loaded")
    
    @classmethod
    
    def calc_MSE(self, y_target, y_predict):
        error = keras.add(tf.cast(y_target,tf.float32),- y_predict)
        return keras.mean(error**2)
    
    def calc_KL_divergence(self, y_target, y_predict): #inputs are exected by keras
        return -0.5 * kresa.sum(1 + self.log_sigma - self.mu - tf.exp(self.log_sigma))

# =============================================================================
#     def calc_comb_loss(self, y_target, y_predict):
#         return self.combined_loss_weight*self.calc_MSE(y_target, y_predict) + self.calc_KL_divergence(y_target, y_predict)
# =============================================================================

    def calc_comb_loss_dummy(self, a=2, b=2):
         def calc_comb_loss(self, y_target, y_predict):
             return self.combined_loss_weight*self.calc_MSE(y_target, y_predict) + self.calc_KL_divergence(y_target, y_predict)
         return calc_comb_loss


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
            
            
    
    
"""Sources:
    https://keras.io/examples/generative/vae/
    https://www.youtube.com/watch?v=A6mdOEPGM1E
"""
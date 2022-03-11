import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, models, Input
from stacked_mnist import StackedMNISTData, DataMode
import os
import matplotlib.pyplot as plt
from verification_net import VerificationNet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()
import keras.backend as K


#Architecture
   

class VAE:
    def __init__(self, latent_dim, filename, trained = True):
        self.n_dim = 28
        self.latent_dim = latent_dim
        self.filename = filename
        self.trained = trained
        
    
        """Encoder"""
        
        EncoderInput = Input(shape = (self.n_dim, self.n_dim, 1),name='EncoderInput')
        x = layers.Conv2D(16, (3,3),strides = 1, padding = "same",activation = "relu")(EncoderInput)#prevous layer output
        x = layers.Conv2D(16, (3,3),strides = 2, padding = "same",activation = "relu")(x)
        x = layers.Conv2D(16, (3,3),strides = 2, padding = "same",activation = "relu")(x)
        
        """Bottleneck of Encoder"""
        
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        mu = layers.Dense(self.latent_dim, name = "mu")(x)
        log_sigma = layers.Dense(self.latent_dim, name = "log_sigma")(x)
        def sample_from_gaussian(inputs):
            mu, log_sigma = inputs
            epsilon = tf.keras.backend.random_normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1])) #beckend got loads of distributions
            return mu + tf.exp(0.5 * log_sigma) * epsilon
            
        EncoderOutput = layers.Lambda(sample_from_gaussian, name = "EncoderOutput")([mu, log_sigma])#Lambdalayer allows function input
        Encoder = models.Model(EncoderInput, [EncoderOutput, mu, log_sigma], name='Encoder')
        
        
        """Decoder"""
        
        DecoderInput = Input(shape = (self.latent_dim, 1), name='DecoderInput')
        x = layers.Dense(7*7*16, activation = "relu")(DecoderInput)
        x = layers.Reshape((7, 7, 16*self.latent_dim))(x)
        x = layers.Conv2DTranspose(16, (3,3), strides = 2,padding = "same", activation = "relu")(x)
        x = layers.Conv2DTranspose(16, (3,3), strides = 2,padding = "same", activation = "relu")(x)
        DecoderOutput = layers.Conv2D(1, (3,3), strides = 1, padding = "same", activation = "sigmoid")(x)
        
        Decoder = models.Model(DecoderInput,DecoderOutput, name='Decoder')
        
        """Compiling Encoder + Decoder"""
        
        Encoder_output = Encoder.output[0]
        out = Decoder(Encoder_output)
        AutoEncoder = models.Model(Encoder.input, out, name='AutoEncoder')
        
        #loss = keras.losses.BinaryCrossentropy()
        
        
        self.optim = keras.optimizers.Adam(learning_rate = 0.001)
        #AutoEncoder.add_loss(self.calc_comb_loss)#This allows losses without ytarget,ypred
        AutoEncoder.compile(optimizer = self.optim,
                            loss = None,
                            metrics = ["accuracy", 
                                       tf.keras.metrics.mean_squared_error,
                                       "BinaryCrossentropy"])
            
        
        
        
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.AutoEncoder = AutoEncoder
        self.Encoder.summary()
        self.Decoder.summary()
        
        if trained:
            self.AutoEncoder.load_weights(filename)
            print("weights loaded")
        
    def train_manualy(self, train_images, val_images, batch_size = 1024, epochs = 5):
        if not self.trained:
            train_images = tf.cast(train_images, tf.float32)
            epochs = epochs*len(train_images)//batch_size
            for epoch in range(epochs):
                tf.random.shuffle(train_images)
                randm_batch = train_images[0:batch_size]
                self.train_step(randm_batch)
                if epoch%10 == 0:
                    print(f"Currently at epoch {epoch} of {epochs}")
            self.AutoEncoder.save_weights(self.filename)
            print("model is trained and weights saved")
        else:
            print("model is already trained")
    
    @tf.function

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            print(type(data))
            z, mu, sigma = self.Encoder(data, training = True)
            data_output = self.Decoder(z)
            BCE_func = tf.keras.losses.BinaryCrossentropy()
            BCE_loss = BCE_func(data, data_output)
            KL_loss = -0.5 * K.mean(1 + sigma - mu - K.exp(sigma),axis = 1)
            comb_loss = 10*BCE_loss + KL_loss
        gradients = tape.gradient(comb_loss, self.AutoEncoder.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.AutoEncoder.trainable_variables))


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
    
    
    
####Old architecture tries
class VAE1:
    def __init__(self, latent_dim, filename, trained = True):
        self.n_dim = 28
        self.latent_dim = latent_dim
        self.filename = filename
        self.trained = trained
        
    
        """Encoder"""
        
        EncoderInput = Input(shape = (self.n_dim, self.n_dim, 1),name='EncoderInput')
        x = layers.Conv2D(8, (3,3),strides = 1, padding = "same",activation = "relu")(EncoderInput)#prevous layer output
        x = layers.Conv2D(8, (3,3),strides = 2, padding = "same",activation = "relu")(x)
        x = layers.Conv2D(8, (3,3),strides = 2, padding = "same",activation = "relu")(x)
        
        """Bottleneck of Encoder"""
        
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        self.mu = layers.Dense(self.latent_dim, name = "mu")(x)
        self.log_sigma = layers.Dense(self.latent_dim, name = "log_sigma")(x)
        def sample_from_gaussian(inputs):
            mu, log_sigma = inputs
            epsilon = tf.keras.backend.random_normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1])) #beckend got loads of distributions
            return mu + tf.exp(0.5 * log_sigma) * epsilon
            
        EncoderOutput = layers.Lambda(sample_from_gaussian, name = "EncoderOutput")([self.mu, self.log_sigma])#Lambdalayer allows function input
        Encoder = models.Model(EncoderInput, EncoderOutput, name='Encoder')
        
        
        """Decoder"""
        
        DecoderInput = Input(shape = (self.latent_dim, 1), name='DecoderInput')
        x = layers.Dense(7*7*8, activation = "relu")(DecoderInput)
        x = layers.Reshape((7, 7, 8*self.latent_dim))(x)
        x = layers.Conv2DTranspose(8, (3,3), strides = 2,padding = "same", activation = "relu")(x)
        x = layers.Conv2DTranspose(8, (3,3), strides = 2,padding = "same", activation = "relu")(x)
        DecoderOutput = layers.Conv2D(1, (3,3), strides = 1, padding = "same", activation = "sigmoid")(x)
        
        Decoder = models.Model(DecoderInput,DecoderOutput, name='Decoder')
        
        """Compiling Encoder + Decoder"""
        
        Encoder_output = Encoder.output
        out = Decoder(Encoder_output)
        AutoEncoder = models.Model(Encoder.input, out, name='AutoEncoder')
        
        #loss = keras.losses.BinaryCrossentropy()
        
        def calc_loss(mu, log_sigma):
            
            def calcBinaryCrossEntropy(y_true, y_pred):
                y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())
                term_1 = y_true * K.log(y_pred + K.epsilon())
                return -K.sum(term_0 + term_1, axis=[1, 2, 3])
                
            def calcKL(mu, log_sigma): #inputs are exected by keras
                return -0.5 * K.sum(1 + log_sigma - mu - K.exp(log_sigma),axis=[1])
            
            def calccomb_loss(y_target, y_predict):
                BCE = calcBinaryCrossEntropy(y_target, y_predict)
                KL = calcKL(y_target, y_predict)
                totLoss = 1000*BCE + KL
                return totLoss
            
            return calccomb_loss
        
        self.wrapped_loss = calc_loss(self.mu, self.log_sigma)
        self.loss = self.calc_comb_loss
        self.optim = keras.optimizers.Adam(learning_rate = 0.001)
        #AutoEncoder.add_loss(self.calc_comb_loss)#This allows losses without ytarget,ypred
        AutoEncoder.compile(optimizer = self.optim,
                            loss = self.loss,
                            metrics = ["accuracy", 
                                       tf.keras.metrics.mean_squared_error,
                                       "BinaryCrossentropy"])
            
        
        
        
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
            
    def train_manualy(self, train_images, val_images, batch_size = 1024, epochs = 5):
        if not self.trained:
            train_images = tf.cast(train_images, tf.float32)
            epochs = epochs*len(train_images)//batch_size
            for epoch in range(epochs):
                tf.random.shuffle(train_images)
                randm_batch = train_images[0:batch_size]
                self.train_step(randm_batch)
                if epoch%10 == 0:
                    print(f"Currently at epoch {epoch} of {epochs}")
            self.AutoEncoder.save_weights(self.filename)
            print("model is trained and weights saved")
        else:
            print("model is already trained")
    
    @tf.function

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            print(type(data))
            z = self.Encoder(data, training = True)
            data_output = self.Decoder(z)
            BCE_func = tf.keras.losses.BinaryCrossentropy()
            comb_loss = BCE_func(data, data_output)
            print("type of tf.keras.losses.BinaryCrossentropy() is: ", type(comb_loss))
            comb_loss_kl = self.calc_comb_loss(data, data_output)
            print("type of real comb loss is: ", type(comb_loss))
            print("same type? : ", type(comb_loss) == type(comb_loss_kl))
        gradients = tape.gradient(comb_loss, self.AutoEncoder.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.AutoEncoder.trainable_variables))
    
    
    def calc_MSE(self, target, pred):
        error = tf.cast(target, tf.float32) - pred
        return tf.reduce_mean(tf.square(error), axis = [1,2,3])
    
    def calc_BinaryCrossEntropy(self, y_true, y_pred): 
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())
        term_1 = y_true * K.log(y_pred + K.epsilon())
        return -K.mean(term_0 + term_1, axis=0)
        
    def calc_KL(self, y_target, y_predict): #inputs are exected by keras
        return -0.5 * K.mean(1 + self.log_sigma - self.mu - K.exp(self.log_sigma),axis = 1)
    
    def calc_comb_loss(self, y_target, y_predict):
        #MSE = self.calc_MSE(y_target, y_predict)
        BCE_func = tf.keras.losses.BinaryCrossentropy()
        BCE = BCE_func(y_target, y_predict)
        
        #BCE = self.calc_BinaryCrossEntropy(y_target, y_predict)
        KL = self.calc_KL(y_target, y_predict)
        return BCE + KL
    
        
        

            
    
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
                
    
"""Sources:
    https://keras.io/examples/generative/vae/
    https://www.youtube.com/watch?v=A6mdOEPGM1E
    https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#train_the_model
    https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
"""
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
        x = layers.MaxPooling2D(pool_size = (2,2)))(x)
        x = layers.Conv2D(64, (3,3),strides = 1, padding = "same",activation = "relu")(x)
        x = layers.MaxPooling2D(pool_size = (2,2)))(x)
        
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
        
        #loss = keras.losses.BinaryCrossentropy()
        loss = self.calc_comb_loss
        optim = keras.optimizers.Adam(learning_rate = 0.01)
        #AutoEncoder.add_loss(self.calc_comb_loss)#This allows losses without ytarget,ypred
        AutoEncoder.compile(optimizer = optim)
            
        
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.AutoEncoder = AutoEncoder
        self.Encoder.summary()
        self.Decoder.summary()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        if trained:
            self.AutoEncoder.load_weights(filename)
            print("weights loaded")
            
    @property
    
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
      """Executes one training step and returns the loss.
    
      This function computes the loss and gradients, and uses the latter to
      update the model's parameters.
      """
      with tf.GradientTape() as tape:
          z = self.Encoder(data)
          data_output = self.Decoder(z)
          reconstruction_loss = tf.reduce_mean(
              tf.reduce_sum(
                  keras.losses.binary_crossentropy(data, reconstruction)
                  )
              )
          kl_loss = -0.5 * tf.reduce_mean(1 + self.log_sigma - self.mu - tf.exp(self.log_sigma))
          kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
          combined_loss = reconstruction_loss + kl_loss
      gradients = tape.gradient(combined_loss, model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.kl_loss_tracker.update_state(kl_loss)
      return {
          "loss": self.total_loss_tracker.result(),
          "reconstruction_loss": self.reconstruction_loss_tracker.result(),
          "kl_loss": self.kl_loss_tracker.result(),
      }
    
    @classmethod
    
    
# =============================================================================
#     def calc_MSE(self, y_target, y_predict):
#         error = keras.add(tf.cast(y_target,tf.float32),- y_predict)
#         return keras.mean(error**2)
# =============================================================================
    
    def calc_KL_divergence(self, y_target, y_predict): #inputs are exected by keras
        return -0.5 * tf.reduce_mean(1 + self.log_sigma - self.mu - tf.exp(self.log_sigma))

    def calc_comb_loss(self, y_target, y_predict):
        return self.calc_KL_divergence(y_target, y_predict)
        return self.combined_loss_weight*keras.losses.BinaryCrossentropy(y_target, y_predict, logits = True) + self.calc_KL_divergence(y_target, y_predict)
    
    def loss(self, model, x, y, training):
      # training=training is needed only if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      y_ = model(x, training=training)
      return calc_comb_loss(y_true=y, y_pred=y_)
  
    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


# =============================================================================
#     def calc_comb_loss_dummy(self, a=2, b=2):
#          def calc_comb_loss(self, y_target, y_predict):
#              return self.combined_loss_weight*self.calc_MSE(y_target, y_predict) + self.calc_KL_divergence(y_target, y_predict)
#          return calc_comb_loss
# =============================================================================


    def train(self, train_images, val_images, batch_size = 1024, epochs = 5):
        if not self.trained:
            self.train_step(self.AutoEncoder, train_images , batch_size, epochs)
            self.AutoEncoder.save_weights(self.filename)
            print("model is trained and weights saved")
        else:
            print("model is already trained")
            
    def train_step(self, model, train_images, batch_size, epochs):
        train_loss_results = []
        train_accuracy_results = []
        N = len(train_images)
        rest = N%batch_size
        train_images = train_images[:-rest]
        num_epochs = 201
    
        for epoch in range(num_epochs):
          epoch_loss_avg = tf.keras.metrics.Mean()
          epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
          # Training loop - using batches of 32
          for x in np.split(train_images,batch_size):
            # Optimize the model
            loss_value, grads = self.grad(model, x, x)
            optimizer.apply_gradients(zip(self.grads, model.trainable_variables))
        
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(x, model(x, training=True))
        
          # End epoch
          train_loss_results.append(epoch_loss_avg.result())
          train_accuracy_results.append(epoch_accuracy.result())
    
          if epoch % 50 == 0:
              print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

            
            
    
    
"""Sources:
    https://keras.io/examples/generative/vae/
    https://www.youtube.com/watch?v=A6mdOEPGM1E
    https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#train_the_model
"""
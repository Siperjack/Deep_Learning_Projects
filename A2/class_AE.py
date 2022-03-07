# =============================================================================
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# 
# =============================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, models
from stacked_mnist import StackedMNISTData, DataMode
import os
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#%%Hyperparameters
n_dim = 28
C = 64
latent_dim = 4
Nchannels = 1
epoch = 5

def AutoEncoder():
    def __init__(self, force_learn: bool = False, file_name: str = "./models/unspecified") -> None:
        self.force_relearn = force_learn
        self.file_name = file_name
        #Encoder
        Encoder = models.Sequential(name='Encoder')
        
        #Block 1
        Encoder.add(layers.Conv2D(64, (5,5),strides = 2, padding = "same", activation = "relu", input_shape = (n_dim, n_dim, Nchannels)))
        Encoder.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 14
        
        #Block 2
        Encoder.add(layers.Conv2D(32, (3,3),strides = 2, padding = "same",activation = "relu"))
        
        #Block 3
        Encoder.add(layers.Flatten())
        Encoder.add(layers.Dense(latent_dim))
        Encoder.add(layers.Normalization())
        
        self.Encoder = Encoder
        #Decoder
        Decoder = models.Sequential(name='Decoder')
        #Block 4
        Decoder.add(layers.Dense(1568, activation = "relu", input_shape = (latent_dim, Nchannels)))
        
        #Block 5
        Decoder.add(layers.Reshape((7, 7, 32*latent_dim)))
        
        #Block 6
        Decoder.add(layers.Conv2DTranspose(16, (4,4), strides = 2,padding = "same", activation = "relu"))
        
        #Block 7
        Decoder.add(layers.Conv2DTranspose(Nchannels, (4,4), strides = 2,padding = "same", activation = "sigmoid"))
        self.Decoder = Decoder
        #Summary
        #Encoder.summary()
        #
        #Decoder.summary()
        #Optimizers and loss
        Encoder_output = Encoder.output
        out = Decoder(Encoder_output)
        self.AutoEncoder = models.Model(Encoder.input, out, name='AutoEncoder')
        #AutoEncoder.summary()
        
        loss = keras.losses.BinaryCrossentropy()
        optim = keras.optimizers.Adam(learning_rate = 0.01)
        self.AutoEncoder.compile(optimizer = optim, loss=loss, metrics = "accuracy")
    
        self.done_training = self.load_weights()

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.AutoEncoder.load_weights(filepath=self.file_name)
            # print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(f"Could not read weights for verification_net from file. Must retrain...")
            AutoEcoder.save_weights(self.file_name)
            done_training = False

        return done_training


    def train(self, generator: StackedMNISTData, epochs: int = 10, OwnData = False) -> bool:
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        self.done_training = self.load_weights()

        if self.force_relearn or self.done_training is False:
            if Nchannels == 1:
                data = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
            else:
                data = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
            train_images, train_labels = data.get_full_data_set(training = True)
            val_images, val_labels = data.get_full_data_set(training = False)
            self.AutoEncoder.fit(
                train_images,
                train_images,#label
                epochs = epoch,
                shuffle = True,
                batch_size = C,
                validation_data=(val_images, val_images)
                )


            # Fit model
            self.AutoEncoder.fit(x=x_train, y=y_train, batch_size=1024, epochs=epochs,
                           validation_data=(x_test, y_test))

            # Save weights and leave
            self.AutoEncoder.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def predict(self, data: np.ndarray) -> tuple:
        """
        Predict the classes of some specific data-set. This is basically prediction using keras, but
        this method is supporting multi-channel inputs.
        Since the model is defined for one-channel inputs, we will here do one channel at the time.

        The rule here is that channel 0 define the "ones", channel 1 defines the tens, and channel 2
        defines the hundreds.

        Since we later need to know what the "strength of conviction" for each class-assessment we will
        return both classifications and the belief of the class.
        For multi-channel images, the belief is simply defined as the probability of the allocated class
        for each channel, multiplied.
        """
        no_channels = data.shape[-1]

        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError("Model is not trained, so makes no sense to try to use it")

        predictions = np.zeros((data.shape[0],))
        beliefs = np.ones((data.shape[0],))
        for channel in range(no_channels):
            channel_prediction = self.AutoEncoder.predict(data[:, :, :, [channel]])
            beliefs = np.multiply(beliefs, np.max(channel_prediction, axis=1))
            predictions += np.argmax(channel_prediction, axis=1) * np.power(10, channel)

        return predictions, beliefs

    def check_class_coverage(self, data: np.ndarray, tolerance: float = .8) -> float:
        """
        Out of the total number of classes that can be generated, how many are in the data-set?
        I'll only could samples for which the network asserts there is at least tolerance probability
        for a given class.
        """
        no_classes_available = np.power(10, data.shape[-1])
        predictions, beliefs = self.predict(data=data)

        # Only keep predictions where all channels were legal
        predictions = predictions[beliefs >= tolerance]

        # Coverage: Fraction of possible classes that were seen
        coverage = float(len(np.unique(predictions))) / no_classes_available
        return coverage

    def check_predictability(self, data: np.ndarray,
                             correct_labels: list = None,
                             tolerance: float = .8) -> tuple:
        """
        Out of the number of data points retrieved, how many are we able to make predictions about?
        ... and do we guess right??

        Inputs here are
        - data samples -- size (N, 28, 28, color-channels)
        - correct labels -- if we have them. List of N integers
        - tolerance: Minimum level of "confidence" for us to make a guess

        """
        # Get predictions; only keep those where all channels were "confident enough"
        predictions, beliefs = self.predict(data=data)
        predictions = predictions[beliefs >= tolerance]
        predictability = len(predictions) / len(data)

        if correct_labels is not None:
            # Drop those that were below threshold
            correct_labels = correct_labels[beliefs >= tolerance]
            accuracy = np.sum(predictions == correct_labels) / len(data)
        else:
            accuracy = None

        return predictability, accuracy

#%%
AE = VerificationNet(force_learn= True, file_name = "./models/BinAE")
#%%
reconstructed_images = Autoencoder.predict(val_images, batch_size = 1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

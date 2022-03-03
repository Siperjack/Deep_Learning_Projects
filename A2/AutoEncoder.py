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

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#%%Hyperparameters
n_dim = 28
C = 1
latent_dim = 2
Nchannels = 1

#%%Encoder
model = models.Sequential()

#Block 1
model.add(layers.Conv2D(32, (3,3),padding = "same", activation = "relu", input_shape = (n_dim, n_dim, Nchannels)))
model.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 14

#Block 2
model.add(layers.Conv2D(32, (3,3),padding = "same",activation = "relu"))
model.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 7

#Block 3
model.add(layers.Flatten())
model.add(layers.Dense(latent_dim))

#%%Decoder

#Block 4
model.add(layers.Dense(1568, activation = "relu"))

#Block 5
model.add(layers.Reshape((7, 7, 32)))

#Block 6
model.add(layers.Conv2DTranspose(32, (4,4),strides = 2, padding = "same", activation = "relu"))

#Block 7
model.add(layers.Conv2DTranspose(32, (4,4),strides = 2,padding = "same", activation = "sigmoid"))

#%%Optimizers and loss
loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optim, loss=loss, metrics = "accuracy")
Autoencoder = model
print(Autoencoder.summary())
#evaluate 

#%%Training

Autoencoder.fit(
    training,
    training,#label
    epochs = epochs,
    batch_size = C,
    validationdata=(testing,testing)
    )

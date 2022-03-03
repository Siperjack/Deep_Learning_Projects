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
model = models.Sequential()
model.add(layers.Conv2D(28, (3,3),activation = "relu", input_shape = (28, 28, 3)))
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Conv2D(28, (3,3),activation = "relu"))
model.add(layers.MaxPooling2D(pool_size = (2,2)))
model.add(layers.Flatten())
model.add(layers.Dense())
loss = keras.losses.CategoricalCrossentropy()
optim = keras.optimizers.Adam(learning_rate = 0.001)
print(model.summary())

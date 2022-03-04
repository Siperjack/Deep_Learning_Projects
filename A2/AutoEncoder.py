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
from verification_net import VerificationNet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#%%Hyperparameters
n_dim = 28
C = 64
latent_dim = 4
Nchannels = 1
epoch = 5


#%%Encoder
model = models.Sequential()

#Block 1
model.add(layers.Conv2D(32, (3,3),strides = 2, padding = "same", activation = "relu", input_shape = (n_dim, n_dim, Nchannels)))
model.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 14

#Block 2
model.add(layers.Conv2D(32, (3,3),strides = 2, padding = "same",activation = "relu"))
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
model.add(layers.Conv2DTranspose(32, (4,4), strides = 2,padding = "same", activation = "relu"))

#Block 7
model.add(layers.Conv2DTranspose(Nchannels, (4,4), strides = 2,padding = "same", activation = "sigmoid"))

#%%Optimizers and loss
loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer = optim, loss=loss, metrics = "accuracy")
Autoencoder = model
#print(Autoencoder.summary())
#evaluate 

    #%%Training
if Nchannels == 1:
    data = StackedMNISTData(mode=stacked_mnist.DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
else:
    data = StackedMNISTData(mode=stacked_mnist.DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
train_images, train_labels = data.get_full_data_set(training = True)
val_images, val_labels = data.get_full_data_set(training = False)
Autoencoder.fit(
    train_images,
    train_images,#label
    epochs = epoch,
    shuffle = True,
    batch_size = C,
    validation_data=(val_images, val_images)
    )

        
#%%
reconstructed_images = Autoencoder.predict(train_labels)
#%%
n = 8
fig, ax = plt.subplots(2,n)
for i in range(n):
    plt.gray()
    ax[0,i].imshow(val_images[i].reshape((n_dim, n_dim)))
    ax[1,i].imshow(reconstructed_images[i].reshape((n_dim, n_dim)))
fig, ax = plt.subplots(2,n)
for i in range(n):
    plt.gray()
    ax[0,i].imshow(val_images[n+i].reshape((n_dim, n_dim)))
    ax[1,i].imshow(reconstructed_images[n+i].reshape((n_dim, n_dim)))        
        
    

#%%Generative model
z = np.random.randn(10, latent_dim)

###These shll be insertet in the middle of the autoencoder(in the input to the decoder part)
#%%
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)


cov = net.check_class_coverage(data=reconstructed_images, tolerance=.8)
pred, acc = net.check_predictability(data=reconstructed_images, correct_labels=train_labels)
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")    
    
    
#%%
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)

# I have no data generator (VAE or whatever) here, so just use a sampled set
img, labels = gen.get_random_batch(training=True,  batch_size=25000)
img, labels = gen.get_random_batch(training=True,  batch_size=25000)
cov = net.check_class_coverage(data=img, tolerance=.98)
pred, acc = net.check_predictability(data=img, correct_labels=labels)
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

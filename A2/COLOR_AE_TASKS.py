import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, models
from stacked_mnist import StackedMNISTData, DataMode
import os
import matplotlib.pyplot as plt
from verification_net import VerificationNet
from class_AE import AE
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#%%Global constants
n_dim = 28
latent_dim = 2
Nchannels = 1
trained = False
epochs = 5
#%% Loading
AE_COLOR_BINARY_COMPLETE = AE(trained = trained, filename = "./models/COLOR_BINARY_COMPLETE")
AutoEncoder = AE_COLOR_BINARY_COMPLETE.AutoEncoder

data = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
train_images, train_labels = data.get_full_data_set(training = True)
val_images, val_labels = data.get_full_data_set(training = False)
AE_COLOR_BINARY_COMPLETE.train(train_images, val_images, epochs = epochs)

#%% Make reconstructions
reconstructed_images = AutoEncoder.predict(val_images)
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
Ngen = 2048*4
z = np.random.uniform(0,1, (Ngen, latent_dim, Nchannels))

uniform_generated = AE_COLOR_BINARY_COMPLETE.Decoder.predict(z)

n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        plt.gray()
        ax[i,j].imshow(uniform_generated[i + n*j].reshape((n_dim, n_dim)))

#%% Running verification net on reconstructed images
gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)


cov = net.check_class_coverage(data=reconstructed_images, tolerance=.8)
pred, acc = net.check_predictability(data=reconstructed_images, correct_labels=val_labels)
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")    
    
#%%Anomalies

AE_COLOR_BINARY_MISSING = AE(trained = trained, filename = "./models/COLOR_BINARY_MISSING")
AutoEncoder_missing = AE_COLOR_BINARY_MISSING.AutoEncoder


data_missing = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)
train_images_missing, train_labels_missing = data.get_full_data_set(training = True)
val_images_missing, val_labels_missing = data.get_full_data_set(training = False)

AE_COLOR_BINARY_MISSING.train(train_images_missing, val_images_missing,epochs = epochs)
#%%
results = AutoEncoder_missing.predict(val_images)
#%%

losses = tf.reduce_mean(results - val_images, axis = [1,2,3])
losses_indexes = np.argsort(losses)

#%% Plot example reconstructed with highest errors

n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        plt.gray()
        ax[i,j].imshow(val_images[losses_indexes[i + n*j]].reshape((n_dim, n_dim)))
#%% Stacked generating
Ngen = 2048*4
z = np.random.uniform(0,1, (Ngen, latent_dim, Nchannels))

uniform_generated = AE_COLOR_BINARY_COMPLETE.Decoder.predict(z)

n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        plt.gray()
        ax[i,j].imshow(uniform_generated[i + n*j].reshape((n_dim, n_dim)))
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
import keras.backend as K

#%%"""Global constants"""
"""Global constants"""
n_dim = 28
latent_dim = 2
Nchannels = 1
trained = True
epochs = 25

debug_mode = True
#%% """Loading complete model and train if not trained"""
"""Loading complete model and train if not trained"""
#untrainedAE_MONO_BINARY_COMPLETE = AE(latent_dim = 4, filename = "./models/MONO_BINARY_COMPLETE_BCE_latent4", trained = trained)#BCE = BinaryCrossentropy
AE_MONO_BINARY_COMPLETE = AE(latent_dim = 2, filename = "./models/MONO_BINARY_COMPLETE_BCE_latent2", trained = trained)
#AE_MONO_BINARY_COMPLETE = AE(latent_dim = 16, filename = "./models/MONO_BINARY_COMPLETE_BCE_latent16", trained = trained)
data = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)

if debug_mode:
    train_images, train_labels = data.get_random_batch (training = True)
    val_images, val_labels = data.get_random_batch (training = False)
else:
    train_images, train_labels = data.get_full_data_set(training = True)
    val_images, val_labels = data.get_full_data_set(training = False)

AE_MONO_BINARY_COMPLETE.train(train_images, val_images, epochs = epochs)

#%%"""Loading missing model and train if not trained"""
"""Loading missing model and train if not trained"""
epochs = epochs//2
#untraindAE_MONO_BINARY_MISSING = AE(latent_dim = 4, trained = trained, filename = "./models/MONO_BINARY_MISSING_BCE_latent4")
AE_MONO_BINARY_MISSING = AE(latent_dim = 2, trained = trained, filename = "./models/MONO_BINARY_MISSING_BCE_latent2")
#AE_MONO_BINARY_MISSING = AE(latent_dim = 16, trained = trained, filename = "./models/MONO_BINARY_MISSING_BCE_latent16")

data_missing = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
if debug_mode:
    train_images_missing, train_labels_missing = data_missing.get_random_batch (training = True)
    val_images_missing, val_labels_missing = data_missing.get_random_batch (training = False)
else:
    train_images_missing, train_labels_missing = data_missing.get_full_data_set(training = True)
    val_images_missing, val_labels_missing = data_missing.get_full_data_set(training = False)

AE_MONO_BINARY_MISSING.train(train_images_missing, val_images_missing,epochs = epochs)
#%% """Make reconstructions and plot"""
"""Make reconstructions and plot"""
reconstructed_images = AE_MONO_BINARY_COMPLETE.predict(val_images)
n = 8
fig, ax = plt.subplots(4,n)
for i in range(n):
    ax[0,i].imshow(val_images[i], cmap='gray')
    ax[1,i].imshow(reconstructed_images[i].reshape((n_dim,n_dim)), cmap='gray') 
    ax[2,i].imshow(val_images[n+i], cmap='gray')
    ax[3,i].imshow(reconstructed_images[n+i].reshape((n_dim,n_dim)), cmap='gray')         
        
#%% """Running verification net on reconstructed images"""
"""Running verification net on reconstructed images"""
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)


cov = net.check_class_coverage(data=reconstructed_images, tolerance=.8)
pred, acc = net.check_predictability(data=reconstructed_images, correct_labels=val_labels)
print("""Running verification net on reconstructed images""")
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")   

#%%"""Generate images from randn and plot"""
"""Generate images from randn and plot"""
if debug_mode:
    Ngen = 1000
else:
    Ngen = 10000
#z = np.random.uniform(0,1, (Ngen, latent_dim, Nchannels))
z = np.random.randn(Ngen, latent_dim, Nchannels)

generated = AE_MONO_BINARY_COMPLETE.predict_from_latent(z)

n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        ax[i,j].imshow(generated[i + n*j].reshape((n_dim, n_dim)),cmap='gray')

#%% """Running verification net on generated images"""
"""Running verification net on generated images"""
cov = net.check_class_coverage(data=generated, tolerance=.8)
pred, acc = net.check_predictability(data=reconstructed_images, tolerance=.8)

print("""Running verification net on generated images""")
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
#print(f"Accuracy: {100 * acc:.2f}%")    
    
#%%"""Plot anomalies for MONO"""
"""Plot anomalies for MONO"""
results = AE_MONO_BINARY_MISSING.predict(val_images)

losses = K.mean(K.square(results - tf.cast(val_images, tf.float32)), axis = [1,2,3])
losses_indexes = tf.argsort(losses)[::-1]

#Plot example reconstructed with highest errors

n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        ax[i,j].imshow(val_images[losses_indexes[i + n*j]].reshape((n_dim, n_dim)),cmap='gray')
        
        
#%% """Make stacked reconstructions and plot"""
"""Make stacked reconstructions and plot"""
data_color = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
if debug_mode:
    train_images_color, train_labels_color = data_color.get_random_batch (training = True)
    val_images_color, val_labels_color = data_color.get_random_batch (training = False)
else:
    train_images_color, train_labels_color = data_color.get_full_data_set(training = True)
    val_images_color, val_labels_color = data_color.get_full_data_set(training = False)    
n = 8
dataset_size = len(val_images_color)
reconstructed_color = np.zeros((dataset_size, 28, 28, 3))

reconstructed_color = AE_MONO_BINARY_COMPLETE.predict(val_images_color)
fig, ax = plt.subplots(2,n)
for i in range(n):
    ax[0,i].imshow(val_images_color[i,:,:,:]*255)
    ax[1,i].imshow(reconstructed_color[i,:,:,:])
    
#%% """Running verification net on reconstructed color"""
"""Running verification net on reconstructed color"""

cov = net.check_class_coverage(data=reconstructed_color, tolerance=.5)
pred, acc = net.check_predictability(data=reconstructed_color, correct_labels=val_labels_color)
print("""Running verification net on reconstructed color""")
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")   
#%% """Generate stacked images from randint and plot"""
"""Generate stacked images from randint and plot"""
Nchannels = 3

#z_stacked = np.random.uniform(0,1, (Ngen, latent_dim, Nchannels))
z_stacked = np.random.randn(Ngen, latent_dim, Nchannels)

generated_stacked = AE_MONO_BINARY_COMPLETE.predict_from_latent(z_stacked)
n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        ax[i,j].imshow(generated_stacked[i + n*j])
        
#%% """Running verification net on generated color"""
"""Running verification net on generated color"""

cov = net.check_class_coverage(data=generated_stacked, tolerance=.5)
pred, acc = net.check_predictability(data=generated_stacked, tolerance = .5)

print("""Running verification net on generated color""")
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
#print(f"Accuracy: {100 * acc:.2f}%")   


#%%"""Plot anomalies for STACKED"""
"""Plot anomalies for STACKED"""
results_stacked = AE_MONO_BINARY_MISSING.predict(val_images_color)
losses_stacked = tf.reduce_mean(K.square(results_stacked - val_images_color), axis = [1,2,3])
losses_stacked_indexes = tf.argsort(losses_stacked)[::-1]

#Plot example reconstructed with highest errors
n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        ax[i,j].imshow(val_images_color[losses_stacked_indexes[i + n*j]]*255)

#%%      
        




        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

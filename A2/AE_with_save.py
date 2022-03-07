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
epoch = 1
   
class AE:
    def __init__(self):
        #Encoder
        model = models.Sequential(name='Encoder')
        
        #Block 1
        model.add(layers.Conv2D(32, (3,3),strides = 1, padding = "same", activation = "relu", input_shape = (n_dim, n_dim, Nchannels)))
        model.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 14
        model.add(layers.Dropout(0.25))
        
        #Block 2
        model.add(layers.Conv2D(64, (3,3),strides = 1, padding = "same",activation = "relu"))
        model.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 14
        model.add(layers.Dropout(0.25))
        
        #Block 3
        model.add(layers.Conv2D(64, (3,3),strides = 1, padding = "same",activation = "relu"))
        model.add(layers.MaxPooling2D(pool_size = (2,2))) #dim 14
        model.add(layers.Dropout(0.25))
        
        #Block 4
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(latent_dim))
        model.add(layers.Normalization())
        
        Encoder = model
        Decoder = models.Sequential(name='Decoder')
        #Block 4
        Decoder.add(layers.Dense(1568, activation = "relu", input_shape = (latent_dim, Nchannels)))
        
        #Block 5
        Decoder.add(layers.Reshape((7, 7, 32*latent_dim)))
        
        #Block 6
        Decoder.add(layers.Conv2DTranspose(32, (4,4), strides = 2,padding = "same", activation = "relu"))
        Decoder.add(layers.Conv2DTranspose(64, (4,4), strides = 1,padding = "same", activation = "relu"))
        #Block 7
        Decoder.add(layers.Conv2DTranspose(Nchannels, (4,4), strides = 2,padding = "same", activation = "sigmoid"))
        Encoder_output = Encoder.output
        out = Decoder(Encoder_output)
        AutoEncoder = models.Model(Encoder.input, out, name='AutoEncoder')
        #AutoEncoder.summary()
        
        loss = keras.losses.BinaryCrossentropy()
        optim = keras.optimizers.Adam(learning_rate = 0.01)
        AutoEncoder.compile(optimizer = optim, loss=loss, metrics = "accuracy")
        #evaluate 
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.AutoEncoder = AutoEncoder
        AutoEncoder.summary()

    #%%Training
    
def train(model, train_images, val_images, filename):
    model.fit(
        train_images,
        train_images,#label
        epochs = epoch,
        shuffle = True,
        batch_size = C,
        validation_data=(val_images, val_images)
        )
    
    model.save_weights(filename)

        
#%% Loading
AE_MONO_BINARY_COMPLETE = AE()
AutoEncoder = AE_MONO_BINARY_COMPLETE.AutoEncoder
AutoEncoder.load_weights("./models/MONO_BINARY_COMPLETE")

#%%
data = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
train_images, train_labels = data.get_full_data_set(training = True)
val_images, val_labels = data.get_full_data_set(training = False)

#%%
#train(AutoEncoder, train_images, val_images, filename = "./models/MONO_BINARY_COMPLETE")
reconstructed_images = AutoEncoder.predict(val_images)
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
Ngen = 2048*4
z = np.random.uniform(0,1, (Ngen, latent_dim, Nchannels))

uniform_generated = AE_MONO_BINARY_COMPLETE.Decoder.predict(z)

n = 5
fig, ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        plt.gray()
        ax[i,j].imshow(uniform_generated[i + n*j].reshape((n_dim, n_dim)))

#%% Running verification net on reconstructed images
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)


cov = net.check_class_coverage(data=reconstructed_images, tolerance=.8)
pred, acc = net.check_predictability(data=reconstructed_images, correct_labels=val_labels)
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")    
    
#%%Anomalies

AE_MONO_BINARY_MISSING = AE()
AutoEncoder_missing = AE_MONO_BINARY_MISSING.AutoEncoder
AutoEncoder.load_weights("./models/MONO_BINARY_MISSING")


data_missing = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
train_images_missing, train_labels_missing = data.get_full_data_set(training = True)
val_images_missing, val_labels_missing = data.get_full_data_set(training = False)

#train(AutoEncoder_missing, train_images_missing, val_images_missing, filename = "./models/MONO_BINARY_MISSING")

results = AutoEncoder_missing.predict(val_images)
#%%
scce = tf.keras.losses.BinaryCrossentropy()
print(np.shape(results))
loss = scce(val_images[:], results[:]).numpy()
print(np.shape(loss))





        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

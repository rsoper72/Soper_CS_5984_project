from keras.models import Model
# train autoencoder model for unsupervised clustering of bot and human response dialogs

from keras.layers import Input, Dense
import pickle
import numpy as np

# load training data
file_name = 'data/auto_x_tr4.pkl'
file_obj = open(file_name,'rb') 
auto_x = pickle.load(file_obj)   
file_obj.close()

# set training parameters
batch_size = 64
epochs = 8

# encoder architecture
input_data = Input(shape=(6050,))
encoded = Dense(512, activation='relu')(input_data)
encoded = Dense(32, activation='relu')(encoded)

# decoder architecture
decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(6050, activation='sigmoid')(decoded)

# model assignments
encoder = Model(input_data, encoded)
autoencoder = Model(input_data, decoded)

# use SGD with Adam optimization and MSE loss
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

import time

start = time.time()

# train autoencoder
autoencoder.fit(auto_x, auto_x,
                batch_size=batch_size, epochs=epochs, validation_split = 0.2)

end = time.time()
elaps = end-start

autoencoder.save('data/autoenc2.h5')

print('Elapsed time for ',epochs,' epochs: ',elaps)
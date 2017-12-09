# train seq2seq RNN for chatbot dialog response : GLoVe encoded word vec case 

from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
import pickle
import numpy as np

# load training data
file_name = 'data/x_tr_r_vec4.pkl'  # encoder input reversed per Sutskever et al., "Sequence to sequence learning with neural networks"
file_obj = open(file_name,'rb') 
x_vec = pickle.load(file_obj)   
file_obj.close()
file_name = 'data/y_tr_in_vec4.pkl'
file_obj = open(file_name,'rb') 
y_in_vec = pickle.load(file_obj)   
file_obj.close()
file_name = 'data/y_tr_tar_vec4.pkl'
file_obj = open(file_name,'rb') 
y_tar_vec = pickle.load(file_obj)   
file_obj.close()
file_name = 'data/wrd_vec4.pkl'
file_obj = open(file_name,'rb') 
wrd_vec = pickle.load(file_obj)   
file_obj.close()

# 50 elements per word for GLoVe encoding 
num_encoder_tokens = 50
num_decoder_tokens = 50

encoder_input_data = np.asarray(x_vec, dtype=float)
decoder_input_data = np.asarray(y_in_vec, dtype=float)
decoder_target_data = np.asarray(y_tar_vec, dtype=float)

# 60 words max per sequence (decoder has extra start/stop word for input/target)
max_encoder_seq_length = 60
max_decoder_seq_length = 61

# model/training parameters
batch_size = 64
epochs = 50
latent_dim = 512

print('Number of samples:', len(encoder_input_data))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('Batch size:',batch_size)
print('Number of epochs:',epochs)
print('Hidden layer size:',latent_dim)

input_token_index = wrd_vec
target_token_index = wrd_vec

# Encoder architecture
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]  # retain encoder state for decoder input

# Decoder architecture
decoder_inputs = Input(shape=(None, num_decoder_tokens))  
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)  # conditioned on encoder state
decoder_dense = Dense(num_decoder_tokens, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)

# Assign seq2seq architecture to model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

import time
start = time.time()

# Training by SGD with Adam optimization and MSE loss
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
		  
end = time.time()
		  
# Save model
model.save('chat_4_glv.h5')

elap = end-start
print('Time lapse for',epochs,'epochs:',elap)
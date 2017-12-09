# train "Turing" seq2seq chatbot detector

from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
import pickle
import numpy as np

# text file echos screen reporting
f_res = open('turing_res1.txt','w')

# load training data
file_name = 'data/x_r_full_tur4_shuf.pkl'
file_obj = open(file_name,'rb') 
x_vec = pickle.load(file_obj)   
file_obj.close()
file_name = 'data/y_in_tur4_shuf.pkl'
file_obj = open(file_name,'rb') 
y_in_vec = pickle.load(file_obj)   
file_obj.close()
file_name = 'data/y_tar_tur4_shuf.pkl'
file_obj = open(file_name,'rb') 
y_tar_vec = pickle.load(file_obj)   
file_obj.close()
file_name = 'data/wrd_vec4.pkl'
file_obj = open(file_name,'rb') 
wrd_vec = pickle.load(file_obj)   
file_obj.close()

# GLoVe word vector length 50
num_encoder_tokens = 50
num_decoder_tokens = 50

encoder_input_data = np.asarray(x_vec[:4000], dtype=float)
decoder_input_data = np.asarray(y_in_vec[:4000], dtype=float)
decoder_target_data = np.asarray(y_tar_vec[:4000], dtype=float)

# max words / sequence = 60 (decoder has stop word ending input sequence)
max_encoder_seq_length = 60
max_decoder_seq_length = 61

# model/training parameters
batch_size = 64
epochs = 10
latent_dim = 128

print('Number of samples:', len(encoder_input_data))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('Batch size:',batch_size)
print('Number of epochs:',epochs)
print('Hidden layer size:',latent_dim)

f_res.write('Number of samples: '+str(len(encoder_input_data))+'\n')
f_res.write('Number of unique input tokens: '+str(num_encoder_tokens)+'\n')
f_res.write('Number of unique output tokens: '+str(num_decoder_tokens)+'\n')
f_res.write('Max sequence length for inputs: '+str(max_encoder_seq_length)+'\n')
f_res.write('Max sequence length for outputs: '+str(max_decoder_seq_length)+'\n')
f_res.write('Batch size: '+str(batch_size)+'\n')
f_res.write('Number of epochs: '+str(epochs)+'\n')
f_res.write('Hidden layer size: '+str(latent_dim)+'\n')

input_token_index = wrd_vec
target_token_index = {'hum':[0,1], 'bot':[1,0]}

# seq2seq encoder architecture
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]  # retain final state to pass to decoder

# seq2seq decoder architecture
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=False, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(2, activation='softmax')  # 2-bit output for human vs. bot dialog ID
decoder_outputs = decoder_dense(decoder_outputs)

# assign architecture to model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

import time
start = time.time()

# SGD training with Adam optimization and cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

end = time.time()
		  
# Save model
model.save('turing_detect_2.h5')

elap = end-start
print('Time lapse for',epochs,'epochs:',elap)

f_res.write('Time lapse for '+str(epochs)+' epochs: '+str(elap)+'\n')


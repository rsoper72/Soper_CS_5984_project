# decode GLoVe encoded chatbot output

import numpy as np
import pickle
import math

# load corpus dictionary
f_name = 'data/wrd_vec4.pkl'
f_obj = open(f_name, 'rb')
wrd_vec = pickle.load(f_obj)
f_obj.close()

# construct token to vector and token to work mappings
tok_vec = {}
tok_word = {}
for (i, word) in enumerate(wrd_vec):
    tok_vec[i] = wrd_vec[word]
    tok_word[i] = word
    if word == '*stop*':
        stop_tok =i
		
f_name = 'data/y_ts_tar_pred_vec4_new.pkl'
f_obj = open(f_name, 'rb')
y_pred = pickle.load(f_obj)
f_obj.close()

y_pred = np.asarray(y_pred)
y_pred.shape

# helper function: determine token based on cosine distance
def tokenize(vec):
    cos_dist = []
    a_sq = 0
    for j in range(50):
        a_sq += vec[j]*vec[j]
    a_sqrt = math.sqrt(a_sq)
    for i in tok_vec:
        vec_i = tok_vec[i]          
        b_sq = 0
        ab = 0
        for j in range(50):
            b_sq += vec_i[j]*vec_i[j]
            ab += vec[j]*vec_i[j]
        cos_dist.append(ab/a_sqrt/math.sqrt(b_sq))
    return np.argmax(cos_dist)

# vector -> token
y_pred_t = []
for i in range(10):
    y_toks = []
    for j in range(61):
        cur_tok = tokenize(y_pred[i][j])
        if cur_tok == stop_tok:
            break
        else:
            y_toks.append(cur_tok)
    y_pred_t.append(y_toks)

# token -> word
y_pred_w = []
for i in range(len(y_pred_t)):
    pred_w = []
    for j in range(len(y_pred_t[i])):
        pred_w.append(tok_word[y_pred_t[i][j]])
    y_pred_w.append(pred_w)

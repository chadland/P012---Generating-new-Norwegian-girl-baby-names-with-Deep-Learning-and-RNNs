# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:58:49 2017

@author: christerhadland
"""

#Imports
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import urllib
from tensorflow.models.rnn.ptb import reader
import pandas as pd
import io
import requests
import random 
import string

#List tensorflow version
print(tf.__version__)

#Load data an functions
# -*- coding: utf-8 -*-
url = "https://www.dropbox.com/s/rxe3vsvdt03jtvi/2017%20-%2001%20-%2009%20-%20SSBJentenavn20062015V2.csv?dl=1"  # dl=1 is important
import urllib
import csv
import codecs
import sys  
#sys.setdefaultencoding('utf8')

#Download file
u = urllib.urlopen(url)
data = u.read()
u.close()
#print (u.headers.getparam("charset"))

#Save csv to disc 
with open(os.path.expanduser('~/SSBJentenavn20062015,csv'), "wb") as f :
    f.write(data)
f.close()

import os
cwd = os.getcwd()

#Convert data to dataset 
df = pd.read_csv(os.path.expanduser('~/SSBJentenavn20062015,csv'),encoding ='iso-8859-1',delimiter=";")

#Convert df to text and remove numbers
df_to_text = ''.join([i for i in df["Navn"].to_string() if not i.isdigit()])
df_to_text = df_to_text.replace(" ", "")

#List the data frame and number of names 
print 'Number of Norwegian girl names of babies born between 2005-2015 with more than 4 occurences: ' + str(len(df))
print 'List of names and counts each year:'
print df
print df_to_text

# -*- coding: utf-8 -*-
vocab = set(df_to_text)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
data = [vocab_to_idx[c] for c in df_to_text]
print vocab
print vocab_size
print idx_to_vocab
print vocab_to_idx
print data

def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield ptb_iterator(data, batch_size, num_steps)


def test_epoch_function(n=1, num_steps=5, batch_size=32):
#Test and understanding of epochs
    l=[]
    #Generate epochs with length and 32 randomly selected sequences
    s=gen_epochs(n, num_steps, batch_size) 
    
    #Extract batches and put them in a list 
    for i in s:
        for t in i:
            l.append(t)
    
    #Batch Length
    print "Batch Length: " + str(len(data)//batch_size) #128 Batches in one pass of the data
    #Epoch Size
    print "Epoch Size: " + str((len(data)//batch_size) // num_steps) #Since length is 5 for each sequence one Epoch Size is 128 Batch Lengths / 5
    
    #Get a batch and print out its content   
    test = np.vectorize(idx_to_vocab.get)(l[0][0])
    for m in test:
        print m    

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, num_epochs, num_steps = 5, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses

def build_multilayer_lstm_graph_with_dynamic_rnn(
    state_size = 512,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 5,
    num_layers = 3,
    learning_rate = 1e-4,
    apply_dropout=False,
    cell_type='LSTM'):

    reset_graph()

    #Intilize droput variable 
    dropout = tf.constant(1.0)    
    
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    
    if cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.GRUCell(state_size)

    #Test with or without dropout    
    if apply_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
    
    if cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
  
    #Test with or without dropout    
    if apply_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
    
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b
    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )
                                                                         
def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return("".join(chars))

def generate_new_names(initilized_model, stored_model, number_of_characters_to_generate, 
                       random_letter, pick_top_chars):
    
    #Select random letter as starting sequence if specified. 
    if random_letter:
        letters = 'abcdefghijklmnopqrstuvwxyzæøå'
        letters = list(letters.upper())    
        starting_letter= random.choice(letters)
    else:
        starting_letter = 'A'
    
    #Generate sequence
    generated_names = generate_characters(initilized_model, 
                                          stored_model, 
                                          number_of_characters_to_generate, 
                                          prompt=starting_letter, 
                                          pick_top_chars=pick_top_chars)

    # -*- coding: utf-8-*-
    #Check which names that are not in the orginal list
    generated_names_list = generated_names.split('\n')
    names_from_ssb_list = df_to_text.split('\n')

    #Exclude list already in the list 
    new_names_generated = list(set(generated_names_list) - set(names_from_ssb_list))

    #Output number of names generated etc. 
    print "Number of generated names: " + str(len(generated_names_list))
    print "Number of new names generated that does not exist in the original list: " + str(len(new_names_generated))
    print "Rate of new names: " + str(len(new_names_generated)/float(len(generated_names_list))) + '\n'

    #Print out names
    print "New Norwegian girl baby names produced by RNN 3 Layer Deep Neural Net: " + str(len(generated_names_list)) + '\n'

    for x in sorted(new_names_generated):
        print x
   
    #Return new names and rate of new names
    return new_names_generated, (len(new_names_generated)/float(len(generated_names_list)))


#Model iterator to conduct experiment 
def model_trainer(
    num_epochs=[10], 
    num_steps=[1,3,5,7], 
    cell_type=['GRU', 'LSTM'], 
    prefix="GirlNameGenRNN_", 
    apply_dropout=[True, False],
    state_size = [512,256],
    batch_size = [32],
    num_layers = [3,4],
    learning_rate = [1e-4],
    number_of_characters_to_generate = 700, 
    random_letter = False , 
    pick_top_chars=5):
    
    #Create empty result list
    model_results=[]
    for a in num_epochs:
        for b in num_steps: 
            for c in cell_type:
                for d in apply_dropout:
                    for e in state_size:
                        for f in batch_size:
                            for g in num_layers:
                                for l in learning_rate:
                                    #Initilize model
                                    current_model = build_multilayer_lstm_graph_with_dynamic_rnn(
                                                    num_steps=b, 
                                                    cell_type=c,
                                                    apply_dropout=d,
                                                    state_size = e,
                                                    batch_size = f,
                                                    num_layers=g,
                                                    learning_rate=l)
                                                   
                                    #Generate and store model
                                    model_name =    '~/' + prefix +  (
                                                    str(a)+ "_" + 
                                                    str(b)+ "_" +
                                                    str(c)+ "_" +
                                                    str(d)+ "_" +
                                                    str(e)+ "_" +
                                                    str(f)+ "_" +
                                                    str(g)+ "_" +
                                                    str(round(1000000*l,0)))
                                                    
                                    model_name= model_name.replace('.','_')
                                    
                                    print model_name
                                    
                                    #Train model 
                                    current_model_loss = train_network(current_model,  
                                                                       num_epochs = a,
                                                                       num_steps=b,
                                                                       batch_size=f,
                                                                       save=os.path.expanduser(model_name))
                                    
                                    #Initilize model to predict characters 
                                    initilized_model=build_multilayer_lstm_graph_with_dynamic_rnn(num_steps=1, 
                                                    cell_type=c,
                                                    apply_dropout=d,
                                                    state_size = e,
                                                    batch_size = 1,
                                                    num_layers=g,
                                                    learning_rate=l)
                                    
                                    #Generate characters
                                    generated_new_names, rate_of_new_names = generate_new_names(initilized_model,os.path.expanduser(model_name),number_of_characters_to_generate, random_letter,pick_top_chars) 
                                                                                                          
                                    #Store model results 
                                    model_results.append([model_name, a, b, c, d, e , f, g, l, current_model_loss, generated_new_names, rate_of_new_names])
                                    
                                    #Save column names
                                    model_results_columns = ['ModelName', 'NumEpochs', 'NumSteps', 'CellType', 
                                                             'Dropout', 'StateSize', 'BatchSize', 'NumLayers', 'LearningRate', 'CurrentModelLoss', 'GeneratedNewNames', 'RateOfNewNamesOfSSBNames', 'CharactersToGenerate', 'RandomLetterAsStartSequence', 'NrOfTopCharsToReturn']
        
        return model_results
# TODO:
#Investigate layers and check to see what triggers sequences given an input
#Get people to generate names
model_results = model_trainer()
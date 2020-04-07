import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import adabound
import os, copy, pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import regex as re
import random

from model import *
from BatchLoader import *
from Optimizer import *
from DataGenerator import *



'''
Takes in preprocessed data stored in an class object named 
'dataprocessor'
--attributes
	.src  #input [batch_size, sequence_length, characters]
	.tgt  #label [batch_size, sequence_length]
	.max_word_length #max length of a word in the vocabulary
	.w2n # word to integer mapping dictionary
	.n2w # integer to word mapping dictionary
	.c2n # character to integer mapping dictionary
	.n2c # integer to charcter mapping dictionary

'''

character_vocabulary = 'abcdefghijklmnopqrstuvwxyz.,/\()-#*%^&[]{}|_;:<>?@$!+~`1234567890°' + "'" +'"'

max_w_length = dataprocessor.max_word_length
batch_size = 20

#embedding hyperparameters
vocab_size = len(dataprocessor.c2n)
word_vocab_size = len(dataprocessor.w2n)

#model hyperparameters
padding_idx = 0
kernels = [1,2,3,4,5]
embedding_dim = 15
num_filter = 20
dropout = 0.5
num_rnn_layers = 2
d_rnn = 200
lr = 1.0
bidirectional = True

#training hyperparameter
epochs = 25
k_fold = 10
shuffle = True

#random seed
seed = 1234 



#TRAINING LOOP
kfold = KFold(k_fold, random_state=seed)
for fold, (train_idx, val_idx)  in enumerate(kfold.split(dataprocessor.src)):

	print('TRAINING {}TH FOLD--------'.format(fold))

	#define model
	model = CharacterRNNLM(max_w_length, embedding_dim, vocab_size + 1,
						 word_vocab_size + 1, padding_idx, kernels,
						  num_filter, d_rnn = d_rnn, num_layers = num_rnn_layers,
						  				 dropout = dropout, bidirectional = bidirectional)

	#define loss and optimizer
	#optimizer = optim.SGD(model.parameters(), lr = lr)

	optimizer = adabound.AdaBound(model.parameters(), lr=1e-4, final_lr=0.01)
	scheduler = None

	Opt = SpecOptimizer(model, optimizer, scheduler, initial_lr = lr, max_norm = 5)
	loss_fn = nn.NLLLoss(ignore_index= padding_idx)
	LossCompute = LossComputer(Opt, loss_fn)

	history = {}

	#define data iterators
	train_DataIterator = DataIterator([dataprocessor.src[i] for i in train_idx.astype(int)],
											[dataprocessor.tgt[i] for i in train_idx.astype(int)],
											max_w_length, batch_size = batch_size, shuffle = shuffle)
	val_DataIterator = DataIterator([dataprocessor.src[i] for i in val_idx.astype(int)], 
											[dataprocessor.tgt[i] for i in val_idx.astype(int)], 
											max_w_length, batch_size = batch_size, shuffle = shuffle)
	#train

	total_loss = []
	total_val_loss = []
	total_val_perplexity = []
	for epoch in range(epochs):
		epoch_loss, epoch_val_loss, epoch_val_perplexity = run_epoch(train_DataIterator(), val_DataIterator(), 
		 														model, LossCompute, epoch)
		total_loss.append(epoch_loss)
		total_val_loss.append(epoch_val_loss)
		total_val_perplexity.append(epoch_val_perplexity)

	
	torch.save({'fold': fold, 'epoch': epoch,'model_state_dict': model.state_dict(),
				 'optimizer_state_dict': optimizer.state_dict(),'training_loss': tuple(total_loss),
				  'validation_loss': tuple(total_val_loss), 'validation_perplexity': tuple(total_val_perplexity)},
				   PATH)



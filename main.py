import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, copy, pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import regex as re
import random

from model import *
from BatchLoader import *
from Optimizer import *



if __name__ == '__main__':
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
	d_ff = 500
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
		model = CharacterRNNLM(embedding_dim, vocab_size + 1,
							 word_vocab_size + 1, padding_idx, kernels,
							  num_filter, d_rnn = d_rnn, d_ff = d_ff, num_layers = num_rnn_layers,
							  				 dropout = dropout, bidirectional = bidirectional)

		#define loss and optimizer
		optimizer = optim.SGD(model.parameters(), lr = lr)
		'''
		Adam reaches convergence faster and does not require
		sophisticated learning rate scheduling
		,but generalizes worse than SGD 
		'''

		#optimizer = optim.Adam(model.parameters(), lr=lr)

		Opt = SpecOptimizer(model, optimizer, initial_lr = lr, max_norm = 5)
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
			#total_val_perplexity.append(epoch_val_perplexity)

			#save model
			torch.save(model.state_dict(), 'model_{}.pth'.format(fold))

		history['train_loss'] = tuple(total_loss)
		history['val_loss'] = tuple(total_val_loss)

		# #save model
		# torch.save(model.state_dict(), 'model_{}.pth'.format(fold))

		#save training history
		with open('model_{}_hist.pickle'.format(fold), 'wb') as handle:
			pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)



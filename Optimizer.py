import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, copy, sys

import numpy as np
import random


class SpecOptimizer:
	'''
	We backpropagate for 35 time steps using stochastic gradient descent
	where the learning rate is initially set to 1.0 and halved if the 
	perplexity does not decrease by more than 1.0 on the validation set 
	after an epoch.

	'''

	def __init__(self, model, optimizer, initial_lr = 0.01, max_norm = 5):
		self.lr = initial_lr
		self._perplexity = 1e12
		self.optimizer = optimizer
		self.max_norm = max_norm
		self._step = 0
		self.model = model
		
		'''
		optimizer contains model parameters 
		'''

	def step(self):
		'''
		performs one step of optimization 
		'''

		self._step += 1

		#truncated bptt

		#gradient clipping
		for name, p in self.model.named_parameters():
			if name[:5] == 'RNNLM':
				_ = nn.utils.clip_grad_norm_(p, max_norm =self.max_norm )


		self.optimizer.step()


	def update_lr(self, perplexity):

		if (self._perplexity - perplexity) < 1.0:
			self.lr /= 2
			for p in self.optimizer.param_groups:
				p['lr'] = self.lr
			print('New learning rate is: {}'.format(self.lr))

		self._perplexity = perplexity


class LossComputer:

	def __init__(self, optimizer, loss_fn):
		self.loss_fn = loss_fn 
		self.optimizer = optimizer

	def __call__(self, out, tgt):

		loss = self.loss_fn(out.transpose(1, 2) ,tgt)
		loss.backward()

		self.optimizer.step()
		self.optimizer.optimizer.zero_grad()

		return loss

	def update_lr(self, perplexity):
		self.optimizer.update_lr(perplexity)



class DataGenerator:
	'''
	creates synthetic data for model verification
	'''


	def __init__(self, dic = None, C_voc_size = 2, W_voc_size = 3, word_length = 2, seq_length = 3, nbatches = 100, vbatches = 10):
		self.dic = {i: tuple([random.randint(1,C_voc_size) for _ in range(word_length)]) for i in range(W_voc_size)}
		self.C_voc_size = C_voc_size
		self.W_voc_size = W_voc_size
		self.word_length = word_length
		self.seq_length = seq_length
		self.nbatches = nbatches
		self.vbatches = vbatches

	def __call__(self, batch_size, train = True):

		if train == True:
			nbatches = self.nbatches
		else:
			nbatches = self.vbatches

		for i in range(nbatches):
			tgt = np.random.randint(0, self.W_voc_size, size=(batch_size, self.seq_length)) #target should be [batch_size, seq_length]
			#source should be [batch_size, seq_length, word_length]

			src = np.zeros([batch_size, self.seq_length, self.word_length],dtype=int)
			for i in range(batch_size):
				for j in range(self.seq_length-1):
					src[i,j,: ] = np.array(self.dic[tgt[i,j+1]], dtype = int)

			src = torch.from_numpy(src[:,:-1,:])
			tgt = torch.from_numpy(tgt[:,1:])
			yield src, tgt


	def easy_gen(self, batch_size):

		for i in range(self.nbatches):
			tgt = np.random.randint(0, self.W_voc_size, size=(batch_size, self.seq_length))
			src = np.zeros([batch_size, self.seq_length, self.word_length],dtype=int)
			for i in range(batch_size):
				for j in range(self.seq_length-1):
					for k in range(self.word_length):
						src[i,j,k ] = tgt[i,j+1]

			src = torch.from_numpy(src[:,:-1,:])
			tgt = torch.from_numpy(tgt[:,1:])

			yield src, tgt


def run_epoch(data, val_data, model, loss_compute, epoch, verbose = 20):

	total_loss = 0
	model.train()
	for i, (src, tgt) in enumerate(data):
		out = model(src)
		loss_one_step = loss_compute(out, tgt)
		total_loss += float(loss_one_step)

		if i % verbose == 0:
			print('At {} Epoch, Step {}, Current loss: {}'.format(epoch, i, loss_one_step))

	total_loss /= i
	
	val_perplexity = 0
	total_val_loss = 0
	model.eval()
	for i, (val_src, val_tgt) in enumerate(val_data):
		val_out = model(val_src)

		#note that perplexity is the inverse of negative log likelihood
		val_loss = loss_compute.loss_fn(val_out.transpose(1, 2), val_tgt)
		val_perplexity += float(torch.exp(val_loss))

		total_val_loss += float(val_loss)
	
	total_val_loss /= i
	val_perplexity /= i
	loss_compute.update_lr(val_perplexity)

	print('At {} Epoch, Avg Loss: {}, Avg Validation Loss: {}, Avg Perplexity on Validation Set: {}'.format(epoch,total_loss, total_val_loss, val_perplexity))

	return total_loss, total_val_loss, val_perplexity
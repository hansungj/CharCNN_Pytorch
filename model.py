import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, copy, sys

import numpy as np
import random


class EmbeddingLayer(nn.Module):

	def __init__(self, embedding_dim, vocab_size, padding_idx = 0):
		'''
		nn.Embedding takes 
			input (*)
			output (*, embedding_dim)
		'''
		super(EmbeddingLayer, self).__init__()
		self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim =  embedding_dim, padding_idx = padding_idx) 


	def forward(self, x):

		'''
		input will be [batch_size, max_sequence_length, max_character_length]
		'''
		return self.embedding(x)


class CharCNN(nn.Module):

	def __init__(self, embedding_dim, kernels = [2,3,4,5,6,7], num_filters = 25):

		'''
		This module will take a list of "n-grams" = "filter widths"
		and create sum of N filters that will be concatenated 

		The difference in output sizes of different CNN filters
		are okay because we will be applying global pooling at the end
		
		Define graph nodes in initialization function 
		connect these nodes in forward() function 
		'''

		super(CharCNN, self).__init__()
		self.num_filters = sum([kern*num_filters for kern in kernels])

		self.kernels = nn.ModuleList([nn.Conv1d( embedding_dim, kernel*num_filters, kernel , stride = 1) for kernel in kernels])

	def forward(self, x):

		'''
		input: [batch_size x temporal x max_c_length x embedding dim]
		output: [batch_size x temporal x num_filter]
	
		'''
		x = [ F.relu(kernel(x)) for kernel in self.kernels]
		x = [ F.max_pool1d(kernel, kernel.size()[-1]) for kernel in x]
		x = torch.cat(x, dim = 1 )
		return torch.squeeze(x)


class HighwayNetwork(nn.Module):

	def __init__(self, word_dim):

		super(HighwayNetwork, self).__init__()
		self.Wh	= nn.Linear(word_dim, word_dim, bias=True)
		self.Wt = nn.Linear(word_dim, word_dim, bias=True)

	def forward(self, x):
		transform_gate = torch.sigmoid(self.Wt(x))
		carry_gate = 1 - transform_gate
		return transform_gate * F.relu(self.Wh(x))  + carry_gate * x

class SubLayer(nn.Module):

	def __init__(self, d_model, dropout = 0.5):
		'''
		applies layer normalization and dropout
	
		'''
		super(SubLayer, self).__init__()

		self.dropout = nn.Dropout(dropout)

		self.alpha = nn.Parameter(torch.ones(d_model))
		self.beta = nn.Parameter(torch.zeros(d_model))
		self.eps = 1e-6

	def forward(self, x):
		mean = x.mean(dim = -1, keepdim = True)
		std = x.std( dim = -1, keepdim = True)
		x = self.alpha * (x - mean) / (std + self.eps) + self.beta
		return self.dropout(x)

class RNNLM(nn.Module):

	def __init__(self, rnn_type, d_model, num_layers, d_rnn, dropout = 0.5, bidirectional = False):

		super(RNNLM, self).__init__()

		if rnn_type in ['LSTM', 'GRU']:
			self.RNN = getattr(nn, rnn_type)(d_model, d_rnn, num_layers, dropout = dropout, bidirectional = bidirectional)

	def forward(self, x):

		x, _ = self.RNN(x)
		return x
		

class Classifier(nn.Module):

	def __init__(self, word_vocab_size, d_ff, input_dim, dropout = 0.5, num_ff_layers = 2):

		super(Classifier, self).__init__()
		self.FFN = nn.ModuleList([nn.Linear(input_dim, d_ff)])
		self.FFN.extend([nn.Linear(d_ff, d_ff) for _ in range(num_ff_layers-1)])
		self.FFN.append(nn.Linear(d_ff, word_vocab_size))
		self.Sublayer = nn.ModuleList([SubLayer(d_ff, dropout) for _ in range(num_ff_layers)])
		self.Out = nn.LogSoftmax(dim=-1) #word_vocab_size

	def forward(self, x):

		for (layer, sublayer) in zip(self.FFN[:-1],self.Sublayer):
			x = layer(x)
			x = F.relu(sublayer(x))
		x = self.FFN[-1](x)
		x = self.Out(x)
		return x 


class CharacterRNNLM(nn.Module):

	def __init__(self, embedding_dim, vocab_size, word_vocab_size, 
						padding_idx = 0, kernels = [2,3,4,5,6,7], 
						num_filters = 100, rnn_type = 'LSTM', num_layers = 3,
						 d_rnn = 200, d_ff = 500, dropout = 0.5, bidirectional = False ):

		super(CharacterRNNLM, self).__init__()
		self.EmbeddingLayer = EmbeddingLayer(embedding_dim, vocab_size, padding_idx)
		self.CharCNN = CharCNN(embedding_dim, kernels, num_filters)

		self.word_dim = self.CharCNN.num_filters
		self.HighwayNetwork = HighwayNetwork(self.word_dim)

		self.RNNLM = RNNLM(rnn_type, self.word_dim, num_layers, d_rnn, dropout, bidirectional)
		self.Classifier = Classifier(word_vocab_size, d_ff, d_rnn)

	def forward(self, x, debug = False):
		if debug:
			print('Input size:', x.size())

		x = self.EmbeddingLayer(x)

		if debug:
			print('After Embedding layer: ', x.size())

		batch_size, seq_len, mac_c_len, emb_dim = x.size()
		x = x.view(batch_size * seq_len , emb_dim, mac_c_len)

		if debug:
			print('After resizing: ', x.size())

		y = self.CharCNN(x)

		if debug:
			print('After CNN layer: ', y.size())

		z = self.HighwayNetwork(y)

		if debug:
			print('After Highway layer: ', z.size())

		z = z.view(seq_len, batch_size, -1)

		if debug:
			print('After final resizing layer: ', z.size())


		z = self.RNNLM(z)
		z = z.view(z.size()[1], z.size()[0], -1)

		if debug:
			print('After RNN LM: ', z.size())

		return self.Classifier(z)


class SpecOptimizer:
	'''
	We backpropagate for 35 time steps using stochastic gradient descent
	where the learning rate is initially set to 1.0 and halved if the 
	perplexity does not decrease by more than 1.0 on the validation set 
	after an epoch.

	'''

	def __init__(self, model, optimizer, initial_lr = 0.01, max_norm = 5):
		self.lr = initial_lr
		self._perplexity = 1e5
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

		if (self._perplexity - perplexity) > 1.0:
			self.lr /= 2
			for p in self.optimizer.param_groups:
				p['lr'] = self.lr

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
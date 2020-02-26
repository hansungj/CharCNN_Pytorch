import torch
import torch.nn as nn
import torch.nn.functional as F

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
		self.bidirectional = bidirectional
		self.d_rnn = d_rnn

		self.EmbeddingLayer = EmbeddingLayer(embedding_dim, vocab_size, padding_idx)
		self.CharCNN = CharCNN(embedding_dim, kernels, num_filters)

		self.word_dim = self.CharCNN.num_filters
		self.HighwayNetwork = HighwayNetwork(self.word_dim)

		self.RNNLM = RNNLM(rnn_type, self.word_dim, num_layers, d_rnn, dropout, bidirectional)

		if bidirectional:
			d_rnn *= 2
		self.Classifier = Classifier(word_vocab_size, d_ff, d_rnn)

	def forward(self, x, debug = False):

		x = self.EmbeddingLayer(x)

		batch_size, seq_len, mac_c_len, emb_dim = x.size()
		x = x.view(batch_size * seq_len , emb_dim, mac_c_len)

		y = self.CharCNN(x)
		z = self.HighwayNetwork(y)

		z = z.view(seq_len, batch_size, -1)
		z = self.RNNLM(z)

		if self.bidirectional:
			f_z, b_z = z[:-2,:,:self.d_rnn], z[2:,:,self.d_rnn:]
			z = torch.cat((f_z,b_z), dim = -1)
		else:
			z = z[:-2,:,:]

		z = z.view(z.size()[1], z.size()[0], -1)

		return self.Classifier(z)


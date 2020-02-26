import torch
import torch.nn as nn

import numpy as np

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

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd 

import random

class DataIterator:

	def __init__(self, src, tgt, max_word_length, batch_size, shuffle = True):

		self.src, self.tgt = self.build(src, tgt, max_word_length, batch_size)
		self.shuffle = shuffle

	def __call__(self):

		if self.shuffle:
			indices = list(range(len(self.src)))
			random.shuffle(indices)
			for i in indices:
				yield self.src[i], self.tgt[i]
		else:
			for src, tgt in zip(self.src, self,tgt):
				yield src, tgt

	def build(self, src, tgt, max_word_length, batch_size):
		'''
		pad inputs character-wise

		##note##
		you might want to make it so that you build at every epoch
		so that it shuffles, if this is a big problem... test it?
		'''

		print('Preparing data...')

		torch_src = []
		torch_tgt = []

		for bi in range(0, len(src), batch_size):
			#find max_length in this batch
			b_mx_len = max([len(x) for x in tgt[bi: bi+batch_size]])
			current_b_len = len(tgt[bi: bi+batch_size]) #in case number of examples dont divide by batch-size nicely

			#src
			src_padded = np.zeros([current_b_len, b_mx_len, max_word_length])
			#tgt
			tgt_padded = np.zeros([current_b_len, b_mx_len])

			for i in range(current_b_len):
				tgt_padded[i,:len(tgt[bi+i])] = tgt[bi+i] #copy over words
				for j in range(len(src[bi+i])):
					src_padded[i, j, :len(src[bi+i][j])] = src[bi+i][j] #copy over characters

			torch_src.append(torch.from_numpy(src_padded).to(torch.int64))
			torch_tgt.append(torch.from_numpy(tgt_padded).to(torch.int64))

		return torch_src, torch_tgt
 








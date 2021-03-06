import torch 
import numpy as np
import pandas as pd 

import random

class DataIterator:

	def __init__(self, src, tgt, max_word_length, batch_size, shuffle = False):

		self.max_word_length = max_word_length
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.src, self.tgt = zip(*sorted(zip(src, tgt), key=lambda x: len(x[1])))

	def __call__(self):
		for src, tgt in self.create_batches():
			yield src, tgt

	def create_batches(self):
		'''
		pad inputs character-wise
		'''
		print('Preparing data...')
		print(self.max_word_length)
		torch_src = []
		torch_tgt = []

		for bi in range(0, len(self.src), self.batch_size):
			#find max_length in this batch
			b_mx_len = max([len(x) for x in self.tgt[bi: bi+self.batch_size]])
			current_b_len = len(self.tgt[bi: bi+self.batch_size]) #in case number of examples dont divide by batch-size nicely

			#src
			src_padded = np.zeros([current_b_len, b_mx_len + 2, self.max_word_length])
			#tgt
			tgt_padded = np.zeros([current_b_len, b_mx_len])

			for i in range(current_b_len):
				tgt_padded[i,:len(self.tgt[bi+i])] = self.tgt[bi+i] #copy over words
				for j in range(len(self.src[bi+i])):
					src_padded[i, j, :len(self.src[bi+i][j])] = self.src[bi+i][j] #copy over characters

			torch_src.append(torch.from_numpy(src_padded).to(torch.int64).cuda())
			torch_tgt.append(torch.from_numpy(tgt_padded).to(torch.int64).cuda())

		return zip(torch_src,torch_tgt)





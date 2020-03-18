import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import random


class SpecOptimizer:


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
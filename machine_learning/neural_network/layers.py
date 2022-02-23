import numpy as np
import pandas as pd
from activation_funcs import *

class hidden_layers:
	def __init__(self, num_neurals, activation):
		self.W = None
		self.num_neurals = num_neurals
		self.activation = activation

	def forward(self, X):
		if self.W is None:
			W_shape = (X.shape[1], self.num_neurals)
			self.W = np.random.normal(loc=0, scale=np.sqrt(2/(X.shape[1]+self.num_neurals)), size=W_shape)	

		activation_func = eval(self.activation)
		self.Z = X @ self.W
		A = activation_func(X @ self.W)

		return A 

	def backward(self, X, dA):
		activation_func_grad = eval(self.activation + '_grad')
		z = self.Z

		dZ = dA * activation_func_grad(z) 

		dW = X.T @ dZ

		return dZ, dW
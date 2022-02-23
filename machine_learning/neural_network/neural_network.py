from layers import hidden_layers
from activation_funcs import *
from utils import *
import numpy as np
import pandas as pd


class neural_network:
	def __init__(self, num_classes, lr):
		self.layers = []
		self.num_classes = num_classes
		self.lr = lr 
		self.m = 0

	def add_layers(self, num_neurals, activation):
		self.layers.append(hidden_layers(num_neurals, activation))		# add all hidden layers in NN

	def forward(self, X):
		all_X = [X]
		for layer in self.layers:
			all_X.append(layer.forward(all_X[-1]))

		return all_X 		# all A in NN

	def compute_cost(self, Y, Y_hat):
		self.m = Y.shape[0]
		loss = -np.sum(Y * np.log(Y_hat)) / self.m
		return loss 

	def grad_last(self, Y, all_X):
		dZ = (all_X[-1] - Y) / self.m
		dW = all_X[-2].T @ dZ 
		return dZ, dW 

	def backward(self, Y, all_X):
		dZ, dW = self.grad_last(Y, all_X)
		dW_list = [dW]

		for i in reversed(range(len(self.layers) - 1)):
			after_layer = self.layers[i + 1]
			layer = self.layers[i]
			X = all_X[i]

			dA_prev = dZ @ after_layer.W.T 			# dA[l-1]
			dZ, dW = layer.backward(X, dA_prev)		# dZ[l-1], dW[l-1]

			dW_list.insert(0, dW)					# list of all dW from 0 to L

		return dW_list

	def update_weights(self, dW_list):
		for i, layer in enumerate(self.layers):
			layer.W = layer.W - self.lr * dW_list[i]

	def predict(self, X_test):
		Y_hat = self.forward(X_test)[-1]
		# print(Y_hat)
		return np.argmax(Y_hat, axis=1) 		# index max of Y_hat (softmax)
from neural_network import neural_network
from activation_funcs import *
from utils import create_one_hot
import numpy as np
import pandas as pd

def train(X, Y, num_epochs, neural_network):
	all_losses = []

	for epoch in range(num_epochs):
		all_X = neural_network.forward(X)
		loss = neural_network.compute_cost(Y, all_X[-1])
		dW_list = neural_network.backward(Y, all_X)
		neural_network.update_weights(dW_list)

		all_losses.append(loss)


if __name__ == '__main__':
	data = pd.read_csv('./Classified_Data.csv')
	samples = []
	targets = []

	for dt in data.values:
	    samples.append(dt[1:-1])
	    targets.append(dt[-1])

	x = np.array(samples)
	X = (x - np.mean(x)) / np.std(x)
	y = np.array(targets, dtype=np.int32)
	Y = create_one_hot(y, 2)

	model = neural_network(2, 0.01)
	model.add_layers(4, 'relu')
	model.add_layers(2, 'sigmoid')

	train(X, Y, 50000, model)
	pred = model.predict(X[0:10])
	print(pred)
import numpy as np
import pandas as pd

def softmax(z):
	exps = np.exp(z - np.max(z, axis=1, keepdims=True))
	A = exps / np.sum(exps, axis=1, keepdims=True)
	return A

def predict(X, W):
	Z = X @ W
	return softmax(Z)

def cost(X, Y, W):
	m = X.shape[0]
	preds = predict(X, W)
	return -np.sum(Y - np.log(preds)) / m

def grad(X, Y, W):
	m = X.shape[0]
	preds = predict(X, W)
	dZ = (preds - Y) / m
	dW = X.T @ dZ
	return dW 

def train(X, Y, W, lr, iter):
	loss_his = []
	for i in range(iter):
		W = W - lr * grad(X, Y, W)
		loss = cost(X, Y, W)
		loss_his.append(loss)
		if np.linalg.norm(grad(X, Y, W)) / X.shape[0] < 1e-6:
			break

	return W, i, loss_his


def create_one_hot(Y, num_classes):
	Y_onehot = np.zeros((Y.shape[0], num_classes), dtype=np.int32)
	Y_onehot[np.arange(Y.shape[0]), Y] = 1
	return Y_onehot

if __name__ == '__main__':
	dataset = pd.read_csv('Classified_Data.csv')
	samples = []
	labels = []

	for data in dataset.values:
	    samples.append(data[1:11])
	    labels.append(data[11])

	X = np.array(samples)
	y = np.array(labels, dtype=np.int32)
	Y = create_one_hot(y, 2)

	W0 = np.zeros((X.shape[1], Y.shape[1]))

	W, i, loss_his = train(X, Y, W0, lr=0.1, iter=20000)
	print(W, i)
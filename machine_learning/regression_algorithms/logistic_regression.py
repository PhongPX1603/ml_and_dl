import numpy as np 
import pandas as pd 

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def predict(X, W):
	return sigmoid(X @ W)

def cost(X, Y, W):
	m = X.shape[0]
	preds = predict(X, W)
	return -np.sum(Y * np.log(preds)) / m

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
		if np.linalg.norm(grad(X, Y, W))/X.shape[0] < 1e-6:
			break

	return W, i, loss_his

def valuator(X, Y, W):
	preds = predict(X, W) > 0.5
	true_pred = preds == Y 
	return np.sum(true_pred)/Y.shape[0]

if __name__ == '__main__':
	dataset = pd.read_csv('Classified_Data.csv')
	samples = []
	labels = []

	for data in dataset.values:
	    samples.append(data[1:11])
	    labels.append(data[11])

	X = np.array(samples)
	Y = np.array([labels]).T

	W0 = np.zeros((X.shape[1], Y.shape[1]))

	W, i, loss_his = train(X, Y, W0, lr=0.1, iter=20000)
	print(W, i)

	accuracy = valuator(X, Y, W)
	print(accuracy)
#ex1: Y = X**2 x W0 + X x W1 + b
import numpy as np

x = [1, 2, 3]
y = [2, 4, 6]

X = np.array([x]).T 	# (m, n=2)
Y = np.array([y]).T 	# (m, 1)

one = np.ones((X.shape[0], 1))
X = np.concatenate((X**2, X, one), axis=1)

def predict(X, W): 
	return X @ W

def cost(X, Y, W):
	m = X.shape[0]
	return 1/(2*m) * np.linalg.norm((Y - predict(X, W)), 2) ** 2

def grad(X, Y, W):
	m = X.shape[0]
	return -1/m * X.T @ (Y - predict(X, W))

def train(X, Y, W, lr, iter):
	loss_his = []
	for i in range(iter):
		loss = cost(X, Y, W)
		loss_his.append(loss)
		W = W - lr * grad(X, Y, W)
		if np.linalg.norm(grad(X, Y, W)) / (X.shape[0]) < 1e-6:
			break

	return W, loss_his, i

if __name__ == '__main__':
	W0 = np.zeros((X.shape[1], Y.shape[1]))
	W, loss_his, i = train(X, Y, W0, lr=0.05, iter=50000)
	print(W, i)
	print(loss_his[0], loss_his[-1])
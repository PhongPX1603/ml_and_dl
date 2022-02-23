x = [1, 2, 3]
y = [2, 4, 6]

def predict(x, w, b):
    return w*x + b

def cost(x, y, w, b):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (w*x[i] + b)) ** 2

    return sum_error/n

def update(x, y, w, b, lr):
    n = len(x)
    w_temp = 0
    b_temp = 0
    
    for i in range(n):
        w_temp += -2*x[i] * (y[i] - (w*x[i] + b))
        b_temp += -2 * (y[i] - (w*x[i] + b))
    
    w = w - (w_temp/n) * lr 
    b = b - (b_temp/n) * lr

    return w, b  

def train(x, y, w, b, lr, iter=1000):
    loss_his = []
    for i in range(iter):
        w, b = update(x, y, w, b, lr)
        loss = cost(x, y, w, b)
        loss_his.append(loss)
        if i>10 and loss_his[i-10] - loss_his[i] < 1e-9:
            break
        
    return w, b, loss_his, i 


if __name__ == '__main__':
	w, b, loss_his, i = train(x, y, w=0, b=0, lr=0.01, iter=10000)
	print(w, b, i)
	for i in range(len(x)):
		print(x[i] * w + b)
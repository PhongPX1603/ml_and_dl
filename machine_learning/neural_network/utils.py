import numpy as np

def normalize(X):
    return X / 255.


def reshape_x(X):
    '''reshape X with N matrix (28 x 28) to X with N vector (784,)
    Args:
        X: np.ndarray, shape NxHxW (with H = W = 28)
    Output:
        X: np.ndarray, shape NxD (with D = H * W)
    '''
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])		# reshape to (m, n)


def add_one(X):		# add bias 
    '''Pad 1 as the 785th feature of train_X and test_X and valid_X
    Args:
        X: np.ndarray, shape N, D (with D=784)
    Output:
        X: np.ndarray, shape N, D + 1
    '''
    X = np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=1)
    
    return X


def create_one_hot(y, num_classes):
    '''
    Convert Y to shape like Y_hat (softmax)
    Example: y = [0, 2, 1, 0] with shape (4,), num_classes = 3
        --> y_onehot = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]] shape: (4, 3)
        Args:
            y: np.ndarray, shape N
            num_classes: int, (usually =np.unique(y).shape[0])
        Outputs:
            y_onehot: np.ndarray, shape Nxnum_classes
    '''
    y_onehot = np.zeros(shape=(y.shape[0], num_classes), dtype=np.int32)
    y_onehot[np.arange(y.shape[0]), y] = 1
    
    return y_onehot
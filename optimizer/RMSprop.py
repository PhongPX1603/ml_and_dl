from torch import optim

learning_rate = 0.01
momentum = 0.9

optimizer = optim.RMSprop(params=model.parameters(), lr=learning_rate, alpha=0.99,
						  eps=1e-08,weight_decay=0, momentum=momentum, centered=False


"""
- params(iterable) – iterable of parameters to optimize or dicts defining parameter groups

- lr(float, optional) – learning rate (default: 1e-2)

- momentum(float, optional) – momentum factor (default: 0)

- alpha(float, optional) – smoothing constant (default: 0.99)

- eps(float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)

- centered(bool, optional) – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

- weight_decay(float, optional) – weight decay (L2 penalty) (default: 0)
"""
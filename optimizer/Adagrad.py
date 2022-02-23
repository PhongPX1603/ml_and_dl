from torch import optim

learning_rate = 0.01

optimizer = optim.Adagrad(params=model.parameters(), lr=learning_rate, lr_decay=0,
						  weight_decay=0, initial_accumulator_value=0, eps=1e-10)

"""
- params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
- lr (float, optional) – learning rate (default: 1e-2)
- lr_decay (float, optional) – learning rate decay (default: 0)
- weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
- eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-10)
"""
from torch import optim

learning_rate = 0.01
betas = (0.9, 0.999)

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, betas=betas,
					   eps=1e-08, weight_decay=0, amsgrad=False)


"""
- params (iterable) – iterable of parameters to optimize or dicts defining parameter groups

- lr (float, optional) – learning rate (default: 1e-3)

- betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient 
										  and its square (default: (0.9, 0.999))

- eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)

- weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

- amsgrad (boolean, optional) – whether to use the AMSGrad variant of this algorithm from the paper 
							    On the Convergence of Adam and Beyond (default: False)
"""
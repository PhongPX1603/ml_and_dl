from torch import optim

learning_rate = 0.01
momentum = 0.9

optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)
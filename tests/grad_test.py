from rnn import RNN
import numpy as np

np.random.seed(1)
net = RNN(100, 10, 100)
x = net.one_hot([0,1,2,3], 100)
y = net.one_hot([1,2,3,4], 100)
net.gradient_check(x, y)

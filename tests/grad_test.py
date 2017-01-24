from rnn import RNN
import numpy as np

np.random.seed(1)
net = RNN(100, 10, 100)
x = [0, 1, 2, 3]
y = [1, 2, 3, 4]
net.gradient_check(x, y)

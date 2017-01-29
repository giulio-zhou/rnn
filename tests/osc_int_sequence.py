from network.lstm import LSTM
from network.rnn import RNN
import numpy as np

# Generate dataset, create random sequences of integers
# Goal: Learn oscillating [-1, 0, 1, 0, -1, ...] pattern
num_vocab = 20
num_hidden = 50
num_examples = 1000
example_dim = 8
X = np.random.randint(1, num_vocab - 1, (num_examples, example_dim))
offset = [-1, 0, 1, 0, -1, 0, 1, 0]
Y = X + np.vstack([offset] * num_examples)
print(X)
print(Y)

# net = RNN(num_vocab, num_hidden, num_vocab)
# net.train(X, Y, 10000, 1e-2)
# test_seq = [1, 2, 3, 4, 5, 6, 7, 8]
# print(net.predict(test_seq))

lstm_net = LSTM(num_vocab, num_hidden, num_vocab)
lstm_net.train(X, Y, 10000, 0.1)
test_seq = np.arange(16)
print(lstm_net.predict(test_seq))

from network.lstm import LSTM
from network.rnn import RNN
import numpy as np

# Generate dataset, create random sequences of integers
# Goal: Learn to add [1, 2, 3, 4] to sequence of four digits
num_vocab = 20
num_hidden = 50
num_examples = 1000
example_dim = 8
X = np.random.randint(0, num_vocab - example_dim, (num_examples, example_dim))
offset = [i + 1 for i in range(example_dim)]
Y = X + np.vstack([offset] * num_examples)
print(X)
print(Y)

net = RNN(num_vocab, num_hidden, num_vocab)
net.train(X, Y, 10000, 1e-2)
test_seq = [1, 2, 3, 4, 5, 6, 7, 8]
print(net.predict(test_seq))

lstm_net = LSTM(num_vocab, num_hidden, num_vocab)
lstm_net.train(X, Y, 10000, 0.1)
test_seq = [1, 2, 3, 4, 5, 6, 7, 8]
print(lstm_net.predict(test_seq))

from rnn import RNN
from lstm import LSTM
import numpy as np

# Generate dataset, create random sequences of integers
# Goal: Learn to add 1 to each value in the sequence
num_vocab = 100
num_hidden = 10
X = np.random.randint(1, num_vocab - 1, (1000, 10))
Y = X + 1

net = RNN(num_vocab, num_hidden, num_vocab)
net.train(X, Y, 10000)
test_seq = [1, 2, 3, 4, 5, 6, 7, 8]
print(net.predict(test_seq))

lstm_net = LSTM(num_vocab, num_hidden, num_vocab)
lstm_net.train(X, Y, 10000, 0.1)
test_seq = [1, 2, 3, 4, 5, 6, 7, 8]
print(lstm_net.predict(test_seq))

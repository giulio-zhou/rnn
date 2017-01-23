from rnn import RNN
import numpy as np

# Generate dataset, create random sequences of integers
# Goal: Learn to add 1 to each value in the sequence
num_vocab = 100
num_hidden = 10
X = np.random.randint(1, num_vocab - 1, (1000, 10))
Y = X + 1

net = RNN(num_vocab, num_hidden, num_vocab)
X_one_hot = net.one_hot(X, num_vocab)
Y_one_hot = net.one_hot(Y, num_vocab)
net.train(X_one_hot, Y_one_hot, 10000)
test_seq = net.one_hot([1, 2, 3, 4, 5, 6, 7, 8], num_vocab)
print(net.predict(test_seq))

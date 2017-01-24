from rnn import RNN
import numpy as np

# Generate dataset, create random sequences of integers
# Goal: Learn to add [1, 2, 3, 4] to sequence of four digits
num_vocab = 20
num_hidden = 50
num_examples = 1000
example_dim = 4
X = np.random.randint(0, num_vocab - example_dim, (num_examples, example_dim))
offset = [i + 1 for i in range(example_dim)]
Y = X + np.vstack([offset] * num_examples)
print(X)
print(Y)

net = RNN(num_vocab, num_hidden, num_vocab)
X_one_hot = net.one_hot(X, num_vocab)
Y_one_hot = net.one_hot(Y, num_vocab)
net.train(X_one_hot, Y_one_hot, 10000, 1e-2)
test_seq = net.one_hot([1, 2, 3, 4], num_vocab)
print(net.predict(test_seq))

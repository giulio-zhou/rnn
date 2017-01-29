from network.lstm import LSTM
from preprocess_text import process_shakespeare, process_sentences_as_words
from preprocess_text import vocabulary_size as vocab_size
from util import pickle, unpickle 
import os
import sys

if __name__ == '__main__':
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    model_output_path = sys.argv[3]
    if os.path.exists(model_path):
        lstm_net = unpickle(model_path)
    else:
        lstm_net = LSTM(vocab_size, 200, vocab_size)
    data = process_shakespeare(sys.argv[1])
    index_to_word, word_to_index, tokenized_sentences = \
            process_sentences_as_words(data)
    # Make training data
    X = map(lambda sent: map(lambda x: word_to_index[x], sent[:-1]),
            tokenized_sentences)
    Y = map(lambda sent: map(lambda x: word_to_index[x], sent[1:]),
            tokenized_sentences)
    lstm_net.train(X, Y, 100000, 0.1)
    pickle(lstm_net, model_output_path)

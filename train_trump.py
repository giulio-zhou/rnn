from lstm import LSTM
from preprocess_text import process_twitter
from preprocess_text import process_sentences_as_char 
from preprocess_text import vocabulary_size as vocab_size
from util import pickle, unpickle 
import os
import sys

if __name__ == '__main__':
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    model_output_path = sys.argv[3]
    num_iter = int(sys.argv[4])
    data = process_twitter(data_path)
    index_to_char, char_to_index, sentences = process_sentences_as_char(data)
    if os.path.exists(model_path):
        lstm_net = unpickle(model_path)
    else:
        lstm_net = LSTM(128 + 2, 50, 128 + 2)
    X = map(lambda sent: map(lambda x: char_to_index[x], sent[:-1]), sentences)
    Y = map(lambda sent: map(lambda x: char_to_index[x], sent[1:]), sentences)
    lstm_net.train(X, Y, num_iter, 1e-3)
    pickle(lstm_net, model_output_path)

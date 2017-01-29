from preprocess_text import sentence_start_token, sentence_end_token
from preprocess_text import vocabulary_size as vocab_size
from preprocess_text import generate_char_sentence
from util import pickle, unpickle
import sys

if __name__ == '__main__':
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    num_sentences = int(sys.argv[3])
    print("Load pickled data...")
    index_to_word, word_to_index = unpickle(data_path)
    print("Load pickled model...")
    model = unpickle(model_path)
    print("Generate text...")
    sentences = []
    for i in range(num_sentences):
        sentence = generate_char_sentence(model, index_to_word, word_to_index)
        sentences.append(sentence)
    for sentence in sentences:
        print(sentence)

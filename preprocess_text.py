import itertools
import nltk
import numpy as np
import pandas as pd
import sys

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def process_shakespeare(path):
    data = pd.read_csv(path, delimiter=';',
            names=['line', 'title', 'stmt', 'part', 'speaker', 'text'])
    return data.text.tolist()

def process_sentences_as_words(data):
    sentences = itertools.chain(
        *[nltk.sent_tokenize(line.decode('utf-8').lower()) for line in data])
    sentences = ["%s %s %s" % (sentence_start_token, sent, sentence_end_token) \
            for sent in sentences]
    print("Number of sentences: %d" % len(sentences))
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Keep the top %d words..." % vocabulary_size)
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = {w: i for i, w in enumerate(index_to_word)}
    print("Replacing all other words with unknown token...")
    print(tokenized_sentences[0])
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    print("Example sentence: %s" % sentences[0])
    print("Sentence after tokenization: %s" % tokenized_sentences[0])
    return index_to_word, word_to_index, tokenized_sentences

data = process_shakespeare(sys.argv[1])
index_to_word, word_to_index, tokenized_sentences = process_sentences_as_words(data)

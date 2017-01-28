import cPickle

def unpickle(path):
    with open(path, 'rb') as input_file:
        obj = cPickle.load(input_file)
    return obj 

def pickle(obj, path):
    with open(path, 'wb') as output_file:
        cPickle.dump(obj, output_file)

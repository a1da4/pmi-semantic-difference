import _pickle


def write_pickle(data, filename):
    fp = open(filename, "wb")
    _pickle.dump(data, fp)


def load_pickle(filename):
    fp = open(filename, "rb")
    return _pickle.load(fp)

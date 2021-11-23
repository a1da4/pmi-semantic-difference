import argparse
from collections import Counter

from ioutils import write_pickle


def main(args):
    word2freq = Counter()

    with open(args.file_path) as fp:
        for line in fp:
            words = line.strip().split()
            for word in words:
                word2freq[word] += 1

    write_pickle(word2freq, "word2freq.pkl")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="the path of word_pos data directory")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

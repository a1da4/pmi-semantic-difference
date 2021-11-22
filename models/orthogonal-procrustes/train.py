import argparse
from gensim.models import word2vec
from collections import Counter


def main(args):
    with open(args.file_path) as fp:
        sentences = []
        for line in fp:
            sent = line.strip().split()
            sentences.append(sent)

    model = word2vec.Word2Vec(
        sentences,
        size=args.dim,
        min_count=args.min_count,
        window=args.window_size,
        sg=1,
        hs=0,
        negative=args.negative,
        seed=args.seed,
    )
    model.save(
        f"W2V_d-{args.dim}_w-{args.window_size}_ns-{args.negative}_seed-{args.seed}"
    )


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="path of target data")
    parser.add_argument("-d", "--dim", type=int, help="int, size of vectors")
    parser.add_argument("-c", "--min_count", type=int, help="int, threshold of counts")
    parser.add_argument("-w", "--window_size", type=int, help="int, size of window")
    parser.add_argument("-n", "--negative", type=int, help="int, negative samples")
    parser.add_argument("-s", "--seed", type=int, help="int, random seed")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

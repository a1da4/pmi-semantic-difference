import os
import re
import argparse
from tqdm import tqdm

from collections import Counter

from ioutils import write_pickle, load_pickle


def main(args):
    """create target id_to_word list."""
    assert len(args.count_dic) >= 2, "input must be more than 2 dictionaries."

    dic_pre = load_pickle(args.count_dic[0])
    list_pre = [w for w in dic_pre if dic_pre[w] >= args.threshold]
    del dic_pre

    for path in args.count_dic[1:]:
        dic_now = load_pickle(path)
        list_now = [w for w in dic_now if dic_now[w] >= args.threshold]
        del dic_now
        match = [w for w in list_pre if w in list_now]

        list_pre = match
        print(len(match))

    del list_pre, list_now
    with open("id_to_word.txt", "w") as fp:
        for i, w in enumerate(match):
            fp.write(f"{i}\t{w}\n")
            i += 1

    d = {}
    for i, w in enumerate(match):
        d[i] = w
    write_pickle(d, "dic_id2word.pkl")

    with open("id_to_word_joint.txt", "w") as fp:
        for i, n in enumerate(range(len(args.count_dic))):
            for w in match:
                fp.write(f'{i}\t{"_"*n + w}\n')


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count_dic",
        nargs="*",
        help="the path of Counter() dictionari[es] want to match.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="minimum count for selecting target word.",
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

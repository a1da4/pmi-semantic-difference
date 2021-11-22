import os
import argparse

from utils import process_lemma_line


def main(args):
    for file_ in sorted(os.listdir(args.data_dir)):
        with open(args.data_dir + "/" + file_, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                # lemma, lemma_pos, pos_tags = process_lemma_line(line)
                word, _, lemma_pos, _ = process_lemma_line(line)
                if lemma_pos == None:
                    continue
                print(f" {word}", end="")
        print()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="the path of data directory")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

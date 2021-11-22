import os
import argparse
from tqdm import tqdm

from utils import process_lemma_line


def main(args):
    decade = args.data_dir.split("/")[-1].split("_")[1]
    doc_words = []
    doc_lemmas = []
    for file_ in tqdm(sorted(os.listdir(args.data_dir))):
        words = []
        lemmas = []
        with open(args.data_dir + "/" + file_, "r", encoding="utf-8") as f:
            for line in f:
                word, lemma, lemma_pos, _ = process_lemma_line(line, is_split_sent=True)
                if lemma_pos == None:
                    continue
                words.append(word)
                lemmas.append(lemma)
                if word in [".", "!", "?"] and len(words) > 1:
                    doc_words.append(words)
                    doc_lemmas.append(lemmas)
                    words = []
                    lemmas = []
        if len(words) > 1 and len(lemmas) > 1:
            doc_words.append(words)
            doc_lemmas.append(lemmas)

    # print words, lemmas
    # with open(f"docs/text_word/{decade}.txt", "w") as fp:
    with open(f"docs/text_word/{decade}_split_sent.txt", "w") as fp:
        for words in doc_words:
            fp.write(f"{' '.join(words)}\n")

    # with open(f"docs/text_lemma/{decade}.txt", "w") as fp:
    with open(f"docs/text_lemma/{decade}_split_sent.txt", "w") as fp:
        for lemmas in doc_lemmas:
            fp.write(f"{' '.join(lemmas)}\n")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="the path of data directory")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

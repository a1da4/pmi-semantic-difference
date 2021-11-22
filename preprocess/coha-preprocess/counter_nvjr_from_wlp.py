import os
import argparse
from tqdm import tqdm

from collections import Counter
from nltk.corpus import stopwords
import chardet

from utils import process_lemma_line
from ioutils import write_pickle


def detect_encoding(file_path):
    with open(file_path, "rb") as fp:
        chardet_dict = chardet.detect(fp.read())
    encoding = chardet_dict["encoding"]

    return encoding


def main(args):
    decade = args.data_dir.split("/")[-1].split("_")[1]
    word_freqs = Counter()
    lemma_freqs = Counter()

    proper_nouns = []
    stop_words = set(stopwords.words("english"))

    for file_ in tqdm(sorted(os.listdir(args.data_dir))):
        try:
            encoding = detect_encoding(args.data_dir + "/" + file_)
            with open(args.data_dir + "/" + file_, "r", encoding=encoding) as f:
                for line in f:
                    word, lemma, lemma_pos, _ = process_lemma_line(line)
                    if lemma_pos == None:
                        continue
                    else:
                        pos = lemma_pos.split("_")[1]
                    if lemma in stop_words:
                        continue
                    elif pos == "np":
                        # np: proper noun
                        continue
                    elif pos[0] in ["n", "v", "j", "r"]:
                        # n: noun, v: verb, j: adj, r: adv
                        lemma_freqs[lemma] += 1
                        word_freqs[word] += 1
                    else:
                        continue
        except:
            print(f"【Unicode is not UTF-8!】: {file_}")
    os.makedirs("freqs", exist_ok=True)
    os.makedirs("freqs/nvjr_lemma", exist_ok=True)
    os.makedirs("freqs/nvjr_word", exist_ok=True)
    write_pickle(lemma_freqs, "freqs/nvjr_lemma/" + decade + "-lemma.pkl")
    write_pickle(word_freqs, "freqs/nvjr_word/" + decade + "-word.pkl")

    # print statics

    print("lemma")
    print(f" more than 100 times:")
    print(
        f" ALL: {sum([1 if lemma_freqs[k] >= 100 else 0 for k in lemma_freqs])} words"
    )
    print(f" more than 50 times:")
    print(f" ALL: {sum([1 if lemma_freqs[k] >= 50 else 0 for k in lemma_freqs])} words")
    print(f" more than 20 times:")
    print(f" ALL: {sum([1 if lemma_freqs[k] >= 20 else 0 for k in lemma_freqs])} words")

    print("word")
    print(f" more than 100 times:")
    print(f" ALL: {sum([1 if word_freqs[k] >= 100 else 0 for k in word_freqs])} words")
    print(f" more than 50 times:")
    print(f" ALL: {sum([1 if word_freqs[k] >= 50 else 0 for k in word_freqs])} words")
    print(f" more than 20 times:")
    print(f" ALL: {sum([1 if word_freqs[k] >= 20 else 0 for k in word_freqs])} words")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="the path of data directory")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

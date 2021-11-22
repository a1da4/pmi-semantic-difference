import argparse
import pickle


def main(args):
    print(args)

    print("1. load dictionaries... ", end="")
    dic_path_base, dic_path_other = args.word2freq
    word2freq_base = pickle.load(open(dic_path_base, "rb"))
    word2freq_other = pickle.load(open(dic_path_other, "rb"))
    print("done")

    print("2. load wordlist... ", end="")
    words_replace = []  # α
    words_target = []  # β
    with open(args.wordlist) as fp:
        for line in fp:
            word_replace, word_target = line.strip().split(" -> ")
            words_replace.append(word_replace)
            words_target.append(word_target)

    print("done")

    print("3. replace words in word2freq... ", end="")
    for word_replace, word_target in zip(words_replace, words_target):
        word2freq_base[word_target] = 0
        word2freq_other[word_replace] = word2freq_other[word_target]
        word2freq_other[word_target] = 0
    print("done")

    print("4. save replaced word2freq...", end="")
    replaced_dic_path_base = dic_path_base[:-4] + "_replaced.pkl"
    replaced_dic_path_other = dic_path_other[:-4] + "_replaced.pkl"
    pickle.dump(word2freq_base, open(replaced_dic_path_base, "wb"))
    pickle.dump(word2freq_other, open(replaced_dic_path_other, "wb"))
    print("done")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--word2freq", nargs=2, help="path of the word2freq dictionaries"
    )
    parser.add_argument("-l", "--wordlist", help="replaced word pairs")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

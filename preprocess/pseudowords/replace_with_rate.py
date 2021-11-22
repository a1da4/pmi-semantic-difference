import argparse
import pickle


def load_pickle(path):
    fp = open(path, "rb")
    return pickle.load(fp)


# define replace frequency for each word in word pairs
def obtain_replace_freq_in_each_pair(
    sampled_list_replace, sampled_list_target, word2freq, replace_rate
):
    """define replace frequency for each word pair (consider frequencies for each word and replace_rate)
    :param sampled_list_replace:
    :param sampled_list_target:
    :param word2freq: dict, word->frequency
    :param replace_rate: float, #beta_post(replaced) / #(total replaced)
    :return: replace_word2freq
    """
    replace_word2freq = {}
    for w_replace, w_target in zip(sampled_list_replace, sampled_list_target):
        print(f" w_rep, w_tar: {w_replace}, {w_target}")
        f_replace = word2freq[w_replace]
        f_target = word2freq[w_target]
        f_rate = f_target / (f_replace + f_target)
        print(f" f_replace: {f_replace}")
        print(f" f_target: {f_target}")
        print(f" f_rate: {f_rate}")
        if f_rate > replace_rate:
            replace_f_replace = f_replace
            replace_f_target = replace_rate / (1 - replace_rate) * f_replace
        elif f_rate < replace_rate:
            replace_f_replace = (1 - replace_rate) / replace_rate * f_target
            replace_f_target = f_target
        else:
            # use all
            replace_f_replace = f_replace
            replace_f_target = f_target
        replace_word2freq[w_replace] = int(replace_f_replace)
        replace_word2freq[w_target] = int(replace_f_target)
        print(f" replace_f_replace: {replace_word2freq[w_replace]}")
        print(f" replace_f_target: {replace_word2freq[w_target]}")
        assert replace_word2freq[w_replace] <= f_replace, "FrequencyOver: replace"
        assert replace_word2freq[w_target] <= f_target, "FrequencyOver: target"
        print()

    return replace_word2freq


def write_replaced_corpus_specified_ratio(
    file_path,
    sampled_list_target,
    sampled_list_replace,
    replace_word2freq,
    replace_rate,
):
    """write replaced corpus whose words are replaced by specified ratio
    :param file_path:
    :param sampled_list_target:
    :param sampled_list_replace:
    :param replace_word2freq:
    :param replace_rate:
    """
    replaced_file_path = file_path + f"_replaced_rate-{replace_rate}.txt"
    with open(replaced_file_path, "w") as wp:
        with open(file_path) as rp:
            for line in rp:
                sentence = []
                words = line.strip().split()
                for word in words:
                    if word in sampled_list_replace:
                        if replace_word2freq[word] > 0:
                            replace_word2freq[word] -= 1
                            replaced_word = word
                        else:
                            replaced_word = "#"
                    elif word in sampled_list_target:
                        if replace_word2freq[word] > 0:
                            replace_word2freq[word] -= 1
                            replaced_word_id = sampled_list_target.index(word)
                            replaced_word = sampled_list_replace[replaced_word_id]
                        else:
                            replaced_word = word
                    else:
                        replaced_word = word
                    sentence.append(replaced_word)
                wp.write(f"{' '.join(sentence)}\n")


def main(args):
    print(args)
    assert args.file_path is not None, "FileNotFoundError"
    assert args.word_to_freq is not None, "Word2FreqNotFoundError"
    assert args.wordpair_list is not None, "WordPairListNotFoundError"

    print("1. load datasets... ")
    word2freq = load_pickle(args.word_to_freq)
    # w_replace: [w_replace -> w_target]
    sampled_list_replace = []
    sampled_list_target = []
    with open(args.wordpair_list) as fp:
        for line in fp:
            w_replace, w_target = line.strip().split(" -> ")
            print(f"{w_replace}, {w_target}")
            sampled_list_replace.append(w_replace)
            sampled_list_target.append(w_target)
    print("done")

    print("2. define replace freq...")
    replace_word2freq_seq = []
    for i in range(1, 6):
        replace_rate = i * 0.1
        print("#" * 20)
        print(f"replace_rate: {replace_rate}")
        replace_word2freq = obtain_replace_freq_in_each_pair(
            sampled_list_replace, sampled_list_target, word2freq, replace_rate
        )
        print(replace_word2freq)
        replace_word2freq_seq.append(replace_word2freq)
    print("done")

    print("3. replace corpus with word(replace, target) in replace_freq times...")
    replace_rate_seq = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(len(replace_rate_seq)):
        replace_rate = replace_rate_seq[i]
        replace_word2freq = replace_word2freq_seq[i]
        write_replaced_corpus_specified_ratio(
            args.file_path,
            sampled_list_target,
            sampled_list_replace,
            replace_word2freq,
            replace_rate,
        )
    print("done")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="path of target file")
    parser.add_argument("-d", "--word_to_freq", help="path of word2freq dict")
    parser.add_argument("-l", "--wordpair_list", help="path of wordpair list")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

import argparse

import random

random.seed(1)


def sample_pairs(replace_word_list_before, replace_word_list_after, num):
    sampled_ids = random.sample(range(len(replace_word_list_before)), num)
    sampled_list_before = [replace_word_list_before[id] for id in sampled_ids]
    sampled_list_after = [replace_word_list_after[id] for id in sampled_ids]

    return sampled_list_before, sampled_list_after


def write_replaced_corpus(target_file_path, target_word_list, replace_word_list):
    """replace words
    :param target_file_path: path of target file
    :param target_word_list: list, words in this list are replaced by words in replace_word_list
    :param replace_word_list: list
    :return: None
    """
    replaced_file_path = target_file_path + "_replaced.txt"
    with open(replaced_file_path, "w") as wp:
        with open(target_file_path) as rp:
            for line in rp:
                sentence = []
                words = line.strip().split()
                for word in words:
                    if word in replace_word_list:
                        replaced_word = "#"
                    elif word in target_word_list:
                        replaced_word_id = target_word_list.index(word)
                        replaced_word = replace_word_list[replaced_word_id]
                    else:
                        replaced_word = word
                    sentence.append(replaced_word)
                wp.write(f"{' '.join(sentence)}\n")
    return


def main(args):
    print(args)
    assert args.file_path is not None, "FileNotFoundError"
    assert args.wordpair_list is not None, "WordPairListNotFoundError"
    replace_word_list_before = []
    replace_word_list_after = []
    with open(args.wordpair_list) as fp:
        for line in fp:
            words = line.strip().split()
            if len(words) != 3:
                continue
            words = [words[1][:-1], words[2]]
            if (
                words[0] not in replace_word_list_before
                and words[1] not in replace_word_list_before
                and words[0] not in replace_word_list_after
                and words[1] not in replace_word_list_after
            ):
                replace_word_list_before.append(words[0])
                replace_word_list_after.append(words[1])
    print(f"replace before: {len(replace_word_list_before)}")
    print(f"replace after: {len(replace_word_list_after)}")
    print(f"uniq words before: {len(set(replace_word_list_before))}")
    print(f"uniq words after: {len(set(replace_word_list_after))}")
    print(
        f"uniq words total: {len(set(replace_word_list_before)|(set(replace_word_list_after)))}"
    )

    num_samples = 50
    sampled_list_before, sampled_list_after = sample_pairs(
        replace_word_list_before, replace_word_list_after, num_samples
    )

    pairs_file_path = args.wordpair_list + f"_sampled-{num_samples}"
    with open(pairs_file_path, "w") as fp:
        for i in range(num_samples):
            wb = sampled_list_before[i]
            wa = sampled_list_after[i]
            fp.write(f"{wb} -> {wa}\n")

    write_replaced_corpus(
        args.file_path[1],
        sampled_list_after,
        sampled_list_before,
    )
    write_replaced_corpus(args.file_path[0], [], sampled_list_after)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", nargs=2, help="path of target file")
    parser.add_argument("-l", "--wordpair_list", help="path of wordpair list")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

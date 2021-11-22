import re
import string

# 01234 + '-'|'.'|',' + 56789
NUM = re.compile(r"^\d*[-\./,]*\d+$")
PUNCT = set(string.punctuation)
PUNCT.add("--")
AFFIX = set(["n't", "'s", "'d", "'t"])


def process_lemma_line(line):
    """preprocess

    :return lemma:
    :return lemma_pos:
    :return pos_tags:
    """
    info = line.split()
    if len(info) != 3:
        return None, None, None, None

    word = clean_word_permissive(info[0])
    if word == None:
        return None, None, None, None

    lemma = info[1]
    pos_tags = clean_pos(info[2])
    if pos_tags[0] == "":
        lemma_pos = None
    else:
        lemma_pos = lemma + "_" + pos_tags[0][:2]

    return word, lemma, lemma_pos, pos_tags


def clean_word_permissive(word):
    """clean word

    :return word: number is tokenized, and (PUNCT, AFFIX) are reshaped
    """
    if word == "@" or word == "<p>":
        return None
    elif word in PUNCT:
        return None
    elif word in AFFIX:
        return None
    else:
        word = word.strip().strip("*").lower()
        if NUM.match(word):
            word = "<NUM>"
    return word


def clean_pos(pos):
    """clean and split tags

    :return tags: list of tags in a word
    """
    tags = []
    for tag in pos.split("_"):
        tag = tag.strip("@")
        tag = tag.strip("%")
        tags.append(tag)
    return tags

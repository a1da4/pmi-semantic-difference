import argparse
import numpy as np
import pickle

import gensim
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes


def smart_procrustes_align_gensim(base_embed, other_embed, is_word2vec, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
    """
    if is_word2vec:
        base_embed.init_sims()
        other_embed.init_sims()
        # make sure vocabulary and indices are aligned
        in_base_embed, in_other_embed = intersection_align_gensim(
            base_embed, other_embed, words=words
        )
        # get the embedding matrices
        base_vecs = in_base_embed.wv.syn0norm
        other_vecs = in_other_embed.wv.syn0norm
    else:
        base_vecs = base_embed
        other_vecs = other_embed

    # get the rotate vector "ortho"
    ortho, _ = orthogonal_procrustes(base_vecs, other_vecs)

    if is_word2vec:
        # Replace original array with modified one
        # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
        other_embed.wv.syn0norm = other_embed.wv.syn0 = (other_embed.wv.syn0norm).dot(
            ortho
        )
    else:
        other_embed = other_embed @ ortho

    return in_base_embed, other_embed


def intersection_align_gensim(m1, m2, words=None):
    """Find same words in word2vec m1 and m2, and fix each vocabulary and models.

    :return m1, m2: fixed word2vec. m1.vocab=m2.vocab
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    print(f"before refering words: {len(common_vocab)} words")
    if words:
        common_vocab &= set(words.values())
    print(f"after refering words: {len(common_vocab)} words")

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True
    )

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(
                index=new_index, count=old_vocab_obj.count
            )
        m.wv.vocab = new_vocab

    return (m1, m2)


def main(base_path, other_path, id_to_word, flag):
    try:
        base_embed = Word2Vec.load(base_path)
        other_embed = Word2Vec.load(other_path)
        is_word2vec = True
    except:
        base_embed = np.load(base_path)
        other_embed = np.load(other_path)
        is_word2vec = False

    print(f"is_word2vec: {is_word2vec}")
    base_embed_aligned, other_embed_aligned = smart_procrustes_align_gensim(
        base_embed, other_embed, is_word2vec=is_word2vec, words=id_to_word
    )

    if flag:
        aligned_path = base_path + "_aligned"
        base_embed_aligned.save(aligned_path)

    if is_word2vec:
        aligned_path = other_path + "_aligned"
        other_embed_aligned.save(aligned_path)
    else:
        aligned_path = other_path[:-4] + "_aligned.npy"
        np.save(aligned_path, other_embed_aligned)

    return aligned_path


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", nargs="*", help="path of model(s)")
    parser.add_argument("-p", "--id_to_word", help="path of dict, index2word")
    args = parser.parse_args()

    fp = open(args.id_to_word, "rb")
    id_to_word = pickle.load(fp)
    base_path = args.model_path[0]
    flag = True
    for other_path in args.model_path[1:]:
        print(f"align {other_path} -> {base_path}")
        base_path = main(base_path, other_path, id_to_word, flag)
        flag = False


if __name__ == "__main__":
    cli_main()

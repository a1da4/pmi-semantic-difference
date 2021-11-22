import numpy as np
import torch

from tqdm import tqdm

#############################################
# similarity funcitons
#############################################
def cos_sim(x, y, eps=1e-8):
    """compute cos similarity
    :param x, y: vector
    :param eps: tiny value to avoid deviding 0

    :return: cos similarity
    """

    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def inner_product(base_vec, other_vec):
    """compute cosine similarity with norm
    :param base_vec, other_vec: vector

    :return: inner product
    """
    nb = base_vec
    no = other_vec

    return np.dot(nb, no)


def nn_match_rate(base_mat, other_mat, id2word, bert_vocab=None, k=1000):
    """compute NN matching rate
    :param mat: vectors
    :param id2word: dictionary(id->word)
    :param bert_vocab: list
    :param k: int, k-NN
    :return: similarities (1, V)
    """
    similarities = np.zeros(len(id2word))
    base_sim_mat = make_similarity_matrix(base_mat, id2word, bert_vocab)
    other_sim_mat = make_similarity_matrix(other_mat, id2word, bert_vocab)
    print("make similarity matrixes...done")
    if bert_vocab is not None:
        bert_vocab = set(id2word.values())
    else:
        bert_vocab = set(bert_vocab)
    for word_id in id2word:
        if id2word[word_id] not in bert_vocab:
            similarities[word_id] = 1
        else:
            base_topk = set(np.argsort(base_sim_mat[word_id])[:k])
            other_topk = set(np.argsort(other_sim_mat[word_id])[:k])
            match_rate = len(base_topk & other_topk) / k
            similarities[word_id] = match_rate

    return similarities


#############################################
# preprocess: model -> matrix
#############################################


def w2v_reshape_into_matrix(w2v_model, id2word):
    """"""
    mat = np.zeros([len(id2word), 100])
    for word_id in id2word:
        target_word = id2word[word_id]
        mat[word_id] += w2v_model[target_word]
    return mat


def bert_reshape_into_matrix(bert_vecs, id2word):
    """"""
    mat = np.zeros([len(id2word), 768])
    bert_vocab = set(bert_vecs.keys())
    for word_id in id2word:
        if id2word[word_id] in bert_vocab:
            target_word = id2word[word_id]
            mat[word_id] += bert_vecs[target_word].numpy().copy()
    return mat


#############################################
# functions
#############################################
def calculate_similarities(
    vec_mat_base, vec_mat_other, id2word, bert_vocab=None, f=cos_sim
):
    """calculate cosine similarities
    :param vec_mat_base, vec_mat_other: numpy, (V, dim)
    :param id2word: dictionary(index->word)
    :param bert_vocab: list, bert vocab
    :param f: similarity function
    :return: similarities (V)
    """
    similarities = np.zeros(len(id2word))
    if f == nn_match_rate:
        similarities = f(vec_mat_base, vec_mat_other, id2word, bert_vocab)
    else:
        if bert_vocab is None:
            target_vocab = id2word.values()
        else:
            target_vocab = list(set(id2word.values()) & set(bert_vocab))
        for k in id2word:
            if id2word[k] in target_vocab:
                base_vec = vec_mat_base[k]
                other_vec = vec_mat_other[k]
                sim = f(base_vec, other_vec)
                similarities[k] = sim
            else:
                similarities[k] = 1

    return similarities


def make_similarity_matrix(target_mat, id2word, bert_vocab=None):
    """create similarity matrix (Vocab, Vocab)
    :param target_mat: word vectors
    :param id2word: dictionary(id->word)
    :param bert_vocab: list
    :return: sim_mat
    """
    V = len(id2word)
    sim_mat = np.zeros([V, V])
    if bert_vocab is None:
        target_vocab = set(id2word.values())
    else:
        target_vocab = set(id2word.values()) & set(bert_vocab)

    for i in tqdm(range(V)):
        if id2word[i] not in target_vocab:
            sim_mat[i] = -1
        else:
            for j in range(i, V):
                if id2word[j] not in target_vocab:
                    sim_mat[i][j] = -1
                    sim_mat[j][i] = -1
                else:
                    base_vec = target_mat[i]
                    other_vec = target_mat[j]
                    sim_mat[i][j] = cos_sim(base_vec, other_vec)
                    sim_mat[j][i] = sim_mat[i][j]
        if i % 100 == 0:
            print(f" %nonzero ... {np.count_nonzero(sim_mat) / (V**2) * 100}")
    return sim_mat


def most_similar(query, word2id, id2word, target_mat, matched_vocab, top=5):
    """search most similar top-k words
    :param query: query word
    :param word2id: dictionary(word->id)
    :param id2word: dictionary(id->word)
    :param target_mat: wordvec
    :param matched_vocab: list, matched with bert vocab
    :param top: top-k

    :return: top-k words sorted cos-similarity
    """

    if query not in word2id:
        print("%s is not found" % query)
        return

    print("\n[query] " + query)
    query_id = word2id[query]
    query_vec = target_mat[query_id]

    vocab_size = len(id2word)
    if -1 in id2word:
        vocab_size -= 1

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_sim(target_mat[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        word = id2word[i]
        if word == query or word not in matched_vocab:
            continue
        print(" %s: %.3f" % (word, similarity[i]))

        count += 1
        if count >= top:
            return

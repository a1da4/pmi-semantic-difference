import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_matrix(file_matrix, V):
    """load matrix
    :param file_matrix: path of pre-trained matrix (output file)
    :param V: vocab size

    :return: matrix(list)
    """
    matrix = [[0 for _ in range(V)] for _ in range(V)]
    with open(file_matrix) as fp:
        for line in fp:
            target_id, context_id_values = line.strip().split("\t")
            context_id_values = context_id_values.split()
            for context_id_value in context_id_values:
                context_id, value = context_id_value.split(":")
                matrix[int(target_id)][int(context_id)] += float(value)

    return matrix

def cos_sim(base_vec, other_vec, eps=1e-8):
    """ compute cosine similarity
    :param base_vec, other_vec: vector
    :param eps: tiny value to avoid deviding 0

    :return: cosine similarity
    """
    nb = base_vec / (np.sqrt(np.sum(base_vec ** 2)) + eps)
    no = other_vec / (np.sqrt(np.sum(other_vec ** 2)) + eps)

    return np.dot(nb, no)

def calculate_similarities(Ws, id_to_word, with_norm):
    """ calculate cosine similarities
    :param Ws: list of word vectors
    :param id_to_word: dict, index -> word
    :param with_norm: bool, consider norm of vectors or not (cos or dot product)
    """
    vocab_size = len(id_to_word)
    similarities = np.zeros(vocab_size)
    for k in id_to_word:
        base_vec = Ws[0][k]
        other_vec = Ws[1][k]
        if with_norm:
            sim = np.dot(base_vec, other_vec)
        else:
            sim = cos_sim(base_vec, other_vec)
        similarities[k] = sim

    return similarities

def roc(Ws, id_to_word, dev_words, with_norm=False):
    """ compute auc value
    """
    vocab_size = len(id_to_word)
    word_to_id = {}
    for index, word in id_to_word.items():
        word_to_id[word] = index
    positive_id = [word_to_id[word] for word in dev_words]
    ans = [1 if index in positive_id else 0 for index in range(vocab_size)]

    TP = []
    FP = []
    similarities = calculate_similarities(Ws, id_to_word, with_norm)
    N_tp = len(positive_id)
    N_tn = len(similarities) - N_tp
    for k in tqdm(range(vocab_size)):
        topk = similarities.argsort()[:k]
        pred = [1 if index in topk else 0 for index in range(vocab_size)]
        true_positive_rate = sum([1 if pred[i]==ans[i]==1 else 0 for i in range(vocab_size)]) / N_tp 
        false_positive_rate = sum([1 if pred[i]==1 and ans[i]==0 else 0 for i in range(vocab_size)]) / N_tn

        TP.append(true_positive_rate)
        FP.append(false_positive_rate)

    AUC = 0
    for i in range(vocab_size - 1):
        AUC += np.abs(FP[i+1] - FP[i])*TP[i] 

    return AUC, FP, TP

def plot_roc(fp, tp, auc, tau, seed, with_norm=False):
    plt.plot(fp, tp, label=f'AUC={auc}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    if with_norm:
        plt.savefig(f'ROC_csvd_cds-75_t-{tau}_seed-{seed}_with-norm.png')
    else: 
        plt.savefig(f'ROC_t-{tau}_seed-{seed}.png')
    plt.close()

def plot_rocs(fps, tps, aucs, taus):
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    for fp, tp, auc, tau in zip(fps, tps, aucs, taus):
        plt.plot(fp, tp, label='tau-{}, AUC={}'.format(tau, format(auc, '3f')))
    plt.legend()
    plt.savefig('ROC_csvd.png')
    plt.close()

def plot_loss(losses, tau, seed):
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(range(len(losses)), losses)
    plt.savefig(f'DWE_csvd_t-{tau}_seed-{seed}.png')
    plt.close()

def plot_losses(losses_list, taus):
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    for losses, tau in zip(losses_list, taus):
        plt.plot(range(len(losses)), losses, label='tau-{}'.format(tau))
    plt.legend()
    plt.savefig('DWE_csvd.png')
    plt.close()


import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from model import SimplifiedDynamicWordEmbeddigs
from util import cos_sim, calculate_similarities, roc, plot_roc, plot_loss


def grid_search(args, id_to_word):
    """grid search for hyper parameters
    :searching param tau:

    :return best_loss, best_Ws, best_Cs, best_tau:
    """
    logging.basicConfig(
        filename=f"search_seed-{args.seed}.log",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    logging.info(f"[INFO] args: {args}")
    time_bins = len(args.ppmi_pathes)
    dev_words = []
    with open(args.dev_list) as fp:
        for line in fp:
            word = line.strip()
            dev_words.append(word)
    best_auc = 0
    best_fp = None
    best_tp = None
    best_losses = None
    best_loss = float("inf")
    best_tau = None
    best_Ws = None
    best_Cs = None
    for tau in tqdm(range(7)):
        tau = 10 ** (tau - 3)
        logging.info(f"[INFO] tau: {tau}")
        model = SimplifiedDynamicWordEmbeddigs(
            time_bins=time_bins, dim=args.dim, tau=tau, es=args.es
        )
        model.load_ppmi_matrix(args.ppmi_pathes, len(id_to_word))

        losses, loss, Ws, Cs, is_es = model.train(args.n_iter, args.seed)
        logging.info(f"[INFO] train finished")
        if is_es:
            logging.info("[INFO] early stopping: compute auc is skipped")
            continue
        auc, fp, tp = roc([Ws[0], Ws[-1]], id_to_word, dev_words, with_norm=False)
        plot_roc(fp, tp, auc, tau, args.seed)
        logging.info(f"auc: {auc}")

        plot_loss(losses, tau, args.seed)
        if auc > best_auc:
            best_auc = auc
            best_fp = fp
            best_tp = tp
            best_losses = losses
            best_loss = loss
            best_tau = tau
            best_Ws = Ws
            best_Cs = Cs
    fp = open(f"./Ws_d-{args.dim}_t-{best_tau}_xavier_seed-{args.seed}.pkl", "wb")
    pickle.dump(best_Ws, fp)
    logging.info("[INFO] grid search finished")
    return


def main(args):
    print(args)
    fp = open(args.id_to_word, "rb")
    id_to_word = pickle.load(fp)
    grid_search(args, id_to_word)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--ppmi_pathes", nargs="*", help="path of ppmi-matrixes.")
    parser.add_argument("-s", "--seed", type=int, default=1, help="int, random seed.")
    parser.add_argument(
        "-d", "--dim", type=int, default=100, help="int, dimension of word embedding."
    )
    parser.add_argument(
        "-n", "--n_iter", type=int, default=5, help="int, iteration of training."
    )
    parser.add_argument(
        "-e", "--es", type=int, default=3, help="int, patients of early stopping."
    )
    parser.add_argument("-p", "--id_to_word", help="path of id_to_word dictionary")
    parser.add_argument("-l", "--dev_list", help="path of dev word list")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

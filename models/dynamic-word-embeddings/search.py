import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from model import DynamicWordEmbeddigs
from util import cos_sim, calculate_similarities, roc, plot_roc, plot_loss


def grid_search(args, id_to_word):
    """grid search for hyper parameters
    :searching param lam, tau(=gam): lambda, tau, gammma

    :return best_loss, best_lam, best_tau:
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
    best_lam = None
    best_tau = None
    best_gam = None
    best_Ws = None
    best_Us = None
    for tau in tqdm(range(7)):
        tau = 10 ** (tau - 3)
        for lam in range(7):
            lam = 10 ** (lam - 3)
            for gam in range(7):
                gam = 10 ** (gam - 3)
                logging.info(f"[INFO] lam: {lam}, tau: {tau} ,gamma: {gam}")
                model = DynamicWordEmbeddigs(
                    time_bins=time_bins,
                    dim=args.dim,
                    lam=lam,
                    tau=tau,
                    gam=gam,
                    es=args.es,
                )
                model.load_ppmi_matrix(args.ppmi_pathes, len(id_to_word))

                losses, loss, Ws, Us, is_es = model.train(args.n_iter, args.seed)
                logging.info(f"[INFO] train finished")
                if is_es:
                    logging.info("[INFO] early stopping: compute auc skipped")
                    continue
                plot_loss(losses, lam, tau, gam, args.es)
                auc, fp, tp = roc(
                    [Ws[0], Ws[-1]], id_to_word, dev_words, with_norm=False
                )
                plot_roc(fp, tp, auc, lam, tau, gam, args.es)
                logging.info("auc: {}".format(auc))
                if auc > best_auc:
                    best_auc = auc
                    best_fp = fp
                    best_tp = tp
                    best_loss = loss
                    best_lam = lam
                    best_tau = tau
                    best_gam = gam
                    best_losses = losses
                    best_Ws = Ws
                    best_Us = Us
    logging.info(
        f"[INFO] Best_parameter:\n loss: {best_loss}\n lambda: {best_lam}\n tau: {best_tau}\n gamma: {best_gam}\n"
    )
    return best_losses, best_Ws, best_Us, best_lam, best_tau, best_gam


def main(args):
    print(args)
    fp = open(args.id_to_word, "rb")
    id_to_word = pickle.load(fp)
    losses, best_Ws, best_Us, best_lam, best_tau, best_gam = grid_search(
        args, id_to_word
    )
    fp = open(
        f"./Ws_asym_d-{args.dim}_l-{best_lam}_t-{best_tau}_g-{best_gam}_es-{args.es}_seed-{args.seed}.pkl",
        "wb",
    )
    pickle.dump(best_Ws, fp)
    logging.info("[INFO] finished")


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
        "-e", "--es", type=int, default=3, help="int, hyperparameter. early stopping."
    )
    parser.add_argument("-p", "--id_to_word", help="path of id_to_word dictionary")
    parser.add_argument("-w", "--dev_list", help="path of dev word list")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

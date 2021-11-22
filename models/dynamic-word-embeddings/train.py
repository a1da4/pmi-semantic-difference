import argparse
import pickle

from model import DynamicWordEmbeddigs
from util import cos_sim, calculate_similarities, roc, plot_roc, plot_loss


def main(args):
    print(args)
    fp = open(args.id_to_word, "rb")
    id_to_word = pickle.load(fp)
    time_bins = len(args.ppmi_pathes)
    dev_words = []
    with open(args.dev_list) as fp:
        for line in fp:
            word = line.strip()
            dev_words.append(word)
    model = DynamicWordEmbeddigs(
        time_bins=time_bins,
        dim=args.dim,
        lam=args.lam,
        tau=args.tau,
        gam=args.gam,
        es=args.es,
    )
    model.load_ppmi_matrix(args.ppmi_pathes, len(id_to_word))
    losses, best_loss, best_Ws, best_Us = model.train(args.n_iter, args.seed)
    fp = open(
        f"./Ws_d-{args.dim}_l-{args.lam}_t-{args.tau}_g-{args.gam}_es-{args.es}.pkl",
        "wb",
    )
    pickle.dump(best_Ws, fp)
    auc, fp, tp = roc(best_Ws, id_to_word, dev_words, with_norm=False)
    plot_roc(fp, tp, auc, args.lam, args.tau, args.gam, args.es)
    plot_loss(losses, args.lam, args.tau, args.gam, args.es)


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
        "-l",
        "--lam",
        type=float,
        default=10.0,
        help="float, hyperparameter. Strength approximating PPMI.",
    )
    parser.add_argument(
        "-t",
        "--tau",
        type=float,
        default=50.0,
        help="float, hyperparameter. Strength of kalman-filter in main embedding.",
    )
    parser.add_argument(
        "-g",
        "--gam",
        type=float,
        default=50.0,
        help="float, hyperparameter. Strength of kalman-filter in main embedding.",
    )
    parser.add_argument("-e", "--es", type=int, default=3, help="int, early stopping")
    parser.add_argument("-p", "--id_to_word", help="path of id_to_word dictionary")
    parser.add_argument("-w", "--dev_list", help="path of dev word list")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

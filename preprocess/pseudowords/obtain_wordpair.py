import argparse
import numpy as np
from tqdm import tqdm

from ioutils import load_pickle
from matrixutils import cos_sim, make_similarity_matrix


def main(args):
    """
    1. V*V ペアで共起行列の内積を計算
    2. 内積が0となるペアを抽出
    """
    print(args)
    id2word = load_pickle(args.id2word)
    if args.similarity_matrixes is not None:
        print("1. load sims_mat ...", end="")
        sims_path_base, sims_path_other = args.similarity_matrixes
        sims_mat_base = np.load(sims_path_base)
        sims_mat_other = np.load(sims_path_other)
        print("done")
    else:
        print("1. create sims_mat ...", end="")
        assert args.cooccur_matrixes is not None, "COOCCUR NOT FOUND ERROR"
        c_path_base, c_path_other = args.cooccur_matrixes
        c_mat_base = np.load(c_path_base)
        c_mat_other = np.load(c_path_other)
        sims_mat_base = make_similarity_matrix(c_mat_base, id2word)
        sims_mat_other = make_similarity_matrix(c_mat_other, id2word)
        del c_mat_base
        del c_mat_other
        sims_path_base = c_path_base[:-4] + "_sims.npy"
        sims_path_other = c_path_other[:-4] + "_sims.npy"
        np.save(sims_path_base, sims_mat_base)
        np.save(sims_path_other, sims_mat_other)
        print("done")

    print("2. obtain target ids ...", end="")
    threshold = 0.0001 # |sim| < 0.01
    is_target_base = (sims_mat_base**2 <= threshold)
    is_target_other = (sims_mat_other**2 <= threshold)
    targets = np.where(is_target_base * is_target_other == True)
    print("done")

    print("3. create ids ...", end="")
    target_ids = []
    print("\n pairs:")
    for row_ar, col_ar in zip(targets[0], targets[1]):
        row_id = int(row_ar)
        col_id = int(col_ar)
        if row_id < col_id:
            print(f" - {id2word[row_id]}, {id2word[col_id]}")
    print("done")
    
def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cooccur_matrixes", nargs=2, help="path of co occur matrixes"
    )
    parser.add_argument(
        "-s", "--similarity_matrixes", nargs=2, help="path of similarity matrixes"
    )
    parser.add_argument("-p", "--id2word", help="path of index2word")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

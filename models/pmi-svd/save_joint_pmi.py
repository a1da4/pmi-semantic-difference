import argparse
import sys

import numpy as np
import logging
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

sys.path.append("sppmi-svd")

from util import *

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dic_id2word", help="path of id2word dict")
parser.add_argument("-m", "--path_models", nargs="*", help="path of models")
parser.add_argument("-d", "--dim", type=int, help="dimension")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.debug(f'[DEBUG] args: {args}')

id2word, word2id = load_pickle(args.dic_id2word)
V = len(id2word)

M = load_matrix(args.path_models[0], V)
for other_path in args.path_models[1:]:
    other_model = load_matrix(other_path, V)
    M = np.vstack((M, other_model))

dim = args.dim
U, S, V = svds(M, k=dim)
joint_word_vec = np.dot(U, np.sqrt(np.diag(S)))
vec_name = f"WV_joint_dim-{str(dim)}.npy"
np.save(vec_name, joint_word_vec)
logging.info('[INFO] finished')

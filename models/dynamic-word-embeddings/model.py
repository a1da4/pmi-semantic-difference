import numpy as np
from tqdm import tqdm

from util import load_matrix


class DynamicWordEmbeddigs:
    def __init__(self, time_bins, dim, lam, tau, gam, es):
        self.time_bins = time_bins
        self.dim = dim
        self.lam = lam
        self.tau = tau
        self.gam = gam
        self.es = es

    def load_ppmi_matrix(self, ppmi_pathes, V):
        """load ppmi matrixes

        :param ppmi_pathes: list of ppmi matrix pathes
        :param V: int, vocab size

        :return Ms: list of ppmi matrixes
        """
        self.Ms = [np.load(ppmi_path) for ppmi_path in ppmi_pathes]
        ppmis = []
        for ppmi_path in ppmi_pathes:
            ppmi_list = load_matrix(ppmi_path, V)
            ppmi_np = np.array(ppmi_list)
            ppmis.append(ppmi_np)
        self.V = V
        return self

    def initialize_embedding(self, seed):
        """initialize word embeddings

        :param time_bins: int, number of time-bins
        :param seed: int, random seed
        :param dim: int, dimension want to decomposition

        :return Ws: list of initialized word embeddings. Ws is main embeddings.
        :return Us: list of initialized word embeddings. Us is introduced to break the symmetry of ppmi matrix M.
        """
        np.random.seed(seed)
        self.Ws = [
            np.random.randn(self.V, self.dim) / np.sqrt(self.dim)
            for _ in range(self.time_bins)
        ]
        self.Us = [
            np.random.randn(self.V, self.dim) / np.sqrt(self.dim)
            for _ in range(self.time_bins)
        ]
        return self

    def update(self, W, M, W_past, W_next):
        """update embedding W(t)[word_index][:]

        :param W: word embedding W(t)[word_index][:]
        :param M: PPMI matrix
        :param W_past: W(t-1)[word_index][:]
        :param W_next: W(t+1)[word_index][:]
        :param lam: normalize parameter, lambda
        :param tau: control alignment parameter, tau
        :param gam: control alignment parameter, gamma
        :param is_B: this time-step t of W(t) is begin(t=0)
        :param is_E: this time-step t of W(t) is end(t=T)

        :return W_hat: updated W(t)
        """
        WtW = np.dot(W.T, W)  # dim*dim
        if self.is_B or self.is_E:
            A = WtW + (self.lam + self.tau + self.gam) * np.eye(self.dim)  # dim*dim
        else:
            A = WtW + (self.lam + 2 * self.tau + self.gam) * np.eye(self.dim)  # dim*dim
        WtM = np.dot(W.T, M)  # dim*V
        B = WtM + self.gam * W.T + self.tau * (W_past.T + W_next.T)  # dim*V

        W_hat = np.linalg.lstsq(A, B)  # dim*V

        return W_hat[0].T  # V*dim

    def compute_loss(self, t):
        """
        :param t: current time-bin
        """
        loss = (
            1 / 2 * np.linalg.norm(self.Ms[t] - self.Ws[t] @ self.Us[t].T) ** 2
            + self.lam
            / 2
            * (np.linalg.norm(self.Ws[t]) ** 2 + np.linalg.norm(self.Us[t]) ** 2)
            + self.gam / 2 * np.linalg.norm(self.Ws[t] - self.Us[t]) ** 2
        )

        if self.is_B:
            return loss
        else:
            loss += (
                self.tau / 2 * np.linalg.norm(self.Ws[t] - self.Ws[t - 1]) ** 2
                + self.tau / 2 * np.linalg.norm(self.Us[t] - self.Us[t - 1]) ** 2
            )
        return loss

    def train(self, n_iter, seed, time_shuffle=True):
        """training embeddings (early stopping is adopted)

        :param n_iter: iteration
        :param seed: random seed
        :param time_shuffle: bool, shuffle the update time-bins in each iteration

        :return losses: list, losses in each iteration
        :return best_loss: float, best loss in all iterations
        :return best_Ws, best_Us: best embeddings (Ws is main embedding)
        """
        self.initialize_embedding(seed)
        losses = []
        is_es = False
        best_loss = float("inf")
        best_Ws = None
        best_Us = None
        patience = 0
        for n in tqdm(range(n_iter)):
            loss = 0
            if time_shuffle and n > 0:
                T = np.random.permutation(self.time_bins)
            else:
                T = range(self.time_bins)
            for t in T:
                W = self.Ws[t]
                U = self.Us[t]
                M = self.Ms[t]
                if t == 0:
                    self.is_B = True
                    W_past = np.zeros((self.V, self.dim))
                    U_past = np.zeros((self.V, self.dim))
                else:
                    self.is_B = False
                    W_past = self.Ws[t - 1]
                    U_past = self.Us[t - 1]
                if t == self.time_bins - 1:
                    self.is_E = True
                    W_next = np.zeros((self.V, self.dim))
                    U_next = np.zeros((self.V, self.dim))
                else:
                    self.is_E = False
                    W_next = self.Ws[t + 1]
                    U_next = self.Us[t + 1]
                self.Ws[t] = self.update(U, M, W_past, W_next)
                self.Us[t] = self.update(W, M, U_past, U_next)
            for t in T:
                if t == 0:
                    self.is_B = True
                else:
                    self.is_B = False
                loss += self.compute_loss(t)
            if loss < best_loss:
                best_loss = loss
                best_Ws = self.Ws
                best_Us = self.Us
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    print("Early Stopping! (patience over 20) iteration: {}".format(n))
                    if n == 20:
                        is_es = True
                    break
            losses.append(loss)
            if len(losses) > self.es:
                not_updated = 0
                # losses[n], losses[n-1], ..., losses[n-self.es]
                for i in range(self.es):
                    post = losses[n - i]
                    pre = losses[n - i - 1]
                    if pre < post:
                        not_updated += 1
                if not_updated == self.es:
                    print("Early Stopping! iteration: {} ".format(n))
                    if n == self.es:
                        is_es = True
                    break
        return losses, best_loss, best_Ws, best_Us, is_es

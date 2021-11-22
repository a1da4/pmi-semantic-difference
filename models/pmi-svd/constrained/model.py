import numpy as np
from tqdm import tqdm

from util import load_matrix

class SimplifiedDynamicWordEmbeddigs():
    def __init__(self, time_bins, dim, tau, es):
        self.time_bins = time_bins
        self.dim = dim
        self.tau = tau
        self.es = es

    def load_ppmi_matrix(self, ppmi_pathes):
        """ load ppmi matrixes
        
        :param ppmi_pathes: list of ppmi matrix pathes

        :return Ms: list of ppmi matrixes 
        :return V: int, vocab size
        """
        ppmis = []
        for ppmi_path in args.ppmi_pathes:
            ppmi_list = load_matrix(ppmi_path, len(id_to_word))
            ppmi_np = np.array(ppmi_list)
            ppmis.append(ppmi_np)
        model.Ms = ppmis
        self.V = self.Ms[0].shape[0]
        return self

    def initialize_embedding(self, seed):
        """ initialize word embeddings

        :param time_bins: int, number of time-bins
        :param seed: int, random seed
        :param dim: int, dimension want to decomposition

        :return Ws: list of initialized word embeddings. Ws is main embeddings. 
        :return Cs: list of initialized word embeddings. Cs is introduced to break the symmetry of ppmi matrix M. 
        """
        np.random.seed(seed)
        # xavier
        self.Ws = [np.random.randn(self.V, self.dim)/np.sqrt(self.dim) for _ in range(self.time_bins)]
        self.Cs = [np.random.randn(self.V, self.dim)/np.sqrt(self.dim) for _ in range(self.time_bins)]
        return self

    def update(self, W, C, M, C_past, C_next, is_target=True):
        """ update embedding W(t)[word_index][:]
        
        :param W: word embedding W(t)[word_index][:]
        :param C: word embedding C(t)[word_index][:]
        :param M: PPMI matrix
        :param C_past: C(t-1)[word_index][:]
        :param C_next: C(t+1)[word_index][:]
        :param is_B: this time-step t of W(t) is begin(t=0)
        :param is_E: this time-step t of W(t) is end(t=T)
        :param is_target: bool, target vector Wt or not

        :return W_hat: updated W(t)
        """
        if is_target:
            D = 1 / np.sqrt(np.linalg.norm(M - W@C.T))
            CtC = np.dot(C.T, C)
            CtM = np.dot(C.T, M)
            A = CtC 
            B = CtM #dim*V

        else:
            D = 1 / np.sqrt(np.linalg.norm(M - W@C.T) + 1e-8)
            if self.is_B:
                E1 = 0.0
            else:
                E1 = 1 / np.sqrt(np.linalg.norm(C_past - C) + 1e-8)
            if self.is_E:
                E2 = 0.0
            else:
                E2 = 1 / np.sqrt(np.linalg.norm(C - C_next) + 1e-8)
            WtW = np.dot(W.T, W) #dim*dim
            WtM = np.dot(W.T, M) #dim*V
            if self.is_B or self.is_E:
                A = D*WtW + (E1+E2)*self.tau*np.eye(self.dim) #dim*dim
            else:
                A = D*WtW + (E1+E2)*2*self.tau*np.eye(self.dim) #dim*dim
            B = D*WtM + self.tau*(E1*C_past.T + E2*C_next.T) #dim*V

        W_hat = np.linalg.lstsq(A, B) #dim*V
        return W_hat[0].T #V*dim

    def normalize(self, t):
        """ vector normalization """
        Zw = np.linalg.norm(self.Ws[t], axis=1)
        Zc = np.linalg.norm(self.Cs[t], axis=1)
        Wt_normalized = self.Ws[t].T / Zw
        Ct_normalized = self.Cs[t].T / Zc
        self.Ws[t] = Wt_normalized.T
        self.Cs[t] = Ct_normalized.T
        return self

    def compute_loss(self, t):
        """
        :param t: current time-bin
        """
        loss = np.linalg.norm(self.Ms[t] - self.Ws[t]@self.Cs[t].T)

        if self.is_B:
            return loss
        else:
            loss += self.tau * np.linalg.norm(self.Cs[t-1] - self.Cs[t])
        return loss

    def train(self, n_iter, seed, time_shuffle=True):
        """ training embeddings (early stopping is adopted)

        :param n_iter: iteration
        :param seed: random seed
        :param time_shuffle: bool, shuffle the update time-bins in each iteration

        :return losses: list, losses in each iteration
        :return best_loss: float, best loss in all iterations
        :return best_Ws, best_Cs: best embeddings (Ws is main embedding)
        """
        is_es = False
        self.initialize_embedding(seed)
        losses = []
        best_loss = float('inf')
        best_Ws = None
        best_Cs = None
        not_updated = 0
        for n in tqdm(range(n_iter)):
            if time_shuffle and n > 0:
                T = np.random.permutation(self.time_bins)
            else:
                T = range(self.time_bins)
            for t in T:
                W = self.Ws[t]
                C = self.Cs[t]
                M = self.Ms[t]
                if t == 0:
                    self.is_B = True
                    W_past = np.zeros((self.V, self.dim))
                    C_past = np.zeros((self.V, self.dim))
                else:
                    self.is_B = False
                    W_past = self.Ws[t-1]
                    C_past = self.Cs[t-1]
                if t == self.time_bins - 1:
                    self.is_E = True
                    W_next = np.zeros((self.V, self.dim))
                    C_next = np.zeros((self.V, self.dim))
                else:
                    self.is_E = False
                    W_next = self.Ws[t+1]
                    C_next = self.Cs[t+1]
                self.Ws[t] = self.update(W, C, M, W_past, W_next, is_target=True)
                self.Cs[t] = self.update(W, C, M, C_past, C_next, is_target=False)
            loss = 0
            for t in T:
                if t==0:
                    self.is_B = True
                else:
                    self.is_B = False
                loss += self.compute_loss(t)
            if loss < best_loss:
                best_loss = loss
                best_Ws = self.Ws
                best_Cs = self.Cs
                patience = 0
            else:
                patience += 1
            losses.append(loss)

            if patience >= 20:
                print('Early Stopping!(patience over 20) iteration: {} '.format(n))
                if n == 20:
                    is_es = True
                break

            if len(losses) > self.es:
                not_updated = 0
                #losses[n], losses[n-1], ..., losses[n-self.es]
                for i in range(self.es):
                    post = losses[n-i]
                    pre = losses[n-i-1]
                    if pre < post:
                        not_updated += 1
                if not_updated == self.es:
                    print('Early Stopping!(loss is increasing) iteration: {} '.format(n))
                    if n == self.es:
                        is_es = True
                    break

        return losses, best_loss, best_Ws, best_Cs, is_es


# exp 9 bayesian personalisation

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class BPR:
    def __init__(self, f=20, lr=0.01, reg=0.01, iters=100):
        self.f, self.lr, self.reg, self.iters = f, lr, reg, iters

    def fit(self, df):
        mat = csr_matrix((np.ones(len(df)), (df.user_id, df.item_id)))
        self.n_users, self.n_items = mat.shape
        self.U = np.random.normal(0, 1/self.f, (self.n_users, self.f))
        self.V = np.random.normal(0, 1/self.f, (self.n_items, self.f))
        triplets = []
        for u in range(self.n_users):
            pos = mat[u].indices
            if not len(pos): continue
            neg = np.setdiff1d(np.arange(self.n_items), pos, assume_unique=True)
            for i in pos:
                triplets.append((u, i, np.random.choice(neg)))
        for _ in range(self.iters):
            np.random.shuffle(triplets)
            for u, i, j in triplets:
                x = np.dot(self.U[u], self.V[i] - self.V[j])
                sig = 1 / (1 + np.exp(x))
                self.U[u] += self.lr * (sig * (self.V[i] - self.V[j]) - self.reg * self.U[u])
                self.V[i] += self.lr * (sig * self.U[u] - self.reg * self.V[i])
                self.V[j] += self.lr * (-sig * self.U[u] - self.reg * self.V[j])

    def predict(self, u, items):
        return np.dot(self.U[u], self.V[items].T)

if __name__ == '__main__':
    np.random.seed(42)
    df = pd.DataFrame({
        'user_id': np.random.randint(0, 100, 1000),
        'item_id': np.random.randint(0, 50, 1000),
        'rating': np.random.randint(1, 6, 1000)
    })
    model = BPR(iters=50)
    model.fit(df)
    preds = model.predict(0, list(range(5)))
    print("Predictions for user 0:", preds.round(3))

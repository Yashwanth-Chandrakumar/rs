#  exp 3: user profile learning

import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class UserProfile:
    def __init__(self, uid):
        self.uid = uid
        self.pref = defaultdict(float)
        self.history = []
    def update(self, feats, rating, lr=0.1):
        r = (rating - 3) / 2
        for k, v in feats.items():
            self.pref[k] += lr * (r - self.pref[k] * v) * v
        self.history.append((feats, rating))
    def vector(self):
        if not self.history: return {}
        vec, total = defaultdict(float), 0
        for feats, rating in self.history:
            r = (rating - 3) / 2; total += abs(r)
            for k, v in feats.items():
                vec[k] += r * v
        if total:
            for k in vec: vec[k] /= total
        return dict(vec)

def recommend(user, items):
    uvec = np.array(list(user.vector().values()))
    scores = {iid: cosine_similarity(uvec.reshape(1, -1),
                np.array(list(feats.values())).reshape(1, -1))[0][0]
              for iid, feats in items.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    user = UserProfile("u1")
    user.update({'a': 1, 'b': 0.5}, 5)
    user.update({'a': 0.5, 'b': 1}, 1)
    items = {1: {'a': 1, 'b': 0}, 2: {'a': 0, 'b': 1}, 3: {'a': 1, 'b': 1}}
    print("Recommendations:", recommend(user, items))

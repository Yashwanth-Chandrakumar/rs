import pandas as pd, numpy as np

class ConstraintRec:
    def __init__(self):
        self.products = None
        self.cons = {}
        self.w = {}
    def load(self, df):
        self.products = df.copy()
    def add(self, attr, typ, val, weight=1.0):
        self.cons[attr] = (typ, val)
        self.w[attr] = weight
    def score(self, row):
        if not self.cons: return 1.0
        tot, s = sum(self.w.values()), 0
        for a, (typ, val) in self.cons.items():
            p = row.get(a, np.nan)
            if pd.isna(p): sc = 0
            elif typ == 'exact': sc = 1 if p == val else 0
            elif typ == 'min': sc = 1 if p >= val else 0
            elif typ == 'max': sc = 1 if p <= val else 0
            else: sc = 0
            s += sc * self.w[a]
        return s / tot
    def recs(self, top_n=None, min_score=0):
        df = self.products.copy()
        df['score'] = df.apply(self.score, axis=1)
        df = df[df['score'] >= min_score].sort_values('score', ascending=False)
        return df.head(top_n) if top_n else df

if __name__ == '__main__':
    data = {
        'id': range(1, 11),
        'name': [f'Product {i}' for i in range(1, 11)],
        'category': ['Electronics', 'Electronics', 'Clothing', 'Clothing',
                     'Books', 'Books', 'Electronics', 'Clothing', 'Books', 'Electronics'],
        'price': [999.99, 499.99, 59.99, 89.99, 19.99, 29.99, 799.99, 129.99, 24.99, 1499.99],
        'rating': [4.5, 4.2, 4.8, 4.0, 4.7, 4.3, 4.6, 4.1, 4.4, 4.9],
        'in_stock': [True, True, False, True, True, True, False, True, True, True]
    }
    df = pd.DataFrame(data)
    rec = ConstraintRec()
    rec.load(df)
    rec.add('category', 'exact', 'Electronics', weight=1.0)
    rec.add('price', 'max', 1000.0, weight=0.8)
    rec.add('rating', 'min', 4.0, weight=0.6)
    rec.add('in_stock', 'exact', True, weight=0.9)
    result = rec.recs(top_n=5, min_score=0.5)
    print(result[['name', 'category', 'price', 'rating', 'score']])

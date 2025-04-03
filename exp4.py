# exp 4 content based recommender

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import PorterStemmer

class ContentBasedRecommender:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        self.scaler = MinMaxScaler()

    def preprocess(self, text):
        text = re.sub(r'[^a-z\s]', '', str(text).lower())
        return ' '.join(self.stemmer.stem(word) for word in text.split())
    
    def fit(self, df):
        self.df = df.copy()
        self.df['proc'] = self.df['description'].apply(self.preprocess)
        self.tfidf = self.vectorizer.fit_transform(self.df['proc'])
        self.sim = cosine_similarity(self.tfidf)
        num_cols = [c for c in ['rating', 'popularity', 'year'] if c in self.df.columns]
        if num_cols:
            num = self.scaler.fit_transform(self.df[num_cols])
            self.features = np.hstack([self.tfidf.toarray(), num])
        else:
            self.features = self.tfidf.toarray()
    
    def similar_items(self, item_id, n=5):
        idx = self.df.index[self.df['item_id'] == item_id][0]
        scores = self.sim[idx]
        top = scores.argsort()[::-1][1:n+1]
        return self.df.iloc[top][['item_id', 'title']].assign(similarity=scores[top].round(3))
    
    def recommend(self, user_profile, n=5):
        score = np.zeros(len(self.df))
        vocab = self.vectorizer.get_feature_names_out()
        for feat, w in user_profile.items():
            feat_stem = self.stemmer.stem(feat)
            indices = [i for i, name in enumerate(vocab) if feat_stem in name]
            for i in indices:
                score += w * self.features[:, i]
        top = score.argsort()[::-1][:n]
        return self.df.iloc[top][['item_id', 'title']].assign(score=score[top].round(3))

# Sample data and testing
def sample_data():
    data = {
        'item_id': range(1, 11),
        'title': ['The Matrix', 'Inception', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump', 
                  'The Shawshank Redemption', 'The Godfather', 'Fight Club', 'Interstellar', 'Gladiator'],
        'description': [
            'A computer programmer discovers a dystopian world inside the Matrix with sci-fi action',
            'A thief who enters the dreams of others to steal secrets in this sci-fi thriller',
            'Batman fights against the criminal mastermind known as the Joker in this action movie',
            'Various interconnected stories of criminals in Los Angeles crime drama',
            'A slow-witted but kind-hearted man witnesses historic events in this drama',
            'A banker is sentenced to life in Shawshank State Penitentiary drama',
            'The aging patriarch of an organized crime dynasty transfers control crime drama',
            'An insomniac office worker and a soap maker form an underground fight club action',
            'A team of explorers travel through a wormhole in space sci-fi adventure',
            'A former Roman General seeks revenge against the corrupt emperor action drama'
        ],
        'year': [1999, 2010, 2008, 1994, 1994, 1994, 1972, 1999, 2014, 2000],
        'rating': [8.7, 8.8, 9.0, 8.9, 8.8, 9.3, 9.2, 8.8, 8.6, 8.5],
        'popularity': [85, 90, 95, 88, 92, 96, 94, 87, 89, 86]
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = sample_data()
    rec = ContentBasedRecommender()
    rec.fit(df)
    print("Similar items to The Matrix:")
    print(rec.similar_items(item_id=1, n=3))
    user_profile = {'action': 0.8, 'sci-fi': 0.9, 'drama': 0.3, 'crime': 0.4}
    print("\nUser profile recommendations:")
    print(rec.recommend(user_profile, n=3))

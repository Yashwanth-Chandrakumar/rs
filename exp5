# exp 5
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimpleCollaborativeFiltering:
    def __init__(self, method='user'):
        self.method = method
        self.matrix = None
        self.sim_matrix = None
    
    def fit(self, ratings_df):
        # Create user-item matrix
        self.matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        # Compute similarity matrix
        if self.method == 'user':
            self.sim_matrix = cosine_similarity(self.matrix)
        else:  # item-based
            self.sim_matrix = cosine_similarity(self.matrix.T)
        return self
    
    def predict(self, user_id, item_id, k=5):
        try:
            if self.method == 'user':
                # Get user index and similar users
                user_idx = self.matrix.index.get_loc(user_id)
                sim_scores = self.sim_matrix[user_idx]
                top_users = np.argsort(sim_scores)[::-1][1:k+1]  # Exclude self
                
                # Get ratings and similarities for the target item
                item_col = self.matrix.columns.get_loc(item_id)
                ratings = np.array([self.matrix.iloc[u, item_col] for u in top_users])
                sims = sim_scores[top_users]
                
                # Filter out users with zero rating
                mask = ratings > 0
                if not any(mask):
                    return self.matrix.iloc[:, item_col].mean()
                # Weighted average prediction
                return np.sum(ratings[mask] * sims[mask]) / np.sum(sims[mask])
            else:
                # Item-based: Get item index and similar items
                item_idx = self.matrix.columns.get_loc(item_id)
                sim_scores = self.sim_matrix[item_idx]
                top_items = np.argsort(sim_scores)[::-1][1:k+1]  # Exclude self
                
                # Get ratings and similarities from the user's row
                user_row = self.matrix.index.get_loc(user_id)
                ratings = np.array([self.matrix.iloc[user_row, i] for i in top_items])
                sims = sim_scores[top_items]
                
                # Filter out items with zero rating
                mask = ratings > 0
                if not any(mask):
                    return self.matrix.iloc[user_row, :].mean()
                # Weighted average prediction
                return np.sum(ratings[mask] * sims[mask]) / np.sum(sims[mask])
        except Exception as e:
            # Default to overall average rating if prediction fails
            print("Prediction error:", e)
            return self.matrix.values.mean()

# Example usage:
if __name__ == '__main__':
    ratings = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3], 
        'item_id': [101, 102, 101, 103, 102], 
        'rating': [5, 3, 4, 1, 2]
    })
    # Initialize and fit the model
    cf = SimpleCollaborativeFiltering(method='user').fit(ratings)
    # Make prediction for user 1 on item 103
    prediction = cf.predict(user_id=1, item_id=103)
    print("Predicted rating for user 1 on item 103:", prediction)

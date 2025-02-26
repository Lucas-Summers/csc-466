import numpy as np
import pandas as pd

class RatingsMatrix:
    def __init__(self, filepath):
        self.ratings = self.load_data(filepath)
        self.user_means = np.nanmean(self.ratings, axis=1)  # Mean per user, ignoring NaNs

    def load_data(self, f, verbose=False):
        df = pd.read_csv(f, header=None)
        df = df.iloc[:, 1:].replace(99, np.nan)
        if verbose:
            print(df.head())
        return df.to_numpy()
    
    def get_user_ratings(self, user_id):
        return self.ratings[user_id]

    def get_item_ratings(self, item_id):
        return self.ratings[:, item_id]

    def mean_utility(self):
        return np.nanmean(self.ratings)
    
    def predict_rating(self, user_id, item_id, similarity, use_adjusted=False):
        '''
        Predict the rating of a user for an item using the similarity function.
        If use_adjusted is True, the adjusted weighted sum is used.
        '''
        original_value = self.ratings[user_id][item_id]
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = np.nan  

        # Compute similarity for all users (vectorized)
        similarity_vector = np.array([
            similarity(self.ratings[user_id], self.ratings[other]) if other != user_id else 0
            for other in range(len(self.ratings))
        ])

        # Mask NaN values in the target item column
        valid_ratings_mask = ~np.isnan(self.ratings[:, item_id])

        if use_adjusted:
            # Adjusted weighted sum: use (rating - user mean)
            adjusted_ratings = (self.ratings[:, item_id] - self.user_means) * valid_ratings_mask
            num = np.nansum(similarity_vector * adjusted_ratings)
            denom = np.nansum(np.abs(similarity_vector) * valid_ratings_mask)
            prediction = self.user_means[user_id] + (num / denom if denom != 0 else 0)
        else:
            # Regular weighted sum
            num = np.nansum(similarity_vector * self.ratings[:, item_id] * valid_ratings_mask)
            denom = np.nansum(np.abs(similarity_vector) * valid_ratings_mask)
            prediction = num / denom if denom != 0 else np.nan

        # Restore original value if it was temporarily replaced
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = original_value  

        return prediction

    def predict_rating_nn(self, user_id, item_id, similarity, k=5, use_adjusted=False):
        original_value = self.ratings[user_id][item_id]
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = np.nan  

        # Compute similarities for all other users (vectorized)
        similarity_vector = np.array([
            similarity(self.ratings[user_id], self.ratings[other]) if other != user_id else -np.inf
            for other in range(len(self.ratings))
        ])

        # Get indices of the k most similar users
        top_k_users = np.argpartition(similarity_vector, -k)[-k:]
        top_k_users = top_k_users[np.argsort(similarity_vector[top_k_users])][::-1]  # Sort descending
        #top_k_users = np.argsort(similarity_vector)[-k:][::-1]  # Sort descending

        # Extract valid ratings from the top k users for the target item
        valid_ratings_mask = ~np.isnan(self.ratings[top_k_users, item_id])

        # Get ratings of top k users for the target item
        neighbor_ratings = self.ratings[top_k_users, item_id] * valid_ratings_mask

        # Compute weighted sum using similarities
        sim_top_k = similarity_vector[top_k_users] * valid_ratings_mask

        if use_adjusted:
            # Use adjusted ratings (subtract user means)
            adjusted_ratings = neighbor_ratings - self.user_means[top_k_users]
            num = np.nansum(sim_top_k * adjusted_ratings)
            denom = np.nansum(np.abs(sim_top_k))
            prediction = self.user_means[user_id] + (num / denom if denom != 0 else 0)
        else:
            # Use regular weighted sum
            num = np.nansum(sim_top_k * neighbor_ratings)
            denom = np.nansum(np.abs(sim_top_k))
            prediction = num / denom if denom != 0 else np.nan

        # Restore the original value if it was temporarily replaced
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = original_value  

        return prediction

def cosine_similarity(ratings1, ratings2):
    mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
    if np.sum(mask) == 0:
        return 0
    
    ratings1, ratings2 = ratings1[mask], ratings2[mask]
    num = np.dot(ratings1, ratings2)
    denom = np.sqrt(np.sum(ratings1 ** 2)) * np.sqrt(np.sum(ratings2 ** 2))
    return num / denom if denom != 0 else 0

def pearson_similarity(ratings1, ratings2):
    mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
    if np.sum(mask) == 0:
        return 0
    
    ratings1, ratings2 = ratings1[mask], ratings2[mask]
    mean1, mean2 = np.mean(ratings1), np.mean(ratings2)
    num = np.sum((ratings1 - mean1) * (ratings2 - mean2))
    denom = np.sqrt(np.sum((ratings1 - mean1) ** 2) * np.sum((ratings2 - mean2) ** 2))
    return num / denom if denom != 0 else 0

if __name__ == "__main__":
    r = RatingsMatrix("csv/jester-data-1.csv")
    print(r.mean_utility())
    print(r.predict_rating(1, 1, cosine_similarity, use_adjusted=False))
    print(r.predict_rating(1, 1, cosine_similarity, use_adjusted=True))
    print(r.predict_rating(1, 1, pearson_similarity, use_adjusted=False))
    print(r.predict_rating(1, 1, pearson_similarity, use_adjusted=True))

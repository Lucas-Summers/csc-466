import numpy as np
import pandas as pd

class RatingsMatrix:
    def __init__(self, filepath):
        self.ratings = self.load_data(filepath)
        self.user_means = np.nanmean(self.ratings, axis=1)  # Mean per user, ignoring NaNs

    def load_data(self, f, verbose=False):
        '''
        Load the joke dataset from a csv into a matrix
        NOTE: all 99 values (no rating available) are replace by nan for easier computation
        '''
        df = pd.read_csv(f, header=None)
        df = df.iloc[:, 1:].replace(99, np.nan)
        if verbose:
            print(df.head())
        return df.to_numpy()
    
    def get_non_nan_params(self):
        '''
        Returns a list of tuples of the form (user_id, item_id) for all non-NaN values in the matrix
        '''
        return [(i, j) for i in range(len(self.ratings)) for j in range(len(self.ratings[i])) if not np.isnan(self.ratings[i][j])]
    
    def get_user_ratings(self, user_id):
        '''
        Returns all ratings for a user
        '''
        return self.ratings[user_id]

    def get_item_ratings(self, item_id):
        '''
        Returns all the ratings for an item
        '''
        return self.ratings[:, item_id]

    def mean_utility(self):
        '''
        Return the average of all ratings in the matrix
        '''
        return np.nanmean(self.ratings)
    
    def predict_rating(self, user_id, item_id, similarity, use_adjusted=False):
        '''
        Predict the rating of a user for an item using the similarity function.
        If use_adjusted is True, the adjusted weighted sum is used.
        '''
        original_value = self.ratings[user_id][item_id]
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = np.nan  

        # Compute similarity for all users
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
            prediction = num / denom if denom != 0 else self.user_means[user_id]
            #prediction = num / denom if denom != 0 else np.nan

        # Restore original value if it was temporarily replaced
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = original_value  

        return prediction

    def predict_rating_nn(self, user_id, item_id, similarity, use_adjusted=False, k=5):
        '''
        Predict the rating of a user for an item using the similarity function.
        k specifies the number of Nearest Neighbors to use in the prediction
        If use_adjusted is True, the adjusted weighted sum is used.
        '''
        original_value = self.ratings[user_id][item_id]
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = np.nan  

        # Compute similarities for all other users
        similarity_vector = np.array([
            similarity(self.ratings[user_id], self.ratings[other]) if other != user_id else -np.inf
            for other in range(len(self.ratings))
        ])
        
        valid_users = np.where(~np.isnan(self.ratings[:, item_id]))[0]  # Users who rated item

        if len(valid_users) < k:  # Ensure we have enough neighbors
            k = len(valid_users)

        # Select top-k users from those who rated the item
        top_k_users = valid_users[np.argsort(similarity_vector[valid_users])[-k:]][::-1]  # Sort descending

        # Extract valid ratings from the top-k users
        neighbor_ratings = self.ratings[top_k_users, item_id]
        sim_top_k = similarity_vector[top_k_users]

        if use_adjusted:
            # Adjusted weighted sum: use (rating - user mean)
            adjusted_ratings = neighbor_ratings - self.user_means[top_k_users]
            num = np.nansum(sim_top_k * adjusted_ratings)
            denom = np.nansum(np.abs(sim_top_k))
            prediction = self.user_means[user_id] + (num / denom if denom != 0 else 0)
        else:
            # Use regular weighted sum
            num = np.nansum(sim_top_k * neighbor_ratings)
            denom = np.nansum(np.abs(sim_top_k))
            prediction = num / denom if denom != 0 else self.user_means[user_id]
            #prediction = num / denom if denom != 0 else np.nan

        # Restore original value if it was temporarily replaced
        if -10.00 <= original_value <= 10.00:
            self.ratings[user_id][item_id] = original_value  

        return prediction

def cosine_similarity(ratings1, ratings2):
    mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
    if np.sum(mask) == 0:
        return 0
    
    ratings1, ratings2 = ratings1[mask], ratings2[mask]
    num = np.dot(ratings1, ratings2)
    denom = np.linalg.norm(ratings1) * np.linalg.norm(ratings2)
    return num / denom if denom != 0 else 0

def pearson_similarity(ratings1, ratings2):
    mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
    if np.sum(mask) == 0:
        return 0
    
    ratings1, ratings2 = ratings1[mask], ratings2[mask]
    mean1, mean2 = np.mean(ratings1), np.mean(ratings2)
    num = np.dot(ratings1 - mean1, ratings2 - mean2)
    denom = np.linalg.norm(ratings1 - mean1) * np.linalg.norm(ratings2 - mean2)
    return num / denom if denom != 0 else 0

if __name__ == "__main__":
    r = RatingsMatrix("csv/jester-data-1.csv")
    print(r.predict_rating_nn(16751, 8, pearson_similarity, use_adjusted=False))

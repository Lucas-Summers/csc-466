import pandas as pd
import numpy as np
import argparse
from cf import predict_rating

def main(start, how_many):
    '''Main function to run the collaborative filtering algorithm'''
    user_ratings = pd.read_csv('user_ratings.csv')
    print(f"Loaded {len(user_ratings)} user ratings")

    print("memory usage: ", user_ratings.memory_usage().sum() / 1024**2, "MB")
    # Identify popular games (top N)
    POPULARITY_THRESHOLD = 0.01  # fraction of top games

    game_popularity = user_ratings['BGGId'].value_counts()
    popular_threshold = int(len(game_popularity) * POPULARITY_THRESHOLD)
    popular_games = game_popularity.nlargest(popular_threshold).index
    print(f"Selected {len(popular_games)} popular games")

    # Filter user_ratings to only include popular games
    filtered_ratings = user_ratings[user_ratings['BGGId'].isin(popular_games)]

    # Remove any duplicates
    filtered_ratings = filtered_ratings.drop_duplicates(subset=['Username', 'BGGId'])

    # Create the pivot table with popular games only
    user_ratings_matrix = filtered_ratings.pivot(index='Username', columns='BGGId', values='Rating')
    print(f"Unfiltered matrix shape: {user_ratings_matrix.shape}")

    # Filter users who have rated at least 50 of these popular games
    user_ratings_matrix = user_ratings_matrix.dropna(thresh=50, axis=0)
    print(f"Matrix shape after user filtering: {user_ratings_matrix.shape}")

    from tqdm import tqdm
    valid_rating_idx = np.where(~np.isnan(user_ratings_matrix.values))
    valid_ratings = list(zip(valid_rating_idx[0], valid_rating_idx[1]))
    print(f"Number of valid ratings: {len(valid_ratings)}")
    errors = []
    for user, item in tqdm(valid_ratings[start*how_many:start*how_many+how_many]):
        pred = predict_rating(user_ratings_matrix.values, user, item)
        actual = user_ratings_matrix.values[user, item]
        errors.append(np.abs(pred - actual))

    np.save(f'out/errors_{start}_{how_many}', errors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run collaborative filtering algorithm')
    parser.add_argument('start', type=int, help='Starting index of ratings to predict')
    parser.add_argument('how_many', type=int, help='Number of ratings to predict')
    args = parser.parse_args()
    print(args.start)
    main(args.start, args.how_many)

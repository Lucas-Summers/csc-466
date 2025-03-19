import argparse
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from cf import predict_rating

def get_id_to_game(games_csv):
    """Get a mapping of game ID to game name."""
    games = pd.read_csv(games_csv)
    id_to_game = dict(zip(games['BGGId'], games['Name']))
    return id_to_game

def get_random_games(user_ratings_csv, num_games):
    print("Loading existing user ratings...")
    """Get a random list of games from the top 100 most popular games."""
    user_ratings = pd.read_csv(user_ratings_csv)
    
    # Identify the top 100 most popular games
    game_popularity = user_ratings['BGGId'].value_counts().nlargest(100).index
    top_100_games = game_popularity.tolist()
    
    # Randomly select games from the top 100
    random_games = random.sample(top_100_games, num_games)
    return random_games

def get_game_name(game_id, id_to_game):
    """Get the game name from the game ID."""
    return id_to_game.get(game_id, "Unknown Game")

def get_suggested_games(user_ratings_csv, ratings, num_recommendations):
    """Get suggested games using collaborative filtering."""
    print("calculating suggestions...")
    user_ratings = pd.read_csv(user_ratings_csv)
    
    # Identify popular games (top N)
    POPULARITY_THRESHOLD = 0.01  # fraction of top games
    game_popularity = user_ratings['BGGId'].value_counts()
    popular_threshold = int(len(game_popularity) * POPULARITY_THRESHOLD)
    popular_games = game_popularity.nlargest(popular_threshold).index
    
    # Filter user_ratings to only include popular games
    filtered_ratings = user_ratings[user_ratings['BGGId'].isin(popular_games)]
    
    # Remove any duplicates
    filtered_ratings = filtered_ratings.drop_duplicates(subset=['Username', 'BGGId'])
    
    # Create the pivot table with popular games only
    user_ratings_matrix = filtered_ratings.pivot(index='Username', columns='BGGId', values='Rating')
    
    # Filter users who have rated at least 50 of these popular games
    user_ratings_matrix = user_ratings_matrix.dropna(thresh=50, axis=0)
    
    # Predict ratings for the current user
    user_vector = np.zeros(user_ratings_matrix.shape[1])
    for game_id, rating in ratings:
        if game_id in user_ratings_matrix.columns:
            user_vector[user_ratings_matrix.columns.get_loc(game_id)] = rating
    
    predicted_ratings = []
    for game_id in tqdm(user_ratings_matrix.columns):
        if game_id not in [game for game, _ in ratings]:
            predicted_rating = predict_rating(user_ratings_matrix.values, user_vector, user_ratings_matrix.columns.get_loc(game_id))
            predicted_ratings.append((game_id, predicted_rating))
    
    # Sort the predicted ratings and return the top n games
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    suggested_games = [game_id for game_id, _ in predicted_ratings[:num_recommendations]]
    
    return suggested_games

def main(user_ratings_csv, games_csv, num_games, num_recommendations):
    id_to_game = get_id_to_game(games_csv)
    games = get_random_games(user_ratings_csv, num_games)
    ratings = []

    print("Please rate the following games from 1 to 10:")

    for game_id in games:
        game_name = get_game_name(game_id, id_to_game)
        url = f"https://boardgamegeek.com/boardgame/{game_id}"
        rating = int(input(f"{game_name} ({url}): "))
        ratings.append((game_id, rating))

    suggested_games = get_suggested_games(user_ratings_csv, ratings, num_recommendations)

    print("\nSuggested Games:")
    for game_id in suggested_games:
        url = f"https://boardgamegeek.com/boardgame/{game_id}"
        print(f"{get_game_name(game_id, id_to_game)} ({url})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Game Rater CLI')
    parser.add_argument('user_ratings_csv', type=str, help='Path to user_ratings.csv')
    parser.add_argument('games_csv', type=str, help='Path to games.csv')
    parser.add_argument('--num_games', type=int, default=10, help='Number of games to rate')
    parser.add_argument('--num_recommendations', type=int, default=3, help='Number of recommendations to output')
    args = parser.parse_args()

    main(args.user_ratings_csv, args.games_csv, args.num_games, args.num_recommendations)
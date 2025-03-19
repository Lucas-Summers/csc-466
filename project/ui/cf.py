import numpy as np

def cosine_similarity(user1, user2):
    """Calculates the cosine similarity between two users."""
    common_ratings = (~np.isnan(user1)) & (~np.isnan(user2))
    if np.sum(common_ratings) == 0:
        return 0

    user1_common = user1[common_ratings]
    user2_common = user2[common_ratings]

    numerator = np.sum(user1_common * user2_common)
    denominator = np.sqrt(np.sum(user1_common ** 2) * np.sum(user2_common ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def predict_rating(ratings, user, itemIdx, k=50):
    """Predict the rating of a user for an item using a weighted sum of k nearest neighbors."""
    ratings = ratings.copy()

    item = ratings[:, itemIdx]

    user_mean = np.nanmean(user)

    # create list of similarities between the user and all other users
    similarities = np.array([cosine_similarity(user, other_user) for other_user in ratings])

    # find the nearest neighbors in the similarity list
    nn_indices = np.argsort(similarities)[::-1]
    if k is not None:
        # if there is a k, filter to k neighbors
        nn_indices = nn_indices[:k]
    rating_present_mask = ~np.isnan(ratings[nn_indices, itemIdx])
    filtered_indices = nn_indices[rating_present_mask]
    nn_sims = similarities[filtered_indices]

    # use the indices of the nearest neighbors to find their ratings for the item
    nn_ratings = ratings[filtered_indices, itemIdx]

    # adjust the ratings by subtracting each user's mean rating
    nn_means = np.array([np.nanmean(ratings[nn_index]) for nn_index in filtered_indices])
    nn_ratings = nn_ratings - nn_means

    # perform calculation for predicted rating using adjusted weighted sum formula
    numerator = np.sum(nn_sims * nn_ratings)
    denominator = np.sum(nn_sims)

    if denominator == 0:
        return user_mean
    else:
        prediction = user_mean + (numerator / denominator)
        # clip to 1-10 range
        return np.clip(prediction, 1, 10)
    

def calculate_mae(tests, ratings):
    """Calculate the MAE of a method on a set of tests. Prints the results of each test
    and keeps track of the results in the confusion matrix."""
    running_ae = []
    print("userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating")
    for user, item in tests:
        pred = predict_rating(ratings, user, item)
        actual = ratings[user, item]
        running_ae.append(np.abs(pred - actual))

        print(f"{user}, {item}, {actual}, {pred}, {np.abs(pred - actual)}")

    mae = np.mean(running_ae)
    std_dev = np.std(running_ae)
    print(f"Standard deviation of errors: {std_dev}")
    print(f"Mean Absolute Error: {mae}")
    return mae

def filter_valid(tests, ratings):
    """Given a list of tests, returns all valid tests."""
    valid_tests = []
    for user, item in tests:
        if np.isnan(ratings[user, item]):
            continue
        else:
            valid_tests.append((user, item))

    return valid_tests

def generate_tests(n_tests, ratings, item=None):
    """Randomly generate a set of validated user/item pairs for testing."""
    tests = []
    n_users, n_items = ratings.shape

    while len(tests) < n_tests:
        user = np.random.randint(n_users)
        if not item:
            item = np.random.randint(n_items)

        # tests is only added to if the user/item pair is valid
        tests += (filter_valid([(user, item)], ratings))

    return tests

def random_sampling(ratings, size=100, repeats=10):
    """Randomly sample user/item pairs for testing and calculate MAE."""
    MAEs = []

    for i in range(repeats):
        tests = generate_tests(size, ratings)

        mae = calculate_mae(tests, ratings)

        print(f"Overall MAE of testing run {i + 1}: {mae}")

        MAEs.append(mae)

    print(f"Mean Overall MAE: {np.mean(MAEs)}")
    print(f"Standard deviation of MAEs: {np.std(MAEs)}")
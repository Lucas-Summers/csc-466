from ratings import RatingsMatrix, cosine_similarity, pearson_similarity
import numpy as np
import pandas as pd
import random
import argparse


def main(method, size, repeats, nnn, adjusted):
    r = RatingsMatrix("csv/jester-data-1.csv")
    similarity = cosine_similarity if "cosine" in args.method else pearson_similarity
    non_nan_ratings = r.get_non_nan_params()
    if size > len(non_nan_ratings):
        print(f"Size is greater than the number of non-NaN ratings {size}, setting size to the number of non-NaN ratings {len(non_nan_ratings)}")
        size = len(non_nan_ratings)

    maes = []
    for i in range(repeats):
        test_cases = random.sample(non_nan_ratings, size)
        errs = []
        print(f"\n-- Repeat {i} --")
        print("userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating")
        for user_id, item_id in test_cases:
            if nnn:
                prediction = r.predict_rating_nn(user_id, item_id, similarity, use_adjusted=adjusted)
            else:
                prediction = r.predict_rating(user_id, item_id, similarity, use_adjusted=adjusted)
            
            actual_rating = r.ratings[user_id][item_id]
            delta_rating = np.abs(actual_rating - prediction)

            print(f"{user_id}, {item_id}, {actual_rating}, {prediction}, {delta_rating}")
            errs.append(delta_rating)

        print("\n-- Metrics --")
        print("MAE:", np.mean(errs))
        maes.append(np.mean(errs))
    
    print("\n-- Summary --")
    print("Mean MAE:", np.mean(maes))
    print("Std Dev MAE:", np.std(maes))




if __name__ == "__main__":
    '''
    EvaluateCFRandom is, essentially, a wrapper around the first evaluation method discussed above. When run with no parameters, it shall print a help message indicating the list of collaborative filtering methods implemented and their Ids. Otherwise, the program shall take two parameters:
$ python EvaluateCFRandom.py Method Size Repeats
Here, Method is the collaborative filtering method and size is the number of test cases to generate. The
program shall execute the first evaluation method and print the output.
    '''
    parser = argparse.ArgumentParser(description="Evaluate a collaborative filtering algorithm")
    parser.add_argument("method", help="The method to evaluate. Expects 'cosine' or 'pearson'")
    parser.add_argument("size", help="The number of test cases to generate", type=int)
    parser.add_argument("repeats", help="The number of times to repeat the test cases", type=int)
    args = parser.parse_args()

    nnn = "nnn" in args.method
    adjusted = "adjusted" in args.method

    main(args.method, args.size, args.repeats, nnn, adjusted)

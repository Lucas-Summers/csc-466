from ratings import RatingsMatrix, cosine_similarity, pearson_similarity
import numpy as np
import pandas as pd
import random
import argparse

def vprint(*args, **kwargs):
    if args[-1] is False:
        return
    if args[-1] is True:
        args = args[:-1]
    print(*args)

def eval_cf_random(method, size, repeats, nnn, adjusted, k=5, verbose=True):
    r = RatingsMatrix("csv/jester-data-1.csv")
    similarity = cosine_similarity if "cosine" in method else pearson_similarity
    non_nan_ratings = r.get_non_nan_params()
    if size > len(non_nan_ratings):
        vprint(f"Size is greater than the number of non-NaN ratings {size}, setting size to the number of non-NaN ratings {len(non_nan_ratings)}", verbose)
        size = len(non_nan_ratings)

    maes = []
    overall_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for i in range(repeats):
        test_cases = random.sample(non_nan_ratings, size)
        errs = []
        vprint(f"\n-- Repeat {i} --", verbose)
        vprint("userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating", verbose)
        for user_id, item_id in test_cases:
            if nnn:
                prediction = r.predict_rating_nn(user_id, item_id, similarity, use_adjusted=adjusted, k=k)
            else:
                prediction = r.predict_rating(user_id, item_id, similarity, use_adjusted=adjusted)
            
            actual_rating = r.ratings[user_id][item_id]
            delta_rating = np.abs(actual_rating - prediction)

            vprint(f"{user_id}, {item_id}, {actual_rating}, {prediction}, {delta_rating}", verbose)
            errs.append(delta_rating)

            # Determine recommendation
            recommend = prediction >= 5
            actual_recommend = actual_rating >= 5

            # Confusion matrix counts
            if recommend and actual_recommend:
                overall_metrics["TP"] += 1
            elif recommend and not actual_recommend:
                overall_metrics["FP"] += 1
            elif not recommend and actual_recommend:
                overall_metrics["FN"] += 1
            else:
                overall_metrics["TN"] += 1

        vprint("\n-- Metrics --", verbose)
        vprint("MAE:", np.mean(errs), verbose)
        maes.append(np.mean(errs))

    # Compute final metrics
    TP, FP, FN, TN = overall_metrics["TP"], overall_metrics["FP"], overall_metrics["FN"], overall_metrics["TN"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    conf_matrix = pd.DataFrame(
        [[overall_metrics["TN"], overall_metrics["FP"]], 
         [overall_metrics["FN"], overall_metrics["TP"]]],
        index=["Actual: Not Recommend", "Actual: Recommend"],
        columns=["Predicted: Not Recommend", "Predicted: Recommend"]
    )
    
    print("\n-- Summary --")
    print("Mean MAE:", np.mean(maes))
    print("Std Dev MAE:", np.std(maes))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    return np.mean(maes), np.std(maes), precision, recall, f1_score, accuracy


if __name__ == "__main__":
    # try `python EvaluateCFRandom pearson 5 5`
    parser = argparse.ArgumentParser(
        description="Evaluate a collaborative filtering algorithm using random sampling.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "method",
        help=(
            "The method to evaluate. Choose from:\n"
            "  - Weighted sum: 'pearson' or 'cosine'\n"
            "  - Adjusted weighted sum: 'adjustedpearson' or 'adjustedcosine'\n"
            "  - Weighted N Nearest Neighbors sum: 'nnnpearson' or 'nnncosine'\n"
            "  - Adjusted weighted N Nearest Neighbors sum: 'nnnadjustedpearson' or 'nnnadjustedcosine'"
        )
    )
    parser.add_argument("size", help="The number of test cases to generate", type=int)
    parser.add_argument("repeats", help="The number of times to repeat the test cases", type=int)
    args = parser.parse_args()

    assert "cosine" in args.method or "pearson" in args.method, "Method must have 'cosine' or 'pearson'"

    nnn = "nnn" in args.method
    adjusted = "adjusted" in args.method

    eval_cf_random(args.method, args.size, args.repeats, nnn, adjusted)

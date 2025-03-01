import argparse
import numpy as np
import pandas as pd
from ratings import RatingsMatrix, cosine_similarity, pearson_similarity

def vprint(*args, **kwargs):
    if args[-1] is False:
        return
    if args[-1] is True:
        args = args[:-1]
    print(*args)

def eval_cf_list(method, filename, nnn, adjusted, k=5, verbose=True):
    r = RatingsMatrix("csv/jester-data-1.csv")
    similarity = cosine_similarity if "cosine" in method else pearson_similarity

    errs = []
    overall_metrics = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    with open (filename, "r") as f:
        vprint("userID, itemID, Actual_Rating, Predicted_Rating, Delta_Rating")
        for line in f:
            user_id, item_id = map(int, line.strip().split(","))
            if np.isnan(r.ratings[user_id][item_id]):
                # skip since we don't have a rating to compare
                continue

            if nnn:
                prediction = r.predict_rating_nn(user_id, item_id, similarity, use_adjusted=adjusted, k=k)
            else:
                prediction = r.predict_rating(user_id, item_id, similarity, use_adjusted=adjusted)
            
            actual_rating = r.ratings[user_id][item_id]
            delta_rating = np.abs(actual_rating - prediction)

            vprint(f"{user_id}, {item_id}, {actual_rating}, {prediction}, {delta_rating}")
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

    print("\n-- Summary --")
    print("MAE:", np.mean(errs))

    # Compute final metrics
    TP, FP, FN, TN = overall_metrics["TP"], overall_metrics["FP"], overall_metrics["FN"], overall_metrics["TN"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    conf_matrix = pd.DataFrame(
        [[TN, FP], 
         [FN, TP]],
        index=["Actual: Not Recommend", "Actual: Recommend"],
        columns=["Predicted: Not Recommend", "Predicted: Recommend"]
    )
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    return np.mean(errs), np.std(errs), precision, recall, f1_score, accuracy


if __name__ == "__main__":
    # try `python EvaluateCFList.py pearson list.txt`
    parser = argparse.ArgumentParser(description="Evaluate a collaborative filtering algorithm")
    parser.add_argument("method", help="The method to evaluate. Expects 'cosine' or 'pearson'; 'nnn' and/or 'adjusted' as modifiers")
    parser.add_argument("filename", help="The filename of the test cases, expects UserID, ItemID")
    args = parser.parse_args()

    # TODO: change to ids instead of names
    assert "cosine" in args.method or "pearson" in args.method, "Method must have 'cosine' or 'pearson'"

    nnn = "nnn" in args.method
    adjusted = "adjusted" in args.method

    eval_cf_list(args.method, args.filename, nnn, adjusted)


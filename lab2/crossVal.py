import math
from csv_reader import read_csv, get_Xy
from c45 import c45
import json
from tqdm import tqdm
import sys
import pandas as pd

# try it out with `python crossVal.py csv/nursery.csv trees/hyp_sample.json trees/best_tree.json`

def nfold(csv_file, hyps, pbar, n=10):
    '''
    Perform n-fold cross validation on the given dataset
    Returns the overall accuracy and confusion matrix
    '''
    domain, class_var, df = read_csv(csv_file)
    shuffled = df.sample(frac=1)

    nth = len(shuffled) // n
    accuracies = []
    overall_confusion_matrix = None
    total_pred, total_truth = [], []
    for i in range(n):
        test = shuffled.iloc[i*nth:(i+1)*nth]
        ts_X, ts_y = get_Xy(class_var, test)
        train = pd.concat([shuffled.iloc[:i*nth], shuffled.iloc[(i+1)*nth:]])
        tr_X, tr_y = get_Xy(class_var, train)

        model = c45(metric=hyps[0], threshold=hyps[1])
        model.fit(tr_X, tr_y, csv_file)
        
        predictions = model.predict(ts_X)
        ground_truth = ts_y.tolist()

        total_pred.extend(predictions)
        total_truth.extend(ground_truth)

        pred_df = pd.DataFrame({"predictions": predictions, "truth": ground_truth})
        correct = pred_df[pred_df["predictions"] == pred_df["truth"]].shape[0]
        total = len(ground_truth)
        # incorrect = total - correct
        accuracy = correct / total
        # error_rate = 1 - accuracy
        
        accuracies.append(accuracy)
        pbar.update(1)

    overall_accuracy = sum(accuracies) / n
    overall_confusion_matrix = pd.crosstab(
        total_pred, total_truth, rownames=["Predicted"], colnames=["Actual"], dropna=False
    )

    return overall_accuracy, overall_confusion_matrix

def read_hyps(hyps_file):
    val_dict = json.load(open(hyps_file, "r"))
    if "InfoGain" not in val_dict or "Ratio" not in val_dict:
        print("Error: Invalid hyps file, expected keys 'InfoGain' and 'Ratio'")
        return None
    
    return (val_dict["InfoGain"], val_dict["Ratio"])

def grid_search(csv_file, hyps_file):
    best_accuracy = 0
    best_confusion_matrix = None
    best_params = None

    info_gains, ratios = read_hyps(hyps_file)

    n = 10
    # info gain searches
    pbar = tqdm(total=len(info_gains) * n)
    for thresh in info_gains:
        pbar.set_description(f"Info Gain: {thresh}")
        acc, confusion_matrix = nfold(csv_file, ("info_gain", thresh), pbar, n)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_confusion_matrix = confusion_matrix
            best_params = ("info_gain", thresh)
        pbar.set_postfix(acc=acc, best_acc=best_accuracy)
    
    # gain ratio searches
    pbar = tqdm(total=len(ratios) * n)
    for thresh in ratios:
        pbar.set_description(f"Gain Ratio: {thresh}")
        acc, confusion_matrix = nfold(csv_file, ("gain_ratio", thresh), pbar, n)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_confusion_matrix = confusion_matrix
            best_params = ("gain_ratio", thresh)
        pbar.set_postfix(acc=acc, best_acc=best_accuracy)

    return best_accuracy, best_confusion_matrix, best_params

if __name__ == "__main__":
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        acc, conf_mat, params = grid_search(sys.argv[1], sys.argv[2])
        print(f"Best parameters: {params}")
        print(f"Best accuracy: {acc}")
        print("Confusion matrix:")
        print(conf_mat)
        if len(sys.argv) == 4:
            domain, class_var, df = read_csv(sys.argv[1])
            eval_X, eval_y = get_Xy(class_var, df)

            model = c45(metric=params[0], threshold=params[1])
            model.fit(eval_X, eval_y, sys.argv[1])
            model.save_tree(sys.argv[3])
            # print(model.to_graphviz_dot())
    else:
        print("Usage: python crossVal.py <CSVFile> <HypsFile> [<save_best_tree.json>]")

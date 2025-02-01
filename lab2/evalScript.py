from tqdm import tqdm
from csv_reader import read_csv, get_Xy
from c45 import c45
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

'''
script for caching the results of grid search into "evaltrees" directory
'''
def csv_save_dir(csv_file):
    return f"evaltrees/{csv_file.split('/')[-1].split('.')[0]}/"

def nfold(csv_file, hyps, pbar, n=10, save_tree=False):
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

        # penalize larger trees
        # target_tree_size = 30
        # if model.tree_size() > target_tree_size:
        #     accuracy /= math.log(model.tree_size(), target_tree_size)
        accuracies.append(accuracy)
        pbar.update(1)

    overall_accuracy = sum(accuracies) / n
    overall_confusion_matrix = pd.crosstab(
        total_pred, total_truth, rownames=["Predicted"], colnames=["Actual"], dropna=False
    )

    # save tree
    if save_tree:
        dir_name = csv_save_dir(csv_file)
        os.makedirs(dir_name, exist_ok=True)
        model.save_tree(f"{dir_name}{hyps[0]}_{hyps[1]}.json")

    return overall_accuracy, overall_confusion_matrix

def grid_search(csv_file, info_hyps, ratio_hyps, save_tree=False):
    n = 10
    records = []

    # info gain searches
    pbar = tqdm(total=len(info_hyps) * n)
    for thresh in info_hyps:
        pbar.set_description(f"Info Gain: {thresh}")
        acc, confusion_matrix = nfold(csv_file, ("info_gain", thresh), pbar, n, save_tree)

        record = {"acc": acc, "confusion_matrix": confusion_matrix.to_string(), "params": ("info_gain", thresh)}
        records.append(record)

        pbar.set_postfix(acc=acc)
    
    # gain ratio searches
    pbar = tqdm(total=len(ratio_hyps) * n)
    for thresh in ratio_hyps:
        pbar.set_description(f"Gain Ratio: {thresh}")
        acc, confusion_matrix = nfold(csv_file, ("gain_ratio", thresh), pbar, n, save_tree)
        
        record = {"acc": acc, "confusion_matrix": confusion_matrix.to_string(), "params": ("gain_ratio", thresh)}
        records.append(record)

        pbar.set_postfix(acc=acc)

    return records

def read_grid_search_results(json_file):
    # each line is a different list of records
    records = []
    with open(json_file, "r") as f:
        for line in f:
            records.extend(json.loads(line))
    return records

if __name__ == "__main__":
    CSVFiles = ["csv/iris.csv", "csv/letter-recog.csv", "csv/nursery.csv"]
    for csv_file in CSVFiles:
        info_gains = [0.6, 0.7, 0.8, 0.9]
        ratios = [0.6, 0.7, 0.8, 0.9]
        recs = grid_search(csv_file, info_gains, ratios, save_tree=True)
        save_dir = csv_save_dir(csv_file)
        json.dump(recs, open(f"{save_dir}grid_search.json", "a"))
        open(f"{save_dir}grid_search.json", "a").write("\n")
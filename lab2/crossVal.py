from csv_reader import read_csv
from c45 import c45
import json
from tqdm import tqdm
import pandas as pd

def nfold(csv_file, hyps, n=10):
    '''
    Perform n-fold cross validation on the given dataset
    Returns the overall accuracy and confusion matrix
    '''
    domain, class_var, df = read_csv(csv_file)
    shuffled = df.sample(frac=1)

    nth = len(shuffled) // n
    accuracies = []
    overall_confusion_matrix = None
    for i in range(n):
        test = shuffled.iloc[i*nth:(i+1)*nth]
        train = pd.concat([shuffled.iloc[:i*nth], shuffled.iloc[(i+1)*nth:]])

        model = c45(metric=hyps[0], threshold=hyps[1])
        model.fit(train.iloc[:, :-1], train.iloc[:, -1], csv_file)
        predictions = model.predict(test.iloc[:, :-1])

        ground_truth = test.iloc[:, -1]
        correct = (predictions == ground_truth).sum()
        total = len(ground_truth)
        # incorrect = total - correct
        accuracy = correct / total
        # error_rate = 1 - accuracy

        confusion_matrix = pd.crosstab(
            ground_truth, predictions, rownames=["Actual"], colnames=["Predicted"]
        )

        if overall_confusion_matrix is None:
            overall_confusion_matrix = confusion_matrix
        else:
            overall_confusion_matrix += confusion_matrix
        accuracies.append(accuracy)

    overall_accuracy = sum(accuracies) / n

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
    
    pbar = tqdm(info_gains)
    for thresh in pbar:
        pbar.set_description(f"Info Gain: {thresh:.2f}")
        acc, confusion_matrix = nfold(csv_file, ("info_gain", thresh))
        if acc >= best_accuracy:
            best_accuracy = acc
            best_confusion_matrix = confusion_matrix
            best_params = ("info_gain", thresh)
        pbar.set_description(f"Info Gain: {thresh:.2f}, Curr Acc: {acc:.2f}, Best Acc: {best_accuracy:.2f}")
    
    pbar = tqdm(ratios)
    for thresh in pbar:
        pbar.set_description(f"Ratio: {thresh:.2f}")
        acc, confusion_matrix = nfold(csv_file, ("gain_ratio", thresh))
        if acc >= best_accuracy:
            best_accuracy = acc
            best_confusion_matrix = confusion_matrix
            best_params = ("gain_ratio", thresh)
        pbar.set_description(f"Ratio: {thresh:.2f}, Curr Acc: {acc:.2f}, Best Acc: {best_accuracy:.2f}")

    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy}")
    print("Confusion matrix:")
    print(best_confusion_matrix)

grid_search("csv/nursery.csv", "trees/hyp_sample.json")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python crossVal.py <CSVFile> <HypsFile>")
#     else:
#         tenfold(sys.argv[1], sys.argv[2])

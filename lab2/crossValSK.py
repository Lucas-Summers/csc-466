from csv_reader import read_csv, get_Xy
import json
from tqdm import tqdm
import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt

# try it out with `python crossValSK.py csv/nursery.csv trees/hyp_sample.json trees/best_tree.png`

def c45(metric, threshold):
    ''
    if metric == "info_gain":
        model = DecisionTreeClassifier(
            criterion="entropy",  # Equivalent to Information Gain
            min_impurity_decrease=threshold  # Threshold
        )
    elif metric == "gain_ratio":
        model = DecisionTreeClassifier(
            criterion="entropy",  # Equivalent to Information Gain
            min_impurity_decrease=threshold  # Threshold
        )
    else:
        raise ValueError("Invalid metric.")

    return model

def nfold(df, hyps, encoder, n=10):
    '''
    Perform n-fold cross validation on the given dataset
    Returns the overall accuracy and confusion matrix
    '''
    shuffled = df.sample(frac=1)

    nth = len(shuffled) // n
    accuracies = []
    overall_confusion_matrix = None
    for i in range(n):
        test = shuffled.iloc[i*nth:(i+1)*nth]
        ts_X, ts_y = get_Xy(class_var, test)
        train = pd.concat([shuffled.iloc[:i*nth], shuffled.iloc[(i+1)*nth:]])
        tr_X, tr_y = get_Xy(class_var, train)
       
        model = c45(metric=hyps[0], threshold=hyps[1])
        model.fit(tr_X, tr_y)

        predictions = model.predict(ts_X)
        ground_truth = tr_y

        if encoder is not None:
            ground_truth = encoder.categories_[-1][ground_truth.astype(int)]
            predictions = encoder.categories_[-1][predictions.astype(int)]

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

def grid_search(df, hyps_file, encoder):
    best_accuracy = 0
    best_confusion_matrix = None
    best_params = None

    info_gains, ratios = read_hyps(hyps_file)
    
    # info gain searches
    pbar = tqdm(info_gains)
    for thresh in pbar:
        pbar.set_description(f"Info Gain: {thresh}")
        acc, confusion_matrix = nfold(df, ("info_gain", thresh), encoder)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_confusion_matrix = confusion_matrix
            best_params = ("info_gain", thresh)
        pbar.set_description(f"Info Gain: {thresh}, Curr Acc: {acc:.2f}, Best Acc: {best_accuracy:.2f}")
    
    # gain ratio searches
    pbar = tqdm(ratios)
    for thresh in pbar:
        pbar.set_description(f"Gain Ratio: {thresh}")
        acc, confusion_matrix = nfold(df, ("gain_ratio", thresh), encoder)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_confusion_matrix = confusion_matrix
            best_params = ("gain_ratio", thresh)
        pbar.set_description(f"Gain Ratio: {thresh}, Curr Acc: {acc:.2f}, Best Acc: {best_accuracy:.2f}")

    return best_accuracy, best_confusion_matrix, best_params

def save_tree(model, feature_names, class_names, filename):
    """ Save the decision tree as an image with original category names """

    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        domain, class_var, df = read_csv(sys.argv[1])
        categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
        if categorical_columns:
            encoder = OrdinalEncoder()
            df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

        acc, conf_mat, params = grid_search(df, sys.argv[2], encoder)
        print(f"Best parameters: {params}")
        print(f"Best accuracy: {acc}")
        print("Confusion matrix:")
        print(conf_mat)
        if len(sys.argv) == 4:
            model = c45(metric=params[0], threshold=params[1])
            ev_X, ev_y = get_Xy(class_var, df)
            model.fit(ev_X, ev_y)
            
            if categorical_columns:
                df[categorical_columns] = encoder.inverse_transform(df[categorical_columns])
            print(ev_X.columns, ev_y.unique())
            save_tree(model, ev_X.columns, ev_y.unique(), sys.argv[3])
    else:
        print("Usage: python crossValSK.py <CSVFile> <HypsFile> [<save_best_tree.png>]")

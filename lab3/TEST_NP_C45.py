import sys
from csv_reader import read_csv, get_Xy_as_np, get_Xy
from c45 import c45
from df_c45 import c45 as df_c45
import json
import numpy as np
from deepdiff import DeepDiff

def diff_test(thresh, metric, df, class_var):
    ## df version
    X_df, y_df = get_Xy(class_var, df)
    df_size = X_df.shape[0]
    random_indices = np.random.choice(df_size, int(df_size * 0.8), replace=False)
    X_df = X_df.iloc[random_indices, :]
    y_df = y_df.iloc[random_indices]
    model_df = df_c45(metric=metric, threshold=thresh)
    model_df.fit(X_df, y_df, csv_file)

    ## np version
    X, y, attrs = get_Xy_as_np(class_var, df)
    X = X[random_indices, :]
    y = y[random_indices]
    model = c45(metric=metric, threshold=thresh)
    model.fit(X, y, attrs, csv_file)

    dd = DeepDiff(model_df.tree, model.tree, ignore_order=True)
    if 'value_changed' in dd:
        changes = dd['value_changed']
        for k in changes:
            print(k, changes[k])
        print(model_df.tree, model.tree)

    df_pred, pred = model_df.predict(X_df), model.predict(X, attrs)

    dd = DeepDiff(df_pred, pred, ignore_order=True)
    if dd:
        print(dd)
        print(len(df_pred), len(pred))  

# try `python TEST_NP_C45.py csv/nursery.csv`
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python TEST_NP_C45.py <TrainingSetFile.csv> [<fileToSave>]")
        sys.exit(1)

    csv_file = sys.argv[1]
    #json_file = sys.argv[2]

    domain, class_var, df = read_csv(csv_file)
    
    threshs = [0.01, 0.05, 0.1, 0.2, 0.4]
    metrics = ["info_gain", "gain_ratio"]
    for thresh in threshs:
        for metric in metrics:
            diff_test(thresh, metric, df, class_var)



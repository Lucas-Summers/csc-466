import pandas as pd
import io

def read_csv(filename):
    '''
    reads the special csv format used in this lab
    returns a tuple with:
    - domain (as a list of int)
    - class_var (as a string)
    - dataframe of values
    '''
    # read file as string
    with open(filename, 'r') as f:
        data = f.read()

    # read special info
    domain = data.split("\n")[1].split(",")
    class_var = data.split("\n")[2]

    # remove second and third line, so it looks like a normal csv
    data = data.split("\n", 2)[0] + "\n" + data.split("\n", 3)[3]

    return domain, class_var, pd.read_csv(io.StringIO(data))

def get_Xy(class_var, df):
    y = df[class_var]
    X = df.drop(columns=[class_var])
    return X, y

def get_Xy_as_np(class_var, df):
    y = df[class_var]
    X = df.drop(columns=[class_var])
    attrs = X.columns.tolist()
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y, attrs
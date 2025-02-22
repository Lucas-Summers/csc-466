import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def remove_outliers_zscore(X, y, threshold=3):
    # Calculate Z-scores
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    z_scores = np.where(std_dev != 0, (X - mean) / std_dev, 0)

    # Filter out rows where any feature has a Z-score greater than the threshold
    mask = np.all(np.abs(z_scores) < threshold, axis=1)

    X_clean = X[mask]
    if y is not None:
        y_clean = y[mask]
    else:
        y_clean = None
    return X_clean, y_clean

def remove_outliers_iqr(X, y):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    # Define outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out rows that are outside of the bounds
    mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)

    X_clean = X[mask]
    if y is not None:
        y_clean = y[mask]
    else:
        y_clean = None
    return X_clean, y_clean

def load_data(csv, target=False):
    df = pd.read_csv(csv, header=None)

    inclusion_mask = df.iloc[0].astype(int)  # Convert first row to integers
    df = df.iloc[1:].reset_index(drop=True)

    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])
    if target:
        dropped = df.loc[:, inclusion_mask == 0].to_numpy().ravel()
    else:
        dropped = None

    df = df.loc[:, inclusion_mask == 1].to_numpy()
    # print(df)

    return df, dropped
    
def preprocess_data(X, y, method="standard"):
    assert method in ["standard", "normal"] 

    if method == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return remove_outliers_zscore(X_scaled, y)
    elif method == "normal":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return remove_outliers_iqr(X_scaled, y)
    else:
        return X

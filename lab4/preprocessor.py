import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder

def remove_outliers_zscore(X, threshold=3):
    # Calculate Z-scores
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    z_scores = np.where(std_dev != 0, (X - mean) / std_dev, 0)

    
    # Filter out rows where any feature has a Z-score greater than the threshold
    X_clean = X[np.all(np.abs(z_scores) < threshold, axis=1)]
    return X_clean

def remove_outliers_iqr(X):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    # Define outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out rows that are outside of the bounds
    X_clean = X[np.all((X >= lower_bound) & (X <= upper_bound), axis=1)]
    return X_clean

def preprocess_data(X, method="standard"):
    assert method in ["standard", "normal"]
    if method == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return remove_outliers_zscore(X_scaled)
    elif method == "normal":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return remove_outliers_iqr(X_scaled)
    else:
        return X

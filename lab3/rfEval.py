from csv_reader import read_csv, get_Xy
from rf import RandomForest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import sys


def get_hyperparameter_ranges(X):
    num_attributes = X.shape[1]
    
    # Determine NumAttributes range
    if num_attributes <= 10:
        num_attributes_range = [1, 2, 3, 4]
    elif num_attributes <= 20:
        num_attributes_range = [1, 2, 3, 4, 5]
    else:
        num_attributes_range = [1, 2, 3, 4, 5, 6, 7]
    
    # Determine NumTrees range based on attribute combinations
    num_trees_range = [25, 50, 100, 200, 500]
    if num_attributes > 10:
        num_trees_range = [60, 120, 240, 500, 1000]
    
    # Determine NumDataPoints range based on dataset size
    num_data_points_range = [0.05, 0.1, 0.2, 0.25]  # percentages of the dataset
    if X.shape[0] < 1000:
        num_data_points_range = [0.1, 0.15, 0.2]
    
    return num_trees_range, num_attributes_range, num_data_points_range


def grid_search(X_train, y_train, X_test, y_test, model_type, num_trees_range, num_attributes_range, num_data_points_range):
    best_model = None
    best_score = 0
    best_params = None
    
    # Perform grid search over hyperparameters
    for num_trees in num_trees_range:
        for num_attributes in num_attributes_range:
            for num_data_points in num_data_points_range:
                print(num_trees, num_attributes, num_data_points)
                if model_type == 'sklearn':
                    model = RandomForestClassifier(n_estimators=num_trees, 
                                                   max_features=num_attributes, 
                                                   max_samples=(1.0-num_data_points),
                                                   criterion="entropy", # info gain
                                                   min_impurity_decrease=0.01)
                else:
                    model = RandomForest(num_attributes, num_data_points, num_trees)  # Custom RF
                
                model.fit(X_train, y_train)
                test_pred = model.predict(X_test)
                score = accuracy_score(y_test, test_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = (num_trees, num_attributes, num_data_points)
    
    return best_model, best_params


def evaluate_model(model, X_train, y_train, X_test, y_test, labels):

    # Training evaluation
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_conf_matrix_df = pd.DataFrame(train_conf_matrix, 
                                        index=[f"Actual {label}" for label in labels], 
                                        columns=[f"Predicted {label}" for label in labels])
    if labels.size == 2:
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_train, y_train_pred, average='binary')
    else:
        train_precision, train_recall, train_f1 = None, None, None
    
    # Test evaluation
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    test_conf_matrix_df = pd.DataFrame(test_conf_matrix, 
                                       index=[f"Actual {label}" for label in labels], 
                                       columns=[f"Predicted {label}" for label in labels])
    if labels.size == 2:
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
    else:
        test_precision, test_recall, test_f1 = None, None, None
    
    
    return {
        "train_accuracy": train_accuracy,
        "train_conf_matrix": train_conf_matrix_df,
        "test_accuracy": test_accuracy,
        "test_conf_matrix": test_conf_matrix_df,
        "train_precision": train_precision, 
        "train_recall": train_recall, 
        "train_f1": train_f1,
        "test_precision": test_precision, 
        "test_recall": test_recall, 
        "test_f1": test_f1
    }


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python rfEval.py <CSVFile> <Method>")
        sys.exit(1)
    assert sys.argv[2] in ['sklearn', '466'], "Error: Method must be 'sklearn' or '466'"

    # Load dataset
    domain, class_var, df = read_csv(sys.argv[1])
    
    labels = np.unique(df[class_var])
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        encoder = OrdinalEncoder()
        df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

    X, y = get_Xy(class_var, df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get hyperparameter ranges
    num_trees_range, num_attributes_range, num_data_points_range = get_hyperparameter_ranges(X)

    # Grid search to find the best model
    best_model, best_params = grid_search(X_train, y_train, X_test, y_test, sys.argv[2], num_trees_range, num_attributes_range, num_data_points_range)

    # Evaluate the best model
    evaluation_results = evaluate_model(best_model, X_train, y_train, X_test, y_test, labels)

    # Output results
    print(f"Selected Method: {sys.argv[2]}")
    print(f"Hyperparameter ranges tested:")
    print(f"NumTrees: {num_trees_range}")
    print(f"NumAttributes: {num_attributes_range}")
    print(f"NumDataPoints: {num_data_points_range}")
    print(f"Best Hyperparameters: {best_params}")
    
    print("\nTraining Accuracy:", evaluation_results["train_accuracy"])
    print("Training Confusion Matrix:\n", evaluation_results["train_conf_matrix"])
    print("Test Accuracy:", evaluation_results["test_accuracy"])
    print("Test Confusion Matrix:\n", evaluation_results["test_conf_matrix"])
    
    print("\nPrecision/Recall/F1 (Train):")
    print(f"Precision: {evaluation_results['train_precision']}, Recall: {evaluation_results['train_recall']}, F1: {evaluation_results['train_f1']}")
    
    print("\nPrecision/Recall/F1 (Test):")
    print(f"Precision: {evaluation_results['test_precision']}, Recall: {evaluation_results['test_recall']}, F1: {evaluation_results['test_f1']}")

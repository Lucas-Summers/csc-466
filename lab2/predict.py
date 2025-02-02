import sys
import pandas as pd
from csv_reader import read_csv, get_Xy
from c45 import c45

# try `python predict.py csv/nursery.csv trees/nursery_sample.json``
if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python predict.py <CSVFile> <JSONFile> [eval]")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    
    domain, class_var, df = read_csv(csv_file)
    X, y = get_Xy(class_var, df)
    tree = c45()

    tree.read_tree(json_file)
    
    predictions = tree.predict(X)

    if len(sys.argv) == 4 and sys.argv[3] == "eval":
        ground_truth = y
        correct = (predictions == ground_truth).sum()
        total = len(ground_truth)
        incorrect = total - correct
        accuracy = correct / total
        error_rate = 1 - accuracy

        print("Predictions for each input data point:")
        for i in range(len(predictions)):
            print(predictions[i])
        print(f"Total number of records classified: {total}")
        print(f"Total number of records correctly classified: {correct}")
        print(f"Total number of records incorrectly classified: {incorrect}")
        print(f"Overall accuracy: {accuracy:.2f}")
        print(f"Error rate: {error_rate:.2f}")

        # Confusion matrix
        print("Confusion Matrix:")
        confusion_matrix = pd.crosstab(
            ground_truth, predictions, rownames=["Actual"], colnames=["Predicted"]
        )
        print(confusion_matrix)
    else:
        for i in range(len(predictions)):
            print(predictions[i])
        #print(tree.to_graphviz_dot())

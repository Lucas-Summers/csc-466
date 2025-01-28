import sys
from csv_reader import read_csv
from c45 import c45
import json

# try `python induceC45.py csv/nursery.csv trees/nursery_test.json``
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python induceC45.py <TrainingSetFile.csv> [<fileToSave>]")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    
    domain, class_var, df = read_csv(csv_file)
    model = c45()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model.fit(X, y, csv_file)
    print(json.dumps(model.tree, indent=2))

    if len(sys.argv) == 3:
        model.save_tree(sys.argv[2])

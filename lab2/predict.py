import sys
from csv_reader import read_csv
from c45 import c45

# try `python predict.py csv/nursery.csv trees/nursery_sample.json``
if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python predict.py <CSVFile> <JSONFile> [eval]")
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2]
    
    domain, class_var, df = read_csv(csv_file)
    tree = c45()
    tree.read_tree(json_file)

    if len(sys.argv) == 4 and sys.argv[3] == "eval":
        pass
    else:
        print(tree.predict(df[:10]))

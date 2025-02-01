import os
from c45 import c45
import matplotlib.pyplot as plt
from evalScript import read_grid_search_results

# create graphs for tree size and accuracy based on results in evaltrees directory

def tree_size_eval():
    # traverse the trees in evaltrees and graph the tree size
    evaltrees_dir = "evaltrees"
    for subdir in os.listdir(evaltrees_dir):
        print(subdir)
        subdir_path = os.path.join(evaltrees_dir, subdir)
        if os.path.isdir(subdir_path):
            info_gains = []
            ratios = []
            for file in os.listdir(subdir_path):
                if file.endswith(".json") and ("info_gain" in file or "gain_ratio" in file):
                    threshold = file.split("_")[-1].removesuffix(".json")
                    threshold = float(threshold)
                    model = c45()
                    model.read_tree(os.path.join(subdir_path, file))
                    size = model.tree_size()
                    if "info_gain" in file:
                        info_gains.append((threshold, size))
                    elif "gain_ratio" in file:
                        ratios.append((threshold, size))
                
            info_gains.sort(key=lambda x: x[0])
            ratios.sort(key=lambda x: x[0])
            plt.figure()
            plt.plot([thresh[0] for thresh in info_gains], [thresh[1] for thresh in info_gains], label="Info Gain")
            plt.plot([thresh[0] for thresh in ratios], [thresh[1] for thresh in ratios], label="Gain Ratio")
            # points
            plt.scatter([thresh[0] for thresh in info_gains], [thresh[1] for thresh in info_gains])
            plt.scatter([thresh[0] for thresh in ratios], [thresh[1] for thresh in ratios])
            plt.xlabel("Threshold")
            plt.ylabel("Tree Size")
            plt.title("Tree Size")
            plt.legend()
            plt.savefig(f"{subdir_path}/tree_size.png")

def visualize():
    # traverse the evaltrees directory and graph the accuracies for each dataset, save image
    for root, dirs, files in os.walk("evaltrees"):
        for file in files:
            if file == "grid_search.json":
                recs = read_grid_search_results(os.path.join(root, file))
                
                # x-axis: threshold, y-axis: accuracy
                accuracies = [rec["acc"] for rec in recs]
            
                # split the params into info_gain and gain_ratio
                params = [rec["params"] for rec in recs]
                info_idx = [i for i, param in enumerate(params) if param[0] == "info_gain"]
                ratio_idx = [i for i, param in enumerate(params) if param[0] == "gain_ratio"]
                
                # sort in ascending order
                zip_info = list(zip([recs[i]["params"][1] for i in info_idx], [recs[i]["acc"] for i in info_idx]))
                zip_ratio = list(zip([recs[i]["params"][1] for i in ratio_idx], [recs[i]["acc"] for i in ratio_idx]))
                zip_info.sort(key=lambda x: x[0])
                zip_ratio.sort(key=lambda x: x[0])
                
                # plot and save
                plt.figure()
                # line plot 
                plt.plot([pair[0] for pair in zip_info], [pair[1] for pair in zip_info], label="Info Gain")
                plt.plot([pair[0] for pair in zip_ratio], [pair[1] for pair in zip_ratio], label="Gain Ratio")
                plt.xlabel("Threshold")
                plt.ylabel("Accuracy")
                plt.title("Grid Search Results")
                # points
                plt.scatter([pair[0] for pair in zip_info], [pair[1] for pair in zip_info])
                plt.scatter([pair[0] for pair in zip_ratio], [pair[1] for pair in zip_ratio])

                plt.legend()
                plt.savefig(f"{root}/grid_search.png")

def print_one_point(subdir, metric, threshold):
    # print the accuracy for a single point
    val = read_grid_search_results(f"evaltrees/{subdir}/grid_search.json")
    for rec in val:
        if rec["params"][0] == metric and rec["params"][1] == threshold:
            print(rec["acc"], rec["confusion_matrix"])
            # tree size
            model = c45()
            model.read_tree(f"evaltrees/{subdir}/{metric}_{threshold}.json")
            print("tree size", model.tree_size())
            break
        
# tree_size_eval()
# visualize()
print_one_point("nursery", "info_gain", 0.2)

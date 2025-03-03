import numpy as np
import matplotlib.pyplot as plt
from ratings import RatingsMatrix, cosine_similarity, pearson_similarity
from EvaluateCFRandom import eval_cf_random
from EvaluateCFList import eval_cf_list
from tqdm import tqdm

def eval_nn(neigbs=range(1, 100, 25), size=100, repeats=3):
    '''
    Evaluate the effect of the number of neighbors on the Mean MAE and Accuracy
    Plots mae and accuracy vs. number of neighbors for both cosine and pearson similarity
    neigbs: list of number of neighbors to evaluate
    '''
    results = []
    for n in tqdm(neigbs):
        mmae, std, prec, rec, f1, acc = eval_cf_random(method="cosine", size=size, repeats=repeats, nnn=True, adjusted=False, k=n, verbose=False)
        results.append((n, mmae, std, prec, rec, f1, acc))
    plot_results(results, 
                "Number of Neighbors",
                "MAE", "Accuracy", "MMAE vs. Number of Neighbors", 
                f"evals/cos_nn_vs_mmae_acc_{neigbs[-1]}_{size}_{repeats}.png")
    
    results = []
    for n in tqdm(neigbs):
        mmae, std, prec, rec, f1, acc = eval_cf_random(method="pearson", size=size, repeats=repeats, nnn=True, adjusted=False, k=n, verbose=False)
        results.append((n, mmae, std, prec, rec, f1, acc))
    plot_results(results, 
                "Number of Neighbors",
                "MAE", "Accuracy", "MMAE vs. Number of Neighbors", 
                f"evals/pear_nn_vs_mmae_acc_{neigbs[-1]}_{size}_{repeats}.png")

def plot_results(results, x_label, y_label, acc_label, title, filename):
    x = [r[0] for r in results]
    y = [r[1] for r in results]
    yerr = [r[2] for r in results]
    acc = [r[6] for r in results]  # Assuming accuracy is the seventh element in the tuple

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color=color)
    ax1.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel(acc_label, color=color)  # we already handled the x-label with ax1
    ax2.plot(x, acc, 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(title)
    # save to im
    plt.savefig(f"{filename}.png")

def test_all_models1():
    num_reps = 10  # Number of repetitions for random evaluation

    models = [
        ("NNN Pearson (Adjusted)", "pearson", True, True),
        ("NNN Cosine (Adjusted)", "cosine", True, True),
        ("Regular Pearson (Adjusted)", "pearson", False, True),
        ("Regular Cosine (Adjusted)", "cosine", False, True),
        ("NNN Pearson (Unadjusted)", "pearson", True, False),
        ("NNN Cosine (Unadjusted)", "cosine", True, False),
        ("Regular Pearson (Unadjusted)", "pearson", False, False),
        ("Regular Cosine (Unadjusted)", "cosine", False, False),
    ]

    mae_list, acc_list = [], []
    mae_random, acc_random = [[] for _ in models], [[] for _ in models]

    # --- List Evaluation ---
    for model_name, sim_metric, nnn, adjusted in models:
        mae, _, _, _, _, acc = eval_cf_list(sim_metric, "list.txt", nnn, adjusted, k=20)
        mae_list.append(mae)
        acc_list.append(acc)

    # --- Random Evaluation ---
    for _ in range(num_reps):
        for i, (model_name, sim_metric, nnn, adjusted) in enumerate(models):
            mae, _, _, _, _, acc = eval_cf_random(sim_metric, 100, 1, nnn, adjusted, k=20)
            mae_random[i].append(mae)
            acc_random[i].append(acc)

    labels = [m[0] for m in models]
    x = np.arange(len(labels))

    # --- PLOT 1: List Evaluation (Bar Charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(x, mae_list, color=["blue", "green", "red", "purple", "cyan", "orange", "pink", "gray"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel("MAE")
    axes[0].set_title("List Evaluation - MAE")

    axes[1].bar(x, acc_list, color=["blue", "green", "red", "purple", "cyan", "orange", "pink", "gray"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("List Evaluation - Accuracy")

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Random Evaluation (Side-by-Side Line Graphs) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    reps = np.arange(1, num_reps + 1)

    # Left: MAE
    for i, label in enumerate(labels):
        axes[0].plot(reps, mae_random[i], marker="o", label=f"{label}")
    axes[0].set_xlabel("Repetitions")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Random Evaluation - MAE")
    axes[0].legend()

    # Right: Accuracy
    for i, label in enumerate(labels):
        axes[1].plot(reps, acc_random[i], linestyle="--", marker="o", label=f"{label}")
    axes[1].set_xlabel("Repetitions")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Random Evaluation - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def test_all_models2():
    models = [
        ("NNN Pearson (Adjusted)", "pearson", True, True),
        ("NNN Cosine (Adjusted)", "cosine", True, True),
        ("Regular Pearson (Adjusted)", "pearson", False, True),
        ("Regular Cosine (Adjusted)", "cosine", False, True),
        ("NNN Pearson (Unadjusted)", "pearson", True, False),
        ("NNN Cosine (Unadjusted)", "cosine", True, False),
        ("Regular Pearson (Unadjusted)", "pearson", False, False),
        ("Regular Cosine (Unadjusted)", "cosine", False, False),
    ]

    mae_list, acc_list = [], []
    mae_random, acc_random = [], []

    # --- List Evaluation ---
    for model_name, sim_metric, nnn, adjusted in models:
        mae, _, _, _, _, acc = eval_cf_list(sim_metric, "list.txt", nnn, adjusted, k=20)
        mae_list.append(mae)
        acc_list.append(acc)
    
    # --- Random Evaluation (Only Once) ---
    for model_name, sim_metric, nnn, adjusted in models:
        mae, _, _, _, _, acc = eval_cf_random(sim_metric, 100, 10, nnn, adjusted, k=20)
        mae_random.append(mae)
        acc_random.append(acc)
    
    labels = [m[0] for m in models]
    x = np.arange(len(labels))
    colors = ["blue", "green", "red", "purple", "cyan", "orange", "pink", "gray"]

    # --- PLOT 1: List Evaluation (Bar Charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(x, mae_list, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel("MAE")
    axes[0].set_title("List Evaluation - MAE")

    axes[1].bar(x, acc_list, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("List Evaluation - Accuracy")

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Random Evaluation (Bar Charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(x, mae_random, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Random Evaluation - MAE")

    axes[1].bar(x, acc_random, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Random Evaluation - Accuracy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #eval_nn()
    #test_all_models1()
    test_all_models2()

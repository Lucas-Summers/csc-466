import numpy as np
import matplotlib.pyplot as plt
from ratings import RatingsMatrix, cosine_similarity, pearson_similarity
from EvaluateCFRandom import eval_cf_random
from EvaluateCFList import eval_cf_list

def test_best_n_neighbors():
    n_values = list(range(1, 50, 2))
    list_mae = []
    random_mae = []
    list_accuracy = []
    random_accuracy = []
    
    # Evaluate CF for each N
    for n in n_values:
        mae_list, _, _, _, _, acc_list = eval_cf_list("pearson", "list.txt", True, False, k=n)
        list_mae.append(mae_list)
        list_accuracy.append(acc_list)
        
        mae_random, _, _, _, _, acc_random = eval_cf_random("pearson", 5, 1, True, False, k=n)
        random_mae.append(mae_random)
        random_accuracy.append(acc_random)
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot MAE for both list-based and random-based
    axes[0].plot(n_values, list_mae, marker='o', color='b', label='List-based MAE')
    axes[0].plot(n_values, random_mae, marker='x', color='r', label='Random-based MAE')
    axes[0].set_xlabel("N (Number of Neighbors)")
    axes[0].set_ylabel("Mean Absolute Error (MAE)")
    axes[0].set_title("MAE for List-based and Random-based CF Evaluation")
    axes[0].legend()

    # Plot accuracy for both list-based and random-based
    axes[1].plot(n_values, list_accuracy, marker='o', color='b', label='List-based Accuracy')
    axes[1].plot(n_values, random_accuracy, marker='x', color='r', label='Random-based Accuracy')
    axes[1].set_xlabel("N (Number of Neighbors)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy for List-based and Random-based CF Evaluation")
    axes[1].legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Best N for each method
    best_n_list = n_values[np.argmin(list_mae)]
    best_n_random = n_values[np.argmin(random_mae)]
    print(f"Best N value (List-based): {best_n_list} with MAE: {min(list_mae)} and Accuracy: {list_accuracy[np.argmin(list_mae)]}")
    print(f"Best N value (Random-based): {best_n_random} with MAE: {min(random_mae)} and Accuracy: {random_accuracy[np.argmin(random_mae)]}")

def test_all_models():
    num_reps = 5  # Number of repetitions for random evaluation

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
        mae, _, _, _, _, acc = eval_cf_list(sim_metric, "list.txt", nnn, adjusted, k=19)
        mae_list.append(mae)
        acc_list.append(acc)

    # --- Random Evaluation ---
    for _ in range(num_reps):
        for i, (model_name, sim_metric, nnn, adjusted) in enumerate(models):
            mae, _, _, _, _, acc = eval_cf_random(sim_metric, 5, 1, nnn, adjusted, k=19)
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
        axes[0].plot(reps, mae_random[i], marker="o", label=f"{label} - MAE")
    axes[0].set_xlabel("Repetitions")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Random Evaluation - MAE")
    axes[0].legend()

    # Right: Accuracy
    for i, label in enumerate(labels):
        axes[1].plot(reps, acc_random[i], linestyle="--", marker="o", label=f"{label} - Accuracy")
    axes[1].set_xlabel("Repetitions")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Random Evaluation - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #test_best_n_neighbors()
    test_all_models()

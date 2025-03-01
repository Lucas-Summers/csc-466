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

def test_adjusted_vs_regular():
    num_reps = 5  # Number of repetitions for random evaluation

    # Store results as lists for random evaluation (each will have 10 values)
    mae_nnn_adjusted, acc_nnn_adjusted = [], []
    mae_nnn, acc_nnn = [], []
    mae_regular, acc_regular = [], []
    mae_regular_adjusted, acc_regular_adjusted = [], []

    # --- List Evaluation (Single Run) ---
    mae_nnn_adjusted_list, _, _, _, _, acc_nnn_adjusted_list = eval_cf_list("pearson", "list.txt", True, True, k=19)
    mae_nnn_list, _, _, _, _, acc_nnn_list = eval_cf_list("pearson", "list.txt", True, False, k=19)
    mae_regular_list, _, _, _, _, acc_regular_list = eval_cf_list("pearson", "list.txt", False, False)
    mae_regular_adjusted_list, _, _, _, _, acc_regular_adjusted_list = eval_cf_list("pearson", "list.txt", False, True)

    # --- Random Evaluation (Looping for 10 Repetitions) ---
    for _ in range(num_reps):
        mae_nnn_adjusted_val, _, _, _, _, acc_nnn_adjusted_val = eval_cf_random("pearson", 5, 1, True, True, k=19)
        mae_nnn_val, _, _, _, _, acc_nnn_val = eval_cf_random("pearson", 5, 1, True, False, k=19)
        mae_regular_val, _, _, _, _, acc_regular_val = eval_cf_random("pearson", 5, 1, False, False)
        mae_regular_adjusted_val, _, _, _, _, acc_regular_adjusted_val = eval_cf_random("pearson", 5, 1, False, True)

        mae_nnn_adjusted.append(mae_nnn_adjusted_val)
        acc_nnn_adjusted.append(acc_nnn_adjusted_val)
        mae_nnn.append(mae_nnn_val)
        acc_nnn.append(acc_nnn_val)
        mae_regular.append(mae_regular_val)
        acc_regular.append(acc_regular_val)
        mae_regular_adjusted.append(mae_regular_adjusted_val)
        acc_regular_adjusted.append(acc_regular_adjusted_val)

    # Labels
    labels = ["NNN Adjusted", "NNN", "Regular", "Regular Adjusted"]
    x = np.arange(len(labels))

    # --- PLOT 1: List Evaluation (Bar Charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE Bar Chart
    axes[0].bar(x, [mae_nnn_adjusted_list, mae_nnn_list, mae_regular_list, mae_regular_adjusted_list], 
                color=["blue", "green", "red", "purple"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("List Evaluation - MAE")

    # Accuracy Bar Chart
    axes[1].bar(x, [acc_nnn_adjusted_list, acc_nnn_list, acc_regular_list, acc_regular_adjusted_list], 
                color=["blue", "green", "red", "purple"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("List Evaluation - Accuracy")

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Random Evaluation (Side-by-Side Line Graphs) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reps = np.arange(1, num_reps + 1)  # X-axis (Repetitions)

    # Left: MAE
    axes[0].plot(reps, mae_nnn_adjusted, "bo-", label="NNN Adjusted - MAE")
    axes[0].plot(reps, mae_nnn, "go-", label="NNN - MAE")
    axes[0].plot(reps, mae_regular, "ro-", label="Regular - MAE")
    axes[0].plot(reps, mae_regular_adjusted, "mo-", label="Regular Adjusted - MAE")
    axes[0].set_xlabel("Repetitions")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Random Evaluation - MAE")
    axes[0].legend()

    # Right: Accuracy
    axes[1].plot(reps, acc_nnn_adjusted, "b--", label="NNN Adjusted - Accuracy")
    axes[1].plot(reps, acc_nnn, "g--", label="NNN - Accuracy")
    axes[1].plot(reps, acc_regular, "r--", label="Regular - Accuracy")
    axes[1].plot(reps, acc_regular_adjusted, "m--", label="Regular Adjusted - Accuracy")
    axes[1].set_xlabel("Repetitions")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Random Evaluation - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def test_pearson_vs_cosine():
    num_reps = 5  # Number of repetitions for random evaluation

    # Store results for random evaluation (each will have 10 values)
    mae_nnn_pearson, acc_nnn_pearson = [], []
    mae_nnn_cosine, acc_nnn_cosine = [], []
    mae_regular_pearson, acc_regular_pearson = [], []
    mae_regular_cosine, acc_regular_cosine = [], []

    # --- List Evaluation (Single Run) ---
    mae_nnn_pearson_list, _, _, _, _, acc_nnn_pearson_list = eval_cf_list("pearson", "list.txt", True, True, k=19)
    mae_nnn_cosine_list, _, _, _, _, acc_nnn_cosine_list = eval_cf_list("cosine", "list.txt", True, True, k=19)
    mae_regular_pearson_list, _, _, _, _, acc_regular_pearson_list = eval_cf_list("pearson", "list.txt", False, True)
    mae_regular_cosine_list, _, _, _, _, acc_regular_cosine_list = eval_cf_list("cosine", "list.txt", False, True)

    # --- Random Evaluation (Looping for 10 Repetitions) ---
    for _ in range(num_reps):
        mae_nnn_pearson_val, _, _, _, _, acc_nnn_pearson_val = eval_cf_random("pearson", 5, 1, True, True, k=19)
        mae_nnn_cosine_val, _, _, _, _, acc_nnn_cosine_val = eval_cf_random("cosine", 5, 1, True, True, k=19)
        mae_regular_pearson_val, _, _, _, _, acc_regular_pearson_val = eval_cf_random("pearson", 5, 1, False, True)
        mae_regular_cosine_val, _, _, _, _, acc_regular_cosine_val = eval_cf_random("cosine", 5, 1, False, True)

        mae_nnn_pearson.append(mae_nnn_pearson_val)
        acc_nnn_pearson.append(acc_nnn_pearson_val)
        mae_nnn_cosine.append(mae_nnn_cosine_val)
        acc_nnn_cosine.append(acc_nnn_cosine_val)
        mae_regular_pearson.append(mae_regular_pearson_val)
        acc_regular_pearson.append(acc_regular_pearson_val)
        mae_regular_cosine.append(mae_regular_cosine_val)
        acc_regular_cosine.append(acc_regular_cosine_val)

    # Labels
    labels = ["NNN Pearson", "NNN Cosine", "Regular Pearson", "Regular Cosine"]
    x = np.arange(len(labels))

    # --- PLOT 1: List Evaluation (Bar Charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE Bar Chart
    axes[0].bar(x, [mae_nnn_pearson_list, mae_nnn_cosine_list, mae_regular_pearson_list, mae_regular_cosine_list], 
                color=["blue", "green", "red", "purple"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("List Evaluation - MAE")

    # Accuracy Bar Chart
    axes[1].bar(x, [acc_nnn_pearson_list, acc_nnn_cosine_list, acc_regular_pearson_list, acc_regular_cosine_list], 
                color=["blue", "green", "red", "purple"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("List Evaluation - Accuracy")

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Random Evaluation (Side-by-Side Line Graphs) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reps = np.arange(1, num_reps + 1)  # X-axis (Repetitions)

    # Left: MAE
    axes[0].plot(reps, mae_nnn_pearson, "bo-", label="NNN Pearson - MAE")
    axes[0].plot(reps, mae_nnn_cosine, "go-", label="NNN Cosine - MAE")
    axes[0].plot(reps, mae_regular_pearson, "ro-", label="Regular Pearson - MAE")
    axes[0].plot(reps, mae_regular_cosine, "mo-", label="Regular Cosine - MAE")
    axes[0].set_xlabel("Repetitions")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Random Evaluation - MAE")
    axes[0].legend()

    # Right: Accuracy
    axes[1].plot(reps, acc_nnn_pearson, "b--", label="NNN Pearson - Accuracy")
    axes[1].plot(reps, acc_nnn_cosine, "g--", label="NNN Cosine - Accuracy")
    axes[1].plot(reps, acc_regular_pearson, "r--", label="Regular Pearson - Accuracy")
    axes[1].plot(reps, acc_regular_cosine, "m--", label="Regular Cosine - Accuracy")
    axes[1].set_xlabel("Repetitions")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Random Evaluation - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def test_best_models():
    num_reps = 5  # Number of repetitions for random evaluation

    # Store results for random evaluation (each will have 10 values)
    mae_nnn_pearson, acc_nnn_pearson = [], []
    mae_regular_pearson, acc_regular_pearson = [], []

    # --- List Evaluation (Single Run) ---
    mae_nnn_pearson_list, _, _, _, _, acc_nnn_pearson_list = eval_cf_list("pearson", "list.txt", True, True, k=19)
    mae_regular_pearson_list, _, _, _, _, acc_regular_pearson_list = eval_cf_list("pearson", "list.txt", False, True)

    # --- Random Evaluation (Looping for 10 Repetitions) ---
    for _ in range(num_reps):
        mae_nnn_pearson_val, _, _, _, _, acc_nnn_pearson_val = eval_cf_random("pearson", 10, 1, True, True, k=19)
        mae_regular_pearson_val, _, _, _, _, acc_regular_pearson_val = eval_cf_random("pearson", 10, 1, False, True)

        mae_nnn_pearson.append(mae_nnn_pearson_val)
        acc_nnn_pearson.append(acc_nnn_pearson_val)
        mae_regular_pearson.append(mae_regular_pearson_val)
        acc_regular_pearson.append(acc_regular_pearson_val)

    # Labels
    labels = ["Adjusted NNN (Pearson, k=19)", "Adjusted Sum (Pearson)"]
    x = np.arange(len(labels))

    # --- PLOT 1: List Evaluation (Bar Charts) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE Bar Chart
    axes[0].bar(x, [mae_nnn_pearson_list, mae_regular_pearson_list], color=["blue", "green"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("List Evaluation - MAE")

    # Accuracy Bar Chart
    axes[1].bar(x, [acc_nnn_pearson_list, acc_regular_pearson_list], color=["blue", "green"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("List Evaluation - Accuracy")

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Random Evaluation (Side-by-Side Line Graphs) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reps = np.arange(1, num_reps + 1)  # X-axis (Repetitions)

    # Left: MAE
    axes[0].plot(reps, mae_nnn_pearson, "bo-", label="Adjusted NNN (Pearson) - MAE")
    axes[0].plot(reps, mae_regular_pearson, "go-", label="Adjusted Sum (Pearson) - MAE")
    axes[0].set_xlabel("Repetitions")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Random Evaluation - MAE")
    axes[0].legend()

    # Right: Accuracy
    axes[1].plot(reps, acc_nnn_pearson, "b--", label="Adjusted NNN (Pearson) - Accuracy")
    axes[1].plot(reps, acc_regular_pearson, "g--", label="Adjusted Sum (Pearson) - Accuracy")
    axes[1].set_xlabel("Repetitions")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Random Evaluation - Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_best_n_neighbors()
    #test_adjusted_vs_regular()
    #test_pearson_vs_cosine()
    #test_best_models()

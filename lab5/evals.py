from EvaluateCFRandom import eval_cf_random
import matplotlib.pyplot as plt
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

eval_nn()

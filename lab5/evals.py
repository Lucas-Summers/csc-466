from EvaluateCFRandom import main as cf_random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

def eval_nn(n=10, size=100, repeats=3):
    '''
    Evaluate the effect of the number of neighbors on the Mean MAE and Accuracy
    Plots mae and accuracy vs. number of neighbors for both cosine and pearson similarity
    neigbs: list of number of neighbors to evaluate
    '''
    os.makedirs("results", exist_ok=True)
    mmae, std, prec, rec, f1, acc = cf_random(method="cosine", size=size, repeats=repeats, nnn=True, adjusted=False, k=n, verbose=False)
    with open(f"results/cosine_{n}_{size}_{repeats}.txt", "w") as f:
        f.write(f"{n},{mmae},{std},{prec},{rec},{f1},{acc}\n")
    mmae, std, prec, rec, f1, acc = cf_random(method="pearson", size=size, repeats=repeats, nnn=True, adjusted=False, k=n, verbose=False)
    with open(f"results/pearson_{n}_{size}_{repeats}.txt", "w") as f:
        f.write(f"{n},{mmae},{std},{prec},{rec},{f1},{acc}\n")


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a collaborative filtering algorithm")
    parser.add_argument("neigbs", help="The number of neighbors to evaluate", type=int)
    parser.add_argument("--size", help="The number of test cases to generate", type=int, default=100)
    parser.add_argument("--repeats", help="The number of times to repeat the test cases", type=int, default=3)
    args = parser.parse_args()
    eval_nn(n=args.neigbs, size=args.size, repeats=args.repeats)

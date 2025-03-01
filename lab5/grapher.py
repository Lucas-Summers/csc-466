import pandas as pd
import matplotlib.pyplot as plt

def plot_stats_from_csv(csv_file="consolidated_results.csv"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, index_col=False)
    
    methods = ['cosine', 'pearson']
    adjustments = ['nonadjusted', 'adjusted']
    colors = ["#4878D0", "#6ACC64", "#D65F5F", "#B47CC7"]
    
    # Plot each statistic with respect to 'n'
    stats = ['mmae', 'prec', 'rec', 'f1', 'acc']
    for stat in stats:
        plt.figure()
        color_index = 0
        for method in methods:
            for adjusted in adjustments:
                # Filter the DataFrame for the specific method, adjustment, size 100, and 10 repeats
                filtered_df = df[(df['method'] == method) & (df['adjusted'] == adjusted) & (df['size'] == 100) & (df['repeats'] == 10)]
                # Sort the filtered DataFrame by the 'n' column
                filtered_df = filtered_df.sort_values('n')
                # Convert the 'n' column to numeric
                filtered_df['n'] = pd.to_numeric(filtered_df['n'])
                
                # Plot the data
                if stat == 'mmae':
                    plt.plot(filtered_df['n'], filtered_df[stat], marker='o', color=colors[color_index], label=f'{method} {adjusted}')
                    plt.fill_between(filtered_df['n'], filtered_df[stat] - filtered_df['std'], filtered_df[stat] + filtered_df['std'], alpha=0.3, color=colors[color_index])
                else:
                    plt.plot(filtered_df['n'], filtered_df[stat], marker='o', color=colors[color_index], label=f'{method} {adjusted}')
                
                color_index += 1

        plt.xlabel('Number of Neighbors (n)')
        plt.ylabel(stat.upper())
        plt.title(f'{stat.upper()} vs. Number of Neighbors (n)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"evals/{stat}_vs_n.png")
        plt.show()

# Call the function to plot the stats
plot_stats_from_csv()
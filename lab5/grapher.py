import pandas as pd
import matplotlib.pyplot as plt

def plot_stats_from_csv(csv_file="consolidated_results.csv"):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, index_col=False)
    # Filter the DataFrame for the cosine method, size 100, and 10 repeats
    filtered_df = df[(df['method'] == 'pearson') & (df['size'] == 100) & (df['repeats'] == 10)]
    # Sort the filtered DataFrame by the 'n' column
    filtered_df = filtered_df.sort_values('n')
    print(filtered_df.head())
    # Convert the 'n' column to numeric
    filtered_df['n'] = pd.to_numeric(filtered_df['n'])
    
    # Plot each statistic with respect to 'n'
    stats = ['mmae', 'prec', 'rec', 'f1', 'acc']
    for stat in stats:
        plt.figure()
        if stat == 'mmae':
            plt.plot(filtered_df['n'], filtered_df[stat], marker='o')
            plt.fill_between(filtered_df['n'], filtered_df[stat] - filtered_df['std'], filtered_df[stat] + filtered_df['std'], alpha=0.3)        
        else:
            plt.plot(filtered_df['n'], filtered_df[stat], marker='o')
        plt.xlabel('Number of Neighbors (n)')
        plt.ylabel(stat.upper())
        plt.title(f'{stat.upper()} vs. Number of Neighbors (n)')
        plt.grid(True)
        plt.savefig(f"{stat}_vs_n.png")
        plt.show()

# Call the function to plot the stats
plot_stats_from_csv()
import os
import csv

def consolidate_results(results_dir="results", output_file="consolidated_results.csv"):
    # List all files in the results directory
    files = os.listdir(results_dir)
    
    # Open the output CSV file for writing
    with open(output_file, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header row
        csvwriter.writerow(["method", "n", "size", "repeats", "adjusted", "mmae", "std", "prec", "rec", "f1", "acc"])

        # Iterate over each file in the results directory
        for file in files:
            if file.endswith(".txt"):
                # Extract method, n, size, and repeats from the filename
                parts = file.split('_')
                method = parts[0]
                n = parts[1]
                size = parts[2]
                adjusted = "nonadjusted"
                if "adjusted" in file:
                    repeats = parts[3]
                    adjusted = "adjusted"
                else:
                    repeats = parts[3].split('.')[0]
                
                # Read the content of the file
                with open(os.path.join(results_dir, file), "r") as f:
                    content = f.readline().strip()
                    # Write the consolidated row to the CSV file
                    csvwriter.writerow([method, n, size, repeats, adjusted] + content.split(',')[1:])

# Call the function to consolidate results
consolidate_results()
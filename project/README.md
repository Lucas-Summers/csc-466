# CSC 466 FINAL ANALYTICAL PROJECT (Winter 2025)

## Team
- Lucas Summers (lsumme01@calpoly.edu)
- Xiuyuan Qiu (xiqiu@calpoly.edu)
- Braeden Alonge (balonge@calpoly.edu)
- Nathan Lim (nlim10@calpoly.edu)

## Overview
This project analyzes a comprehensive dataset of board games to understand what 
factors contribute to high ratings. The analysis uses principal component analysis, 
random forest/linear regression, collaborative filtering, and various clustering 
techniques to identify patterns and insights.

## Dataset

Data can be downloaded here:

https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek/

*NOTE:* All CSV files must be in the `data/` dir at the root of the project


## Requirements
```
scikit-learn
numpy
pandas
tqdm
matplotlib
seaborn
notebook
```
If not already installed, run `pip install -r requirements.txt`.

## Running the Notebooks
```
cd notebooks/
jupyter notebook
```

Next, open the `0_index` notebook which will guide you through the individual notebooks

*NOTE:* run notebooks in order as some rely on data produced in previous notebooks

- Intermediate dataframes and data shared between notebooks are stored in the `frames/` dir
- All plots generated in notebooks are stored in the `plots/` dir









import pandas as pd
import matplotlib.pyplot as plt


def unique_values_count(df, plot=True):
    """
    Count the number of unique values in each column and optionally display a bar plot.

    Parameters:
        df (pd.DataFrame): Input dataset
        plot (bool): Whether to display a bar graph

    Returns:
        pd.Series: Number of unique values per column
    """
    # Count unique values per column
    unique_counts = df.nunique(dropna=False)  # include NaN as a unique value

    # Plot
    if plot:
        plt.figure(figsize=(10, 5))
        unique_counts.plot(kind="bar", color="green")
        plt.title("Unique Values per Column")
        plt.ylabel("Number of Unique Values")
        plt.xlabel("Columns")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return unique_counts

df = pd.read_csv("../data/messy_data.csv")
unique_counts = unique_values_count(df.iloc[:, 2:])
print(unique_counts)
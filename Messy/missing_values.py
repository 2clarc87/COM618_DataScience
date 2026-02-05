import pandas as pd
import matplotlib.pyplot as plt


def missing_values_by_column(df, missing_markers=None, plot=True):
    if missing_markers is None:
        missing_markers = ["?", None, "", "NA"]

    # Replace custom missing markers with NaN
    df_clean = df.replace(missing_markers, pd.NA)

    # Count missing values per column
    missing_counts = df_clean.isna().sum()

    # Plot
    if plot:
        plt.figure(figsize=(10, 5))
        missing_counts.plot(kind="bar")
        plt.title("Missing Values per Column")
        plt.ylabel("Count of Missing Values")
        plt.xlabel("Columns")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return missing_counts

df = pd.read_csv("../data/messy_data.csv")
missing_counts = missing_values_by_column(df)
print(missing_counts)

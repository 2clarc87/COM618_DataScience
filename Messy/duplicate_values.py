import pandas as pd
import matplotlib.pyplot as plt


def duplicate_analysis(df):
    duplicate_rows = df.duplicated().sum()
    return duplicate_rows


df = pd.read_csv("../data/messy_data.csv")
duplicate_rows = duplicate_analysis(df)
print(f"Duplicate rows: {duplicate_rows}")
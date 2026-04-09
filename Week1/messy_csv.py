import pandas as pd
import numpy as np
import random
import string

# -----------------------------
# Configuration
# -----------------------------
INPUT_CSV = "data/clean_data.csv"
OUTPUT_CSV = "data/messy_data.csv"

MESSINESS_LEVEL = 0.15  # 0.0 = clean, 0.3 = very messy

MISSING_TOKENS = ["?", "None", np.nan, "NA"]
YES_NO_VARIANTS = ["Yes", "YES", "yes", "No", "NO", "no"]
DRUG_STATES = ["No", "Steady", "Up", "Down", "NO", "steady"]

# -----------------------------
# Helper functions
# -----------------------------
def random_typo(value):
    if not isinstance(value, str) or len(value) < 3:
        return value
    i = random.randint(0, len(value) - 2)
    return value[:i] + value[i + 1] + value[i] + value[i + 2:]

def maybe_missing(value):
    if random.random() < MESSINESS_LEVEL:
        return random.choice(MISSING_TOKENS)
    return value

def corrupt_numeric(value):
    if pd.isna(value):
        return value
    r = random.random()
    if r < 0.33:
        return str(value)          # numeric as string
    elif r < 0.66:
        return value * random.choice([10, -1])  # outlier
    else:
        return value

def corrupt_diag(code):
    if pd.isna(code):
        return code
    code = str(code)
    if random.random() < MESSINESS_LEVEL:
        if random.random() < 0.5:
            return code + random.choice(string.ascii_uppercase)
        else:
            return code.replace(".", random.choice(["", ","]))
    return code

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# FORCE everything to object so corruption is always legal
df = df.astype("object")

# -----------------------------
# Column groups
# -----------------------------
numeric_cols = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

diag_cols = ["diag_1", "diag_2", "diag_3"]
yes_no_cols = ["diabetesMed", "change"]

drug_cols = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "insulin"
]

categorical_cols = df.select_dtypes(include=["object", "string"]).columns

# -----------------------------
# 🔑 CRITICAL FIX
# Allow numeric columns to hold messy values
# -----------------------------
df[numeric_cols] = df[numeric_cols].astype("object")

# -----------------------------
# Apply messiness
# -----------------------------
total_ops = len(df) * len(df.columns)
op_count = 0

for col in df.columns:
    for i in range(len(df)):
        if op_count % 10_000 == 0:
            print(f"{op_count / total_ops:.1%} complete")
        op_count += 1

        value = df.at[i, col]

        # Random missingness
        value = maybe_missing(value)

        # Column-specific corruption
        # if col in numeric_cols and random.random() < MESSINESS_LEVEL:
        #     value = corrupt_numeric(value)
        #
        # elif col in diag_cols:
        #     value = corrupt_diag(value)

        if col in yes_no_cols and random.random() < MESSINESS_LEVEL:
            value = random.choice(YES_NO_VARIANTS)

        elif col in drug_cols and random.random() < MESSINESS_LEVEL:
            value = random.choice(DRUG_STATES)

        elif col in categorical_cols and random.random() < MESSINESS_LEVEL:
            if isinstance(value, str):
                value = value.lower() if random.random() < 0.5 else value#random_typo(value)

        df.at[i, col] = value

# -----------------------------
# Duplicate some rows
# -----------------------------
n_dupes = int(len(df) * MESSINESS_LEVEL)
dupes = df.sample(n_dupes, random_state=42)
df = pd.concat([df, dupes], ignore_index=True)

# -----------------------------
# Shuffle rows
# -----------------------------
df = df.sample(frac=1).reset_index(drop=True)

# -----------------------------
# Save messy data
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"Messy dataset saved to {OUTPUT_CSV}")

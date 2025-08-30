import pandas as pd
import os

# List of dataset files
files = [
    "Dataset/CEAS_08.csv",
    "Dataset/Enron.csv",
    "Dataset/Ling.csv",
    "Dataset/Nazario.csv",
    "Dataset/Nigerian_Fraud.csv",
    "Dataset/phishing_email.csv",
    "Dataset/SpamAssasin.csv",
    "Dataset/spam_Emails_data.csv"   
]

all_dfs = []

for file in files:
    if not os.path.exists(file):
        print(f" Skipping {file} (file not found)")
        continue

    try:
        df = pd.read_csv(file)

        if "CEAS_08.csv" in file or "Nazario.csv" in file or "Nigerian_Fraud.csv" in file or "SpamAssasin.csv" in file:
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
            df = df[["label", "text"]]

        elif "Enron.csv" in file or "Ling.csv" in file:
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
            df = df[["label", "text"]]

        elif "phishing_email.csv" in file:
            df = df.rename(columns={"text_combined": "text"})
            df = df[["label", "text"]]

        elif "spam_Emails_data.csv" in file:
            df = df[["label", "text"]]

        else:
            print(f" Skipping {file} (unsupported format)")
            continue

        df = df.dropna()
        all_dfs.append(df)

    except Exception as e:
        print(f" Error reading {file}: {e}")

# Merge all datasets
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n Final merged dataset shape: {merged_df.shape}")
    print(merged_df["label"].value_counts())
    merged_df.to_csv("Dataset/final_emails.csv", index=False)
    print("\n Saved merged dataset to Dataset/final_emails.csv")
else:
    print(" No datasets were loaded!")

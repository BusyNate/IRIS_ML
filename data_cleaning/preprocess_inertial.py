# preprocess_inertial.py
import pandas as pd
from scipy.signal import savgol_filter
import os

def load_and_clean_csv(file_path, output_path):
    """Clean inertial dataset: drop NaN, normalize, smooth"""
    df = pd.read_csv(file_path)

    # Drop missing values
    df = df.dropna()

    # Normalize numeric values
    for col in df.columns:
        if col not in ["timestamp", "label"]:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Apply smoothing
    for col in df.columns:
        if col not in ["timestamp", "label"]:
            df[col] = savgol_filter(df[col], window_length=3, polyorder=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned inertial dataset saved to {output_path}")

if __name__ == "__main__":
    load_and_clean_csv("dataset/inertial/raw_inertial.csv", "outputs/clean_inertial.csv")

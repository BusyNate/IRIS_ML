# preprocess_blink_from_dataset.py
import pandas as pd
import numpy as np
import os

def preprocess_blink_dataset(input_csv, output_csv):
    """Clean blink dataset: keep duration, blink type, and mean eye openness"""

    # Load dataset
    df = pd.read_csv(input_csv)

    # Convert "1.2,3.4,5.6" -> average
    def series_mean(series_str):
        try:
            values = [float(x) for x in str(series_str).split(",") if x.strip() != ""]
            return np.mean(values) if values else np.nan
        except:
            return np.nan

    # Add left & right eye averages
    df["left_eye_mean"] = df["eye_openness_left_series"].apply(series_mean)
    df["right_eye_mean"] = df["eye_openness_right_series"].apply(series_mean)

    # Keep only important columns
    clean_df = df[["Participant", "blink_duration", "left_eye_mean", "right_eye_mean", "blink_type"]].dropna()

    # Normalize (0-1 scaling)
    for col in ["blink_duration", "left_eye_mean", "right_eye_mean"]:
        clean_df[col] = (clean_df[col] - clean_df[col].min()) / (clean_df[col].max() - clean_df[col].min())

    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned blink dataset saved to {output_csv}")

if __name__ == "__main__":
    preprocess_blink_dataset("dataset.csv", "outputs/clean_blink.csv")

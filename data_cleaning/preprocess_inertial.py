import pandas as pd
from scipy.signal import savgol_filter

def clean_inertial_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df.dropna()

    # Normalize only sensor columns, skip timestamp
    for col in df.columns:
        if col not in ["timestamp"]:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Apply smoothing only on sensor columns
    for col in df.columns:
        if col not in ["timestamp"]:
            df[col] = savgol_filter(df[col], 5, 2)

    df.to_csv(output_csv, index=False)
    print(f"Inertial dataset cleaned and saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "data/dataset.csv"
    output_csv = "outputs/clean_inertial.csv"
    clean_inertial_data(input_csv, output_csv)

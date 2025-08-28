import pandas as pd
from scipy.signal import savgol_filter

def clean_inertial_data(input_csv, output_csv):
    """Load inertial dataset, normalize values, smooth signals"""
    df = pd.read_csv(input_csv)

    # Drop rows with missing values
    df = df.dropna()

    # Normalize accelerometer/gyro values (scale 0–1)
    for col in df.columns:
        if col not in ["timestamp", "label"]:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Apply smoothing filter
    for col in df.columns:
        if col not in ["timestamp", "label"]:
            df[col] = savgol_filter(df[col], 5, 2)

    df.to_csv(output_csv, index=False)
    print(f"Inertial dataset cleaned and saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "../data/inertial_raw.csv"         # raw inertial data
    output_csv = "../outputs/clean_inertial.csv"   # cleaned output
    clean_inertial_data(input_csv, output_csv)

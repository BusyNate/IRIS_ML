import pandas as pd

def clean_blink_data(input_csv, output_csv):
    """Load blink dataset, normalize pixel values, save cleaned version"""
    df = pd.read_csv(input_csv)

    # Drop rows with missing values
    df = df.dropna()

    # Normalize all numeric columns (scale 0–1)
    for col in df.columns:
        if col != "label":  # don't normalize labels
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df.to_csv(output_csv, index=False)
    print(f"Blink dataset cleaned and saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "../data/dataset.csv"         # raw dataset
    output_csv = "../outputs/clean_blink.csv" # cleaned output
    clean_blink_data(input_csv, output_csv)

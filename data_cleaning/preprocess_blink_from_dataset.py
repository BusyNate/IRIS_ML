import pandas as pd

def clean_blink_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = df.dropna()

    # Normalize only raw-voltage column, skip timestamp
    for col in df.columns:
        if col not in ["timestamp"]:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df.to_csv(output_csv, index=False)
    print(f"Blink dataset cleaned and saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "data/dataset.csv"
    output_csv = "outputs/clean_blink.csv"
    clean_blink_data(input_csv, output_csv)

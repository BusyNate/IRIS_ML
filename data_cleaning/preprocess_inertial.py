import pandas as pd
import os

def clean_inertial_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are completely NaN
    df.dropna(how='all', inplace=True)

    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0

    df.to_csv(output_csv, index=False)
    print(f"Inertial data cleaned and saved to {output_csv}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "inertial_dataset.csv")  
    output_csv = os.path.join(base_dir, "inertial_dataset_cleaned.csv")
    clean_inertial_data(input_csv, output_csv)

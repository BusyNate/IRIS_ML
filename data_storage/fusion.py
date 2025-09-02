import pandas as pd
import os

def fuse_data(file_list, output_csv):
    combined_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)
        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(how='all', inplace=True)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Normalize numeric columns in combined data
    numeric_cols = combined_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        min_val = combined_df[col].min()
        max_val = combined_df[col].max()
        if min_val != max_val:
            combined_df[col] = (combined_df[col] - min_val) / (max_val - min_val)
        else:
            combined_df[col] = 0

    combined_df.to_csv(output_csv, index=False)
    print(f"Fused data saved to {output_csv}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_fuse = [
        os.path.join(base_dir, "..\\data_cleaning\\blink_dataset_cleaned.csv"),
        os.path.join(base_dir, "..\\data_cleaning\\inertial_dataset_cleaned.csv")
    ]
    output_csv = os.path.join(base_dir, "fused_data.csv")
    fuse_data(files_to_fuse, output_csv)

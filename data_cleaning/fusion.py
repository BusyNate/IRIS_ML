# fusion.py
import pandas as pd
import os

def fuse_predictions(blink_csv, inertial_csv, output_csv):
    """Combine blink-based and inertial-based predictions into one file"""
    blink_df = pd.read_csv(blink_csv)
    inertial_df = pd.read_csv(inertial_csv)

    # Align by row index (simplest fusion)
    fused = pd.concat([blink_df, inertial_df], axis=1)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fused.to_csv(output_csv, index=False)
    print(f"✅ Fused dataset saved to {output_csv}")

if __name__ == "__main__":
    fuse_predictions("outputs/clean_blink.csv", "outputs/clean_inertial.csv", "outputs/fused_data.csv")

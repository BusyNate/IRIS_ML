import pandas as pd

def fuse_predictions(blink_csv, inertial_csv, output_csv):
    blink_df = pd.read_csv(blink_csv)
    inertial_df = pd.read_csv(inertial_csv)

    # Merge them side by side, keep timestamp only once
    fused = pd.concat([blink_df, inertial_df.drop(columns=["timestamp"])], axis=1)

    fused.to_csv(output_csv, index=False)
    print(f"Fused predictions saved to {output_csv}")

if __name__ == "__main__":
    blink_csv = "outputs/clean_blink.csv"
    inertial_csv = "outputs/clean_inertial.csv"
    output_csv = "outputs/fused_predictions.csv"

    fuse_predictions(blink_csv, inertial_csv, output_csv)

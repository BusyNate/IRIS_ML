import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configurable parameters
sampling_rate = 50  # Hz (adjust to match ESP32)
window_seconds = 2  # window length in seconds
step_seconds = 1  # overlap step in seconds

window_size = int(window_seconds * sampling_rate)
step = int(step_seconds * sampling_rate)

# Feature extraction for MPU6050 (only ay, gx)
def extract_features_mpu(X, y, window_size, step):
    Xs, ys = [], []
    for i in range(0, len(X) - window_size, step):
        window = X.iloc[i:i+window_size]
        features = []
        for col in ["acc_y", "gyro_x"]:
            features += [
                window[col].mean(),
                window[col].std(),
                window[col].min(),
                window[col].max(),
            ]
        Xs.append(features)
        ys.append(y[i+window_size-1])
    return np.array(Xs), np.array(ys)

# Load dataset
df = pd.read_csv("C:/Users/mndiw/Documents/IRIS_ML/data/inertial/raw_inertial.csv")  # timestamp, ay, gx, label
X = df[["acc_y", "gyro_x"]]
y = df["label"]

# Extract features
X_feat, y_feat = extract_features_mpu(X, y, window_size, step)

if len(X_feat) <= 0:
    raise ValueError(f"Dataset too small! Need at least {window_size} rows, got {len(X)}.")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_feat, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"MPU6050 RF Accuracy: {acc:.2f}")
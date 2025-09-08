import numpy as np
import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_cleaning.preprocess_blink_from_dataset import clean_blink_data

#pre-processed datasets(false data)
RAW_DATA_PATH = "C:/Users/mndiw/Documents/IRIS_ML/data/dataset.csv"
CLEAN_DATA_PATH = "C:/Users/mndiw/Documents/IRIS_ML/outputs/clean_blink.csv"

clean_blink_data(RAW_DATA_PATH, CLEAN_DATA_PATH)

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "C:/Users/mndiw/Documents/IRIS_ML/data/dataset.csv"
FEATURE_COLS = ["eye_openness_left_series", "eye_openness_right_series"]
LABEL_COL = "blink_type"

SAMPLING_RATE = 50  # Hz (match ESP32 later)
WINDOW_SECONDS = 3
STEP_SECONDS = 1.5

window_size = int(WINDOW_SECONDS * SAMPLING_RATE)
step = int(STEP_SECONDS * SAMPLING_RATE)

# -----------------------------
# Helpers
# -----------------------------

# Parse stringified lists or single values into arrays
def parse_series(x):
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype=float)
        except Exception:
            return np.array([np.nan])
    elif isinstance(x, (int, float, np.floating)):
        return np.array([x])
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=float)
    else:
        return np.array([np.nan])

# Always flatten arrays before computing stats
def safe_stats(series):
    arrs = [np.array(v, dtype=float).ravel() for v in series if v is not None]
    if len(arrs) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    arr = np.concatenate(arrs)
    return [arr.mean(), arr.std(), arr.min(), arr.max()]

# Feature extraction
def extract_features_photodiode(X, y, window_size, step):
    Xs, ys = [], []
    for i in range(0, len(X) - window_size, step):
        window = X.iloc[i:i+window_size]
        features = []
        for col in X.columns:
            features += safe_stats(window[col])
        Xs.append(features)
        ys.append(y.iloc[i+window_size-1])  # label at window end
    return np.array(Xs), np.array(ys)

# -----------------------------
# Main pipeline
# -----------------------------
# Load dataset
df = pd.read_csv(DATA_PATH)

# Apply parser to feature columns
for col in FEATURE_COLS:
    df[col] = df[col].apply(parse_series)

X = df[FEATURE_COLS]
y = df[LABEL_COL]

# Extract features
X_feat, y_feat = extract_features_photodiode(X, y, window_size, step)

if len(X_feat) == 0:
    raise ValueError(f"Dataset too small! Need at least {window_size} rows, got {len(X)}")

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
print(f"Photodiode RF Accuracy: {acc:.2f}")
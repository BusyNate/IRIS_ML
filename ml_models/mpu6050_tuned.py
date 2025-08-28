import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#load Ds
DATA_PATH = "C:/Users/mndiw/Documents/IRIS_ML/data/dataset.csv"
df = pd.read_csv(DATA_PATH)

#ignore unnecessary columns
id_cols = [col for col in df.columns if "participant" in col.lower() or "id" in col.lower()]
df = df.drop(columns=id_cols, errors="ignore")

#creaet feats and targets
target_col = "blink_type"
X = df.drop(columns=[target_col])
y = df[target_col]

#encodes any remains categorial x's/columns ensures all data is read through
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

#encoding y since it is categorical
if y.dtype == "object":
    le_y = LabelEncoder()
    y = le_y.fit_transform(y.astype(str))

#test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#trainign
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model trained successfully. Accuracy: {acc:.2f}")
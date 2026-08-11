import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("aug_train.csv")

print("Original Shape:", df.shape)
print(df.dtypes)

# Remove ID column
if "enrollee_id" in df.columns:
    df.drop("enrollee_id", axis=1, inplace=True)

# Fill Missing Values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode Categorical Columns
label_encoders = {}

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Save Model
joblib.dump(model, "job_selection_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model Saved Successfully!")


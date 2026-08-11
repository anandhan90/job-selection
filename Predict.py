import pandas as pd
import joblib

# Load Model
model = joblib.load("job_selection_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Load Test Data
df = pd.read_csv("test_data.csv")

# Remove ID
if "enrollee_id" in df.columns:
    df.drop("enrollee_id", axis=1, inplace=True)

# Fill Missing Values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode
for col in df.columns:
    if col in encoders:
        le = encoders[col]

        df[col] = df[col].map(
            lambda s: s if s in le.classes_ else le.classes_[0]
        )

        le.classes_ = list(le.classes_)
        df[col] = le.transform(df[col])

# Predict
prediction = model.predict(df)

# Output
df["Prediction"] = prediction

print(df[["Prediction"]])

df.to_csv("prediction_output.csv", index=False)

print("Prediction Completed")
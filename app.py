import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("job_selection_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset only to get column names
df = pd.read_csv("aug_train.csv")

if "enrollee_id" in df.columns:
    df = df.drop("enrollee_id", axis=1)

if "target" in df.columns:
    feature_columns = df.drop("target", axis=1).columns.tolist()
else:
    feature_columns = df.columns.tolist()


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Job Selection Prediction",
    page_icon="💼",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("💼 Job Selection Prediction")
st.write("Predict whether a candidate is likely to be selected for a job.")

st.divider()


# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    st.subheader("Candidate Details")

    input_data = {}

    # Create input fields based on dataset columns
    for col in feature_columns:

        if col in label_encoders:
            # Categorical column
            options = list(label_encoders[col].classes_)

            input_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                options
            )

        else:
            # Numeric column
            input_data[col] = st.number_input(
                col.replace("_", " ").title(),
                value=0.0
            )

    submitted = st.form_submit_button(
        "🔮 Predict Job Selection"
    )


# -----------------------------
# Prediction
# -----------------------------
if submitted:

    try:
        input_df = pd.DataFrame([input_data])

        # Encode categorical values
        for col, le in label_encoders.items():

            if col in input_df.columns:
                input_df[col] = le.transform(
                    input_df[col].astype(str)
                )

        # Make sure column order is same as training
        input_df = input_df[feature_columns]

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0]
            confidence = max(probability) * 100
        else:
            confidence = None

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Candidate is likely to be selected!")

        else:
            st.error("❌ Candidate is unlikely to be selected.")

        if confidence is not None:
            st.info(f"Prediction Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error("Prediction Error")
        st.write(str(e))
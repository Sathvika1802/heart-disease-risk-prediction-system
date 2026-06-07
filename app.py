import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/heart_disease_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
model_results = joblib.load("models/model_results.pkl")

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)


st.title("❤️ Heart Disease Risk Prediction System")

st.write(
    "This machine learning application predicts whether a patient may have a higher or lower risk of heart disease "
    "based on clinical health indicators."
)

st.warning(
    "Disclaimer: This tool is for educational and research purposes only and should not be used as a medical diagnosis system."
)


st.sidebar.header("About the Project")
st.sidebar.write(
    "This project uses a Logistic Regression model trained on a heart disease dataset. "
    "The model predicts heart disease risk using patient clinical features."
)

st.sidebar.header("Model Used")
st.sidebar.write("Final Model: **Logistic Regression**")
st.header("Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input("Age", placeholder="Enter age")
    sex = st.selectbox("Sex", ["Select", "Male", "Female"])

    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        ["Select", "Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"]
    )

    resting_blood_pressure = st.text_input(
        "Resting Blood Pressure",
        placeholder="Enter resting blood pressure"
    )

with col2:
    cholestoral = st.text_input(
        "Cholesterol",
        placeholder="Enter cholesterol level"
    )

    fasting_blood_sugar = st.selectbox(
        "Fasting Blood Sugar",
        ["Select", "Lower than 120 mg/ml", "Greater than 120 mg/ml"]
    )

    rest_ecg = st.selectbox(
        "Resting ECG",
        ["Select", "Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
    )

    Max_heart_rate = st.text_input(
        "Maximum Heart Rate",
        placeholder="Enter maximum heart rate"
    )

with col3:
    exercise_induced_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Select", "No", "Yes"]
    )

    oldpeak = st.text_input(
        "Oldpeak",
        placeholder="Enter oldpeak value"
    )

    slope = st.selectbox(
        "Slope",
        ["Select", "Upsloping", "Flat", "Downsloping"]
    )

    vessels_colored_by_flourosopy = st.selectbox(
        "Vessels Colored by Fluoroscopy",
        ["Select", "Zero", "One", "Two", "Three", "Four"]
    )

    thalassemia = st.selectbox(
        "Thalassemia",
        ["Select", "Normal", "Fixed Defect", "Reversable Defect", "No"]
    )

if st.button("Predict Heart Disease Risk"):
    missing_fields = (
        not age
        or not resting_blood_pressure
        or not cholestoral
        or not Max_heart_rate
        or not oldpeak
        or "Select" in [
            sex,
            chest_pain_type,
            fasting_blood_sugar,
            rest_ecg,
            exercise_induced_angina,
            slope,
            vessels_colored_by_flourosopy,
            thalassemia
        ]
    )

    if missing_fields:
        st.warning("Please fill in all patient details before prediction.")

    else:
        try:
            input_data = pd.DataFrame({
                "age": [int(age)],
                "sex": [sex],
                "chest_pain_type": [chest_pain_type],
                "resting_blood_pressure": [int(resting_blood_pressure)],
                "cholestoral": [int(cholestoral)],
                "fasting_blood_sugar": [fasting_blood_sugar],
                "rest_ecg": [rest_ecg],
                "Max_heart_rate": [int(Max_heart_rate)],
                "exercise_induced_angina": [exercise_induced_angina],
                "oldpeak": [float(oldpeak)],
                "slope": [slope],
                "vessels_colored_by_flourosopy": [vessels_colored_by_flourosopy],
                "thalassemia": [thalassemia]
            })

            # Keep same column order as training
            input_data = input_data[feature_names]

            st.subheader("Patient Input Summary")
            st.dataframe(input_data)

            prediction = model.predict(input_data)[0]
            prediction_probability = model.predict_proba(input_data)[0]

            no_disease_prob = prediction_probability[0] * 100
            disease_prob = prediction_probability[1] * 100

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Higher Risk of Heart Disease")
                st.write(f"Model confidence for heart disease risk: **{disease_prob:.2f}%**")
            else:
                st.success("Lower Risk / No Heart Disease")
                st.write(f"Model confidence for lower risk: **{no_disease_prob:.2f}%**")

            st.write("### Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Class": ["Lower Risk / No Heart Disease", "Higher Risk of Heart Disease"],
                "Probability (%)": [no_disease_prob, disease_prob]
            })

            st.dataframe(probability_df)
            st.bar_chart(probability_df.set_index("Class"))

        except ValueError:
            st.error(
                "Please enter valid numeric values for age, resting blood pressure, cholesterol, maximum heart rate, and oldpeak."
            )

st.header("Model Performance")

st.write(
    "The project compared Logistic Regression, Random Forest, and SVM models after removing duplicate records from the dataset."
)

st.dataframe(model_results)

st.bar_chart(model_results)
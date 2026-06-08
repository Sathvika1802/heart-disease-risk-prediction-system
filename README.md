# Heart Disease Risk Prediction System

This project is a machine learning-based web application that predicts whether a patient may have a lower or higher risk of heart disease based on clinical health indicators. The model was trained using patient health data, evaluated with multiple machine learning algorithms, and integrated into a Streamlit web application for interactive prediction.

## Features

* Enter patient clinical details through a web interface
* Predict whether the patient has lower or higher heart disease risk
* Display prediction confidence score
* Show prediction probability for both classes
* Compare Logistic Regression, Random Forest, and SVM models
* Remove duplicate records to reduce data leakage and improve evaluation reliability
* Includes a Streamlit web application for interactive prediction

## Dataset

The dataset used for this project was obtained from Kaggle and contains patient clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, oldpeak, slope, fluoroscopy-related vessel count, thalassemia, and a target label for heart disease risk.

Target classes:

* `0` = Lower risk / no heart disease
* `1` = Higher risk of heart disease

Duplicate records were removed before training to reduce data leakage and improve evaluation reliability.

## Model Details

* Final Model: Logistic Regression
* Compared Models: Logistic Regression, Random Forest, SVM
* Best Model Selected Using: F1-score
* Accuracy: 83.61%
* Precision: 84.85%
* Recall: 84.85%
* F1 Score: 84.85%
* Target Classes: Lower risk / no heart disease, Higher risk of heart disease

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Google Colab
* VS Code

## Project Files

* `Heart_Disease_Risk_Prediction.ipynb` - Google Colab notebook containing data loading, preprocessing, duplicate removal, model training, evaluation, feature importance, and model saving steps.
* `app.py` - Streamlit web application for entering patient details and predicting heart disease risk.
* `models/heart_disease_model.pkl` - Saved trained Logistic Regression model pipeline.
* `models/feature_names.pkl` - Saved feature names used during model training.
* `models/model_results.pkl` - Saved model comparison results.
* `requirements.txt` - Required Python packages.
* `README.md` - Project documentation.

## How to Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python3 -m streamlit run app.py
```

## Disclaimer

This project is for educational and portfolio purposes only. It is not intended for clinical diagnosis or medical decision-making.

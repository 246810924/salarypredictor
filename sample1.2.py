import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f7f9fc;
    }
    .stApp {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1d391kg h1 {
        color: #4CAF50;
    }
    .css-1d391kg h2 {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI setup
st.title("Employee Salary Predictor")
st.markdown("Upload your dataset and predict employee salaries using machine learning!")

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    try:
        # Load dataset
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview:")
        st.write(data.head())

        # Preprocessing
        st.write("### Preprocessing")

        # Handle missing values
        if st.checkbox("Fill missing values with median", value=True):
            data.fillna(data.median(numeric_only=True), inplace=True)

        # User selects target column
        target_column = st.selectbox("Select target column (salary):", options=data.columns)

        # Select feature columns
        feature_columns = st.multiselect(
            "Select feature columns for prediction:",
            options=[col for col in data.columns if col != target_column],
            default=[col for col in data.columns if col != target_column]
        )

        # Encode categorical columns dynamically during predictions
        categorical_columns = data[feature_columns].select_dtypes(include=['object']).columns
        label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

        # Splitting the data
        X = data[feature_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Training
        st.write("### Model Training")
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Save the trained model
        if st.button("Save Model"):
            import joblib
            joblib.dump(model, "salary_predictor_model.pkl")
            st.success("Model saved as 'salary_predictor_model.pkl'!")

        # Predict on user input
        st.write("### Make Predictions")
        user_input = {}
        for feature in feature_columns:
            if feature in categorical_columns:
                options = label_encoders[feature].inverse_transform(data[feature].unique())
                user_input[feature] = st.selectbox(f"Select value for {feature}:", options=options)
            else:
                user_input[feature] = st.number_input(f"Enter value for {feature}:", step=0.1)

        if st.button("Predict Salary"):
            input_df = pd.DataFrame([user_input])

            # Encode categorical features in user input
            for feature in categorical_columns:
                if feature in user_input:
                    input_df[feature] = label_encoders[feature].transform(input_df[feature])

            input_scaled = scaler.transform(input_df)
            salary_prediction = model.predict(input_scaled)
            st.write(f"Predicted Salary: {salary_prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class GMVPredictor:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.model = None

    def load_data(self, uploaded_file):
        """Load data from uploaded CSV file"""
        self.data = pd.read_csv(uploaded_file)
        return self.data

    def preprocess_data(self):
        """Comprehensive data preprocessing method"""
        # Handle missing values
        median_income = np.nanmedian(self.data['Monthly Income'])
        self.data["Monthly Income"].fillna(median_income, inplace=True)
        
        median_transaction = np.nanmedian(self.data['Time since last transaction'])
        self.data["Time since last transaction"].fillna(median_transaction, inplace=True)
        
        # Categorical encoding
        for col in ['Gender', 'Education Level', 'Marital Status']:
            dummy = pd.get_dummies(self.data[col], prefix=col)
            self.data = pd.concat([self.data, dummy], axis=1)
            del self.data[col]
        
        # Numerical feature engineering
        self.data['Date of Birth'] = pd.to_datetime(self.data['Date of Birth'])
        self.data['Age'] = (pd.DatetimeIndex(self.data['Date of Birth']).year - 2020)
        self.data['Age'].fillna(method='ffill', inplace=True)
        
        self.data['Current job'] = pd.to_datetime(self.data['Current job'])
        self.data['Current job'] = (pd.DatetimeIndex(self.data['Current job']).year - 2020)
        self.data['Current job'].fillna(method='ffill', inplace=True)
        
        # Normalization
        for col in ['Income', 'Credit score']:
            self.data[col] = preprocessing.normalize([self.data[col]])
        
        # Label encoding for remaining categorical columns
        le = LabelEncoder()
        categorical_cols = ['Gender', 'Education Level', 'Marital Status']
        for col in categorical_cols:
            self.data[col] = le.fit_transform(self.data[col])
        
        # Split features and target
        self.X = self.data.drop('Avg GMV', axis=1)
        self.y = self.data['Avg GMV']

    def train_model(self):
        """Train linear regression model"""
        x_train, x_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42
        )
        self.model = LinearRegression().fit(x_train, y_train)
        return x_test, y_test

    def predict(self, input_data):
        """Make predictions using trained model"""
        return self.model.predict(input_data)

def main():
    st.set_page_config(page_title="GMV Predictor", page_icon=":chart_with_upwards_trend:")
    st.title("Gross Merchandise Value (GMV) Prediction")

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a page", 
        ["Home", "Data Upload", "Data Exploration", "Model Training", "Prediction"]
    )

    predictor = GMVPredictor()

    if app_mode == "Home":
        st.write("""
        # Welcome to GMV Predictor
        This application helps you predict Gross Merchandise Value using machine learning.

        ### Steps:
        1. Upload your dataset
        2. Explore your data
        3. Train the model
        4. Make predictions
        """)

    elif app_mode == "Data Upload":
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = predictor.load_data(uploaded_file)
            st.dataframe(data.head())
            st.write(f"Dataset Shape: {data.shape}")

    elif app_mode == "Data Exploration":
        st.header("Data Exploration")
        if predictor.data is None:
            st.warning("Please upload a dataset first!")
        else:
            # Basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(predictor.data.describe())

            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(predictor.data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    elif app_mode == "Model Training":
        st.header("Model Training")
        if predictor.data is None:
            st.warning("Please upload a dataset first!")
        else:
            if st.button("Preprocess Data"):
                predictor.preprocess_data()
                st.success("Data Preprocessed Successfully!")

            if st.button("Train Model"):
                x_test, y_test = predictor.train_model()
                st.success("Model Trained Successfully!")
                
                # Model Performance
                st.subheader("Model Performance")
                predictions = predictor.predict(x_test)
                mse = np.mean((predictions - y_test)**2)
                st.metric("Mean Squared Error", f"{mse:.4f}")

    elif app_mode == "Prediction":
        st.header("Make Predictions")
        if predictor.model is None:
            st.warning("Please train the model first!")
        else:
            # Dynamic input fields based on features
            input_data = {}
            for col in predictor.X.columns:
                input_data[col] = st.number_input(f"Enter {col}", value=0.0)

            if st.button("Predict GMV"):
                input_df = pd.DataFrame([input_data])
                prediction = predictor.predict(input_df)
                st.success(f"Predicted GMV: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()

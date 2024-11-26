import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import io

class DatasetGenerator:
    def __init__(self):
        self.fake = Faker()
        np.random.seed(42)

    def generate_dataset(self, num_samples, income_mean, income_std, 
                          credit_score_mean, credit_score_std):
        """
        Generate a synthetic dataset with user-defined parameters
        """
        data = {
            'Monthly Income': np.random.normal(income_mean, income_std, num_samples),
            'Time since last transaction': np.random.uniform(1, 365, num_samples),
            'Gender': np.random.choice(['Male', 'Female'], num_samples),
            'Education Level': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_samples),
            'Marital Status': np.random.choice(['Single', 'Married', 'Divorced'], num_samples),
            'Date of Birth': [self.fake.date_of_birth(minimum_age=18, maximum_age=65).strftime('%Y-%m-%d') for _ in range(num_samples)],
            'Current job': [self.fake.date_between(start_date='-10y', end_date='today').strftime('%Y-%m-%d') for _ in range(num_samples)],
            'Income': np.random.uniform(20000, 200000, num_samples),
            'Credit score': np.random.normal(credit_score_mean, credit_score_std, num_samples)
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Generate target variable Avg GMV with some realistic correlation
        df['Avg GMV'] = (
            0.5 * df['Monthly Income'] + 
            0.3 * df['Income'] + 
            0.2 * df['Credit score'] + 
            np.random.normal(0, 1000, num_samples)
        )

        # Round Avg GMV to 2 decimal places
        df['Avg GMV'] = df['Avg GMV'].round(2)

        return df

def main():
    st.set_page_config(
        page_title="GMV Dataset Generator", 
        page_icon=":bar_chart:", 
        layout="wide"
    )

    st.title("🚀 Synthetic Dataset Generator for GMV Prediction")

    # Sidebar for configuration
    st.sidebar.header("Dataset Generation Parameters")
    
    # Number of samples
    num_samples = st.sidebar.slider(
        "Number of Samples", 
        min_value=100, 
        max_value=10000, 
        value=500, 
        step=100
    )

    # Income parameters
    st.sidebar.subheader("Monthly Income Distribution")
    income_mean = st.sidebar.number_input(
        "Mean Monthly Income ($)", 
        min_value=1000, 
        max_value=20000, 
        value=5000
    )
    income_std = st.sidebar.number_input(
        "Income Standard Deviation ($)", 
        min_value=100, 
        max_value=5000, 
        value=1500
    )

    # Credit Score parameters
    st.sidebar.subheader("Credit Score Distribution")
    credit_score_mean = st.sidebar.number_input(
        "Mean Credit Score", 
        min_value=300, 
        max_value=850, 
        value=680
    )
    credit_score_std = st.sidebar.number_input(
        "Credit Score Standard Deviation", 
        min_value=10, 
        max_value=200, 
        value=100
    )

    # Generate dataset button
    if st.sidebar.button("Generate Dataset"):
        # Instantiate generator
        generator = DatasetGenerator()
        
        # Generate dataset
        df = generator.generate_dataset(
            num_samples, 
            income_mean, 
            income_std, 
            credit_score_mean, 
            credit_score_std
        )

        # Display dataset details
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Dataset statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", len(df))
            st.metric("Mean Monthly Income", f"${df['Monthly Income'].mean():.2f}")
        
        with col2:
            st.metric("Mean Avg GMV", f"${df['Avg GMV'].mean():.2f}")
            st.metric("Mean Credit Score", f"{df['Credit score'].mean():.2f}")

        # Visualization
        st.subheader("Dataset Distributions")
        
        # Income Distribution
        col1, col2 = st.columns(2)
        with col1:
            st.write("Monthly Income Distribution")
            st.bar_chart(df['Monthly Income'])
        
        with col2:
            st.write("Average GMV Distribution")
            st.bar_chart(df['Avg GMV'])

        # Download options
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download CSV",
            data=buffer,
            file_name='gmv_synthetic_dataset.csv',
            mime='text/csv',
            key='download-csv'
        )

if __name__ == "__main__":
    main()

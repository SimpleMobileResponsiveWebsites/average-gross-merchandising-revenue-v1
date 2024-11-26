import pandas as pd
import numpy as np
from faker import Faker
import random

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()

# Generate sample data
def generate_gmv_dataset(num_samples=500):
    data = {
        'Monthly Income': np.random.normal(5000, 1500, num_samples),
        'Time since last transaction': np.random.uniform(1, 365, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Education Level': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], num_samples),
        'Marital Status': np.random.choice(['Single', 'Married', 'Divorced'], num_samples),
        'Date of Birth': [fake.date_of_birth(minimum_age=18, maximum_age=65).strftime('%Y-%m-%d') for _ in range(num_samples)],
        'Current job': [fake.date_between(start_date='-10y', end_date='today').strftime('%Y-%m-%d') for _ in range(num_samples)],
        'Income': np.random.uniform(20000, 200000, num_samples),
        'Credit score': np.random.normal(680, 100, num_samples)
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

# Generate and save dataset
dataset = generate_gmv_dataset()
dataset.to_csv('data_train.csv', index=False)

print(dataset.head())
print("\nDataset Shape:", dataset.shape)
print("\nColumns:", list(dataset.columns))

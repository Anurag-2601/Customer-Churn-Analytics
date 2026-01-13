"""
Generate sample customer churn dataset for analysis
"""
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate customer data
data = {
    'CustomerID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 70, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Tenure': np.random.randint(0, 72, n_samples),  # months with company
    'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
    'TotalCharges': None,  # Will calculate based on tenure and monthly charges
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
}

df = pd.DataFrame(data)

# Calculate TotalCharges based on Tenure and MonthlyCharges
df['TotalCharges'] = (df['Tenure'] * df['MonthlyCharges']).round(2)

# Generate Churn based on realistic factors
churn_probability = 0.2  # Base churn rate

# Factors that increase churn probability
churn_score = np.zeros(n_samples)
churn_score += (df['Contract'] == 'Month-to-month').values * 0.3
churn_score += (df['Tenure'] < 12).values * 0.25
churn_score += (df['MonthlyCharges'] > 80).values * 0.2
churn_score += (df['TechSupport'] == 'No').values * 0.15
churn_score += (df['OnlineSecurity'] == 'No').values * 0.1

# Normalize and add random noise
churn_score = churn_score / churn_score.max()
churn_score += np.random.normal(0, 0.1, n_samples)
churn_score = np.clip(churn_score, 0, 1)

# Generate Churn based on probability
df['Churn'] = (churn_score > np.percentile(churn_score, 73)).astype(int)
df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})

# Save to CSV
df.to_csv('../data/customer_churn_data.csv', index=False)

print(f"Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"Churn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())

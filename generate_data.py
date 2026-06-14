"""
Generate realistic synthetic Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def generate_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Telco customer churn dataset.
    
    Args:
        n_samples: Number of customer records to generate
        random_state: Seed for reproducibility
        
    Returns:
        pd.DataFrame: Synthetic churn dataset
    """
    np.random.seed(random_state)
    
    # Generate base features
    data = {
        'customer_id': [f'CUST_{str(i).zfill(5)}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
        'tenure': np.random.exponential(30, n_samples).astype(int) % 72,
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
        'internet_service': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples, p=[0.41, 0.34, 0.25]),
        'monthly_charges': np.random.uniform(20, 120, n_samples).round(2),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    }
    
    # Calculate total_charges based on tenure
    data['total_charges'] = (data['monthly_charges'] * data['tenure']).round(2)
    
    # Generate churn based on patterns
    churn_prob = np.zeros(n_samples)
    
    # Month-to-month contracts have higher churn
    churn_prob[data['contract_type'] == 'Month-to-month'] += 0.35
    churn_prob[data['contract_type'] == 'One year'] += 0.15
    
    # Longer tenure reduces churn
    tenure_factor = np.array(data['tenure']) / 72.0
    churn_prob -= tenure_factor * 0.30
    
    # Fiber optic internet increases churn (quality issues)
    churn_prob[data['internet_service'] == 'Fiber optic'] += 0.20
    
    # Senior citizens churn more
    churn_prob[data['senior_citizen'] == 1] += 0.10
    
    # Electronic check payment method increases churn
    churn_prob[data['payment_method'] == 'Electronic check'] += 0.15
    
    # Add randomness
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, n_samples), 0, 1)
    
    data['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame(data)
    
    # Introduce small amount of missing values
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'total_charges'] = np.nan
    
    return df


if __name__ == '__main__':
    # Generate and save dataset
    df = generate_churn_dataset(n_samples=5000)
    df.to_csv('data/churn_data.csv', index=False)
    print(f"✓ Dataset generated: {len(df)} records")
    print(df.head())
    print(f"\nChurn rate: {df['churn'].mean():.2%}")
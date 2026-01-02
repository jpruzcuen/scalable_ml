"""
Generate dummy prediction data for testing the dashboard
This simulates what the inference pipeline will produce
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating dummy prediction data...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate 200 random locations across Sweden
n_locations = 200

# Sweden bounds
lat_min, lat_max = 55.0, 69.0
lon_min, lon_max = 11.0, 24.0

# Swedish regions
regions = ['Stockholm', 'Västra Götaland', 'Skåne', 'Västerbotten', 
           'Norrbotten', 'Dalarna', 'Gävleborg', 'Jämtland']

# Generate dummy data
data = {
    'row_id': range(n_locations),
    'Lat': np.random.uniform(lat_min, lat_max, n_locations),
    'Lon': np.random.uniform(lon_min, lon_max, n_locations),
    'Month': [datetime(2026, 2, 1)] * n_locations,  # Next month prediction
    
    # Weather lag features
    'ssrd_lag1': np.random.uniform(5e6, 2e7, n_locations),
    'ssrd_lag2': np.random.uniform(5e6, 2e7, n_locations),
    'ssrd_lag3': np.random.uniform(5e6, 2e7, n_locations),
    'swvl1_lag1': np.random.uniform(0.1, 0.3, n_locations),
    'swvl1_lag2': np.random.uniform(0.1, 0.3, n_locations),
    'swvl1_lag3': np.random.uniform(0.1, 0.3, n_locations),
    'swvl2_lag1': np.random.uniform(0.1, 0.3, n_locations),
    'swvl2_lag2': np.random.uniform(0.1, 0.3, n_locations),
    'swvl2_lag3': np.random.uniform(0.1, 0.3, n_locations),
    't2m_lag1': np.random.uniform(270, 290, n_locations),
    't2m_lag2': np.random.uniform(270, 290, n_locations),
    't2m_lag3': np.random.uniform(270, 290, n_locations),
    'tp_lag1': np.random.uniform(0, 0.01, n_locations),
    'tp_lag2': np.random.uniform(0, 0.01, n_locations),
    'tp_lag3': np.random.uniform(0, 0.01, n_locations),
    
    # NDVI lag features
    'NDVI': np.random.uniform(0.3, 0.9, n_locations),
    'NDVI_lag1': np.random.uniform(0.3, 0.9, n_locations),
    'NDVI_lag2': np.random.uniform(0.3, 0.9, n_locations),
    
    # Prediction (0 or 1) - but we'll use probability for better visualization
    'Pressence_pred': np.random.randint(0, 2, n_locations),
    
    # Add outbreak probability (0-1) for better visualization
    'outbreak_probability': np.random.uniform(0, 1, n_locations),
    
    # Add kommun and lan for filtering
    'Kommun': [f'Kommun_{i}' for i in range(n_locations)],
    'Lan': np.random.choice(regions, n_locations)
}

df = pd.DataFrame(data)

# Make some locations high risk (for testing)
high_risk_mask = np.random.choice([True, False], size=n_locations, p=[0.2, 0.8])
df.loc[high_risk_mask, 'outbreak_probability'] = np.random.uniform(0.7, 1.0, high_risk_mask.sum())
df.loc[high_risk_mask, 'Pressence_pred'] = 1

# Save to CSV
import os
os.makedirs('./data/predictions', exist_ok=True)
df.to_csv('./data/predictions/dummy_predictions.csv', index=False)

print(f"✅ Generated {len(df)} dummy predictions")
print(f"   Saved to: ./data/predictions/dummy_predictions.csv")
print(f"\nColumns: {list(df.columns)}")
print(f"\nPrediction distribution:")
print(f"   Presence (1): {(df['Pressence_pred'] == 1).sum()}")
print(f"   Absence (0): {(df['Pressence_pred'] == 0).sum()}")
print(f"\nOutbreak probability:")
print(f"   Mean: {df['outbreak_probability'].mean():.2%}")
print(f"   High risk (>70%): {(df['outbreak_probability'] > 0.7).sum()}")

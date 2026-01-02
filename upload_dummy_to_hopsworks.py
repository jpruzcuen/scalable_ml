"""
Generate dummy merged features and upload to Hopsworks for testing
This lets you test the full pipeline without real data
"""
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

print("=" * 60)
print("Creating DUMMY Merged Features for Hopsworks Testing")
print("=" * 60)

# Generate dummy data
print("\n📊 Generating dummy data...")
np.random.seed(42)

n_locations = 200
n_months = 24  # 2 years of data

# Sweden bounds
lat_min, lat_max = 55.0, 69.0
lon_min, lon_max = 11.0, 24.0

# Swedish regions
regions = ['Stockholm', 'Västra Götaland', 'Skåne', 'Västerbotten', 
           'Norrbotten', 'Dalarna', 'Gävleborg', 'Jämtland']

# Create base locations
locations = pd.DataFrame({
    'row_id': range(n_locations),
    'Lat': np.random.uniform(lat_min, lat_max, n_locations),
    'Lon': np.random.uniform(lon_min, lon_max, n_locations),
    'Kommun': [f'Kommun_{i}' for i in range(n_locations)],
    'Lan': np.random.choice(regions, n_locations)
})

# Create time series for each location
all_data = []
for month_offset in range(n_months):
    month_date = datetime(2024, 1, 1) + timedelta(days=30*month_offset)
    
    for _, loc in locations.iterrows():
        record = {
            'row_id': loc['row_id'],
            'Lat': loc['Lat'],
            'Lon': loc['Lon'],
            'Kommun': loc['Kommun'],
            'Lan': loc['Lan'],
            'Month': month_date,
            'Date': month_date.strftime('%Y-%m-%d'),
            
            # Target variable
            'Pressence': np.random.choice([0, 1], p=[0.8, 0.2]),
            'Quantity': np.random.randint(0, 10),
            
            # Weather features (current + lags)
            't2m': np.random.uniform(270, 290),
            't2m_lag1': np.random.uniform(270, 290),
            't2m_lag2': np.random.uniform(270, 290),
            't2m_lag3': np.random.uniform(270, 290),
            
            'tp': np.random.uniform(0, 0.01),
            'tp_lag1': np.random.uniform(0, 0.01),
            'tp_lag2': np.random.uniform(0, 0.01),
            'tp_lag3': np.random.uniform(0, 0.01),
            
            'swvl1': np.random.uniform(0.1, 0.3),
            'swvl1_lag1': np.random.uniform(0.1, 0.3),
            'swvl1_lag2': np.random.uniform(0.1, 0.3),
            'swvl1_lag3': np.random.uniform(0.1, 0.3),
            
            'swvl2': np.random.uniform(0.1, 0.3),
            'swvl2_lag1': np.random.uniform(0.1, 0.3),
            'swvl2_lag2': np.random.uniform(0.1, 0.3),
            'swvl2_lag3': np.random.uniform(0.1, 0.3),
            
            'ssrd': np.random.uniform(5e6, 2e7),
            'ssrd_lag1': np.random.uniform(5e6, 2e7),
            'ssrd_lag2': np.random.uniform(5e6, 2e7),
            'ssrd_lag3': np.random.uniform(5e6, 2e7),
            
            # NDVI features
            'NDVI': np.random.uniform(0.3, 0.9),
            'NDVI_lag1': np.random.uniform(0.3, 0.9),
            'NDVI_lag2': np.random.uniform(0.3, 0.9),
            'NDVI_clim': np.random.uniform(0.5, 0.7),
            'NDVI_anom': np.random.uniform(-0.2, 0.2),
            'NDVI_anom_lag1': np.random.uniform(-0.2, 0.2),
            'NDVI_anom_lag2': np.random.uniform(-0.2, 0.2),
        }
        all_data.append(record)

dummy_df = pd.DataFrame(all_data)

print(f"✅ Generated {len(dummy_df)} dummy records")
print(f"   Locations: {n_locations}")
print(f"   Time points: {n_months} months")
print(f"   Features: {len(dummy_df.columns)} columns")
print(f"   Date range: {dummy_df['Month'].min()} to {dummy_df['Month'].max()}")

# Convert to lowercase (Hopsworks requirement)
print("\n🔤 Converting column names to lowercase...")
dummy_df.columns = dummy_df.columns.str.lower()

# Connect to Hopsworks
print("\n🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
fs = project.get_feature_store()
print(f"✅ Connected to project: {project.name}")

# Create feature group
print("\n📦 Creating dummy feature group...")
beetle_fg = fs.get_or_create_feature_group(
    name='beetle_features_dummy',
    description='DUMMY merged features for testing Hopsworks integration (not real data!)',
    version=1,
    primary_key=['row_id'],
    event_time='month',
    online_enabled=False
)
print("✅ Feature group ready")

# Upload
print("\n⬆️  Uploading dummy data to Hopsworks...")
beetle_fg.insert(dummy_df, write_options={"wait_for_job": True})

print(f"\n✅ SUCCESS! Uploaded {len(dummy_df)} dummy records to Hopsworks")
print(f"Feature group: beetle_features_dummy (version 1)")
print(f"Columns: {len(dummy_df.columns)} features")

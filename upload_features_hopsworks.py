import pandas as pd
import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 60)
print("Uploading Features to Hopsworks from CSV")
print("=" * 60)

# ========================================
# CONFIGURE THIS FILE PATH
# ========================================
csv = './data/features.csv'  # <-- CHANGE THIS TO OUR FILE PATH
# ========================================

# Load CSV
print(f"\n Loading CSV file: {csv}")
df = pd.read_csv(csv)

print(f" Loaded {len(df)} rows")
print(f" Columns: {len(df.columns)}")
print(f" Column names: {list(df.columns)[:10]}...")

# ensuring Month column is datetime
if 'Month' in df.columns:
    df['Month'] = pd.to_datetime(df['Month'])
    print(f"   Date range: {df['Month'].min()} to {df['Month'].max()}")

# Convert to lowercase (Hopsworks requirement)
print("\n Converting column names to lowercase...")
df.columns = df.columns.str.lower()
print(f"   Updated columns: {list(df.columns)[:10]}...")

# Connecting to Hopsworks
print("\n Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
fs = project.get_feature_store()
print(f"Connected to project: {project.name}")

# Creating feature group
print("\n Creating feature group...")
beetle_fg = fs.get_or_create_feature_group(
    name='beetle_features',
    description='Merged beetle observations, weather (ERA5), and NDVI features with lags for outbreak prediction',
    version=1,
    primary_key=['row_id'],
    event_time='month',
    online_enabled=False
)
print("Feature group ready")

# Upload to hopsworks
print("\n Uploading data to Hopsworks...")
beetle_fg.insert(df, write_options={"wait_for_job": True})

print(f"\n SUCCESS! Uploaded {len(df)} records to Hopsworks")
print(f"Feature group: beetle_features (version 1)")
print(f"Columns: {len(df.columns)} features")

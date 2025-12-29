"""
Script to upload beetle observation CSV to Hopsworks
Run instead of the full notebook pipeline
"""

import pandas as pd
import hopsworks
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv('HOPSWORKS_API_KEY')
if not api_key:
    print("ERROR: HOPSWORKS_API_KEY not found in .env file")
    print("Please create a .env file with: HOPSWORKS_API_KEY='your_key'")
    exit(1)

print(f" API Key loaded (starts with: {api_key[:10]}...)")

print("=" * 60)
print("Uploading Beetle Observations to Hopsworks")
print("=" * 60)

# Loading the CSV file
print("\n Loading CSV file...")
obs_df = pd.read_csv('./data/beetle/artportalen/artportalen_final.csv')

print(f" Loaded {len(obs_df)} observations")
print(f"Columns: {list(obs_df.columns)}")

# Adding Month column for event_time (if not already present)
if 'Month' not in obs_df.columns:
    print("\n Adding Month column...")
    obs_df['Month'] = pd.to_datetime(obs_df['Date']).dt.to_period("M").dt.to_timestamp()
else:
    obs_df['Month'] = pd.to_datetime(obs_df['Month'])

print(f"Date range: {obs_df['Month'].min()} to {obs_df['Month'].max()}")

# Convert all column names to lowercase (to match with hopsworks requirement)
print("\n Converting column names to lowercase...")
obs_df.columns = obs_df.columns.str.lower()
print(f"Columns: {list(obs_df.columns)}")

# Connect to Hopsworks
print("\n Connecting to Hopsworks...")
print("(This may take 30-60 seconds...)")
try:
    project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
    print(f" Connected to project: {project.name}")
except Exception as e:
    print(f" Error connecting: {e}")
    exit(1)

# Get feature store
fs = project.get_feature_store()
print(f" Connected to feature store")

# Create/get feature group
print("\n Creating feature group...")
beetle_obs_fg = fs.get_or_create_feature_group(
    name='beetle_observations',
    description='Bark beetle observations from Artportalen with presence/absence labels',
    version=1,
    primary_key=['row_id'],
    event_time='month',  # lowercase to match converted column names
    online_enabled=False
)
print(" Feature group ready")

# Inserting data
print("\n  Uploading data to Hopsworks...")
beetle_obs_fg.insert(obs_df, write_options={"wait_for_job": True})

print(f"\n SUCCESS! Uploaded {len(obs_df)} observations to Hopsworks")
print(f"Feature group: beetle_observations (version 1)")
print("\n" + "=" * 60)

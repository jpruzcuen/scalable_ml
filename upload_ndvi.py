import pandas as pd
import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

print("Uploading NDVI Features to Hopsworks...")

# Load CSV
ndvi_df = pd.read_csv('./data/ndvi/ndvi_final.csv')
ndvi_df['Month'] = pd.to_datetime(ndvi_df['Month'])

# Convert to lowercase
ndvi_df.columns = ndvi_df.columns.str.lower()

print(f"Loaded {len(ndvi_df)} NDVI records")

# Connect
project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
fs = project.get_feature_store()

# Create feature group
ndvi_fg = fs.get_or_create_feature_group(
    name='ndvi_features',
    description='HLS NDVI satellite data with anomalies and lags',
    version=1,
    primary_key=['row_id'],
    event_time='month',
    online_enabled=False
)

# Upload
ndvi_fg.insert(ndvi_df, write_options={"wait_for_job": True})
print(f"Uploaded {len(ndvi_df)} NDVI records!")

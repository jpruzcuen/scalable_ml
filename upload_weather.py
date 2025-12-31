import pandas as pd
import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

print("Uploading Weather Features to Hopsworks...")

# Load CSV
weather_df = pd.read_csv('./data/weather_monthly/weather_final.csv')
weather_df['Month'] = pd.to_datetime(weather_df['Month'])

# Convert to lowercase
weather_df.columns = weather_df.columns.str.lower()

print(f"Loaded {len(weather_df)} weather records")

# Connect
project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
fs = project.get_feature_store()

# Create feature group
weather_fg = fs.get_or_create_feature_group(
    name='weather_features',
    description='ERA5 monthly weather data with lag features',
    version=1,
    primary_key=['row_id'],
    event_time='month',
    online_enabled=False
)

# Upload
weather_fg.insert(weather_df, write_options={"wait_for_job": True})
print(f"Uploaded {len(weather_df)} weather records!")

import pandas as pd
import ee
from tqdm import tqdm

from download_utils import extract_monthly_ndvi

ee.Authenticate()
ee.Initialize()

save = False

obs = pd.read_csv('../data/beetle/artportalen/artportalen_final.csv')
obs["Month"] = pd.to_datetime(obs['Date']).dt.to_period("M").dt.to_timestamp()

obs.drop(columns=['Kommun', 'Lan','Quantity','Date','Pressence'], inplace=True)

print(f"Obs has {len(obs)} datapoints")

try:
    ndvi_features = pd.read_csv('../data/ndvi/ndvi_features.csv')
    ndvi_features["Month"] = pd.to_datetime(ndvi_features['Month'])

    ndvi_final = obs.merge(ndvi_features.drop(columns=["Lat", "Lon"]), on=["row_id", "Month"], how="left")
    ndvi_final.drop(columns=['Month_num'], inplace=True)

except:

    try:
        ndvi_df = pd.read_csv('../data/ndvi/ndvi_raw.csv')
        ndvi_df['Month'] = pd.to_datetime(ndvi_df['Month'])

    except:
        modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
        ndvi_df = extract_monthly_ndvi(modis, obs)

    if save: ndvi_df.to_csv('../data/ndvi/ndvi_raw.csv', index=False)

    ndvi_df["Month_num"] = ndvi_df["Month"].dt.month

    ndvi_climatology = (
        ndvi_df
        .groupby(["Lat", "Lon", "Month_num"], as_index=False)
        .agg(NDVI_clim=("NDVI", "mean"))
    )

    ndvi_features = ndvi_df.merge(ndvi_climatology, on=["Lat", "Lon", "Month_num"],how="left")
    ndvi_features["NDVI_anom"] = ndvi_features["NDVI"] - ndvi_features["NDVI_clim"]

    ndvi_features = ndvi_features.sort_values(["Lat", "Lon", "Month"])

    MAX_LAG = 2

    for lag in range(1, MAX_LAG + 1):
        ndvi_features[f"NDVI_lag{lag}"] = (
            ndvi_features
            .groupby(["Lat", "Lon"])["NDVI"]
            .shift(lag)
        )

        ndvi_features[f"NDVI_anom_lag{lag}"] = (
            ndvi_features
            .groupby(["Lat", "Lon"])["NDVI_anom"]
            .shift(lag)
        )

    if save: ndvi_features.to_csv('../data/ndvi/ndvi_features.csv', index=False)

    

ndvi_final
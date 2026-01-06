import pandas as pd
import xarray as xr
import numpy as np
import os
import zipfile
import cdsapi
import pandas as pd
import ee
from tqdm import tqdm


######################################## WEATHER ########################################

# Weather data is downloaded from ERA5 Reanalysis

CDS_URL = os.environ["CDS_API_URL"]
CDS_KEY = os.environ["CDS_API_KEY"]

def download_monthly_weather_from_obs(obs_df):
    '''
    obs_df: dataframe with observation data
    '''

    c = cdsapi.Client(url = CDS_URL, key=CDS_KEY)
    zip_path = "../data/weather_monthly/era5_sweden_monthly.zip"

    c.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "surface_solar_radiation_downwards"
            ],
            "year": [str(i) for i in obs_df['Date'].dt.year.unique()],
            "month": [
                "01","02","03","04","05","06",
                "07","08","09","10","11","12"
            ],
            "time": "00:00",
            "area": [
                69.5, 10.5, # Box for Sweden
                55.0, 24.5
            ],
            "format": "netcdf",
        },
        zip_path
    )

    with zipfile.ZipFile(zip_path, "r") as z:
        extracted_files = z.namelist()
        z.extractall('../data/weather_monthly/')

    original_path = os.path.join('../data/weather_monthly/', extracted_files[0])
    new_path = os.path.join('../data/weather_monthly/', 'weather.nc')
    os.rename(original_path, new_path)

    weather = xr.open_dataset(new_path, engine='netcdf4')

    return weather

def download_daily_weather(start_date, end_date):

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    date_range = pd.date_range(start, end)

    # Group dates by year-month and get only the days in the range
    year_month_days = {}
    for date in date_range:
        year_month = (date.year, date.month)
        if year_month not in year_month_days:
            year_month_days[year_month] = []
        year_month_days[year_month].append(date.day)

    c = cdsapi.Client(url = CDS_URL, key=CDS_KEY)
    
    i = 0
    # Make a separate request for each year-month combination
    for (year, month), days in year_month_days.items():
        zip_path = os.path.join('../data/weather_daily', f"era5_sweden_daily{i}.zip")
        c.retrieve(
            "reanalysis-era5-land",
            {
                "variable": [
                    "2m_temperature",
                    "total_precipitation",
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "surface_solar_radiation_downwards"
                ],
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in sorted(days)],
                "time": "12:00",
                "area": [
                    69.5, 10.5,
                    55.0, 24.5
                ],
                "format": "netcdf",
            },
            zip_path
        )
        i +=1
        print(f"\n Downloaded month {month} in {year} \n")


def combine_daily_weather():

    data_dir = '../data/weather_daily'
    datasets = []
    extract_dirs = []

    # Loop through all zip files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.zip'):
            zip_path = os.path.join(data_dir, filename)

            with zipfile.ZipFile(zip_path, 'r') as z:
                # Create a unique subdirectory for each zip file to avoid name conflicts
                zip_name = filename.replace('.zip', '')
                extract_dir = os.path.join(data_dir, zip_name)
                os.makedirs(extract_dir, exist_ok=True)
                extract_dirs.append(extract_dir)
                
                z.extractall(extract_dir)

                # Load all extracted .nc files from this subdirectory
                for nc_file in os.listdir(extract_dir):
                    if nc_file.endswith('.nc') or nc_file.endswith(' 2') or nc_file.endswith(' 3'):
                        nc_path = os.path.join(extract_dir, nc_file)
                        ds = xr.open_dataset(nc_path, engine='netcdf4')

                        datasets.append(ds)

    # Concatenate along time dimension
    weather = xr.concat(datasets, dim='valid_time')

    # Delete all subdirectories
    import shutil
    for extract_dir in extract_dirs:
        shutil.rmtree(extract_dir)

    return weather

def sample_era5_to_points(obs_df: pd.DataFrame, ds: xr.Dataset, 
        lat_col="Lat", lon_col="Lon", date_col="Date", method="nearest", max_lag=3):
    
    """
    Sample ERA5-Land monthly variables at observation points.
    Inputs:
    -------
    obs_df: dataframe with observation data
    ds: xarray dataset with weather data

    Ouputs:
    -------
    Returns a new df with climate variables appended to the coordinate-time datapoints
    """

    df = obs_df.copy()
    df["Month_number"] = df[date_col].dt.month

    # Get weather variables
    weather_vars = [v for v in ds.data_vars]
    
    # Create a multi-index for all observations x all lags
    n_obs = len(df)
    n_lags = max_lag + 1
    
    # Repeat lat/lon for each lag
    lats = np.repeat(df[lat_col].values, n_lags)
    lons = np.repeat(df[lon_col].values, n_lags)
    
    # Create all lagged times at once
    times = []
    for lag in range(n_lags):
        lagged_month = df["Month"] - pd.DateOffset(months=lag)
        times.append(lagged_month.values)
    times = np.concatenate(times)
    
    # Single selection for all points and all times
    xr_points = xr.Dataset({
        "latitude": ("points", lats),
        "longitude": ("points", lons),
        "valid_time": ("points", times),
    })
    
    sampled = ds.sel(
        latitude=xr_points["latitude"],
        longitude=xr_points["longitude"],
        valid_time=xr_points["valid_time"],
        method=method
    )
    
    # Convert to DataFrame
    weather_df = sampled.to_dataframe().reset_index()
    weather_df = weather_df[weather_vars]
    
    # Reshape: split by lag and rename columns
    all_results = []
    for lag in range(n_lags):
        start_idx = lag * n_obs
        end_idx = (lag + 1) * n_obs
        lag_df = weather_df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        if lag > 0:
            lag_df = lag_df.rename(columns={v: f"{v}_lag{lag}" for v in weather_vars})
        
        all_results.append(lag_df)
    
    # Concatenate horizontally
    combined_weather = pd.concat(all_results, axis=1)
    
    # Join back to original df
    df = pd.concat([df[['Lat','Lon','Month','row_id']].reset_index(drop=True), combined_weather], axis=1)

    return df



######################################## NDVI ########################################

# NDVI data is downloaded from Google's Earth Engine


def df_to_ee_points(df):
    """
    Convert df rows to EE points (Earth Engine format) and add metadata (row id and date of the month)
    """
    features = []

    for _, row in df.iterrows():  # Changed from 'obs' to 'df'
        features.append(
            ee.Feature(
                ee.Geometry.Point(row["Lon"], row["Lat"]),
                {
                    "row_id": int(row["row_id"]),
                    #"month": row["Month"].strftime("%Y-%m-01"),
                    "Lat": float(row["Lat"]),
                    "Lon": float(row["Lon"]),
                }
            )
        )

    return ee.FeatureCollection(features)

def era5_grid_to_points_df(ds):
    """
    Convert era5 grid to df
    """
    lats = ds.latitude.values
    lons = ds.longitude.values

    rows = []
    idx = 0
    for lat in lats:
        for lon in lons:
            rows.append({
                "row_id": idx,
                "Lat": float(lat),
                "Lon": float(lon),
            })
            idx += 1

    return pd.DataFrame(rows)


def monthly_ndvi_image(dataset, year, month, region=None):
    """
    Select satellite image from modis 
    """
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    monthly = dataset.filterDate(start, end)

    results = ee.Image(
            ee.Algorithms.If(
                monthly.size().gt(0),
                monthly.mean().multiply(0.0001).rename("NDVI"),
                ee.Image().rename("NDVI")
            )
        ).set("year", year).set("month", month)

    if region is not None:
        results = results.clip(region)

    return results

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def download_monthly_ndvi_from_obs(dataset, points_df, months, chunk_size=500):
    """

    """
    #TODO : Update to older version
    results = []
    
    for m in tqdm(months, desc="Months"):
        img = monthly_ndvi_image(dataset, m.year, m.month)
        
        # Track which row_ids we received data for this month
        received_row_ids = set()
        
        chunks = list(chunk_list(points_df.index.tolist(), chunk_size))
        
        for chunk in chunks:
            chunk_df = points_df.loc[chunk]
            points_fc = df_to_ee_points(chunk_df)

            sampled = img.sampleRegions(
                collection=points_fc,
                scale=250,
                geometries=False
            )

            info = sampled.getInfo()

            for f in info["features"]:
                p = f["properties"]
                row_id = p["row_id"]
                received_row_ids.add(row_id)

                results.append({
                    "row_id": row_id,
                    "Month": m,
                    "NDVI": p.get("NDVI"),
                    "Lat": p["Lat"],
                    "Lon": p["Lon"],
                })
        
        # Add missing row_ids with None NDVI values
        for _, row in points_df.iterrows():
            row_id = int(row["row_id"])
            if row_id not in received_row_ids:
                results.append({
                    "row_id": row_id,
                    "Month": m,
                    "NDVI": None,
                    "Lat": row["Lat"],
                    "Lon": row["Lon"],
                })

    return pd.DataFrame(results)

def download_monthly_ndvi(dataset, points_df, months, chunk_size=500):
    """
    Download the NDVI values for a given month list across coordinates.
    
    For each month, samples all coordinates in points_df against that month's image.
    Handles missing data from Earth Engine (null/masked pixels).
    """
    results = []
    
    for m in tqdm(months, desc="Months"):
        img = monthly_ndvi_image(dataset, m.year, m.month)
        
        # Track which row_ids we received data for this month
        received_row_ids = set()
        
        chunks = list(chunk_list(points_df.index.tolist(), chunk_size))
        
        for chunk in chunks:
            chunk_df = points_df.loc[chunk]
            points_fc = df_to_ee_points(chunk_df)

            sampled = img.sampleRegions(
                collection=points_fc,
                scale=250,
                geometries=False
            )

            info = sampled.getInfo()

            for f in info["features"]:
                p = f["properties"]
                row_id = p["row_id"]
                received_row_ids.add(row_id)

                results.append({
                    "row_id": row_id,
                    "Month": m,
                    "NDVI": p.get("NDVI"),
                    "Lat": p["Lat"],
                    "Lon": p["Lon"],
                })
        
        # Add missing row_ids with None NDVI values
        for _, row in points_df.iterrows():
            row_id = int(row["row_id"])
            if row_id not in received_row_ids:
                results.append({
                    "row_id": row_id,
                    "Month": m,
                    "NDVI": None,
                    "Lat": row["Lat"],
                    "Lon": row["Lon"],
                })

    return pd.DataFrame(results)

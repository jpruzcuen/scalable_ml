import pandas as pd
from download_utils import download_daily_weather, combine_daily_weather
import xarray as xr
import geopandas
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

def download_weather_for_inference():
    save = True

    end = pd.Timestamp.today()
    start = end.replace(day=1) - relativedelta(months=2) ## -2 months to be able to create lagged features

    download_daily_weather(start, end)

    weather_daily = combine_daily_weather()

    weather_agg = weather_daily.resample(valid_time='MS').map(
        lambda x: xr.Dataset({
            "t2m": x["t2m"].mean(dim="valid_time"),
            "tp": x["tp"].sum(dim="valid_time"),
            "swvl1": x["swvl1"].mean(dim="valid_time"),
            "swvl2": x["swvl2"].mean(dim="valid_time"),
            "ssrd": x["ssrd"].sum(dim="valid_time"),
        })
    )
    weather_agg


    # Convert to dataframe for easier manipulation
    weather_features = weather_agg.to_dataframe().reset_index()
    weather_features = weather_features.sort_values(['latitude', 'longitude', 'valid_time']).reset_index(drop=True)

    # Create lagged features for each lat/lon combination
    max_lag = 2
    weather_vars = ['t2m', 'tp', 'swvl1', 'swvl2', 'ssrd']

    # Group by location and create lags
    lagged_data = []
    for (lat, lon), group in weather_features.groupby(['latitude', 'longitude']):
        group = group.sort_values('valid_time').reset_index(drop=True)
        
        # Lagged features
        for lag in range(1, max_lag + 1):
            for var in weather_vars:
                group[f'{var}_lag{lag+1}'] = group[var].shift(lag)
        
        lagged_data.append(group)

    weather_final = pd.concat(lagged_data, ignore_index=True)
    weather_final.dropna(inplace=True)
    weather_final.drop(columns=['number'], inplace=True)

    d = {}
    for v in weather_vars:
        d[v] = v+'_lag1'
    d['latitude'] = 'Lat'
    d['longitude'] = 'Lon'

    weather_final.rename(columns=d, inplace=True)
    weather_final = weather_final[sorted(list(weather_final.columns))]

    if save: weather_final.to_csv('../data/weather_daily/weather_' + str(end.month) + '_' + str(end.year) + '_.csv', index=False)

    return weather_final
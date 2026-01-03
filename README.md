# Bark beetle outbreak prediction system

A machine learning system for predicting spruce bark beetle outbreaks in Sweden using weather data (ERA5), satellite imagery (NDVI), and species observations from Artportalen.

## Overview

This system monitors and predicts bark beetle outbreak likelihood across Sweden on a monthly basis using:
- **Weather Data**: ERA5 monthly averages (temperature, precipitation, soil moisture, solar radiation)
- **Satellite Data**: HLS NDVI vegetation indices
- **Observations**: Beetle presence/absence data from Artportalen.se

## Prerequisites 
- Python 3.11 or 3.12
- Conda
- Hopsworks account


### Configure Hopsworks

Create a `.env` file in the project root:

```bash
HOPSWORKS_API_KEY='your_hopsworks_api_key_here'
```

## Running the Application
```bash
streamlit run main.py
```

## Features

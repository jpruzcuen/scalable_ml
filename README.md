# Bark beetle outbreak prediction system

A machine learning system for predicting spruce bark beetle (_Ips typographus_) outbreaks in Sweden. These beetles are a well known pest destroying a lot of forest area in Europe every year. They are actually the #1 cause of forest damage in Sweden every year. The prediction model uses weather data, NDVI values from satellite imagery and species observations from Artportalen.

## Overview

This system monitors and predicts bark beetle outbreak likelihood across Sweden on a monthly basis using:
- **Weather Data**: temperature, precipitation, soil moisture and solar radiation from ERA5 Land (Copernicus datasets). Monthly averages are estimated with daily measurements.
- **Satellite Data**: NDVI vegetation indices from Google's Earth Engine. Values are estimations over 16 day periods.

## Model Training

For training the binary classification model, observational data from Artportalen was used. These are human-made recordings of dates and coordinates where spruce bark beetles were observed in Sweden between 2018 and 2025. These serve as datapoints with positive label class. To train a model with supervised training we required datapoints with the negative class. For pressence-only datasets like this, the way to generate a "background" signal is to find observational data of another species B that is similar to the one being studied A. This species should obviously live in the same conditions, such that it would be likely that studies that observed B could've likely observed A too. For this we used observations of björksplintborre (_Scolytus ratzeburgii_) in Sweden. 

The model was trained to predict the likelyhood of an outbreak next month (t+1) based on the data of this month (t) and the last two moths (t-1 and t-2). For NDVI data in particular, only only {t, t-1} months of lagged features were used.


## Model Inference

As mentioned there are 2 data sources: weather and NDVI. The weather data is downloaded every day. Then it's sent to the feature pipeline to create lagged features and finally save this data in Hopsworks. NDVI data is downlaoded every 20 days because the composites are updated every 16 days (approximately). 

The inference pipeline is run every day and consists of 2 steps: download data from Hopsworks and generate predictions (inference).

## Application

The UI was built using the _streamlit_ library.

## Technologies Used

1. Hopsworks is used as the cloud platform for data and model storage.
2. Github Actions is used to automatically run the inference pipeline.
3. Copernicus datasets are used to download weather data. 
4. Earth Engine is used to download NDVI data. 


## To run the project locally

### Configure Hopsworks
Create a `.env` file in the project root:

```bash
HOPSWORKS_PROJECT='your_hopsworks_project_here'
HOPSWORKS_API_KEY='your_hopsworks_api_key_here'
```
### APIs
In addition to Hopsworks, you will need to create API keys for Copernicus and Google's Earth Engine.


### Running the Application
```bash
streamlit run main.py
```

### Prerequisites 
- Python 3.11 or 3.12
- Conda
- Hopsworks account
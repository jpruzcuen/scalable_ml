# scalable_ml

# Bark beetle outbreak prediction system

A machine learning system for predicting spruce bark beetle outbreaks in Sweden using weather data (ERA5), satellite imagery (NDVI), and species observations from Artportalen.

## Prerequisites 
- `uv` package installer

## Environment Setup
```bash
source .venv/bin/activate
uv sync
```

### Configure Hopsworks

Create a `.env` file in the project root:

```bash
HOPSWORKS_API_KEY='your_hopsworks_api_key_here'
```

## Running the Application
```bash
streamlit run main.py
```

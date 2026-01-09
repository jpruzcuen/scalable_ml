from setuptools import setup, find_packages

setup(
    name="scalable-ml",
    version="0.1.0",
    description="Add your description here",
    readme="README.md",
    python_requires=">=3.12",
    install_requires=[
        "cdsapi>=0.7.0",
        "earthengine-api>=0.1.0",
        "hopsworks>=3.7.0,<4.0.0",
        "jupyter>=1.1.1",
        "matplotlib>=3.10.8",
        "numpy>=1.26.0,<2.0.0",
        "openpyxl>=3.1.5",
        "pandas>=2.1.0,<2.2.0",
        "plotly>=6.5.0",
        "pyproj>=3.7.2",
        "streamlit>=1.52.2",
        "tqdm>=4.67.1",
        "xarray>=2024.0.0,<2025.0.0",
        "geopandas>=1.0.0",
        "netcdf4>=1.6.0",
        "xgboost>=2.0.0",
        "pyjks<20.0.0",
    ],
)

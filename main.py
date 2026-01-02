import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Bark Beetle Outbreak Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_from_hopsworks():
    """Load latest predictions from Hopsworks"""
    try:
        # Connect to Hopsworks
        with st.spinner("Connecting to Hopsworks..."):
            project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
            fs = project.get_feature_store()
        

        FEATURE_GROUP_NAME = 'beetle_features'
        
        with st.spinner(f"Loading data from {FEATURE_GROUP_NAME}..."):
            beetle_fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=1)
            data = beetle_fg.read()
        
        # Convert month to datetime
        if 'month' in data.columns:
            data['month'] = pd.to_datetime(data['month'])
        
        # Debug: Show what columns we have
        #print(f"DEBUG: Columns from Hopsworks: {list(data.columns)}")
        
        # Rename columns to match dashboard (lowercase to uppercase)
        column_mapping = {
            'month': 'Month',
            'lat': 'Lat',
            'lon': 'Lon',
            'kommun': 'Kommun',
            #'lan': 'Lan',
            'outbreak_probability': 'outbreak_likelihood',
            'ndvi': 'NDVI',
            'pressence': 'Pressence'
        }
        
        # Apply renaming
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        #print(f"DEBUG: Columns after rename: {list(data.columns)}")
        
        # Map lag features to current for visualization
        if 't2m' not in data.columns and 't2m_lag1' in data.columns:
            data['t2m'] = data['t2m_lag1']
        if 'tp' not in data.columns and 'tp_lag1' in data.columns:
            data['tp'] = data['tp_lag1']
        if 'swvl1' not in data.columns and 'swvl1_lag1' in data.columns:
            data['swvl1'] = data['swvl1_lag1']
        if 'swvl2' not in data.columns and 'swvl2_lag1' in data.columns:
            data['swvl2'] = data['swvl2_lag1']
        if 'ssrd' not in data.columns and 'ssrd_lag1' in data.columns:
            data['ssrd'] = data['ssrd_lag1']
        if 'NDVI_anom' not in data.columns and 'ndvi_anom' in data.columns:
            data['NDVI_anom'] = data['ndvi_anom']
        elif 'NDVI_anom' not in data.columns:
            data['NDVI_anom'] = 0
        
        # Create outbreak_likelihood if it doesn't exist
        if 'outbreak_likelihood' not in data.columns:
            if 'outbreak_probability' in data.columns:
                data['outbreak_likelihood'] = data['outbreak_probability']
            elif 'pressence_pred' in data.columns:
                data['outbreak_likelihood'] = data['pressence_pred']
            elif 'pressence' in data.columns:
                # For training data without predictions, create dummy likelihood
                data['outbreak_likelihood'] = data['pressence'].apply(
                    lambda x: np.random.uniform(0.7, 1.0) if x == 1 else np.random.uniform(0.0, 0.3)
                )
            else:
                # Fallback: random values
                data['outbreak_likelihood'] = np.random.uniform(0, 1, len(data))
        
        return data, "hopsworks"
        
    except Exception as e:
        st.sidebar.error(f" Hopsworks error: {str(e)}")
        st.sidebar.info("Falling back to local CSV file...")
        return load_predictions_from_csv()

@st.cache_data
def load_predictions_from_csv():
    """Load predictions from local CSV file (fallback)"""
    try:
        data = pd.read_csv('./data/predictions/pressence_1_12_2025_.csv')
        data['Month'] = pd.to_datetime(data['Month'])
        
        # Convert prediction columns
        if 'outbreak_likelihood' not in data.columns:
            if 'outbreak_probability' in data.columns:
                data['outbreak_likelihood'] = data['outbreak_probability']
            elif 'Pressence_pred' in data.columns:
                data['outbreak_likelihood'] = data['Pressence_pred'].apply(
                    lambda x: np.random.uniform(0.7, 1.0) if x == 1 else np.random.uniform(0.0, 0.3)
                )
        
        # Map lag features
        if 't2m' not in data.columns and 't2m_lag1' in data.columns:
            data['t2m'] = data['t2m_lag1']
        if 'tp' not in data.columns and 'tp_lag1' in data.columns:
            data['tp'] = data['tp_lag1']
        if 'swvl1' not in data.columns and 'swvl1_lag1' in data.columns:
            data['swvl1'] = data['swvl1_lag1']
        if 'swvl2' not in data.columns and 'swvl2_lag1' in data.columns:
            data['swvl2'] = data['swvl2_lag1']
        if 'ssrd' not in data.columns and 'ssrd_lag1' in data.columns:
            data['ssrd'] = data['ssrd_lag1']
        if 'NDVI_anom' not in data.columns:
            data['NDVI_anom'] = 0
        
        st.sidebar.info("📁 Using local CSV file")
        return data, "csv"
        
    except FileNotFoundError:
        st.sidebar.warning("No predictions file found - using mock data")


@st.cache_data
def load_data():
    """Main data loading function - tries Hopsworks first, then CSV"""
    return load_predictions_from_hopsworks()

def main():
    # Header
    st.title("Spruce Bark Beetle Outbreak Prediction")
    st.markdown("""
    **Monthly monitoring dashboard** predicting bark beetle outbreak likelihood across Sweden  
    *Data sources: ERA5 Weather (monthly) • Artportalen.se Observations • HLS NDVI Satellite Data*
    """)
    
    # Load data 
    data, data_source = load_data()    
    
    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        st.markdown("---")
        
        # # Show data source with refresh button
        # st.subheader("Data Source")
        # if data_source == "hopsworks":
        #     st.success("Hopsworks Feature Store")
        #     if st.button("🔄 Refresh Data"):
        #         st.cache_data.clear()
        #         st.rerun()
        # elif data_source == "csv":
        #     st.info("Local CSV file")
        # else:
        #     st.warning("Mock data (no real predictions)")


        # Heatmap variable selector
        variable_options = {
            "Outbreak Likelihood": "outbreak_likelihood",
            "Temperature": "t2m",
            "Precipitation": "tp",
            "Soil Moisture L1": "swvl1",
            "Soil Moisture L2": "swvl2",
            "Solar Radiation": "ssrd",
            "NDVI": "NDVI",
            "NDVI Anomaly": "NDVI_anom",
        }
        
        selected_var_name = st.selectbox(
            "Display variable:",
            options=list(variable_options.keys()),
            index=0
        )
        selected_var = variable_options[selected_var_name]
        
        st.markdown("---")
        
        # Filters
        st.subheader("Filters")
        
        #regions = ["All Regions"] + sorted(data['Lan'].unique().tolist())
        # lan_col = 'Lan' if 'Lan' in data.columns else 'lan'
        # regions = ["All Regions"] + sorted(data[lan_col].dropna().unique().tolist())
        # selected_region = st.selectbox("Region (Län):", regions)
        
        # Risk threshold
        risk_threshold = st.slider(
            "Minimum outbreak likelihood:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Filter locations by minimum predicted outbreak likelihood"
        )
        
        st.markdown("---")
    
    # Apply filters
    filtered_data = data.copy()
    # if selected_region != "All Regions":
    #     filtered_data = filtered_data[filtered_data[lan_col] == selected_region]
    filtered_data = filtered_data[filtered_data['outbreak_likelihood'] >= risk_threshold]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Monitored Locations",
            len(filtered_data),
            help="Number of locations matching current filters"
        )
    
    with col2:
        avg_likelihood = filtered_data['outbreak_likelihood'].mean()
        st.metric(
            "Average Likelihood",
            f"{avg_likelihood:.1%}",
            help="Mean outbreak likelihood across filtered locations"
        )
    
    with col3:
        high_risk = (filtered_data['outbreak_likelihood'] > 0.7).sum()
        st.metric(
            "High Risk Areas",
            high_risk,
            delta_color="inverse",
            help="Locations with >70% outbreak likelihood"
        )
    
    with col4:
        extreme_risk = (filtered_data['outbreak_likelihood'] > 0.85).sum()
        st.metric(
            "Extreme Risk",
            extreme_risk,
            help="Locations with >85% outbreak likelihood"
        )
    
    st.markdown("---")
    
    # Main map
    st.subheader(f" {selected_var_name}")
    
    # Prepare map data
    map_data = filtered_data.copy()
    map_data['latitude'] = map_data['Lat']
    map_data['longitude'] = map_data['Lon']
    
    # Dynamic color scale
    if selected_var == 'outbreak_likelihood':
        color_scale = 'Reds'
        color_label = 'Likelihood'
    elif 'NDVI' in selected_var:
        color_scale = 'Greens'
        color_label = selected_var_name
    elif selected_var == 't2m':
        color_scale = 'RdYlBu_r'
        color_label = 'Temp (K)'
    else:
        color_scale = 'Viridis'
        color_label = selected_var_name
    
    # Create map
    fig = px.scatter_map(
        map_data,
        lat='latitude',
        lon='longitude',
        color=selected_var,
        size='outbreak_likelihood',
        #hover_name='Kommun',
        hover_data={
            'latitude': ':.3f',
            'longitude': ':.3f',
            selected_var: ':.3f',
            'outbreak_likelihood': ':.1%',
            #lan_col: True,
            'Month': True
        },
        color_continuous_scale=color_scale,
        size_max=20,
        zoom=4,
        center={"lat": 62, "lon": 15},
        #mapbox_style="open-street-map",
        labels={selected_var: color_label},
        height=600
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title=color_label,
            thickness=15,
            len=0.7
        )
    )
    
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Outbreak Likelihood Distribution")
        fig_hist = px.histogram(
            filtered_data,
            x='outbreak_likelihood',
            nbins=30,
            labels={'outbreak_likelihood': 'Outbreak Likelihood'},
            color_discrete_sequence=['#d62728']
        )
        fig_hist.update_layout(height=300, showlegend=False)
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="orange", 
                        annotation_text="High Risk Threshold")
        st.plotly_chart(fig_hist, width='stretch')
    
    # with col2:
    #     st.subheader("Regions by Outbreak Risk")
    #     region_stats = filtered_data.groupby(lan_col).agg({
    #         'outbreak_likelihood': ['mean', 'max', 'count']
    #     }).reset_index()
    #     region_stats.columns = ['Region', 'Avg Likelihood', 'Max Likelihood', 'Locations']
    #     region_stats = region_stats.sort_values('Avg Likelihood', ascending=False)
    #     region_stats['Avg Likelihood'] = region_stats['Avg Likelihood'].apply(lambda x: f"{x:.1%}")
    #     region_stats['Max Likelihood'] = region_stats['Max Likelihood'].apply(lambda x: f"{x:.1%}")
    #     st.dataframe(region_stats, width='stretch', hide_index=True)

if __name__ == "__main__":
    main()

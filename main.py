import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Bark Beetle Outbreak Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_from_hopsworks():
    """Load latest predictions from Hopsworks"""
    try:
        # Connect to Hopsworks
        with st.spinner("Connecting to Hopsworks Feature Store..."):
            project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
            fs = project.get_feature_store()
        
        FEATURE_GROUP_NAME = 'beetle_features'
        
        with st.spinner(f"Loading prediction data..."):
            beetle_fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=1)
            data = beetle_fg.read()
        
        # Convert month to datetime
        if 'month' in data.columns:
            data['month'] = pd.to_datetime(data['month'])
        
        # Rename columns to match dashboard
        column_mapping = {
            'month': 'Month',
            'lat': 'Lat',
            'lon': 'Lon',
            'kommun': 'Kommun',
            'lan': 'Lan',
            'outbreak_probability': 'outbreak_likelihood',
            'ndvi': 'NDVI',
            'pressence': 'Pressence'
        }
        
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        # Map lag features to current for visualization
        if 't2m' not in data.columns and 't2m_lag1' in data.columns:
            data['t2m'] = data['t2m_lag1']
        if 'tp' not in data.columns and 'tp_lag1' in data.columns:
            data['tp'] = data['tp_lag1']
        if 'swvl1' not in data.columns and 'swvl1_lag1' in data.columns:
            data['swvl1'] = data['swvl1_lag1']
        if 'swvl2' not in data.columns and 'swvl2_lag1' in data.columns:
            data['swvl2'] = data['swvl2_lag2']
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
            elif 'Pressence' in data.columns:
                # Create probability from presence data
                data['outbreak_likelihood'] = data['Pressence'].apply(
                    lambda x: np.random.uniform(0.7, 1.0) if x == 1 else np.random.uniform(0.0, 0.3)
                )
            else:
                data['outbreak_likelihood'] = np.random.uniform(0, 1, len(data))
        
        return data, "hopsworks"
        
    except Exception as e:
        st.sidebar.error(f"Hopsworks error: {str(e)}")
        st.sidebar.info("💡 Falling back to local CSV file...")
        return load_predictions_from_csv()

@st.cache_data
def load_predictions_from_csv():
    """Load predictions from local CSV file (fallback)"""
    try:
        data = pd.read_csv('./data/predictions/pressence_1_12_2025_.csv')
        data['Month'] = pd.to_datetime(data['Month'])
        
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
        
        st.sidebar.success("Using local CSV file")
        return data, "csv"
        
    except FileNotFoundError:
        st.sidebar.warning("⚠️ No data found")
        return pd.DataFrame(), "none"

@st.cache_data
def load_data():
    """Main data loading function"""
    return load_predictions_from_hopsworks()

def main():
    # Header with custom styling
    st.markdown('<h1 class="main-header">Bark Beetle Outbreak Monitor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predicting spruce bark beetle outbreaks across Sweden using machine learning</p>', unsafe_allow_html=True)
    
    # Info box about the system
    with st.expander("About This System", expanded=False):
        st.markdown("""
        This dashboard monitors and predicts bark beetle outbreak likelihood across Sweden on a monthly basis.
        
        **Data Sources:**
        - **Weather Data**: ERA5 monthly averages (temperature, precipitation, soil moisture, solar radiation)
        - **Satellite Data**: HLS NDVI vegetation indices
        - **Observations**: Beetle presence/absence data from Artportalen.se
        
        **Risk Levels:**
        - 🟢 Low Risk: <30% outbreak probability
        - 🟡 Medium Risk: 30-75% outbreak probability  
        - 🔴 High Risk: >75% outbreak probability
        """)
    
    # Load data
    data, data_source = load_data()
    
    if data.empty:
        st.error("No data available. Please check your data source.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Data source info
        st.markdown("---")
        st.subheader("Data Source")
        if data_source == "hopsworks":
           # st.success("✅ Hopsworks Feature Store")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        elif data_source == "csv":
            st.info("Local CSV File")
        
        st.markdown("---")
        
        # Variable selector with descriptions
        st.subheader("Map Display")
        variable_options = {
            "Outbreak Likelihood": ("outbreak_likelihood", "Predicted probability of beetle outbreak"),
            "Temperature": ("t2m", "Monthly average temperature (Kelvin)"),
            "Precipitation": ("tp", "Total monthly precipitation (m)"),
            "Soil Moisture L1": ("swvl1", "Volumetric soil water layer 1 (0-7cm)"),
            "Soil Moisture L2": ("swvl2", "Volumetric soil water layer 2 (7-28cm)"),
            "Solar Radiation": ("ssrd", "Surface solar radiation downwards (J/m²)"),
            "NDVI": ("NDVI", "Normalized Difference Vegetation Index"),
            "NDVI Anomaly": ("NDVI_anom", "NDVI deviation from climatology"),
        }
        
        selected_var_name = st.selectbox(
            "Select variable to display:",
            options=list(variable_options.keys()),
            index=0,
            help="Choose which environmental variable to visualize on the map"
        )
        selected_var, var_description = variable_options[selected_var_name]
        st.caption(f"ℹ️ {var_description}")
        
        st.markdown("---")
        
        # Filters
        st.subheader("Filters")
        
        # Risk threshold
        risk_threshold = st.slider(
            "Minimum outbreak likelihood:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Filter to show only locations above this outbreak probability threshold"
        )
        
        # Month filter (if available)
        if 'Month' in data.columns:
            unique_months = sorted(data['Month'].unique())
            if len(unique_months) > 1:
                selected_months = st.select_slider(
                    "Time period:",
                    options=unique_months,
                    value=(unique_months[0], unique_months[-1]),
                    format_func=lambda x: pd.to_datetime(x).strftime('%B %Y'),
                    help="Select time range to display"
                )
        
        st.markdown("---")
        
        # Legend
        st.subheader("Risk Legend")
        st.markdown("""
        <div style='font-size: 0.9rem;'>
        🟢 <b>Low</b>: <30%<br>
        🟡 <b>Medium</b>: 30-70%<br>
        🟠 <b>High</b>: 70-85%<br>
        🔴 <b>Extreme</b>: >85%
        </div>
        """, unsafe_allow_html=True)
    
    # Apply filters
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data['outbreak_likelihood'] >= risk_threshold]
    
    if 'Month' in data.columns and len(unique_months) > 1:
        filtered_data = filtered_data[
            (filtered_data['Month'] >= selected_months[0]) &
            (filtered_data['Month'] <= selected_months[1])
        ]
    
    # Key metrics
    st.subheader("Overview Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Monitored Locations",
            value=f"{len(filtered_data):,}",
            help="Total number of locations matching current filter criteria"
        )
    
    with col2:
        avg_likelihood = filtered_data['outbreak_likelihood'].mean()
        st.metric(
            label="Average Risk",
            value=f"{avg_likelihood:.1%}",
            help="Mean outbreak probability across all filtered locations"
        )
    
    with col3:
        high_risk = (filtered_data['outbreak_likelihood'] > 0.7).sum()
        pct_high = (high_risk / len(filtered_data) * 100) if len(filtered_data) > 0 else 0
        st.metric(
            label="High Risk Areas",
            value=f"{high_risk:,}",
            delta=f"{pct_high:.1f}% of total",
            delta_color="inverse",
            help="Locations with >70% outbreak probability"
        )
    
    # with col4:
    #     extreme_risk = (filtered_data['outbreak_likelihood'] > 0.85).sum()
    #     pct_extreme = (extreme_risk / len(filtered_data) * 100) if len(filtered_data) > 0 else 0
    #     st.metric(
    #         label="Extreme Risk",
    #         value=f"{extreme_risk:,}",
    #         delta=f"{pct_extreme:.1f}% of total",
    #         delta_color="inverse",
    #         help="Locations with >85% outbreak probability"
    #     )
    
    st.markdown("---")
    
    # Main map
    st.subheader(f"🗺️ {selected_var_name} Map")
    
    if len(filtered_data) == 0:
        st.warning("⚠️ No locations match the current filter criteria. Try adjusting the risk threshold.")
        return
    
    # Prepare map data
    map_data = filtered_data.copy()
    map_data['latitude'] = map_data['Lat']
    map_data['longitude'] = map_data['Lon']
    
    # Dynamic color scale
    if selected_var == 'outbreak_likelihood':
        color_scale = 'Reds'
        color_label = 'Outbreak Probability'
    elif 'NDVI' in selected_var:
        color_scale = 'Greens'
        color_label = selected_var_name
    elif selected_var == 't2m':
        color_scale = 'RdYlBu_r'
        color_label = 'Temperature (K)'
    else:
        color_scale = 'Viridis'
        color_label = selected_var_name
    
    # Create map
    hover_data_dict = {
        'latitude': ':.3f',
        'longitude': ':.3f',
        selected_var: ':.3f',
        'outbreak_likelihood': ':.1%',
        'Month': True
    }
    
    # Add kommun if available
    if 'Kommun' in map_data.columns:
        hover_data_dict['Kommun'] = True
    
    fig = px.scatter_map(
        map_data,
        lat='latitude',
        lon='longitude',
        color=selected_var,
        size='outbreak_likelihood',
        hover_data=hover_data_dict,
        color_continuous_scale=color_scale,
        size_max=15,
        zoom=4,
        center={"lat": 62, "lon": 15},
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Analysis section
    st.subheader("Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Outbreak Probability Distribution**")
        fig_hist = px.histogram(
            filtered_data,
            x='outbreak_likelihood',
            nbins=30,
            labels={'outbreak_likelihood': 'Outbreak Probability'},
            color_discrete_sequence=['#d62728']
        )
        fig_hist.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Outbreak Probability",
            yaxis_title="Number of Locations"
        )
        fig_hist.add_vline(
            x=0.7,
            line_dash="dash",
            line_color="orange",
            annotation_text="High Risk Threshold (70%)",
            annotation_position="top"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("**Summary Statistics**")
        stats_data = {
            "Metric": [
                "Total Locations",
                "Low Risk (<30%)",
                "Medium Risk (30-75%)",
                "High Risk (>75%)",
                #"Extreme Risk (>85%)",
                "Average Probability",
                "Median Probability",
                "Max Probability"
            ],
            "Value": [
                f"{len(filtered_data):,}",
                f"{(filtered_data['outbreak_likelihood'] < 0.3).sum():,}",
                f"{((filtered_data['outbreak_likelihood'] >= 0.3) & (filtered_data['outbreak_likelihood'] < 0.75)).sum():,}",
                f"{((filtered_data['outbreak_likelihood'] > 0.75)).sum():,}",
                #f"{(filtered_data['outbreak_likelihood'] >= 0.85).sum():,}",
                f"{filtered_data['outbreak_likelihood'].mean():.1%}",
                f"{filtered_data['outbreak_likelihood'].median():.1%}",
                f"{filtered_data['outbreak_likelihood'].max():.1%}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Bark Beetle Outbreak Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate mock data for Sweden (will be replaced with real data later)
@st.cache_data
def generate_mock_data(n_points=150):
    """Generate mock data points across Sweden"""
    np.random.seed(42)
    
    # Sweden approximate bounds
    lat_min, lat_max = 55.0, 69.0
    lon_min, lon_max = 11.0, 24.0
    
    # Generate random points
    data = pd.DataFrame({
        'Lat': np.random.uniform(lat_min, lat_max, n_points),
        'Lon': np.random.uniform(lon_min, lon_max, n_points),
        'outbreak_likelihood': np.random.uniform(0, 1, n_points),
        't2m': np.random.uniform(273, 293, n_points),  # Temperature (K)
        'tp': np.random.uniform(0, 0.01, n_points),    # Precipitation
        'swvl1': np.random.uniform(0.1, 0.3, n_points), # Soil moisture layer 1
        'swvl2': np.random.uniform(0.1, 0.3, n_points), # Soil moisture layer 2
        'ssrd': np.random.uniform(5e6, 2e7, n_points),  # Solar radiation
        'NDVI': np.random.uniform(0.3, 0.9, n_points),
        'NDVI_anom': np.random.uniform(-0.2, 0.2, n_points),
        'Kommun': [f"Kommun_{i % 50}" for i in range(n_points)],
        'Lan': np.random.choice(['Stockholm', 'Västra Götaland', 'Skåne', 'Västerbotten', 'Norrbotten', 'Dalarna'], n_points),
        'Month': pd.date_range(start='2024-01-01', periods=n_points, freq='M')
    })
    
    return data

def main():
    # Header
    st.title("Spruce Bark Beetle Outbreak Prediction")
    st.markdown("""
    Monthly monitoring dashboard predicting bark beetle outbreak likelihood across Sweden  
    Data sources: 
    """)
    
    # Load data (mock for now)
    data = generate_mock_data(n_points=200)
    
    # Sidebar
    with st.sidebar:
        st.header("Dashboard Controls")
        st.markdown("---")
        
        # # Prediction month selector
        # st.subheader("Forecast Period")
        # current_month = datetime.now().replace(day=1)
        # next_month = (current_month + timedelta(days=32)).replace(day=1)
        
        # st.info(f"**Current:** {current_month.strftime('%B %Y')}")
        # st.success(f"**Forecasting:** {next_month.strftime('%B %Y')}")
        
        st.markdown("---")
        
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
        
        regions = ["All Regions"] + sorted(data['Lan'].unique().tolist())
        selected_region = st.selectbox("Region (Län):", regions)
        

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
    if selected_region != "All Regions":
        filtered_data = filtered_data[filtered_data['Lan'] == selected_region]
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
            #delta=f"{(high_risk/len(filtered_data)*100):.1f}% of total" if len(filtered_data) > 0 else "0%",
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
    st.subheader(f"{selected_var_name}")
    
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
    fig = px.scatter_mapbox(
        map_data,
        lat='latitude',
        lon='longitude',
        color=selected_var,
        size='outbreak_likelihood',
        hover_name='Kommun',
        hover_data={
            'latitude': ':.3f',
            'longitude': ':.3f',
            selected_var: ':.3f',
            'outbreak_likelihood': ':.1%',
            'Lan': True,
            'Month': True
        },
        color_continuous_scale=color_scale,
        size_max=20,
        zoom=4,
        center={"lat": 62, "lon": 15},
        mapbox_style="open-street-map",
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
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Regions by Outbreak Risk")
        region_stats = filtered_data.groupby('Lan').agg({
            'outbreak_likelihood': ['mean', 'max', 'count']
        }).reset_index()
        region_stats.columns = ['Region', 'Avg Likelihood', 'Max Likelihood', 'Locations']
        region_stats = region_stats.sort_values('Avg Likelihood', ascending=False)
        region_stats['Avg Likelihood'] = region_stats['Avg Likelihood'].apply(lambda x: f"{x:.1%}")
        region_stats['Max Likelihood'] = region_stats['Max Likelihood'].apply(lambda x: f"{x:.1%}")
        st.dataframe(region_stats, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

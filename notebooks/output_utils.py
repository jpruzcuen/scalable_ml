import geopandas
import matplotlib.pyplot as plt

######################################## PLOTS ########################################

def plot_heatmap_on_map(dataframe, variable, title, lat_col='Lat', lon_col='Lon'):

    gdf = geopandas.GeoDataFrame(
        dataframe, geometry=geopandas.points_from_xy(dataframe[lon_col], dataframe[lat_col])
        )

    world = geopandas.read_file('../data/country/ne_110m_admin_0_sovereignty.shp')

    # Plot Sweden with t2m as a heatmap
    ax = world[world['NAME'] == 'Sweden'].plot(
        color='lightgrey', edgecolor='black', figsize=(12, 10))

    # Plot the GeoDataFrame with t2m values as a heatmap
    gdf.plot(ax=ax, column=variable, cmap='RdYlBu_r', markersize=20, alpha=0.5, 
            legend=True, legend_kwds={'label': title, 'shrink': 0.6})

    ax.set_title(f'{title} Distribution over Sweden')
    plt.tight_layout()
    plt.show()


######################################## FILE HANDLING ########################################

def extract_year_month(filename):
    parts = filename.replace(".csv", "").split("_")
    day = int(parts[1])
    month = int(parts[2])
    year = int(parts[3])
    return year, month, day
import geopandas
import matplotlib.pyplot as plt

######################################## PLOTS ########################################

def plot_heatmap_on_map(dataframe, variable, title, 
                        lat_col='lat', lon_col='lon'):

    gdf = geopandas.GeoDataFrame(
        dataframe, geometry=geopandas.points_from_xy(dataframe[lon_col], dataframe[lat_col])
        )

    world = geopandas.read_file('../data/country/ne_110m_admin_0_sovereignty.shp')

    # Plot Sweden with t2m as a heatmap
    ax = world[world['NAME'] == 'Sweden'].plot(
        color='lightgrey', edgecolor='black', figsize=(8, 6))

    if variable == 'inference':
        # Plot presence and absence with different colors
        gdf_absent = gdf[gdf['pressence_pred'] == 0]
        gdf_present = gdf[gdf['pressence_pred'] == 1]

        gdf_absent.plot(ax=ax, color='white', edgecolor='white', markersize=5, alpha=0.5, label='Absent (0)')
        gdf_present.plot(ax=ax, color='red', markersize=5, alpha=0.7, label='Present (1)')
        ax.legend()
    
    else:
        # Plot the GeoDataFrame with t2m values as a heatmap
        gdf.plot(ax=ax, column=variable, cmap='RdYlBu_r', markersize=20, alpha=0.5, 
            legend=True, legend_kwds={'label': title, 'shrink': 0.6})

    ax.set_title(title)

    plt.tight_layout()
    plt.show()


######################################## FILE HANDLING ########################################

def extract_year_month(filename):
    parts = filename.replace(".csv", "").split("_")
    day = int(parts[1])
    month = int(parts[2])
    year = int(parts[3])
    return year, month, day
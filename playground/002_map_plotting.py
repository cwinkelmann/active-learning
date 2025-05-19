# Packages used by this tutorial
import geopandas # manipulating geographic data
import shapely # manipulating geometries
import pygris # easily acquiring shapefiles from the US Census
import matplotlib.pyplot as plt # visualization
# Importing the main package
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow

# Downloading the state-level dataset from pygris
states = pygris.states(cb=True, year=2022, cache=False)

from matplotlib_map_utils.core.inset_map import InsetMap, inset_map, ExtentIndicator, indicate_extent, DetailIndicator, indicate_detail

# This is just a function to create a new, blank map with matplotlib, with our default settings
def new_map(rows=1, cols=1, figsize=(5,5), dpi=150, ticks=False):
    # Creating the plot(s)
    fig, ax = plt.subplots(rows,cols, figsize=figsize, dpi=dpi)
    # Turning off the x and y axis ticks
    if ticks==False:
        if rows > 1 or cols > 1:
            for a in ax.flatten():
                a.set_xticks([])
                a.set_yticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    # Returning the fig and ax
    return fig, ax


# This is using some utilities within the mmu package to help filter our states
from matplotlib_map_utils.utils import USA
usa = USA()
# Filtering based on FIPS codes
contiguous = states.query(f"GEOID in {usa.filter_contiguous(True)}")
query_object = usa.filter_abbr("AK")
alaska = states.query(f"GEOID == '{query_object}'").to_crs(3467)


# Setting up the main plot
fig, ax = new_map(ticks=True)
# Plotting the contiguous USA
contiguous.plot(ax=ax)

# Adding an inset map to the plot
iax = inset_map(ax, location="lower left", size=0.8, pad=0.1, xticks=[], yticks=[])
# Plotting alaska in the inset map
alaska.plot(ax=iax)

na = NorthArrow(location="lower right", rotation={"degrees":0}, scale=0.1)
# Note that you have to use .copy() here too!
ax.add_artist(na.copy())

plt.show()
import geopandas

# load the data
gdf = geopandas.read_file('tiles.json')
print(gdf)
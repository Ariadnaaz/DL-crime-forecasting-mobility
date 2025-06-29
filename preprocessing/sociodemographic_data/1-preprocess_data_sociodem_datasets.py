import geopandas as gpd
import pandas as pd
import os 
import folium
import numpy as np
import shapely.geometry
from pathlib import Path
from pandarallel import pandarallel
import pygris
from IPython.display import display # Optional if using Jupyter/Colab

pandarallel.initialize(nb_workers=min(os.cpu_count(), 12),progress_bar=True)

"""### Useful functions"""

def extract_multipolygon_city(file_path,city_name):
    '''
    Extracts the entry in the geojson file corresponding to the city selected and outputs the
    corresponding geodataframe with the multipolygon.

        Parameters:
            file_path (str): File path to the geojson
            city_name (str): Name of the city we selected

        Returns:
            feature (geopandas): The geopandas dataframe for that city
    '''
    d = pd.read_json(file_path)
    for feature in d["features"]:
        if feature[0]['properties']['city'] == city_name:
            return gpd.GeoDataFrame.from_features(feature)

def generate_grid(selected_city,grid_size,plot=True):
  gdf = extract_multipolygon_city(file_path='../../city_multipolygons.geojson',city_name=selected_city)

  bbox = gdf.total_bounds
  min_lon, min_lat, max_lon, max_lat = (bbox[2],bbox[1],bbox[0],bbox[3])

  lon = np.linspace(min_lon, max_lon, grid_size+1)
  lat = np.linspace(min_lat, max_lat, grid_size+1)

  latlons = []
  for i in range(len(lat)-1):
      for k in range(len(lon)-1):
          latlons.append((lat[k], lon[i], lat[k+1], lon[i+1]))

  if plot==True:
    m = folium.Map(location=((min_lat+max_lat)/2,(min_lon+max_lon)/2), zoom_start=11)
    for k in latlons:
        folium.Rectangle([(k[0], k[1]), (k[2], k[3])],
                        color='red',
                        fill='pink',
                        fill_opcity=0.5).add_to(m)
    cgeo = (
            gdf.set_crs("epsg:4326")
            .sample(1)
            .pipe(lambda d: d.to_crs(d.estimate_utm_crs()))["geometry"]
            #.centroid.buffer(10000)
            .to_crs("epsg:4326")
            .__geo_interface__
        )

    geo_j = folium.GeoJson(data=cgeo)
    geo_j.add_to(m)
    display(m)
    return (min_lon, min_lat, max_lon, max_lat, latlons)
  else:
    return (min_lon, min_lat, max_lon, max_lat, latlons)


"""### 1. Get cell to GEOIDs mapping (only run this for different grid size)"""

def find_polygons_intersect(row,df): # row is from df_poly_cells, df is the one from pygris.block_groups()
  geoid_list = []
  for p_idx,elem in df.iterrows():
    if row['polygon'].intersects(elem['geometry']):
      geoid_list.append(elem['GEOID'])
  if not geoid_list:
    geoid_list = np.nan
  return pd.Series({'block_groups': geoid_list})

def find_geoids_per_grid_cell(selected_city,city_folder,state,county,output_folder,grid_size=50):
  # get grid boxes
  min_lon, min_lat, max_lon, max_lat, grid_bboxes = generate_grid(selected_city,grid_size=grid_size,plot=False)

  # turn boxes into polygons and that list into a dataframe
  polygon_list = []
  for elem in grid_bboxes:
    polygon = shapely.geometry.box(elem[1],elem[0],elem[3],elem[2],ccw=True)
    polygon_list.append(polygon)
  df_poly_cells = pd.DataFrame(polygon_list,columns=['polygon'])
  print("Shape cell polygons dataframe: ", df_poly_cells.shape)

  # get list of block groups and their geometry
  df_bg = pygris.block_groups(state = state, county = county, cache = True)

  print("Finding the GEOID of all block groups intersecting with each cell...")
  temp = df_poly_cells.parallel_apply(find_polygons_intersect,df=df_bg,axis=1)
  df_final = df_poly_cells.merge(temp,left_index=True, right_index= True)
  print("Shape dataframe after adding GEOID list: ", df_final.shape)

  # make final output folder if it doesn't exist
  os.makedirs(f'{output_folder}/', exist_ok=True)

  df_final.to_csv(f'{output_folder}/{city_folder}_cell_to_geoids_map.csv')
  print("Final cell to geoids map saved!")

find_geoids_per_grid_cell(selected_city='Baltimore',city_folder='Baltimore',state='MD',county='510',output_folder='Sociodem_data_corrected/Grid_cells_0.2gu',grid_size=39)
find_geoids_per_grid_cell(selected_city='Chicago',city_folder='Chicago',state='IL',county='031',output_folder='Sociodem_data_corrected/Grid_cells_0.2gu',grid_size=85)
find_geoids_per_grid_cell(selected_city='Los Angeles',city_folder='Los_Angeles',state='CA',county='037',output_folder='Sociodem_data_corrected/Grid_cells_0.2gu',grid_size=133)
find_geoids_per_grid_cell(selected_city='Philadelphia',city_folder='Philadelphia',state='PA',county='101',output_folder='Sociodem_data_corrected/Grid_cells_0.2gu',grid_size=64)

"""### 2. Get data per GEOID"""

def get_data_per_geoid_city_year(city_folder,state_fips,year,output_folder):

    dic_vars = {'X01_AGE_AND_SEX': ['B01001e1','B01001e26','B01002e2','B01002e3'],
                'X02_RACE': ['B02001e1','B02001e2','B02001e3','B02001e4','B02001e5','B02001e6','B02001e7'],
                'X12_MARITAL_STATUS_AND_HISTORY': ['B12001e2','B12001e11','B12001e3','B12001e4','B12001e9','B12001e10','B12001e12','B12001e13','B12001e18','B12001e19'],
                'X15_EDUCATIONAL_ATTAINMENT': ['B15003e1','B15003e2','B15003e17','B15003e22','B15003e24'],
                'X19_INCOME': ['B19013e1'],
                'X23_EMPLOYMENT_STATUS': ['B23025e1','B23025e4','B23025e5','B23025e6','B23025e7']
                }

    gdf_list = []
    for layer_name in dic_vars:
        print("LAYER: ", layer_name)
        gdf = gpd.read_file(f'raw_data/ACS_{year}_5YR_BG_{state_fips}.gdb.zip',layer=layer_name)

        # keep only the variables we are interested in and the corresponding GEOID
        vars_list = dic_vars[layer_name]
        gdf = gdf[['GEOID']+vars_list]
        gdf.set_index('GEOID',drop=True,inplace=True)
        gdf_list.append(gdf)

    df_gdf = pd.concat(gdf_list, axis=1)

    print("Calculating percentages...")
    # number females
    df_gdf['B01001e26'] = df_gdf['B01001e26']*100/df_gdf['B01001e1']

    # race variables
    df_gdf['B02001e2'] = df_gdf['B02001e2']*100/df_gdf['B02001e1']
    df_gdf['B02001e3'] = df_gdf['B02001e3']*100/df_gdf['B02001e1']
    df_gdf['B02001e4'] = df_gdf['B02001e4']*100/df_gdf['B02001e1']
    df_gdf['B02001e5'] = df_gdf['B02001e5']*100/df_gdf['B02001e1']
    df_gdf['B02001e6'] = df_gdf['B02001e6']*100/df_gdf['B02001e1']
    df_gdf['B02001e7'] = df_gdf['B02001e7']*100/df_gdf['B02001e1']

    # marital status variables
    # male
    df_gdf['B12001e3'] = df_gdf['B12001e3']*100/df_gdf['B12001e2']
    df_gdf['B12001e4'] = df_gdf['B12001e4']*100/df_gdf['B12001e2']
    df_gdf['B12001e9'] = df_gdf['B12001e9']*100/df_gdf['B12001e2']
    df_gdf['B12001e10'] = df_gdf['B12001e10']*100/df_gdf['B12001e2']
    # female
    df_gdf['B12001e12'] = df_gdf['B12001e12']*100/df_gdf['B12001e11']
    df_gdf['B12001e13'] = df_gdf['B12001e13']*100/df_gdf['B12001e11']
    df_gdf['B12001e18'] = df_gdf['B12001e18']*100/df_gdf['B12001e11']
    df_gdf['B12001e19'] = df_gdf['B12001e19']*100/df_gdf['B12001e11']

    # education variables
    df_gdf['B15003e2'] = df_gdf['B15003e2']*100/df_gdf['B15003e1']
    df_gdf['B15003e17'] = df_gdf['B15003e17']*100/df_gdf['B15003e1']
    df_gdf['B15003e22'] = df_gdf['B15003e22']*100/df_gdf['B15003e1']
    df_gdf['B15003e24'] = df_gdf['B15003e24']*100/df_gdf['B15003e1']

    # employment variables
    df_gdf['B23025e4'] = df_gdf['B23025e4']*100/df_gdf['B23025e1']
    df_gdf['B23025e5'] = df_gdf['B23025e5']*100/df_gdf['B23025e1']
    df_gdf['B23025e6'] = df_gdf['B23025e6']*100/df_gdf['B23025e1']
    df_gdf['B23025e7'] = df_gdf['B23025e7']*100/df_gdf['B23025e1']

    # drop unnecessary columns
    df_gdf.drop(columns=['B01001e1','B02001e1','B12001e2','B12001e11','B15003e1','B23025e1'],inplace=True)
    os.makedirs(f'{output_folder}/', exist_ok=True)
    df_gdf.to_csv(f'{output_folder}/{city_folder}_sociodem_{year}.csv')
    print("File saved!")

for year in [2019,2020,2021]:
  print(f"Doing year {year}...")
  get_data_per_geoid_city_year(city_folder='Baltimore',state_fips='24',year=year,output_folder='Sociodem_data_Major_corrected')

print("#### CHICAGO ####")
for year in [2019,2020,2021]:
  print(f"Doing year {year}...")
  get_data_per_geoid_city_year(city_folder='Chicago',state_fips='17',year=year,output_folder='Sociodem_data_Major_corrected')

print("#### LOS ANGELES ####")
for year in [2020,2021]: # 2019
  print(f"Doing year {year}...")
  get_data_per_geoid_city_year(city_folder='Los_Angeles',state_fips='06',year=year,output_folder='Sociodem_data_Major_corrected')

print("#### PHILADELPHIA ####")
for year in [2019,2020,2021]:
  print(f"Doing year {year}...")
  get_data_per_geoid_city_year(city_folder='Philadelphia',state_fips='42',year=year,output_folder='Sociodem_data_Major_corrected')

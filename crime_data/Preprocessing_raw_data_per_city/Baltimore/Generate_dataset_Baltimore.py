import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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

df = pd.read_csv('raw_data/Part1_Crime_5ys.csv',low_memory=False)
print("Size dataframe initially: ", df.shape)

# change type of CrimeDateTime from object to datetime
df['CrimeDateTime'] = df['CrimeDateTime'].map(lambda x: x.replace("+00", ""))
df['CrimeDateTime'] = pd.to_datetime(df['CrimeDateTime'],format="%m/%d/%Y %I:%M:%S %p",errors = 'coerce') # we mark as NaT dates that are too old given datetime lowerbound

# sort by date in ascending order
df.sort_values(by='CrimeDateTime', inplace = True)

# drop rows with with NaT in CrimeDateTime column
df.dropna(subset='CrimeDateTime',inplace=True)
print("Shape dataframe after removing dates too old: ", df.shape)

# make compatible
df.rename(columns={'CrimeDateTime': 'crime_date_time', 'Description':'crime_type','Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# reset index
df.reset_index(drop=True,inplace=True)

# select dates for 2019 to 2023
lower_bound = "2019/01/01 00:00:00"
upper_bound = "2024/01/01 00:00:00"
df_filter = df.copy()
df_filter = df_filter.loc[(df_filter["crime_date_time"] >= lower_bound) & (df_filter["crime_date_time"] < upper_bound)]
df_filter.reset_index(drop=True,inplace=True)
print("Shape dataframe after selecting crimes from 2019 to 2023 incl.", df_filter.shape)

# remove points that are outside the city borders
# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../../city_multipolygons.geojson',city_name='Baltimore')

# remove points that aren't within the multipolygon
df_clean = df_filter.copy()
for i, entry in df_filter.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df_filter.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv("Baltimore_crimes_clean_all_5ys.csv")
print("Shape final dataframe: ", df_clean.shape)

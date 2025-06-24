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

df1 = pd.read_csv('raw_data/incidents_part1_part2_2019.csv',low_memory=False)
df2 = pd.read_csv('raw_data/incidents_part1_part2_2020.csv',low_memory=False)
df3 = pd.read_csv('raw_data/incidents_part1_part2_2021.csv',low_memory=False)
df4 = pd.read_csv('raw_data/incidents_part1_part2_2022.csv',low_memory=False)
df5 = pd.read_csv('raw_data/incidents_part1_part2_2023.csv',low_memory=False)
df = pd.concat([df1,df2,df3,df4,df5])
print("Size dataframe initially: ", df.shape)

# change type of CrimeDateTime from object to datetime
df['dispatch_date_time'] = df['dispatch_date_time'].map(lambda x: x.replace("+00", ""))
df['dispatch_date_time'] = pd.to_datetime(df['dispatch_date_time'],format='%Y-%m-%d %H:%M:%S',errors = 'coerce')

# drop rows with with NaT in CrimeDateTime column
df.dropna(subset='dispatch_date_time',inplace=True)
print("Shape dataframe after removing dates too old: ", df.shape)

# reset in order to have the indexes starting from 0
df.reset_index(drop=True,inplace=True)

# make compatible
df.rename(columns={'dispatch_date_time': 'crime_date_time', 'text_general_code':'crime_type','lat':'latitude','lng':'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# remove points that are outside the city borders
# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../../city_multipolygons.geojson',city_name='Philadelphia')

# remove points that aren't within the multipolygon
#df.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)
df_clean = df.copy()
for i, entry in df.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv(f"Philadelphia_crimes_clean_all_5ys.csv")
print("Shape final dataframe: ", df_clean.shape)

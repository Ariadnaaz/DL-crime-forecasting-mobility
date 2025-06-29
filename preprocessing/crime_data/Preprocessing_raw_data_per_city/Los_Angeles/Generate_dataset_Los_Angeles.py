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

"""### 1. Create preprocessed dataset for 2019"""

df = pd.read_csv('raw_data/Crime_Data_from_2010_to_2019.csv',dtype={'DATE OCC':'str','TIME OCC':'str'},low_memory=False)
print("Size dataframe initially: ", df.shape)

# convert date column to datetime format
df['DateCrime'] = df['DATE OCC'].str.slice(0, 10)
df['DateCrime'] = pd.to_datetime(df['DateCrime'],format='%m/%d/%Y')
# put time in the format hh:mm:ss
df['TIME OCC'] = df['TIME OCC'].astype(str).str[:2] + ':' + df['TIME OCC'].astype(str).str[2:] + ':00'
# convert to timedelta now that we have the strings in the right format
df['TIME OCC'] = pd.to_timedelta(df['TIME OCC'])
# create CrimeDateTime column with the date and time of the crime together
df['CrimeDateTime'] = pd.to_datetime(df['DateCrime'] + df['TIME OCC'], format='%Y-%m-%d %H:%M:%S')
print("Shape dataframe after removing rows with incorrect times: ", df.shape)

# keep dates for 2019
df = df.loc[(df["CrimeDateTime"] >= "2019/01/01") & (df["CrimeDateTime"] < "2020/01/01")]
print("Shape dataframe after selecting crimes for 2019: ", df.shape)

# make compatible
df.rename(columns={'CrimeDateTime': 'crime_date_time', 'Crm Cd Desc':'crime_type','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# remove points that are outside the city borders
# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../../city_multipolygons.geojson',city_name='Los Angeles')

# remove points that aren't within the multipolygon
df.reset_index(drop=True,inplace=True)
df_clean = df.copy()
for i, entry in df.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv("Los_Angeles_crimes_clean_2019.csv")
print("Shape final dataframe: ", df_clean.shape)


"""### 2. Create preprocessed dataset for 2020 to 2023 """

df = pd.read_csv('raw_data/Crime_Data_from_2020_to_Present_5ys.csv',dtype={'DATE OCC':'str','TIME OCC':'str'},low_memory=False)
print("Size dataframe initially: ", df.shape)

# convert date column to datetime format
df['DateCrime'] = df['DATE OCC'].str.slice(0, 10)
df['DateCrime'] = pd.to_datetime(df['DateCrime'],format='%m/%d/%Y')
# put time in the format hh:mm:ss
df['TIME OCC'] = df['TIME OCC'].astype(str).str[:2] + ':' + df['TIME OCC'].astype(str).str[2:] + ':00'
# convert to timedelta now that we have the strings in the right format
df['TIME OCC'] = pd.to_timedelta(df['TIME OCC'])
# create CrimeDateTime column with the date and time of the crime together
df['CrimeDateTime'] = pd.to_datetime(df['DateCrime'] + df['TIME OCC'], format='%Y-%m-%d %H:%M:%S')
print("Shape dataframe after removing rows with incorrect times: ", df.shape)

# keep dates for 2020 to 2022
df = df.loc[(df["CrimeDateTime"] >= "2020/01/01") & (df["CrimeDateTime"] < "2024/01/01")]
print("Shape dataframe after selecting crimes for 2020 to 2022: ", df.shape)

# make compatible
df.rename(columns={'CrimeDateTime': 'crime_date_time', 'Crm Cd Desc':'crime_type','LAT': 'latitude', 'LON': 'longitude'}, inplace=True)

# keep only relevant columns
df = df[['crime_date_time','crime_type','latitude','longitude']]

# remove points that are outside the city borders
# extract multipolygon of the city
gdf = extract_multipolygon_city(file_path='../../../city_multipolygons.geojson',city_name='Los Angeles')

# remove points that aren't within the multipolygon
df.reset_index(drop=True,inplace=True)
df_clean = df.copy()
for i, entry in df.iterrows():
    if gdf['geometry'].contains(Point(entry['longitude'], entry['latitude']))[0]:
        None
    else:
        df_clean = df_clean.drop(df.index[i])

# reset index
df_clean.reset_index(drop=True,inplace=True)

df_clean.to_csv("Los_Angeles_crimes_clean_2020_to_2023.csv")
print("Shape final dataframe: ", df_clean.shape) # approx 30 min to run

"""### 3. Merge the two preprocessed datasets"""

df1 = pd.read_csv("Los_Angeles_crimes_clean_2019.csv",index_col=0)
df2 = pd.read_csv("Los_Angeles_crimes_clean_2020_to_2023.csv",index_col=0)

df_final = pd.concat([df1,df2])

# reset index
df_final.reset_index(drop=True,inplace=True)

df_final.to_csv("Los_Angeles/Los_Angeles_crimes_clean_all_5ys.csv")
print("Shape final dataframe: ", df_final.shape)

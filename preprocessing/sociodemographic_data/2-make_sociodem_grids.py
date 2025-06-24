import geopandas as gpd
import pandas as pd
import os 
import numpy as np
import json
from pandarallel import pandarallel
from pathlib import Path
import os

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


def find_variable_description(var_code):
    with open("sociodem_vars_map.txt", "r") as fp: # this file is created when running the previous code
        # Load the dictionary from the file
        d = json.load(fp)
    # find the dictionary key that contains that var
    desc = [value for key, value in d.items() if var_code in key][0]
    return desc

"""### 1. Make final grid files for all years together"""

def get_data_per_cell(row,df):
  """
    Aggregates sociodemographic data for a spatial grid cell by averaging data
    from all intersecting block groups.

    Parameters
    ----------
    row : pandas.Series
        A row from the grid-to-block-group mapping dataframe (`df_map`).
        Must contain a 'block_groups' column listing GEOIDs as a stringified list.
    df : pandas.DataFrame
        The sociodemographic data indexed by GEOID (block group ID).

    Returns
    -------
    pandas.Series
        A Series containing the average sociodemographic features for the cell.
        Returns a Series of NaNs if no valid block group data is found.
  """
  list_bg = row['block_groups']
  if isinstance(list_bg, str): # we check that list is not empty
    data_list = []
    for geoid in eval(list_bg):
      # check if geoid is in data file
      if geoid in df.index:
        data_list.append(df.loc[geoid])
    if data_list: # check if we did't make any dataframes bc of missing geoids in data
      df_bgs = pd.concat(data_list, axis = 1)
      df_avg = df_bgs.T.mean()
    else:
      # CAREFUL NUM VARIABLES IS HARDCODED
      df_avg = pd.DataFrame(np.full((1,26),np.nan),columns=df.columns.values.tolist()).squeeze()
  else:
    df_avg = pd.DataFrame(np.full((1,26),np.nan),columns=df.columns.values.tolist()).squeeze()
  return df_avg

def make_final_grid_sociodem(city_name,city_folder,input_folder,map_folder,output_folder):
  """
    Creates a cell-level sociodemographic dataset for a given city by aggregating
    block group-level data over a spatial grid across multiple years.

    For each grid cell, the function:
    - Identifies intersecting block groups
    - Averages their sociodemographic values
    - Removes cells outside the city boundary
    - Merges data across 2019, 2020, and 2021
    - Saves the result to a CSV file

    Parameters
    ----------
    city_name : str
        Name of the city (used to filter the multipolygon boundaries).
    city_folder : str
        Name of the folder containing city-specific data.
    input_folder : str
        Folder containing input sociodemographic CSV files per year.
    map_folder : str
        Folder containing the grid-to-block-group mapping CSV file.
    output_folder : str
        Folder to save the final output CSV file.

    Returns
    -------
    None
        The function saves the final cell-level sociodemographic data to disk.
  """
  # we assume all years use same geoid...
  df_map = pd.read_csv(f'{map_folder}/{city_folder}_cell_to_geoids_map.csv',index_col=0)
  df_map['polygon'] = gpd.GeoSeries.from_wkt(df_map['polygon'])

  df_all_list = []
  for year in [2019,2020,2021]:
    print(f"Doing year {year}...")
    df_data = pd.read_csv(f'{input_folder}/{city_folder}_sociodem_{year}.csv')
    df_data['GEOID'] = df_data['GEOID'].str.replace('15000US', '')
    df_data.set_index('GEOID',drop=True,inplace=True)

    print("Finding the GEOID of all block groups intersecting with each cell...")
    temp = df_map.parallel_apply(get_data_per_cell,df=df_data,axis=1)
    df_year = df_map.merge(temp,left_index=True, right_index= True)

    # set to NaN the rows where polygon doesn't intersect with city polygon
    gdf = extract_multipolygon_city(file_path='../city_multipolygons.geojson',city_name=city_name)
    for i,row in df_year.iterrows():
      if not row['polygon'].intersects(gdf.geometry[0]):
        df_year.loc[i] = np.nan

    # remove polygon and block_group
    df_year.drop(columns=['polygon','block_groups'],inplace=True)
    df_year.columns = list(map(lambda x: str(x) + f"_{str(year)}", df_year.columns.tolist()))
    df_all_list.append(df_year)

  # concat the 4 years
  print("Concatenating and saving final dataframe...")
  df_all = pd.concat(df_all_list,axis=1)

  # make final output folder if it doesn't exist
  #Path(f"drive/MyDrive/Sociodemographic_data/{output_folder}").mkdir(exist_ok=True)

  df_all.T.to_csv(f"{output_folder}/{city_folder}_sociodem_all_final_grid.csv")
  print("Shape final dataframe: ", df_all.T.shape)
  print("File saved!")

make_final_grid_sociodem(city_name='Baltimore',
                         city_folder='Baltimore',
                         input_folder='Sociodem_data_corrected',
                         map_folder='Sociodem_data_corrected/Grid_cells_0.2gu',
                         output_folder='Sociodem_data_corrected/Grid_cells_0.2gu/Final_all')

make_final_grid_sociodem(city_name='Chicago',
                         city_folder='Chicago',
                         input_folder='Sociodem_data_corrected',
                         map_folder='Sociodem_data_corrected/Grid_cells_0.2gu',
                         output_folder='Sociodem_data_corrected/Grid_cells_0.2gu/Final_all')

make_final_grid_sociodem(city_name='Los Angeles',
                         city_folder='Los_Angeles',
                         input_folder='Sociodem_data_corrected',
                         map_folder='Sociodem_data_corrected/Grid_cells_0.2gu',
                         output_folder='Sociodem_data_corrected/Grid_cells_0.2gu/Final_all')

make_final_grid_sociodem(city_name='Philadelphia',
                         city_folder='Philadelphia',
                         input_folder='Sociodem_data_corrected',
                         map_folder='Sociodem_data_corrected/Grid_cells_0.2gu',
                         output_folder='Sociodem_data_corrected/Grid_cells_0.2gu/Final_all')

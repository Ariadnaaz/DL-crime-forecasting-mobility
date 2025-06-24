import pandas as pd
import numpy as np
import datetime
from pandarallel import pandarallel
from pathlib import Path
import os

pandarallel.initialize(nb_workers=min(os.cpu_count(), 12),progress_bar=True)

"""### 1. Generate yearly matrix of crime counts per hour (better run in a cluster bc otherwise you might run out of RAM)"""

def hour_of_year(dt):
    beginning_of_year = datetime.datetime(dt['crime_date_time'].year, 1, 1, tzinfo=dt['crime_date_time'].tzinfo)
    return pd.Series({"hour_year":(dt['crime_date_time'] - beginning_of_year).total_seconds() // 3600})

def make_list_hours(row,total_h):
    hours_list = np.zeros(total_h);
    hours_list[int(row['hour_year'])] = 1
    hours_list = hours_list.astype(int)
    return pd.Series({"hour_year_list": hours_list.tolist()})

def make_final_grid_files_year_crime(city_folder,crime,in_out_folder,years,grid_size):

  df_crimes = pd.read_csv(f'{in_out_folder}/Clean_all_grid/{city_folder}_{crime}_clean_all_grid.csv',index_col=0,parse_dates=['crime_date_time'],date_parser=pd.to_datetime)
  print("Shape whole dataset: ", df_crimes.shape)

  print("Translate date of crime to hour of the year")
  df_hour = df_crimes.merge(df_crimes.parallel_apply(hour_of_year,axis=1),left_index=True, right_index= True)
  print("Initial shape: ",df_hour.shape)

  for y in years:
    print("\nYEAR: ",y)
    df_year = df_hour.copy()
    df_year = df_year[df_year['crime_date_time'].dt.year == y]
    df_year.reset_index(inplace=True)
    df_year = df_year[['cell','hour_year']]
    print("Shape after taking only this year: ",df_year.shape)

    # calculate number of hours in that year
    beginning_of_y1 = datetime.datetime(y, 1, 1)
    beginning_of_y2 = datetime.datetime(y+1, 1, 1)
    num_hours_total = int((beginning_of_y2 - beginning_of_y1).total_seconds() // 3600)

    # add column with hour list
    print("Add column with hour list...")
    df_list = df_year.merge(df_year.parallel_apply(make_list_hours,total_h=num_hours_total,axis=1),left_index=True, right_index= True)

    # split the hour list into seperate columns
    print("Split the hour list into seperate columns...")
    df1 = pd.DataFrame(df_list['hour_year_list'].tolist(),columns=list(range(num_hours_total)))
    df_complete = pd.concat([df_list, df1], axis=1)

    # clean up dataframe
    df_filtered = df_complete.copy()
    df_filtered.drop(columns=['hour_year_list'],inplace=True)
    df_filtered['cell'] = df_filtered['cell'].astype(int)

    # make sure we didn't select the wrong grid size
    if df_filtered['cell'].to_numpy().max() > grid_size**2-1:
        print("Incorrect grid_size given input data used!")
        break

    # get total number of crimes for that hour in each cell
    print("Make dataframe grouping by cell...")
    column_names = np.arange(num_hours_total).astype('int').tolist()
    df_grouped = df_filtered.groupby(['cell'])[column_names].sum()
    df_grouped = df_grouped.astype(int)

    # fill cells that aren't part of city with nan
    idx = pd.Series(list(range(0,grid_size**2)))
    df_final = df_grouped.reindex(idx)

    # make folder to save yearly file per city if it doesn't already exist
    os.makedirs(f"{in_out_folder}/Final_grid/", exist_ok=True)

    # save final dataset
    df_final.to_csv(f'{in_out_folder}/Final_grid/{city_folder}_{crime}_{y}_final_grid.csv')
    print("Final dataset saved!")

for crime_type in ['Burglary','Motor Vehicle Theft', 'Assault', 'Robbery', 'Homicide']:
  print(f"######### {crime_type} #########")
  make_final_grid_files_year_crime(city_folder='Baltimore',
                                   crime=crime_type,
                                   in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                                   years=[2019,2020,2021,2022,2023],
                                   grid_size=39)
  print("\n")

for crime_type in ['Burglary','Motor Vehicle Theft', 'Assault', 'Robbery', 'Homicide']:
  print(f"######### {crime_type} #########")
  make_final_grid_files_year_crime(city_folder='Chicago',
                                   crime=crime_type,
                                   in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                                   years=[2019,2020,2021,2022,2023],
                                   grid_size=85)
  print("\n")

for crime_type in ['Burglary','Motor Vehicle Theft', 'Robbery', 'Homicide','Assault']:
  print(f"######### {crime_type} #########")
  make_final_grid_files_year_crime(city_folder='Los_Angeles',
                                   crime=crime_type,
                                   in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                                   years=[2019,2020,2021,2022,2023],
                                   grid_size=133)
  print("\n")

for crime_type in ['Burglary','Motor Vehicle Theft', 'Robbery', 'Homicide','Assault']:
  print(f"######### {crime_type} #########")
  make_final_grid_files_year_crime(city_folder='Philadelphia',
                                   crime=crime_type,
                                   in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                                   years=[2019,2020,2021,2022,2023],
                                   grid_size=64)
  print("\n")

"""### 2. Make final grid matrix for each city for each crime all years together"""

def make_final_grid_city_crime(city_folder,crime,in_out_folder,years):
  # concat the grid for each year after renaming columns
  df_all_list = []
  for year in years:
    print(f"Doing year {year}...")
    df_year = pd.read_csv(f'{in_out_folder}/Final_grid/{city_folder}_{crime}_{year}_final_grid.csv',index_col=0)

    # rename columns to indicate year
    df_year.columns = list(map(lambda x: str(x) + f"_{str(year)}", df_year.columns.tolist()))
    print("size: ",df_year.shape)
    df_all_list.append(df_year)

  # concat the 4 years
  print("Concatenating and saving final dataframe...")
  df_all = pd.concat(df_all_list,axis=1)

  # save final output
  os.makedirs(f"{in_out_folder}/Final_all/", exist_ok=True)
  df_all.T.to_csv(f"{in_out_folder}/Final_all/{city_folder}_{crime}_all_final_grid.csv")
  print("Shape final dataframe: ", df_all.T.shape)
  print("File saved!\n")

for city_folder in ['Baltimore']:
  print(f"####### CITY: {city_folder} #######")
  for crime_type in ['Burglary','Motor Vehicle Theft', 'Assault', 'Robbery', 'Homicide']:
    print(f"### {crime_type} ###")
    make_final_grid_city_crime(city_folder=city_folder,
                               crime=crime_type,
                               in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                               years=[2019,2020,2021,2022,2023],
                               )
  print("\n")

for city_folder in ['Chicago']:
  print(f"####### CITY: {city_folder} #######")
  for crime_type in ['Burglary','Motor Vehicle Theft', 'Assault', 'Robbery', 'Homicide']:
    print(f"### {crime_type} ###")
    make_final_grid_city_crime(city_folder=city_folder,
                               crime=crime_type,
                               in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                               years=[2019,2020,2021,2022,2023],
                               )
  print("\n")

for city_folder in ['Los_Angeles']: # might need to run this one in a cluster if you run out of RAM
  print(f"####### CITY: {city_folder} #######")
  for crime_type in ['Burglary','Motor Vehicle Theft', 'Assault', 'Robbery', 'Homicide']:
    print(f"### {crime_type} ###")
    make_final_grid_city_crime(city_folder=city_folder,
                               crime=crime_type,
                               in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                               years=[2019,2020,2021,2022,2023],
                               )
  print("\n")

for city_folder in ['Philadelphia']:
  print(f"####### CITY: {city_folder} #######")
  for crime_type in ['Burglary','Motor Vehicle Theft', 'Assault', 'Robbery', 'Homicide']:
    print(f"### {crime_type} ###")
    make_final_grid_city_crime(city_folder=city_folder,
                               crime=crime_type,
                               in_out_folder='Crime_data_outputs/Grid_cells_0.2gu',
                               years=[2019,2020,2021,2022,2023],
                               )
  print("\n")

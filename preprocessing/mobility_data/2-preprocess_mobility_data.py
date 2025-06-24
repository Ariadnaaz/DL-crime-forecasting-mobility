import pandas as pd
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime, timedelta
import re

# Path to your folder 
folder_path = 'raw_data/'

# Function to generate the first Mondays of each week in the given date range
def generate_mondays(start_date, end_date):
    mondays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() == 0:  # Monday
            mondays.append(current_date)
        current_date += timedelta(days=7)  # Increment by 7 days to get the next Monday
    return mondays

# Generate the expected dates
start_date = datetime(2018, 12, 31)
end_date = datetime(2024, 1, 1)
mondays = generate_mondays(start_date, end_date)
dates_str = set(date.strftime('%Y-%m-%d') for date in mondays)  # Use a set for quick lookup

# Initialize a dictionary to store file names
file_names = {date: [] for date in dates_str}

# Function to count files per week
def file_names_per_week(directory, file_names):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            # Extract the date from the filename using regex
            match = re.search(r'DATE_RANGE_START-(\d{4}-\d{2}-\d{2})\.csv', filename)
            if match:
                date_str = match.group(1)
                if date_str in file_names:  # Quick lookup using set
                    file_names[date_str].append(filename)
    return file_names

# Get file names (usually needs to be executed several times to really load all the files)
dic_filenames = file_names_per_week(folder_path, file_names)

'''### Group files into one file per week for each year'''

# helper DF used to select only these cities
list_cities = [['Baltimore', 'MD'], ['Chicago', 'IL'], ['Los Angeles', 'CA'], ['Philadelphia', 'PA']]
drp = pd.DataFrame(list_cities, columns=['city', 'region']).assign(key=1)

columns_to_keep = ['placekey', 'parent_placekey', 'safegraph_brand_ids', 'location_name',
       'brands', 'store_id', 'top_category', 'sub_category', 'naics_code',
       'latitude', 'longitude', 'street_address', 'city', 'region',
       'postal_code', 'open_hours', 'category_tags', 'opened_on', 'closed_on',
       'tracking_closed_since', 'geometry_type', 'polygon_wkt',
       'polygon_class', 'enclosed', 'is_synthetic',
       'includes_parking_lot', 'wkt_area_sq_meters',
       'date_range_start', 'date_range_end', 'raw_visit_counts',
       'raw_visitor_counts', 'visits_by_day', 'visits_by_each_hour', 'poi_cbg']

# Function to process each file
def process_file(file_name):
    try:
        df = pd.read_csv(folder_path + "/" + file_name, compression="gzip",usecols=columns_to_keep)
        df_filtered = df.merge(drp, how='inner', on=['city', 'region'])
        #df_filtered.drop(columns=columns_to_drop, inplace=True)
        return df_filtered
    except (gzip.BadGzipFile, pd.errors.EmptyDataError) as e:
        print(f"Error processing file {file_name}: {e}")
        return None

# Function to process files for a week (is not iso week exactly, but files are correct)
def process_week(week,year):
    out = pd.to_datetime(f"{week}{year}Mon", format='%W%Y%a')
    date_week = str(out)[:10] # extracts the date
    print(f"Number of files for week {week}, so {date_week}:", len(dic_filenames[date_week]))

    df_all = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_name): file_name for file_name in dic_filenames[date_week]}

        # Using tqdm to display a progress bar for file processing
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing week {week}"):
            result = future.result()
            if result is not None:
                df_all.append(result)

    if df_all:
        df_week = pd.concat(df_all, ignore_index=True)
        print(np.shape(df_week))

        df_week.dropna(subset=['visits_by_each_hour'],inplace=True)
        print(np.shape(df_week))

        # clean up visits_by_each_hour and turn into a list of integers
        df_week['visits_by_each_hour'] = [str(x).replace('[','').replace(']','').replace('"','').replace('\\', '').split(",") for x in df_week['visits_by_each_hour']]
        df_week['visits_by_each_hour'] = df_week['visits_by_each_hour'].apply(lambda lst: list(map(int, lst)))

        # make column with number of hours in visits_by_each_hour to check if there are 168
        df_week['Number_hours'] = df_week['visits_by_each_hour'].apply(len)

        # keep only rows with 168 values in the visits_by_each_hour
        df_filtered = df_week[df_week['Number_hours'] == 168]
        df_filtered.reset_index(drop=True,inplace=True)
        print("Size dataset after keeping only entries with 168 hours: ", df_filtered.shape)

        # generate columns for each of the 168 hours of the week
        df1 = pd.DataFrame(df_filtered['visits_by_each_hour'].tolist(),columns=list(range(1,169)))

        # merge the two dataframes
        df_complete = pd.concat([df_filtered, df1], axis=1)
        os.makedirs("preprocessed_data/", exist_ok=True)
        df_complete.to_csv(f"preprocessed_data/Mobility_data_{year}_week_{week:02}.csv")
        print(f"Original size for week {week}: ", df_complete.shape)


for year in [2019,2020,2021,2022,2023]:
  for i in range(1,53):
      process_week(week=i,year=year)

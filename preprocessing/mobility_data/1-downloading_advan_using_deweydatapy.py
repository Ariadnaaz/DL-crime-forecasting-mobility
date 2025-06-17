import deweydatapy as ddp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gzip
import os
import pandas as pd
from datetime import datetime, timedelta
import re

# API Key
apikey_ = # put your own API Key

# Advan product path
pp_advan_wp = "https://app.deweydata.io/external-api/v3/products/50f21464-9d1b-4f44-89b5-f633ac31e64f/files"

meta = ddp.get_meta(apikey_, pp_advan_wp, print_meta = True)

# get the list of files withinn than time range
files_df = ddp.get_file_list(apikey_, pp_advan_wp,
                             start_date = '2018-12-31',
                             end_date = '2024-01-01',
                             print_info = True);

# download all the list of files
ddp.download_files(files_df, "raw_data/", skip_exists = True)


"""## If some files are incorrect and need to be downloaded again"""
# Specify the file path in your Google Drive
folder_path = 'raw_data/'

def find_wrong_files_and_redownload(folder_path, year, week):
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
    dates_str = [date.strftime('%Y-%m-%d') for date in mondays]

    # Initialize a dictionary to store file names
    file_names = {date: [] for date in dates_str}

    # Function to count files per week
    def file_names_per_week(directory, file_names):
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                match = re.search(r'DATE_RANGE_START-(\d{4}-\d{2}-\d{2})\.csv', filename)
                if match:
                    date_str = match.group(1)
                    if date_str in file_names:
                        file_names[date_str].append(filename)
        return file_names

    # Get file names
    dic_filenames = file_names_per_week(folder_path, file_names)

    # Get date for Monday for that week number of that year
    out = pd.to_datetime(pd.Series(week).astype(str) + str(year) + 'Mon', format='%W%Y%a')
    date_week = str(out.iloc[0])[:10]

    # Select file names for that week
    print(f"Number of files for week {week}, so {date_week}:", len(dic_filenames[date_week]))

    if len(dic_filenames[date_week]) > 0:
        wrong_files = []

        def check_file(file_name):
            try:
                pd.read_csv(folder_path + "/" + file_name, compression="gzip")
            except gzip.BadGzipFile:
                #print("Wrong file: ", file_name)
                return file_name
            return None

        # Using ThreadPoolExecutor to check files in parallel
        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(check_file, file_name): file_name for file_name in dic_filenames[date_week]}

            # Using tqdm to display a progress bar
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Checking files"):
                file_name = future.result()
                if file_name is not None:
                    wrong_files.append(file_name)

        if wrong_files:
            print(f"Start removal of {len(wrong_files)} files...")

            # Removing files in parallel
            with ThreadPoolExecutor() as executor:
                executor.map(lambda file_name: os.remove(folder_path + file_name) if os.path.exists(folder_path + file_name) else None, wrong_files)

            print(f'{len(wrong_files)} files have been deleted.')
            # Trigger download process for the wrong files
            ddp.download_files(files_df, "raw_data/", skip_exists=True)
        else:
            print("No files to remove")

for year in [2019,2020,2021,2022,2023]:
    for i in range(0,53):
        find_wrong_files_and_redownload(folder_path='raw_data/',
                                    year=year,
                                    week=i)

import pandas as pd
import numpy as np

"""## Select only the 6 crime types and make naming compatible for all 5 cities

### 1. Baltimore
"""

city_folder = 'Baltimore'

# load dataset clean
df_clean = pd.read_csv(f'Preprocessing_raw_data_per_city/{city_folder}/{city_folder}_crimes_clean_all_5ys.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

list_crimes = ['AGG. ASSAULT','COMMON ASSAULT','AUTO THEFT','BURGLARY','HOMICIDE','ROBBERY']

# keep only rows with those 6 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
df_filtered['crime_type'].replace("AUTO THEFT","Motor Vehicle Theft",inplace=True)
df_filtered['crime_type'].replace("AGG. ASSAULT","Assault",inplace=True)
df_filtered['crime_type'].replace("COMMON ASSAULT","Assault",inplace=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()


# save final dataset
os.makedirs('Crime_data_outputs/5_years/', exist_ok=True)
df_filtered.to_csv(f'Crime_data_outputs/5_years/{city_folder}_selected_crimes_clean_all.csv')
print("Saved dataset")
print(df_filtered['crime_type'].unique().tolist())

"""### 2. Chicago"""

city_folder = 'Chicago'

# load dataset clean
df_clean = pd.read_csv(f'Preprocessing_raw_data_per_city/{city_folder}/{city_folder}_crimes_clean_all_5ys.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

list_crimes = ['ASSAULT','MOTOR VEHICLE THEFT','BURGLARY','HOMICIDE','ROBBERY']

# keep only rows with those 6 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()

# save final dataset
df_filtered.to_csv(f'Crime_data_outputs/5_years/{city_folder}_selected_crimes_clean_all.csv')
print("Saved dataset")
print(df_filtered['crime_type'].unique().tolist())

"""### 3. Los Angeles"""

city_folder = 'Los_Angeles'

# load dataset clean
df_clean = pd.read_csv(f'Preprocessing_raw_data_per_city/{city_folder}/{city_folder}_crimes_clean_all_5ys.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

list_crimes = ['VEHICLE - STOLEN','BURGLARY','ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT','INTIMATE PARTNER - AGGRAVATED ASSAULT','INTIMATE PARTNER - SIMPLE ASSAULT','BATTERY - SIMPLE ASSAULT','CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT','CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT','ROBBERY','ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER','CRIMINAL HOMICIDE','MANSLAUGHTER, NEGLIGENT']

# keep only rows with those 6 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
assault_list = '|'.join(['ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT','INTIMATE PARTNER - AGGRAVATED ASSAULT','INTIMATE PARTNER - SIMPLE ASSAULT','BATTERY - SIMPLE ASSAULT','CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT','CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT','ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER'])
homicide_list = '|'.join(['CRIMINAL HOMICIDE','MANSLAUGHTER, NEGLIGENT'])
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(assault_list,"Assault",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(homicide_list,"Homicide",regex=True)
df_filtered['crime_type'].replace("VEHICLE - STOLEN","Motor Vehicle Theft",inplace=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.title() # converting each string to title case
df_filtered['crime_type'].replace('Child Abuse (Physical) - Simple Assault',"Assault",inplace=True)
df_filtered['crime_type'].replace('Child Abuse (Physical) - Aggravated Assault',"Assault",inplace=True)

# save final dataset
df_filtered.to_csv(f'Crime_data_outputs/5_years/{city_folder}_selected_crimes_clean_all.csv')
print("Saved dataset")
print(df_filtered['crime_type'].unique().tolist())

"""### 4. Philadelphia"""

city_folder = 'Philadelphia'

# load dataset clean
df_clean = pd.read_csv(f'Preprocessing_raw_data_per_city/{city_folder}/{city_folder}_crimes_clean_all_5ys.csv',low_memory=False,index_col=0)
print("Shape clean dataset: ", df_clean.shape)

# get list of crime categories
group = df_clean['crime_type'].unique().tolist()
print(group)

list_crimes = ['Burglary Non-Residential', 'Aggravated Assault No Firearm', 'Aggravated Assault Firearm', 'Burglary Residential', 'Robbery No Firearm', 'Robbery Firearm', 'Other Assaults','Motor Vehicle Theft', 'Homicide - Criminal ', 'Homicide - Criminal', 'Homicide - Justifiable ', 'Homicide - Gross Negligence']

# keep only rows with those 5 crimes
df_filtered = df_clean.copy()
df_filtered = df_filtered[df_filtered['crime_type'].isin(list_crimes)]
print("Shape after selecting the 5 types of crimes: ", df_filtered.shape)

# make naming consistent
assault_list = '|'.join([ 'Aggravated Assault No Firearm', 'Aggravated Assault Firearm','Other Assaults'])
homicide_list = '|'.join(['Homicide - Criminal ', 'Homicide - Criminal', 'Homicide - Justifiable ','Homicide - Justifiable', 'Homicide - Gross Negligence'])
burglary_list = '|'.join(['Burglary Non-Residential','Burglary Residential'])
robbery_list = '|'.join(['Robbery No Firearm','Robbery Firearm'])
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(assault_list,"Assault",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(homicide_list,"Homicide",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(burglary_list,"Burglary",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.replace(robbery_list,"Robbery",regex=True)
df_filtered['crime_type'] = df_filtered['crime_type'].str.title()

# save final dataset
df_filtered.to_csv(f'Crime_data_outputs/5_years/{city_folder}_selected_crimes_clean_all.csv')
print("Saved dataset")
print(df_filtered['crime_type'].unique().tolist())

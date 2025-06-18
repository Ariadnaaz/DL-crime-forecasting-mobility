import pandas as pd
from shapely.geometry import Point
import shapely.geometry
import datetime
import geopandas as gpd
import numpy as np
import folium
import pickle
from IPython.display import display
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

def generate_grid(selected_city,grid_size,plot=True):
  gdf = extract_multipolygon_city(file_path='../city_multipolygons.geojson',city_name=selected_city)

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

def plot_grid_and_point(city_name,point_lat,point_long):
  # generate grid boxes
  print("Generate grid boxes...")
  min_lon, min_lat, max_lon, max_lat, grid_bboxes = generate_grid(selected_city=city_name,grid_size=50,plot=False)
  m = folium.Map(location=((min_lat+max_lat)/2,(min_lon+max_lon)/2), zoom_start=11)

  # turn boxes into polygons and that list into a dataframe
  polygon_list = []
  for elem in grid_bboxes:
    polygon = shapely.geometry.box(elem[1],elem[0],elem[3],elem[2],ccw=True)
    polygon_list.append(polygon)
  df_poly = pd.DataFrame(polygon_list,columns=['polygon'])
  print("Shape cell polygons dataframe: ", df_poly.shape)

  for elem in polygon_list:
    geo_j = folium.GeoJson(data=elem)
    geo_j.add_to(m)

  folium.CircleMarker([point_lat,point_long],color='red',radius=0.5).add_to(m)
  display(m)

# create full dataset of unique POIs for a city
def make_list_unique_pois(city_selected,input_folder):
  df1 = pd.read_csv(f'preprocessed_data/{input_folder}/Mobility_data_2019_week_01.csv',index_col=0)
  df1.set_index('placekey',inplace=True,drop=True)
  # select only rows for that city
  df1 = df1[df1['city'] == city_selected]
  # keep only location coordinate (we don't care about other variables)
  df1 = df1[['latitude','longitude']]
  for year in ['2019','2020','2021','2022','2023']:
    print(f"Doing year {year}...")
    for week in range(1,53):
      #print("week: ",week)
      df2 = pd.read_csv(f'preprocessed_data/{input_folder}/Mobility_data_{year}_week_{week:02}.csv',index_col=0)
      #df2.columns = map(str.upper, df2.columns) # in case column names are not all caps
      #print(df2.columns)
      df2.set_index('placekey',inplace=True,drop=True)
      df2 = df2[df2['city'] == city_selected]
      df2 = df2[['latitude','longitude']]
      # merge new dataframe with previous one and drop duplicates
      df1 = pd.concat([df1,df2]).drop_duplicates(keep='first')
    print(f"Shape df1 after {year}: ", df1.shape)
  return df1

def find_polygon_mobility(row,df):
    for p_idx,elem in df.iterrows():
      #print(elem['polygon'])
      if elem['polygon'].contains(Point(row['longitude'],row['latitude'])):
        polygon_num = p_idx
        break
      else:
        polygon_num = np.nan
    return pd.Series({'cell': polygon_num})

"""### 1. Generate mapping between pois and grid num (run in cluster, takes too long to run here 7h+)"""

def generate_mobility_pois_to_grid_num_map(city_name,city_folder,input_folder,output_folder,grid_size=50): # needs the weekly files of all years

  # create full dataset of unique POIs for this city
  print("Generate dataframe of unique pois... ")
  df_pois = make_list_unique_pois(city_selected=city_name,input_folder=input_folder)
  df_pois.reset_index(inplace=True)
  print("Size pois dataset: ", df_pois.shape)

  # generate grid boxes
  print("Generate grid boxes...")
  min_lon, min_lat, max_lon, max_lat, grid_bboxes = generate_grid(selected_city=city_name,grid_size=grid_size,plot=False)

  # turn boxes into polygons and that list into a dataframe
  polygon_list = []
  for elem in grid_bboxes:
    polygon = shapely.geometry.box(elem[1],elem[0],elem[3],elem[2],ccw=True)
    polygon_list.append(polygon)
  df_poly = pd.DataFrame(polygon_list,columns=['polygon'])
  print("Shape cell polygons dataframe: ", df_poly.shape)

  print("Finding the grid cell for each POI...")
  test = df_pois.parallel_apply(find_polygon_mobility,df=df_poly,axis=1)
  df_final = df_pois.merge(test,left_index=True, right_index= True)
  print("Shape dataframe after adding cell number: ", df_final.shape)

  print("Generating mapping dictionary")
  mapping = dict(df_final[['placekey', 'cell']].values)

  # make final output folder if it doesn't exist
  Path(f"drive/MyDrive/PhD_thesis/Mobility_data/preprocessed_data/{output_folder}").mkdir(exist_ok=True,parents=True)

  with open(f'preprocessed_data/{output_folder}/{city_folder}_mobility_all_grid_map.pkl', 'wb') as fp:
    pickle.dump(mapping, fp)
    print('POI to grid cell dictionary saved successfully to file!')

generate_mobility_pois_to_grid_num_map(city_name='Baltimore',city_folder='Baltimore',
                                       input_folder='Data_preprocessed_02082024',
                                       output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                       grid_size=39
                                       )

generate_mobility_pois_to_grid_num_map(city_name='Chicago',city_folder='Chicago',
                                       input_folder='Data_preprocessed_02082024',
                                       output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                       grid_size=85
                                       )

generate_mobility_pois_to_grid_num_map(city_name='Los Angeles',city_folder='Los_Angeles',
                                       input_folder='Data_preprocessed_02082024',
                                       output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                       grid_size=133
                                       )

generate_mobility_pois_to_grid_num_map(city_name='Philadelphia',city_folder='Philadelphia',
                                       input_folder='Data_preprocessed_02082024',
                                       output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                       grid_size=64
                                       )


"""### 2. Create files with grid cells per year (pre-grids)"""

def add_poi_category_column(df_initial):
    df = df_initial.copy()

    group_to_poi = {
        "const": [ # Utilities and Construction
            'Building Material and Supplies Dealers',
            'Foundation, Structure, and Building Exterior Contractors',
            'Building Equipment Contractors',
            'Building Finishing Contractors',
            'Other Specialty Trade Contractors',
            'Utility System Construction',
            'Electric Power Generation, Transmission and Distribution',
        ],
        "manu": [ # Manufacturing
            'Bakeries and Tortilla Manufacturing',
            'Beverage Manufacturing',
            'Other Miscellaneous Manufacturing',
            'Glass and Glass Product Manufacturing',
            'Other Wood Product Manufacturing',
            'Metalworking Machinery Manufacturing',
            'Steel Product Manufacturing from Purchased Steel',
            'Greenhouse, Nursery, and Floriculture Production',
            'Printing and Related Support Activities'
        ],
        "sale": [ # Retail and Wholesale trade
            'Shoe Stores',
            'Grocery Stores',
            'Clothing Stores',
            'Electronics and Appliance Stores',
            'Health and Personal Care Stores',
            'Gasoline Stations',
            'Used Merchandise Stores',
            'Automotive Parts, Accessories, and Tire Stores',
            'Jewelry, Luggage, and Leather Goods Stores',
            'Beer, Wine, and Liquor Stores',
            'Specialty Food Stores',
            'Home Furnishings Stores',
            'Other Motor Vehicle Dealers',
            'Office Supplies, Stationery, and Gift Stores',
            'General Merchandise Stores, including Warehouse Clubs and Supercenters',
            'Department Stores',
            'Book Stores and News Dealers',
            'Automobile Dealers',
            'Furniture Stores',
            'Lawn and Garden Equipment and Supplies Stores',
            'Hardware, and Plumbing and Heating Equipment and Supplies Merchant Wholesalers',
            'Machinery, Equipment, and Supplies Merchant Wholesalers',
            "Drugs and Druggists' Sundries Merchant Wholesalers",
            'Chemical and Allied Products Merchant Wholesalers',
            'Petroleum and Petroleum Products Merchant Wholesalers',
            'Motor Vehicle and Motor Vehicle Parts and Supplies Merchant Wholesalers',
            'Miscellaneous Durable Goods Merchant Wholesalers',
            'Grocery and Related Product Merchant Wholesalers',
            'Household Appliances and Electrical and Electronic Goods Merchant Wholesalers',
            'Lumber and Other Construction Materials Merchant Wholesalers',
            'Direct Selling Establishments',
            'Other Miscellaneous Store Retailers',
            'Professional and Commercial Equipment and Supplies Merchant Wholesalers',
            'Sporting Goods, Hobby, and Musical Instrument Stores',
        ],
        "transp": [ # Transportation and warehousing
            'Support Activities for Air Transportation',
            'Specialized Freight Trucking',
            'Rail Transportation',
            'Taxi and Limousine Service',
            'Other Transit and Ground Passenger Transportation',
            'Transit and Ground Passenger Transportation',
            'Scenic and Sightseeing Transportation',
            'Support Activities for Road Transportation',
            'Freight Transportation Arrangement',
            'Support Activities for Water Transportation',
            'Warehousing and Storage',
            'Automotive Equipment Rental and Leasing',
            'Commercial and Industrial Machinery and Equipment Rental and Leasing',
            'Interurban and Rural Bus Transportation',
            'Postal Service'
        ],
        "services": [ # Business and Professional Services
            'Investigation and Security Services',
            'Activities Related to Credit Intermediation',
            'Offices of Real Estate Agents and Brokers',
            'Other Professional, Scientific, and Technical Services',
            'Accounting, Tax Preparation, Bookkeeping, and Payroll Services',
            'Management, Scientific, and Technical Consulting Services',
            'Advertising, Public Relations, and Related Services',
            'Legal Services',
            'Data Processing, Hosting, and Related Services',
            'Architectural, Engineering, and Related Services',
            'Specialized Design Services',
            'Business Support Services',
            'Employment Services',
            'Travel Arrangement and Reservation Services',
            'Freight Transportation Arrangement',
            'Management of Companies and Enterprises',
            'Activities Related to Real Estate',
            'Agencies, Brokerages, and Other Insurance Related Activities',
            'Cable and Other Subscription Programming',
            'Consumer Goods Rental',
            'Depository Credit Intermediation',
            'General Rental Centers',
            'Insurance Carriers',
            'Lessors of Real Estate',
            'Motion Picture and Video Industries',
            'Nondepository Credit Intermediation',
            'Other Financial Investment Activities',
            'Other Information Services',
            'Radio and Television Broadcasting',
            'Sound Recording Industries',
            'Wired and Wireless Telecommunications Carriers'
        ],
        "educ": [ # Educational services
            'Elementary and Secondary Schools',
            'Colleges, Universities, and Professional Schools',
            'Junior Colleges',
            'Technical and Trade Schools',
            'Other Schools and Instruction',
            'Educational Support Services'
        ],
        "health": [ # Health care and social assistance
            'Offices of Physicians',
            'Offices of Other Health Practitioners',
            'Specialty (except Psychiatric and Substance Abuse) Hospitals',
            'General Medical and Surgical Hospitals',
            'Psychiatric and Substance Abuse Hospitals',
            'Outpatient Care Centers',
            'Nursing Care Facilities (Skilled Nursing Facilities)',
            'Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly',
            'Home Health Care Services',
            'Individual and Family Services',
            'Community Food and Housing, and Emergency and Other Relief Services',
            'Residential Intellectual and Developmental Disability, Mental Health, and Substance Abuse Facilities',
            'Child Day Care Services',
            'Medical and Diagnostic Laboratories',
            'Nursing and Residential Care Facilities',
            'Offices of Dentists',
            'Other Ambulatory Health Care Services'
        ],
        "recr": [ # Arts, entertainment, and recreation
            'Museums, Historical Sites, and Similar Institutions',
            'Amusement Parks and Arcades',
            'Spectator Sports',
            'Other Amusement and Recreation Industries',
            'Performing Arts Companies',
            'Promoters of Performing Arts, Sports, and Similar Events',
            'Social Advocacy Organizations',
            'Civic and Social Organizations',
            'Gambling Industries'
        ],
        "food": [ # Accommodation and food services
            'Traveler Accommodation',
            'Special Food Services',
            'Restaurants and Other Eating Places',
            'Drinking Places (Alcoholic Beverages)',
            'RV (Recreational Vehicle) Parks and Recreational Camps'
        ],
        "public": [ # Public administration
            'Justice, Public Order, and Safety Activities',
            'Administration of Economic Programs',
            'Administration of Human Resource Programs',
            'National Security and International Affairs'
        ],
        "other": [ # Other services
            'Florists',
            'Other Personal Services',
            'Religious Organizations',
            'Personal and Household Goods Repair and Maintenance',
            'Drycleaning and Laundry Services',
            'Death Care Services',
            'Personal Care Services',
            'Social Assistance',
            'Grantmaking and Giving Services',
            'Couriers and Express Delivery Services',
            'Waste Management and Remediation Services',
            'Remediation and Other Waste Management Services',
            'Services to Buildings and Dwellings',
            'Waste Treatment and Disposal',
            'Waste Collection',
            'Automotive Repair and Maintenance',
            'Electronic and Precision Equipment Repair and Maintenance',
        ]
    }

    # Invert the dictionary to create POI to Group mapping
    poi_to_group = {poi: group for group, pois in group_to_poi.items() for poi in pois}

    # Map the POI categories to their groups
    df['poi_category'] = df['top_category'].map(poi_to_group).fillna('other')

    return df

def hour_of_week(year,df):
  beginning_of_year = datetime.datetime(year, 1, 1)
  # had to add .tz_localize(None) bc otherwise it gave me the time offset compared to UTC time
  start_week = df['date_range_start'].iloc[0].tz_localize(None)
  end_week = df['date_range_end'].iloc[0].tz_localize(None)
  start = int((start_week - beginning_of_year).total_seconds() // 3600)
  end = int((end_week - beginning_of_year).total_seconds() // 3600)
  return (start,end)

def make_grid_files_year_mobility_poi_categ(city_name,city_folder,path_map,input_folder,output_folder,grid_size,poi_categ): # make sure it's correct when using again (had to rewrite bc it didn't save)
  # we assume output folder is inside input folder
  # Read dictionary pkl file
  with open(f'preprocessed_data/{path_map}/{city_folder}_mobility_all_grid_map.pkl', 'rb') as fp:
      map_dic = pickle.load(fp)
  map_dic

  for year in [2019,2020,2021,2022,2023]:
    print(f"Doing year {year}...")

    # load rows for that week for the first week
    df1 = pd.read_csv(f'preprocessed_data/{input_folder}/Mobility_data_{year}_week_01.csv',index_col=0,parse_dates=['date_range_start','date_range_end'])
    df1 = add_poi_category_column(df1)
    #print("Original size: ", df1.shape)

    if poi_categ == None:# in case we want our original mobility feature
        df1 = df1[df1['city']==city_name]
    else:
        df1 = df1[(df1['city']==city_name) & (df1['poi_category'] == poi_categ)]
    print("Size after selecting city and categ: ",df1.shape)
    #print(df1.head(5))

    # add cell number and only keep rows with cell num
    df1['cell'] = df1['placekey'].map(map_dic)
    df1.dropna(subset=['cell'],inplace=True)
    df1['cell'] = df1['cell'].astype('int')

    # group by cell and sum values in the visits per hour
    df_grouped = df1.groupby(['cell'])[np.arange(1,169).astype('str').tolist()].sum()
    df_grouped = df_grouped.astype(int)
    # rename columns
    start, end = hour_of_week(year,df1)
    df_grouped.columns = list(range(start,end))
    # fill cells that aren't part of city with nan
    idx = pd.Series(list(range(0,grid_size**2)))
    df1 = df_grouped.reindex(idx)

    for week in range(2,53):
      # load rows for that week for the first week
      df2 = pd.read_csv(f'preprocessed_data/{input_folder}/Mobility_data_{year}_week_{week:02}.csv',index_col=0,parse_dates=['date_range_start','date_range_end'])
      df2 = add_poi_category_column(df2)

      if poi_categ == None:# in case we want our original mobility feature
        df2 = df2[df2['city']==city_name]
      else: # we select only the entries belonging to a certain city and certain poi category
        df2 = df2[(df2['city']==city_name) & (df2['poi_category'] == poi_categ)]

      # add cell number and only keep rows with cell num
      df2['cell'] = df2['placekey'].map(map_dic)
      df2.dropna(subset=['cell'],inplace=True)
      df2['cell'] = df2['cell'].astype('int')
      # group by cell and sum values in the visits per hour
      df_grouped = df2.groupby(['cell'])[np.arange(1,169).astype('str').tolist()].sum()
      df_grouped = df_grouped.astype(int)
      # rename columns
      start, end = hour_of_week(year,df2)
      df_grouped.columns = list(range(start,end))
      # fill cells that aren't part of city with nan
      idx = pd.Series(list(range(0,grid_size**2)))
      df2 = df_grouped.reindex(idx)
      df1 = pd.concat([df1,df2],axis=1) # this concat assumes cell was the index column

    df_year = df1.copy()
    df_year.to_csv(f"preprocessed_data/{output_folder}/{city_folder}_mobility_{year}_{poi_categ}_pre_grid.csv")
    print("Shape final file: ",df_year.shape)
    print("Saved file!")

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_grid_files_year_mobility_poi_categ(city_name='Baltimore',
                                city_folder='Baltimore',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=39,
                                poi_categ=categ)

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_grid_files_year_mobility_poi_categ(city_name='Chicago',
                                city_folder='Chicago',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=85,
                                poi_categ=categ)

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_grid_files_year_mobility_poi_categ(city_name='Los Angeles',
                                city_folder='Los_Angeles',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=133,
                                poi_categ=categ)

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_grid_files_year_mobility_poi_categ(city_name='Philadelphia',
                                city_folder='Philadelphia',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=64,
                                poi_categ=categ)

"""### 2.2 Create pre-grid for diversity feature"""

def calculate_shannon_indexes(df_orig):
    df=df_orig.copy()

    # Step 1: Group by 'cell' and 'poi_category' to count occurrences
    poi_counts = df.groupby(['cell', 'poi_category']).size().reset_index(name='count')

    # Step 2: Calculate the total count of poi_category within each cell
    poi_counts['total'] = poi_counts.groupby('cell')['count'].transform('sum')

    # Step 3: Calculate the proportion for each poi_category within each cell
    poi_counts['proportion'] = poi_counts['count'] / poi_counts['total']

    # Step 4: Calculate the Shannon Diversity Index for each cell
    poi_counts['shannon'] = poi_counts['proportion'] * np.log(poi_counts['proportion'])
    shannon_diversity = poi_counts.groupby('cell')['shannon'].sum().reset_index()

    # Step 5: Negate the value for the Shannon Diversity Index
    shannon_diversity['shannon_diversity'] = -shannon_diversity['shannon']

    df_results = shannon_diversity[['cell', 'shannon_diversity']].set_index('cell')

    # Result: df_results contains unique cells with both indices as columns
    return df_results

def hour_of_week(year,df):
  beginning_of_year = datetime.datetime(year, 1, 1)
  # had to add .tz_localize(None) bc otherwise it gave me the time offset compared to UTC time
  start_week = df['date_range_start'].iloc[0].tz_localize(None)
  end_week = df['date_range_end'].iloc[0].tz_localize(None)
  start = int((start_week - beginning_of_year).total_seconds() // 3600)
  end = int((end_week - beginning_of_year).total_seconds() // 3600)
  return (start,end)

def make_grid_files_year_mobility_poi_diversity(city_name,city_folder,path_map,input_folder,output_folder,grid_size):
  # Read dictionary pkl file
  with open(f'preprocessed_data/{path_map}/{city_folder}_mobility_all_grid_map.pkl', 'rb') as fp:
      map_dic = pickle.load(fp)
  map_dic

  for year in [2019,2020,2021,2022,2023]:
    print(f"Doing year {year}...")
    print("Week: 1")
    # load rows for that week for the first week
    df1 = pd.read_csv(f'preprocessed_data/{input_folder}/Mobility_data_{year}_week_01.csv',index_col=0,parse_dates=['date_range_start','date_range_end'])
    # select only entries for the city we want
    df1 = df1[df1['city']==city_name]
    # add column with poi category based on top_category column
    df1 = add_poi_category_column(df1)
    # add cell number and only keep rows with cell num
    df1['cell'] = df1['placekey'].map(map_dic)
    df1.dropna(subset=['cell'],inplace=True)
    df1['cell'] = df1['cell'].astype('int')

    # get the shannon index for each cell and remove negative 0
    df_grouped = calculate_shannon_indexes(df1)
    df_grouped = df_grouped.applymap(lambda x: 0.000000 if x == -0.000000 else x)

    # repeat the weekly data across 168 hours (7 days * 24 hours)
    df_grouped = pd.concat([df_grouped] * 168, axis=1, ignore_index=True)

    # rename columns to reflext hour
    start, end = hour_of_week(year,df1)
    df_grouped.columns = list(range(start,end))

    # fill cells that aren't part of city with nan
    idx = pd.Series(list(range(0,grid_size**2)))
    df1 = df_grouped.reindex(idx)

    for week in range(2,53):
      print("week:", week)
      # load rows for that week for the first week
      df2 = pd.read_csv(f'preprocessed_data/{input_folder}/Mobility_data_{year}_week_{week:02}.csv',index_col=0,parse_dates=['date_range_start','date_range_end'])
      df2 = add_poi_category_column(df2)

      # select only entries for the city we want
      df2 = df2[df2['city']==city_name]

      # add cell number and only keep rows with cell num
      df2['cell'] = df2['placekey'].map(map_dic)
      df2.dropna(subset=['cell'],inplace=True)
      df2['cell'] = df2['cell'].astype('int')

      # get the shannon index for each cell and remove negative 0
      df_grouped = calculate_shannon_indexes(df2)
      df_grouped = df_grouped.applymap(lambda x: 0.000000 if x == -0.000000 else x)

      # repeat the weekly data across 168 hours (7 days * 24 hours)
      df_grouped = pd.concat([df_grouped] * 168, axis=1, ignore_index=True)

      # rename columns to reflext hour
      start, end = hour_of_week(year,df2)
      df_grouped.columns = list(range(start,end))

      # fill cells that aren't part of city with nan
      idx = pd.Series(list(range(0,grid_size**2)))
      df2 = df_grouped.reindex(idx)

      # concat with previous weeks (merge on columns, so horizontal merge)
      df1 = pd.concat([df1,df2],axis=1)

    df_year = df1.copy() 
    print(df_year.shape)
    df_year.to_csv(f"preprocessed_data/{output_folder}/{city_folder}_mobility_{year}_diversity_pre_grid.csv")
    print("Saved file!")

make_grid_files_year_mobility_poi_diversity(city_name='Baltimore',
                                city_folder='Baltimore',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=39)

make_grid_files_year_mobility_poi_diversity(city_name='Chicago',
                                city_folder='Chicago',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=85)

make_grid_files_year_mobility_poi_diversity(city_name='Los Angeles',
                                city_folder='Los_Angeles',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=133)

make_grid_files_year_mobility_poi_diversity(city_name='Philadelphia',
                                city_folder='Philadelphia',
                                path_map='Data_preprocessed_02082024/Grid_cells_0.2gu/maps',
                                input_folder='Data_preprocessed_02082024',
                                output_folder='Data_preprocessed_02082024/Grid_cells_0.2gu/Pre_grid',
                                grid_size=64)

"""### 3. Create files final grid per city all years together (for poi_categ or diversity features)"""

def make_final_grid_city_mobility(city_folder,in_out_folder,poi_categ):
  #extra_hours_lastyear = 0 # for 2022 that is 24 bc there is one day of 2023 in the last week. For 2023 that is 0

  # concat the grid for each year after renaming columns
  df_all_list = []
  for year in [2019,2020,2021,2022,2023]:
    print(f"Doing year {year}...")
    df_year = pd.read_csv(f'preprocessed_data/{in_out_folder}/Pre_grid/{city_folder}_mobility_{year}_{poi_categ}_pre_grid.csv',index_col=0)

    # calculate extra hours at the end of each year
    last_day = datetime.datetime(year, 12, 31)
    week1_next_year = pd.to_datetime(pd.Series(1).astype(str)+str(year+1)+'Mon', format='%W%Y%a')[0]
    extra_hours = int((week1_next_year - last_day).total_seconds() // 3600) - 24 # because it counts one day extra

    if extra_hours != 0:
        # rename columns to indicate year and correct index for the one of the next year
        new_col_names1 = list(map(lambda x: str(x) + f"_{str(year)}", df_year.columns[:-extra_hours].tolist()))
        new_col_names2 = list(map(lambda x: str(x) + f"_{str(year+1)}", list(range(0,extra_hours))))
        df_year.columns = new_col_names1 + new_col_names2
    else: # because using -0 in the slizing selects zero elements in total
        # rename columns to indicate year in case of no extra hours from following year
        new_col_names1 = list(map(lambda x: str(x) + f"_{str(year)}", df_year.columns.tolist()))
        df_year.columns = new_col_names1

    print(df_year.shape)
    df_all_list.append(df_year)

  print("Concatenating and saving final dataframe...")
  df_all = pd.concat(df_all_list,axis=1)
  #df_all = df_all.iloc[:, :-extra_hours_lastyear] # remove extra hours from the next year (not necessary if last year is 2023)

  df_all.T.to_csv(f"preprocessed_data/{in_out_folder}/Final_all/{city_folder}_mobility_all_{poi_categ}_final_grid.csv")
  print("Shape final dataframe: ", df_all.T.shape)

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_final_grid_city_mobility(city_folder='Baltimore',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ=categ) # size of pre-grid files should be (grid_size^2, 8736)

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_final_grid_city_mobility(city_folder='Chicago',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ=categ)

# This one needs to runn in the cluster because it crashes at the final concat!
for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_final_grid_city_mobility(city_folder='Los_Angeles',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ=categ)

for categ in ['const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
    print("## CATEGORY: ", categ)
    make_final_grid_city_mobility(city_folder='Philadelphia',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ=categ)

# code for diversity (could also put it together with the others, but I have it seperate bc I created the pre-grids later)
make_final_grid_city_mobility(city_folder='Baltimore',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ='diversity')

make_final_grid_city_mobility(city_folder='Chicago',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ='diversity')

# session crashes bc of using all available RAM with this one - might need to run it in a cluster
make_final_grid_city_mobility(city_folder='Los_Angeles',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ='diversity')

make_final_grid_city_mobility(city_folder='Philadelphia',
                                in_out_folder='Data_preprocessed_02082024/Grid_cells_0.2gu',
                                poi_categ='diversity')

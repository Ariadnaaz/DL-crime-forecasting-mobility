import numpy as np
import pandas as pd
import sys
import time
import os

## Useful functions ##

# normalize the data
def NormalizeData(data):
  n = (data - np.min(data)) / (np.max(data) - np.min(data))
  if np.isnan(n).all(): # because if we have all 0's then it turn into array of nan
    n = np.nan_to_num(n)
  return n

def NormalizeDataLocal(data,mini,maxi): # can use this if code is too slow
  n = (data - mini) / (maxi - mini)
  if np.isnan(n).all(): # because if we have all 0's then it turn into array of nan
    n = np.nan_to_num(n)
  return n

## Put data into the correct shape
def extract_one_variable(df,variable,grid_size,normalize=False,mini=None,maxi=None):
  df_var_list = df.loc[variable].values.tolist()
  df_var_list.reverse()
  m = np.array(df_var_list).reshape(grid_size,grid_size,1).T # populate in correct order and add n_samples dimension
  if normalize:
    m = NormalizeDataLocal(m,mini,maxi)
  return np.expand_dims(m, axis=-1) # we add the channel dimension

## Make full matrix for each hour
def make_full_matrix_per_hour(hour_var,df_cri,df_mob_list,df_soc,grid_size):

  # extract the year from the hour variable
  year = hour_var[-4:]

  # correct in case of 2022 or 2023, since we will use sociodem data for 2021 
  if year == '2022' or year=='2023':
    year = '2021'

  list_ch = []

  # append the crime data for that variable
  crime_m = extract_one_variable(df=df_cri,variable=hour_var,grid_size=grid_size,normalize=False)
  list_ch.append(crime_m)

  # append the mobility data for that variable 
  for df_mob in df_mob_list:
    list_ch.append(extract_one_variable(df=df_mob,variable=hour_var,grid_size=grid_size,normalize=False))

  # find the sociodemographic variables correspodning to that year
  soc_var_year_list = df_soc[[year in x for x in df_soc.index]].index.tolist()

  # append all 26 sociodemographic variables
  for var in soc_var_year_list:
    list_ch.append(extract_one_variable(df=df_soc,variable=var,grid_size=grid_size,normalize=False))

  # axis=3 means we merge in the fourth dimension
  final = np.concatenate(list_ch,axis=3)
  return final

def load_preprocessed_data(city_folder,gu,crime_types_list,granularity):

    if granularity == 24:
      # group in sets of 24h (so daily granularity)
      index_19 =[f'{x}_2019' for x in range(6,365)]
      index_20 =[f'{x}_2020' for x in range(0,366)]
      index_21 =[f'{x}_2021' for x in range(0,365)]
      index_22 =[f'{x}_2022' for x in range(0,365)]
      index_23 =[f'{x}_2023' for x in range(0,365)]
    elif granularity == 12:
      # group in sets of 12h
      index_19 =[f'{x}_2019' for x in range(12,730)]
      index_20 =[f'{x}_2020' for x in range(0,732)]
      index_21 =[f'{x}_2021' for x in range(0,730)]
      index_22 =[f'{x}_2022' for x in range(0,730)]
      index_23 =[f'{x}_2023' for x in range(0,730)]
    elif granularity == 8:
      # group in sets of 8h
      index_19 =[f'{x}_2019' for x in range(18,1095)]
      index_20 =[f'{x}_2020' for x in range(0,1098)]
      index_21 =[f'{x}_2021' for x in range(0,1095)]
      index_22 =[f'{x}_2022' for x in range(0,1095)]
      index_23 =[f'{x}_2023' for x in range(0,1095)]

    idx_list = index_19 + index_20 + index_21 + index_22 + index_23

    df_cri_list = []
    for crime_type in crime_types_list:
        df_cri_temp = pd.read_csv(f"../../preprocessing/crime_data/Crime_data_outputs/Grid_cells_{gu}gu/Final_all/{city_folder}_{crime_type}_all_final_grid.csv",index_col=0)
        df_cri_temp = df_cri_temp[144:] # we skip the first 6 days of 2019 bc we don't have mobility data there
        df_cri_temp.reset_index(inplace=True,drop=True)
        df_cri_temp = df_cri_temp.groupby(df_cri_temp.index//granularity).sum()
        df_cri_temp.index = idx_list
        df_cri_list.append(df_cri_temp)

    # add the counts of each crime type
    df_cri = df_cri_list[0]
    for i in range(1,len(crime_types_list)):
        df_cri += df_cri_list[i]
    print("Crime dataset shape: ", df_cri.shape)

    # loop over the 11 poi categories (+diversity) and store them separate in a list (CAREFUL HERE DATA PREPROCESS FOLDER IS HARDCODED)
    df_mob_list = []
    for categ in ['diversity','const', 'manu', 'sale', 'transp', 'services', 'educ', 'health', 'recr', 'food', 'public', 'other']:
        df_mob = pd.read_csv(f"../../preprocessing/mobility_data/preprocessed_data/Data_preprocessed_02082024/Grid_cells_{gu}gu/Final_all/{city_folder}_mobility_all_{categ}_final_grid.csv",index_col=0)
        df_mob.reset_index(inplace=True,drop=True)
        df_mob = df_mob.groupby(df_mob.index//granularity).sum()
        df_mob.index = idx_list
        df_mob_list.append(df_mob)
        print("Mobility dataset shape: ", df_mob.shape)

    df_soc = pd.read_csv(f'../../preprocessing/sociodemographic_data/Sociodem_data_corrected/Grid_cells_{gu}gu/Final_all/{city_folder}_sociodem_all_final_grid.csv',index_col=0)
    print("Sociodemographic dataset shape: ", df_soc.shape)

    # set some values before normalization (that's why not necessary in next function)
    df_cri[df_cri >= 1] = 1 # SO THIS IS CRIME/NO CRIME CODE

    return (df_cri,df_mob_list,df_soc)

def generate_train_val_datasets_original(city_folder,n_frames,grid_size,df_cri,df_mob_list,df_soc):

    start_time = time.time()
    # we use a sliding window here (so n_samples==number of sequences)
    n_samples = df_cri.shape[0]-n_frames+1
    print(f"We should get {n_samples} samples")

    # make the full dataset (uses year from hour_var)
    full_dataset_l = []
    for var in df_cri.index.tolist(): # var is the index names I set in the function load_preprocessed_data
        full_dataset_l.append(make_full_matrix_per_hour(hour_var=var,df_cri=df_cri,df_mob_list=df_mob_list,df_soc=df_soc,grid_size=grid_size))

    full_dataset = np.concatenate(full_dataset_l,axis=0)
    print("Shape full dataset: ", full_dataset.shape)
    end_time = time.time()
    print(f"Time taken to generate full dataset: {end_time - start_time} seconds")

    # Create a view of the dataset with a sliding window using NumPy's stride_tricks
    start_time = time.time()
    shape = (n_samples, n_frames, *full_dataset.shape[1:])
    strides = (full_dataset.strides[0], *full_dataset.strides)
    dataset = np.lib.stride_tricks.as_strided(full_dataset, shape=shape, strides=strides, writeable=False)
    print("Shape dataset after sliding window: ", dataset.shape)
    end_time = time.time()
    print(f"Time taken to generate the sequences: {end_time - start_time} seconds")

    # Chronological splitting with buffer of n_frames-1 between train and validation
    start_time = time.time()
    num_samples = dataset.shape[0]
    train_size = int(0.9 * num_samples)

    buffer = n_frames - 1  # we have to skip n_frames-1 samples to make sure there's no overlap between train and test
    train_dataset = dataset[:train_size - buffer].copy()
    val_dataset = dataset[train_size:].copy()
    end_time = time.time()
    print("shape train_dataset:", train_dataset.shape) # to make sure we apply the mask properly
    print("shape val_dataset:", val_dataset.shape)

    print(f"Number of samples={num_samples}, train_size={train_size}, buffer={buffer}, shape train_dataset={train_dataset.shape}, shape val_dataset={val_dataset.shape}")

    print(f"Time taken to split train and validation (chronological with buffer): {end_time - start_time} seconds")

    # extract min and max for each feature
    start_time = time.time()
    min_vals = np.nanmin(train_dataset, axis=(0,1,2,3), keepdims=True)
    max_vals = np.nanmax(train_dataset, axis=(0,1,2,3), keepdims=True)
    end_time = time.time()
    print(f"Time taken to calculate min and max: {end_time - start_time} seconds")

    # load masks for NaN values
    mask = np.load(f"./pre_training_data/NaN_masks/{city_folder}_mask_final.npy")

    start_time = time.time()
    mask_broadcasted_train = np.broadcast_to(mask[None, None, :, :, None], train_dataset.shape)
    train_dataset[mask_broadcasted_train == 0] = 0
    mask_broadcasted_val = np.broadcast_to(mask[None, None, :, :, None], val_dataset.shape)
    val_dataset[mask_broadcasted_val == 0] = 0
    end_time = time.time()
    print(f"Time taken to replace nan with 0: {end_time - start_time} seconds")

    # NORMALIZATION IS DONE HERE USING MIN/MAX FROM TRAIN SET ALWAYS
    # normalize the train_dataset in-place
    start_time = time.time()
    np.subtract(train_dataset, min_vals, out=train_dataset)
    np.divide(train_dataset, max_vals - min_vals, out=train_dataset)
    # normalize the val_dataset in-place
    np.subtract(val_dataset, min_vals, out=val_dataset)
    np.divide(val_dataset, max_vals - min_vals, out=val_dataset)
    end_time = time.time()
    print(f"Time taken to perform the min-max normalization: {end_time - start_time} seconds")

    # Restore NaN values in both datasets (so NaN can be tracked in the algorithm to apply mask before metric)
    train_dataset[mask_broadcasted_train == 0] = np.nan
    val_dataset[mask_broadcasted_val == 0] = np.nan

    def create_shifted_frames(data):
        x = data[:,:-1,:,:,:] #we remove the last frame
        y = data[:,-1,:,:,0] # we take only the crime channel from the last frame
        y = np.expand_dims(y, axis=-1)
        return x, y

    start_time = time.time()
    # Apply the processing function to the datasets.
    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    end_time = time.time()
    print(f"Time taken to make shifted frames: {end_time - start_time} seconds")

    # Inspect the dataset.
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    return (x_train,y_train,x_val,y_val)

def generate_random_submatrices_fast(grid_size,grid_size_sub,num_sub,x,y):

  num_samples = x.shape[0]
  # we add a frame dimension
  y = np.expand_dims(y, axis=1)

  x_sub_list = []
  y_sub_list = []
  index_sub_list = []

  for sample in range(num_samples):
    count = 0
    it = 0
    while count < num_sub:

      # generate random starting indices
      i = np.random.randint(0, grid_size - grid_size_sub)
      j = np.random.randint(0, grid_size - grid_size_sub)

      # Extract 16x16 sections until we reach num_sub
      y_sub = y[sample,:,i:i+grid_size_sub, j:j+grid_size_sub,:]

      sum_num = np.nansum(np.array(y_sub)[0,:,:,0].flatten().tolist())

      min_num_crimes = 2
      if sum_num>min_num_crimes:
        x_sub_list.append(x[sample,:,i:i+grid_size_sub, j:j+grid_size_sub,:].tolist())
        y_sub_list.append(y_sub.tolist())
        index_sub_list.append((i,j)) # save the indexes
        count += 1
      elif sum_num <=min_num_crimes:
        it +=1
        if it >= 50: # set a maximum amount of iterations before we give up on that sample
          print(f"Sample {sample}: couldn't find submatrix! Count={count}")
          break
      elif np.isnan(sum_num): # otherwise code got stuck here for all nan observations
        print(f"Sample {sample}: is all nan!")
        break

  print("Submatrices created!")
  return (np.array(x_sub_list),np.array(y_sub_list)[:,0,:,:,:],index_sub_list)

def make_datasets_of_submatrices_final(city_folder,granularity,gu,grid_size,grid_size_sub,n_frames,num_sub,crime_types_list,output_folder,output_name):
    # load preprocessed data
    print("1. Loading preprocessed data...")
    start_time = time.time()
    df_cri,df_mob_list,df_soc = load_preprocessed_data(city_folder,gu,crime_types_list,granularity)
    end_time = time.time()
    print(f"Time taken to load prepricessed data: {end_time - start_time} seconds")

    # generate original train/val sets
    print("\n2. Generate_original train/val sets...")
    start_time = time.time()
    x_train,y_train,x_val,y_val = generate_train_val_datasets_original(city_folder,n_frames,grid_size,df_cri,df_mob_list,df_soc)
    end_time = time.time()
    print(f"Total time to generate train val datasets: {end_time - start_time} seconds")

    # generate random submatrices for each set
    print("\n3. Generate random submatrices...")
    start_time = time.time()
    x_t_sub,y_t_sub,index_sub_t = generate_random_submatrices_fast(grid_size,grid_size_sub,num_sub,x_train,y_train)
    x_v_sub,y_v_sub,index_sub_v = generate_random_submatrices_fast(grid_size,grid_size_sub,num_sub,x_val,y_val)
    end_time = time.time()
    print(f"Total time to generate submatrices: {end_time - start_time} seconds")

    # save all as one big compressed numpy
    print("\n4. Save big compressed numpy...")
    os.makedirs(f"{output_folder}", exist_ok=True)
    np.savez_compressed(f'{output_folder}/{city_folder}_{output_name}_chrono',
                        x_train=x_t_sub,y_train=y_t_sub,x_val=x_v_sub,y_val=y_v_sub,i_train=index_sub_t,i_val=index_sub_v)
    print(f"Final sizes training: x {x_t_sub.shape}, y {y_t_sub.shape}")
    print(f"Final sizes validation: x {x_v_sub.shape}, y {y_v_sub.shape}")

    print("\nDatasets saved!\n")

if sys.argv[6] == 'all':
  crimes_l = ['Assault', 'Homicide', 'Robbery','Burglary', 'Motor Vehicle Theft']
elif sys.argv[6] == 'property':
  crimes_l = ['Burglary', 'Motor Vehicle Theft']
elif sys.argv[6] == 'violent':
  crimes_l = ['Assault', 'Homicide', 'Robbery',]

print(f"Generate: {sys.argv[1]},{sys.argv[6]}")

make_datasets_of_submatrices_final(city_folder=sys.argv[1],
                                   granularity=int(sys.argv[7]), # we use 12 for 12 hours
                                   gu=0.2, # greometric unit of analysis (0.2km^2)
                                   grid_size=int(sys.argv[2]), # size of the main grid, which is different for each city
                                   grid_size_sub=int(sys.argv[8]), # size of the subgrid, which we set to 16
                                   n_frames=int(sys.argv[3]), # length of sequence (LB+1)
                                   num_sub=5, # number of subgrids extracted from main grid
                                   crime_types_list=crimes_l, # specification of crime aggregation (all, violent, or property)
                                   output_folder=sys.argv[5],
                                   output_name=sys.argv[6]
                                  )

import numpy as np
import sys
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
import pickle
import os

city_folder = sys.argv[1]
crime_agg = sys.argv[2]
input_folder = sys.argv[3]
grid_size = int(sys.argv[4])
seed = int(sys.argv[5])
cities_list = None
print(("RF:",city_folder,cities_list,crime_agg,input_folder))

# Set a seed for reproducibility
np.random.seed(seed)
print("SEED: ", seed)

def concat_all_cities(data_name,input_folder,ref_city,cities_list,crime_agg):
    x1 = np.load(f'../../pre_training_data/{input_folder}/{ref_city}_{crime_agg}_chrono.npz')[data_name]
    if cities_list != None:
        for city in cities_list:
            x2 = np.load(f'../../pre_training_data/{input_folder}/{city}_{crime_agg}_chrono.npz')[data_name]
            x1 = np.concatenate((x1, x2))
    return x1

x_train = concat_all_cities(data_name='x_train',input_folder=input_folder,ref_city=city_folder,cities_list=cities_list,crime_agg=crime_agg)[:,-1,:,:,:] 
y_train = concat_all_cities(data_name='y_train',input_folder=input_folder,ref_city=city_folder,cities_list=cities_list,crime_agg=crime_agg)
i_train = concat_all_cities(data_name='i_train',input_folder=input_folder,ref_city=city_folder,cities_list=cities_list,crime_agg=crime_agg) 
x_val = concat_all_cities(data_name='x_val',input_folder=input_folder,ref_city=city_folder,cities_list=cities_list,crime_agg=crime_agg)[:,-1,:,:,:] 
y_val = concat_all_cities(data_name='y_val',input_folder=input_folder,ref_city=city_folder,cities_list=cities_list,crime_agg=crime_agg) 
i_val = concat_all_cities(data_name='i_val',input_folder=input_folder,ref_city=city_folder,cities_list=cities_list,crime_agg=crime_agg) 

# Shuffle training data
train_indices = np.random.permutation(x_train.shape[0])
x_train = x_train[train_indices]
y_train = y_train[train_indices]
i_train = i_train[train_indices]

# Shuffle validation data
val_indices = np.random.permutation(x_val.shape[0])
x_val = x_val[val_indices]
y_val = y_val[val_indices]
i_val = i_val[val_indices]

# new shapes for 5 years - revision version
x_train = x_train[:12546,:,:,:]
y_train = y_train[:12546,:,:,:]
i_train = i_val[:12546,:]
x_val = x_val[:1510,:,:,:]
y_val = y_val[:1510,:,:,:]
i_val = i_val[:1510,:]

# setting all NaN values to 0
x_train[np.isnan(x_train)] = 0
y_train[np.isnan(y_train)] = 0
x_val[np.isnan(x_val)] = 0
y_val[np.isnan(y_val)] = 0

print(f"Shapes: {np.shape(x_train)} {np.shape(y_train)} {np.shape(x_val)} {np.shape(y_val)}")

# we have to flatten ALL dimensions excpet the number of samples (dimensions for sub=10)
x_train = np.reshape(x_train,(np.shape(x_train)[0],9984)) #16x16x39
y_train = np.reshape(y_train,(np.shape(x_train)[0],grid_size*grid_size)) #16x16
x_val = np.reshape(x_val,(np.shape(x_val)[0],9984))
y_val = np.reshape(y_val,(np.shape(x_val)[0],grid_size*grid_size))
print(f"Shapes: {np.shape(x_train)} {np.shape(y_train)} {np.shape(x_val)} {np.shape(y_val)}")

# shuffle 
x_train, y_train= sklearn.utils.shuffle(x_train,y_train,random_state=20)
x_val, y_val= sklearn.utils.shuffle(x_val,y_val,random_state=20)

# define model, train and predict
model = RandomForestClassifier(criterion='entropy',class_weight='balanced',n_jobs=-1) 
model.fit(x_train, y_train)
print(model.classes_)
y_pred_prob = model.predict_proba(x_val)
print("Shape y_pred_prob:", np.shape(y_pred_prob))

# Ensure dimensions of y_pred match y_test
y_pred = np.array(y_pred_prob) 
y_val = np.array(y_val)
y_pred = y_pred[:,:,1].T  # Transpose to match dimensions 
print("Shape y_pred:", np.shape(y_pred))

# put predictions as square matrices instead of vectors (MAKE SURE RESHAPE PUTS CELLS IN  THE RIGHT ORDER)
all_targets = y_val.reshape(np.shape(y_val)[0], 16, 16)
all_predictions = y_pred.reshape(np.shape(y_pred)[0], 16, 16)

print("shape of all targets: ", np.shape(all_targets))
print("shape of all predictions: ", np.shape(all_predictions))
print("shape of all index: ", np.shape(i_val))

os.makedirs("predictions", exist_ok=True) # create output folder if it does not already exist

# save corresponding files
with open(f"predictions/rf_i_{city_folder}_{crime_agg}_{seed}.pkl", "wb") as f:  
    pickle.dump(i_val, f)
print("i saved!\n")
    
with open(f"predictions/rf_predictions_{city_folder}_{crime_agg}_{seed}.pkl", "wb") as f_p:  
    pickle.dump(all_predictions, f_p)
print("predictions saved!\n")
    
with open(f"predictions/rf_targets_{city_folder}_{crime_agg}_{seed}.pkl", "wb") as f_t:  
    pickle.dump(all_targets, f_t)
print("targets saved!\n")

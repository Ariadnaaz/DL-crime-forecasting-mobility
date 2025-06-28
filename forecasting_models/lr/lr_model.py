import numpy as np
import sys
import sklearn
from sklearn.linear_model import LogisticRegression
import pickle
import os

city_folder = sys.argv[1] # = ref_city
crime_agg = sys.argv[2]
input_folder = sys.argv[3]
seed = int(sys.argv[4])
cities_list = None
print(("LR:",city_folder,crime_agg,input_folder,cities_list))

# Set a seed for reproducibility
np.random.seed(seed)
print("SEED: ", seed)

# load data
print("Load dataset...")

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
print(f"Shapes: {np.shape(x_train)} {np.shape(y_train)} {np.shape(x_val)} {np.shape(y_val)}")

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

print(f"Shapes after permutation: {np.shape(x_train)} {np.shape(y_train)} {np.shape(x_val)} {np.shape(y_val)}")

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

print(f"Shapes after resize: {np.shape(x_train)} {np.shape(y_train)} {np.shape(x_val)} {np.shape(y_val)}")

# we have to flatten ALL dimensions excpet the number of samples
print("Flatten dataset...")
x_train = np.reshape(x_train,(np.shape(x_train)[0],9984)) # 16x16x39
y_train = np.reshape(y_train,(np.shape(x_train)[0],256)) # 16x16
x_val = np.reshape(x_val,(np.shape(x_val)[0],9984))
y_val = np.reshape(y_val,(np.shape(x_val)[0],256))
print(f"Shapes: {np.shape(x_train)} {np.shape(y_train)} {np.shape(x_val)} {np.shape(y_val)}")

# shuffle dataset
print("Suffle...")
x_train, y_train= sklearn.utils.shuffle(x_train,y_train,random_state=20)
x_val, y_val= sklearn.utils.shuffle(x_val,y_val,random_state=20)

# Train logistic regression models for each target
print("Train logistic regression model for each target")
logistic_regression_models = []
for target_index in range(y_train.shape[1]):
    logistic_regression_model = LogisticRegression(multi_class='ovr',class_weight='balanced',max_iter=400)
    if np.sum(y_train[:, target_index])!=0:
        logistic_regression_model.fit(x_train, y_train[:, target_index])
        logistic_regression_models.append(logistic_regression_model)
    else:
        logistic_regression_models.append(None)

# Predict probabilities for each target
print("Predict probabilities for each target")
y_pred_prob = []
for logistic_regression_model in logistic_regression_models:
    if logistic_regression_model != None:
        y_pred_prob.append(logistic_regression_model.predict_proba(x_val)[:, 1])
    else:
        y_pred_prob.append([0] * 11) 

# Ensure dimensions of y_pred match y_test
y_pred = np.array(y_pred_prob).T  # transpose to match dimensions
y_val = np.array(y_val)

# put predictions as square matrices instead of vectors 
all_targets = y_val.reshape(np.shape(y_val)[0], 16, 16)
all_predictions = y_pred.reshape(np.shape(y_pred)[0], 16, 16)

print("shape of all targets: ", np.shape(all_targets))
print("shape of all predictions: ", np.shape(all_predictions))
print("shape of all index: ", np.shape(i_val))

# create output folder if it does not exist already
os.makedirs("predictions", exist_ok=True)

# save corresponding files
with open(f"predictions/lr_i_{city_folder}_{crime_agg}_{seed}.pkl", "wb") as f:  
    pickle.dump(i_val, f)
print("i saved!\n")
    
with open(f"predictions/lr_predictions_{city_folder}_{crime_agg}_{seed}.pkl", "wb") as f_p:  
    pickle.dump(all_predictions, f_p)
print("predictions saved!\n")
    
with open(f"predictions/lr_targets_{city_folder}_{crime_agg}_{seed}.pkl", "wb") as f_t:  
    pickle.dump(all_targets, f_t)
print("targets saved!\n")

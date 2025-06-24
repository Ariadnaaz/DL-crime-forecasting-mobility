import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchviz import make_dot
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_curve

print("Should return True if a GPU is available: ",torch.cuda.is_available())  
print("Number of GPUs available: ",torch.cuda.device_count())  
print("Name of the first GPU: ",torch.cuda.get_device_name(0)) 

from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import scipy, scipy.ndimage
import pickle
import sys
import numpy as np
import pandas as pd
from convlstm4 import ConvLSTM

# Set the seed for reproducibility
seed = int(sys.argv[6]) # 42, 0, 123, 999
print("SEED: ", seed)
torch.manual_seed(seed)

# Optionally set the seed for CUDA (if using GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    
# Custom worker_init_fn to ensure each worker has a different seed
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
class PrecisionLoss2(nn.Module):
    def __init__(self, epsilon=1e-8, pos_weight=None):
        super(PrecisionLoss2, self).__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: predicted scores from the model (before applying sigmoid)
        targets: ground truth binary labels (0 or 1)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        if self.pos_weight is not None:
            # Apply class weight to the positive class
            targets = targets * self.pos_weight
        
        # Smooth approximation of true positives
        TP = torch.sum(probs * targets)
        
        # Smooth approximation of false positives
        FP = torch.sum(probs * (1 - targets))
        
        # Precision
        precision = TP / (TP + FP + self.epsilon)
        
        # Loss is defined as 1 - precision (to maximize precision, minimize the loss)
        loss = 1 - precision
        return loss

class FocalLoss2(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float): The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size 2 for binary classification. optional.
        reduction (str): Specifies the reduction to apply to the output.
        It should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): Smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma: float,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index: int = -100,
            eps: float = 1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The prediction values should be between 0 and 1, \
             make sure to pass the values to sigmoid for binary classification'
        )

        # Convert target to float
        target = target.float()

        # Compute the focal loss components
        bce_loss = nn.functional.binary_cross_entropy(x, target, reduction='none')
        pt = torch.where(target == 1, x, 1 - x)  # Probability of the true class
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.weights is not None:
            alpha_t = self.weights[1] * target + self.weights[0] * (1 - target)
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * bce_loss

        # Handle ignore index
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            loss = loss * mask

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_neighbours(matrix,indices,distance):
    """
    matrix: numpy array
    indices: list with two integers
    distance: integer bigger than 1
    """
    # get indices of that element in the matrix
    indices = tuple(np.transpose(np.atleast_2d(indices)))
    # create a matrix of ones with same size as the input matrix
    dist = np.ones(np.shape(matrix))
    # put a zero at the location of the element we are looking at
    dist[indices] = 0
    # surround it by increasing numbers 1,2,3...
    dist = scipy.ndimage.distance_transform_cdt(dist, metric='chessboard')
    # get indeces of the values whoes distance is the requested (if we have more than 1, we have to add all previous)
    nb_indices = np.transpose(np.nonzero(dist == 1))
    for i in range(2,distance+1):
        nb_indices = np.concatenate((nb_indices,np.transpose(np.nonzero(dist == i))))
    return [matrix[tuple(ind)] for ind in nb_indices]


def get_custom_conf_matrix(orig,predi,dist=1):
    pred = predi.copy()
    for (i,j), value in np.ndenumerate(orig):
        if orig[i,j]==0 and pred[i,j]==0: # 0: true negative
            continue
        elif orig[i,j]==1 and pred[i,j]==1: # 1: true positive
            continue
        elif orig[i,j]==0 and pred[i,j]==1:
            if 1 in get_neighbours(orig,[i,j],dist):
                pred[i,j]=2 # 2: false positive with neighbouring positive
            else:
                pred[i,j]=3 # 3: false positive without neighbouring positive
        elif orig[i,j]==1 and pred[i,j]==0: # 4: false negative
            pred[i,j]=4

    result = pred.flatten().tolist()

    tn = result.count(0)
    tp = result.count(1)
    fp = result.count(3)
    fn = result.count(4)

    if tp==0 and result.count(2)==0:
        f1_orig = 0
        f1_new = 0
    elif tp==0:
        f1_orig = 0
        f1_new = (tp+result.count(2))/((tp+result.count(2))+0.5*(fp+fn))
    else:
        f1_orig = tp/(tp+0.5*((fp+result.count(2))+fn))
        f1_new = (tp+result.count(2))/((tp+result.count(2))+0.5*(fp+fn))

    if tp+fp+result.count(2) == 0:
        prec = 0
        prec_new = 0
    else:
        prec = tp/(tp+fp+result.count(2))
        prec_new = (tp+result.count(2))/(tp+result.count(2)+fp)
      
    if tp+fn==0:
        rec = 0
    else:  
        rec = tp/(tp+fn)
        
    if tp+result.count(2)+fn == 0:
        rec_new = 0
    else:
        rec_new = (tp+result.count(2))/(tp+result.count(2)+fn)

    return rec,prec,f1_orig,rec_new,prec_new,f1_new,pred

class ConvLSTMModel(nn.Module):
    def __init__(self, num_layers, input_chn, hidden_chn, kernel_size):
        super(ConvLSTMModel, self).__init__()
        self.num_layers = num_layers
    
        self.conv_lstm = ConvLSTM(input_chn, hidden_chn, kernel_size, num_layers, channel_last=True,
                 batch_first=True, bias=True, return_all_layers=True)
        self.final_conv = nn.Conv2d(in_channels=hidden_chn, out_channels=1, kernel_size=(3,3), padding='same')
        self.dropout = nn.Dropout(p=0.8)
    
    def forward(self, x):
        x = self.conv_lstm(x)[1][0][0] # selecting last_output, then extract sublist, then select h
        x = self.dropout(x)
        x = self.final_conv(x)
        return x

def train_model(model, train_loader, val_loader, criterions, optimizer, n_epoch, device, seed):
    criterion, criterion_precision, criterion_focal = criterions
    best_val_loss = +float('inf')
    model.train()
    
    # Define the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            # move data to the GPU
            data, target = data.to(device), target.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            output = model(data)
            output = torch.squeeze(output, 1) # make size match
            loss = criterion(output, target)
            
            # backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, avg_rec, avg_prec, avg_f1, avg_rec_new, avg_prec_new, avg_f1_new = evaluate_model(model, val_loader, criterions, device, seed)
        
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
def evaluate_model(model, val_loader, criterions, device, seed):
    criterion, criterion_precision, criterion_focal = criterions
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in val_loader:
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            output = torch.squeeze(output, 1) # make size match
            loss = criterion(output,target) 
            val_loss += loss.item()
            
            # Collecting predictions and targets for metric calculations
            all_targets.append(target.cpu().numpy())
            all_predictions.append(output.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Concatenate all predictions and targets
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    # put model output logits into probabilities
    all_predictions = torch.sigmoid(torch.from_numpy(all_predictions)).numpy()

    # save prediction and truth before applying threshold
    os.makedirs("predictions/", exist_ok=True)
    with open(f"predictions/convlstm_predictions_{ref_city}_{crime_agg}_{seed}.pkl", "wb") as f_p:  
        pickle.dump(all_predictions, f_p)
    with open(f"predictions/convlstm_targets_{ref_city}_{crime_agg}_{seed}.pkl", "wb") as f_t:  
        pickle.dump(all_targets, f_t)
    print("Saved final predictions and targets!")
    
    # apply thr to go from probability to class number
    thr=0.5
    all_predictions[all_predictions >= thr] = 1
    all_predictions[all_predictions < thr] = 0
    
    # Initialize lists to store metrics for each sample
    f1_list = []
    f1_new_list = []
    rec_list = []
    prec_list = []
    rec_new_list = []
    prec_new_list = []
    prec_auto = []
    rec_auto = []
    # Loop over each sample
    for i in range(np.shape(all_predictions)[0]):
        pred_flat = all_predictions[i,:] 
        true_flat = all_targets[i,:] 
        
        rec,prec,f1,rec_new,prec_new,f1_new,pred = get_custom_conf_matrix(true_flat,pred_flat,dist=1)
     
        rec_list.append(rec)
        prec_list.append(prec)
        f1_list.append(f1)
        rec_new_list.append(rec_new)
        prec_new_list.append(prec_new)
        f1_new_list.append(f1_new)
    
    # Compute the average precision, recall, and f1 score across all samples and classes
    avg_prec = np.mean(prec_list)
    avg_rec = np.mean(rec_list)
    avg_f1 = np.mean(f1_list)
    avg_prec_new = np.mean(prec_new)
    avg_rec_new = np.mean(rec_new)
    avg_f1_new = np.mean(f1_new)

    return avg_val_loss, avg_rec, avg_prec, avg_f1, avg_rec_new, avg_prec_new, avg_f1_new


def concat_all_cities(data_name,input_folder,ref_city,cities_list,crime_agg):
    x1 = np.load(f'../pre_training_data/{input_folder}/{ref_city}_{crime_agg}_chrono.npz')[data_name]
    if cities_list != None:
        for city in cities_list:
            x2 = np.load(f'../pre_training_data/{input_folder}/{city}_{crime_agg}_chrono.npz')[data_name]
            x1 = np.concatenate((x1, x2))
    return x1

def select_features(mode,x_train,x_val):
    if mode == 'CS': # remove mobility channel(s) from data
        x_train = np.delete(x_train,[1,2,3,4,5,6,7,8,9,10,11,12],axis=4)
        x_val = np.delete(x_val,[1,2,3,4,5,6,7,8,9,10,11,12],axis=4)
        return (x_train,x_val)
    elif mode == 'CM':  # keep only C and M channels (so remove sociodem)
        x_train = x_train[:,:,:,:,:13]
        x_val = x_val[:,:,:,:,:13]
        return (x_train,x_val)
    elif mode == 'C':  # keep only C channel
        x_train = x_train[:,:,:,:,0]
        x_val = x_val[:,:,:,:,0]
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)
        return (torch.from_numpy(x_train),torch.from_numpy(x_val))
    else:
        print("No valid mode selected, so we keep all the channels!")
        return (x_train,x_val)


def main(batch_size, n_epochs, input_folder, ref_city, cities_list, crime_agg, mode, num_layers, input_chn, hidden_chn, kernel_size, seed):
    # load the data as tensors
    x_train = torch.from_numpy(concat_all_cities(data_name='x_train',input_folder=input_folder,ref_city=ref_city,cities_list=cities_list,crime_agg=crime_agg)).float() 
    y_train = torch.from_numpy(concat_all_cities(data_name='y_train',input_folder=input_folder,ref_city=ref_city,cities_list=cities_list,crime_agg=crime_agg)).float()
    i_train = concat_all_cities(data_name='i_train',input_folder=input_folder,ref_city=ref_city,cities_list=cities_list,crime_agg=crime_agg)
    x_val = torch.from_numpy(concat_all_cities(data_name='x_val',input_folder=input_folder,ref_city=ref_city,cities_list=cities_list,crime_agg=crime_agg)).float()
    y_val = torch.from_numpy(concat_all_cities(data_name='y_val',input_folder=input_folder,ref_city=ref_city,cities_list=cities_list,crime_agg=crime_agg)).float()
    i_val = concat_all_cities(data_name='i_val',input_folder=input_folder,ref_city=ref_city,cities_list=cities_list,crime_agg=crime_agg)
    
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
    
    # make number of samples consistent (5 years and 39 features) 
    x_train = x_train[:12546,:,:,:,:]
    y_train = y_train[:12546,:,:,:]
    i_train = i_val[:12546,:]
    x_val = x_val[:1510,:,:,:,:]
    y_val = y_val[:1510,:,:,:]
    i_val = i_val[:1510,:]
    
    # Setting all NaN values to 0 (since they have been correctly identified already)
    x_train[np.isnan(x_train)] = 0
    y_train[np.isnan(y_train)] = 0
    x_val[np.isnan(x_val)] = 0
    y_val[np.isnan(y_val)] = 0
    
    # select features we want to use. Options: 'CS', 'CM', 'C', or nothing (so all features)
    x_train, x_val = select_features(mode=mode,x_train=x_train,x_val=x_val)
    
    print(f"Shapes: {np.shape(x_train)} {np.shape(y_train)}")
    print(f"Shapes: {np.shape(x_val)} {np.shape(y_val)}")
    
    # remove channel dimension so it will work at the end
    y_train = y_train.permute(0, 3, 1, 2)
    y_val = y_val.permute(0, 3, 1, 2)
    y_train = torch.squeeze(y_train, 1)
    y_val = torch.squeeze(y_val, 1)
    
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,worker_init_fn=worker_init_fn)
    test_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # Model
    model = ConvLSTMModel(num_layers, input_chn, hidden_chn, kernel_size)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')  
    model.to(device) # to work on GPU
    
    # Calculate class weights
    targets_flat = y_train.view(-1)  # Flatten targets to (n_samples * 22 * 22)

    # Calculate the number of positive and negative examples
    num_positives = targets_flat.sum().item()
    num_negatives = targets_flat.size(0) - num_positives

    pos_weight = num_negatives / num_positives # Calculate pos_weight for the positive class

    # Convert pos_weight to a PyTorch tensor
    pos_weight_tensor = torch.tensor(pos_weight).float().to(device) # targets.device
    print("pos_weight_tensor: ",pos_weight_tensor)
    print([1, pos_weight_tensor.cpu().item()])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) 
    criterion_precision = PrecisionLoss2(pos_weight=pos_weight_tensor)
    criterion_focal = FocalLoss2(gamma=6,weights=torch.tensor([1, pos_weight_tensor.cpu().item()])) 
    criterions = (criterion, criterion_precision, criterion_focal)
    optimizer = optim.Adam(model.parameters(), lr=0.00001) 
    
    print("Training model...") 
    train_model(model, train_loader, test_loader, criterions, optimizer, n_epochs, device, seed)

    # make output folder if it does not exist
    os.makedirs("predictions", exist_ok=True)

    # save index of subgrids used
    with open(f"predictions/convlstm_i_{ref_city}_{crime_agg}_{seed}.pkl", "wb") as f:  
        pickle.dump(i_val, f)
    
if __name__ == "__main__":
    
    # params data loading
    input_folder = sys.argv[3]
    ref_city = sys.argv[1]
    cities_list = None
    crime_agg = sys.argv[2]
    mode = sys.argv[4]
    seq_length = int(sys.argv[5])
    
    # params architecture model
    num_layers = 3
    hidden_chn = seq_length-1 #needs to match sequence legth (so LB-1)
    kernel_size = (3,3)

    if  mode == 'C':
        input_chn = 1
    elif mode == 'CM':
        input_chn = 13
    elif mode == 'CS':
        input_chn = 27
    else:
        input_chn = 39

    # params training
    batch_size = 55
    n_epochs = 200

    print(f"Data settings: ref_city={ref_city}, cities_list={cities_list}, crime_agg={crime_agg}, mode={mode}")
    print(f"Settings: n_epochs={n_epochs}, batch_size={batch_size}, num_layers={num_layers}, hidden_chn={hidden_chn}")
    main(batch_size, n_epochs, input_folder, ref_city, cities_list, crime_agg,mode, num_layers, input_chn, hidden_chn, kernel_size, seed)

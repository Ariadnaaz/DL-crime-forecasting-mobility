import numpy as np
import pickle
import sys
from sklearn.metrics import confusion_matrix
import scipy.ndimage
import os

def get_neighbours(matrix, indices, distance, mask=None):
    """
    matrix: numpy array
    indices: list with two integers
    distance: integer >= 1
    mask: optional mask (1 = valid, 0 = invalid)
    """
    indices = tuple(np.transpose(np.atleast_2d(indices)))
    dist = np.ones(np.shape(matrix))
    dist[indices] = 0

    dist = scipy.ndimage.distance_transform_cdt(dist, metric='chessboard')
    nb_indices = np.transpose(np.nonzero(dist == 1))
    for i in range(2, distance + 1):
        nb_indices = np.concatenate((nb_indices, np.transpose(np.nonzero(dist == i))))

    neighbours = []
    for ind in nb_indices:
        i, j = ind
        if 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1]:
            if mask is None or mask[i, j] == 1:  # only use valid cells
                neighbours.append(matrix[i, j])
    return neighbours


def get_custom_conf_matrix(orig, predi, dist=1, mask=None):
    pred = predi.copy()
    for (i, j), value in np.ndenumerate(orig):
        if mask is not None and mask[i, j] == 0:
            continue  # skip masked cells 
        if orig[i, j] == 0 and pred[i, j] == 0:  # TN
            continue
        elif orig[i, j] == 1 and pred[i, j] == 1:  # TP
            continue
        elif orig[i, j] == 0 and pred[i, j] == 1:
            if 1 in get_neighbours(orig, [i, j], dist, mask=mask):
                pred[i, j] = 2  # FP with neighboring crime
            else:
                pred[i, j] = 3  # FP with no neighbor
        elif orig[i, j] == 1 and pred[i, j] == 0:  # FN
            pred[i, j] = 4

    result = pred[mask == 1].flatten().tolist() if mask is not None else pred.flatten().tolist()
    
    tn = result.count(0)
    tp = result.count(1)
    fp = result.count(3)
    fn = result.count(4)
    fp_nb = result.count(2)

    # F1 & Precision/Recall handling
    if tp == 0 and fp_nb == 0:
        f1_orig = f1_new = 0
    elif tp == 0:
        f1_orig = 0
        f1_new = (tp + fp_nb) / ((tp + fp_nb) + 0.5 * (fp + fn))
    else:
        f1_orig = tp / (tp + 0.5 * ((fp + fp_nb) + fn))
        f1_new = (tp + fp_nb) / ((tp + fp_nb) + 0.5 * (fp + fn))

    if tp + fp + fp_nb == 0:
        prec = prec_new = 0
    else:
        prec = tp / (tp + fp + fp_nb)
        prec_new = (tp + fp_nb) / (tp + fp_nb + fp)

    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    rec_new = (tp + fp_nb) / (tp + fp_nb + fn) if (tp + fp_nb + fn) > 0 else 0

    return rec, prec, f1_orig, rec_new, prec_new, f1_new, pred

def calculate_performance_per_threhold(ref_city, crime_agg, seed):
    full_mask = np.load(f"../../pre_training_data/NaN_masks/{ref_city}_mask_final.npy") # i think mask is not necessary for these metrics

    with open(f"predictions/convlstm_i_{ref_city}_{crime_agg}_{seed}.pkl", 'rb') as f:
        subgrid_indices = pickle.load(f)

    with open(f"predictions/convlstm_predictions_{ref_city}_{crime_agg}_{seed}.pkl", 'rb') as file:
        data_pred = pickle.load(file)

    with open(f"predictions/convlstm_targets_{ref_city}_{crime_agg}_{seed}.pkl", 'rb') as file:
        data_orig = pickle.load(file)
        
    print("All data opened!")

    originals = data_orig
    predictions = data_pred

    assert len(subgrid_indices) == predictions.shape[0], "Mismatch between subgrid indices and number of samples"

    
    perf_list = []

    for thr in np.arange(0.50, 1.01, 0.01):
        rec_list, prec_list, f1_list = [], [], []
        rec_new_list, prec_new_list, f1_new_list = [], [], []

        for sample in range(predictions.shape[0]):
            origi = originals[sample]
            predi = predictions[sample]
            predi2 = (predi >= thr).astype(int)

            top_i, top_j = subgrid_indices[sample]
            mask_subgrid = full_mask[top_i:top_i + 16, top_j:top_j + 16] 
                
            rec, prec, f1, rec_new, prec_new, f1_new, _ = get_custom_conf_matrix(
                origi, predi2, dist=1, mask=mask_subgrid
            )

            rec_list.append(rec)
            prec_list.append(prec)
            f1_list.append(f1)
            rec_new_list.append(rec_new)
            prec_new_list.append(prec_new)
            f1_new_list.append(f1_new)

        perf_list.append([
            thr,
            np.mean(rec_list), np.mean(prec_list), np.mean(f1_list),
            np.mean(rec_new_list), np.mean(prec_new_list), np.mean(f1_new_list)
        ])

    os.makedirs("perf_thrs", exist_ok=True)
    output_path = f"perf_thrs/convlstm_perf_thrs_{ref_city}_{crime_agg}_{seed}.pkl"
    with open(output_path, "wb") as fp:
        pickle.dump(perf_list, fp)
    print(f"Saved performance metrics to {output_path}")
    
calculate_performance_per_threhold(ref_city = sys.argv[1],
                                   crime_agg = sys.argv[2],
                                   seed = sys.argv[3])

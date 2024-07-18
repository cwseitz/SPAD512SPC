from scipy.spatial import distance_matrix
import numpy as np

def match_coordinates(est_coords, true_coords):
    dist_matrix = distance_matrix(est_coords, true_coords)
    errors_x = np.zeros(len(est_coords))
    errors_y = np.zeros(len(est_coords))
    
    flat_indices = np.argsort(dist_matrix, axis=None)
    
    matched_est = np.zeros(len(est_coords), dtype=bool)
    matched_true = np.zeros(len(true_coords), dtype=bool)
    
    for index in flat_indices:
        i, j = divmod(index, len(true_coords))
        if not matched_est[i] and not matched_true[j]:
            matched_est[i] = True
            matched_true[j] = True
            errors_x[i] = est_coords[i][0] - true_coords[j][0]
            errors_y[i] = est_coords[i][1] - true_coords[j][1]
        if matched_est.all() and matched_true.all():
            break
    
    return errors_x, errors_y

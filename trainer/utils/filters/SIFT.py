
import cv2
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine

DIST_METRICS = ['euclidean', 'manhattan', 'chebyshev', 'cosine']

def compute_feature(frame):
    sift = cv2.xfeatures2d.SIFT_create()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, des = sift.detectAndCompute(image, None)
    if des is not None:
        return np.mean(des, axis=0).astype('float32')
    else:
        return np.zeros(128)

def get_distance_fn(dist_metric):
    if dist_metric == 'euclidean':
        return euclidean
    elif dist_metric == 'manhattan':
        return cityblock
    elif dist_metric == 'chebyshev':
        return chebyshev
    elif dist_metric == 'cosine':
        return cosine
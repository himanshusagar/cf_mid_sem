import numpy as np
from sklearn.decomposition import PCA
from lib import dataset_manager
from lib import data_utils

def apply_PCA(data_X):
    pcaModel = PCA()




def apply_algo():
    trainData, testData = dataset_manager.fetch_data_100k_ratings("1")
    trainData = data_utils.normalise_mean_std_dev(trainData)
    #################
    R = trainData
    C = np.corrcoef(R)

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from lib import dataset_manager
from lib import data_utils
from lib import lib_ops

import pandas as pd
from scipy import stats as stats



def apply_algo():
    trainData, testData = dataset_manager.fetch_data_100k_ratings("2")

    #################
   # data_utils.fillZeroesWithColMean(np.arange(10.0).reshape((2,5)))
    A_star = data_utils.fillValueWithColMean(trainData)
    trainData=""

    i_score = ""
    i_load = ""

    iter = 0
    criteria = 0.0

    while(criteria < 0.1):
        print("")
        print("ITERATION ::::" + str(iter) + " ::::::::::::::::::::::::")
        iter  = iter + 1

        ##Run Iterative PCA
        #Center Col Wise
        A_c , itemMean = data_utils.normalise_mean_col_wise(A_star , isMeanReq=True)

        dataSize = np.shape(A_c)
        #Apply SVD with k = min(rows , cols)

        A_dash_c, i_score , i_load = lib_ops.apply_svd_get_nTh_approx(A_c, min( 900 , dataSize[0], dataSize[1]))

        #Reverse centering
        print(  "A_dash_c" +  str(np.shape(A_dash_c) ))
        A_dash = itemMean + A_dash_c

        ##Re impute Values where rating  =1
        A_star_changed = dataset_manager.fill_missing_value(A_star , A_dash , trainData , 0.0)

        ##Check how A is different:
        sub =  np.subtract(A_star_changed , A_star)
        A_star = A_star_changed

        criteria = np.sum(sub)

        print("Criteria" + str(criteria))


    scoreMatrix = i_score
    loadMatrix = i_load


if __name__=="__main__":
    apply_algo()
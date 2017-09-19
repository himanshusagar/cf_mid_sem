import numpy as np

#
# def normalise_mean_std_dev_col_wise(data):
#     data = (data - np.mean(data , axis=1))/ data.std(axis=1)
#     return data


def normalise_mean_col_wise(data , isMeanReq = True):
    itemMean = np.mean(data , axis=0 , keepdims=True)
    data = (data - itemMean)
    print("Normalise Col isMeanReq" + str(isMeanReq))
    if(isMeanReq):
        return data,itemMean

    return data


def fillValueWithColMean(A , value=0.0):

    A[ A == value] = np.nan

    itemMean = np.nanmean(A, axis=0)
    nanIndices = np.where(np.isnan(A))

    isAnyNan = np.shape(nanIndices[nanIndices == True])[0] > 0

    print("Fil Cols : Is ANY NAN " + str(isAnyNan))
    if(isAnyNan):
        A[nanIndices] = np.take(itemMean, nanIndices[1])

    return A



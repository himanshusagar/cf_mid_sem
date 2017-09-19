import numpy as np
from sklearn.metrics import mean_absolute_error


def calc_nmae(rating , train):

    tot_mae = 0
    rows = np.shape(train)[0]
    cols= np.shape(train)[1]
    denominator = 0

    for user in range(rows):
        for item in range(cols):
            if(train[user][item] > 0):
                true_rat = train[user][item]
                pre_rat = rating[user][item]
                i_mae = np.absolute(true_rat - pre_rat)
                tot_mae = tot_mae + i_mae
                denominator = denominator + 1

    solution = tot_mae/(rows * cols)
    print("Final = " + str( solution))

    return solution
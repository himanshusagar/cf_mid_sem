import numpy as np


import  dataset_manager
import  data_utils
#
# from lib import dataset_manager
# from lib import data_utils
#from midsem.nmae import calc_nmae

import nmae


N_FACTORS = 50

USERS  = 943
ITEMS = 1682

INITIAL_MEAN = 0
INITIAL_STD_DEVIATION = .1


def softmax( t , s):
    s = np.multiply(s, np.ones(np.shape(t)))
    mod_t = np.absolute(t)
    print(np.shape(s))

    return np.multiply(np.sign(t) , np.maximum( np.zeros((N_FACTORS,1682)) ,  mod_t - s   ))




def apply_algo(index):
    trainData, testData , maskTrain = dataset_manager.fetch_data_100k_ratings(str(index))

    R = trainData
    A = maskTrain

    beta = np.max( np.linalg.eigvals( np.dot( np.transpose(A )  , A ) ) )


    lambda_U = 1e3
    lambda_V = 1e-1

    average = np.average(A)

    U_o = np.random.normal( INITIAL_MEAN, INITIAL_STD_DEVIATION,
                              ( USERS , N_FACTORS))


    V_o = np.random.normal( INITIAL_MEAN, INITIAL_STD_DEVIATION,
                              ( N_FACTORS , ITEMS))



    iter = 0
    criteria = 0.0



    U_k = U_o
    V_k = V_o

    Y = R ##np.dot(R , A )

    k = 0


    while(k < N_FACTORS):


        print("")
        print("ITERATION ::::::::" + str(iter) + " ::::::::::::::::::::::::")
        iter  = iter + 1


        #Y_offset = Y - np.dot(A , np.dot(U_k , V_k ))

        Y_offset = Y - np.dot(U_k, V_k)

        #Z = np.dot(U_k , V_k ) +  (1/beta) * np.dot( np.transpose(A)  , Y_offset)
        Z = np.dot(U_k, V_k) +  np.multiply( (1 / beta) , Y_offset )


        U_k_plus_1 = np.dot(V_k , np.transpose(V_k)) + np.multiply( lambda_U  , np.identity( N_FACTORS ) )
        inverse = np.linalg.pinv(np.dot(Z, np.transpose(V_k)))
        print(np.shape(inverse))

        U_k_plus_1 = np.transpose(np.dot(  U_k_plus_1 , inverse  ) )

        print(np.shape(U_k_plus_1))

        #For W
        Y_offset_k_plus_1 = Y - np.dot(U_k_plus_1 , V_k)

#        W = np.dot(U_k_plus_1, V_k) + (1 / beta) * np.dot(np.transpose(A), Y_offset_k_plus_1)

        W = np.dot(U_k_plus_1 , V_k) + np.multiply( (1/beta)  ,  Y_offset_k_plus_1 )

        alpha = 1.01 * np.max( np.linalg.eigvals( np.dot( np.transpose(U_k) , U_k )))


        subVal = np.multiply((1 / alpha), np.dot( np.transpose(U_k) , np.subtract(W , np.dot(U_k_plus_1 ,V_k))  )  )

        soft_param_a = np.add(V_k  , subVal  )
        soft_param_b = lambda_V / (2 * alpha)

        V_k_plus_1 = softmax(soft_param_a , soft_param_b)


        U_k = U_k_plus_1
        V_k = V_k_plus_1

       # criteria = np.sum(sub)

       # print("Criteria" + str(criteria))
        k = k + 1

    New_Rating = np.dot( U_k , V_k )

    nmae.calc_nmae(New_Rating , trainData)

    print("Done")

if __name__=="__main__":

    list =[]
    for i in range(1 , 6):
        i_mae = apply_algo(i)
        list.append(i_mae)

    dataset_manager.dump_error("mae_error.txt", list)


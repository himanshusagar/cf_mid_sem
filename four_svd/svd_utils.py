import numpy as np


def mae(predictions):

    mae_ = np.mean([float(abs(true_r - predict_r))
                    for (_, _, true_r, predict_r ) in predictions])

    print('MAE:  {0:1.4f}'.format(mae_))

    return mae_




def evaluate(algo, data,  verbose=1):
    m = "MAE"
    print("::::::Calculating MAE:::::::")
    for fold_i, (trainset, testset) in enumerate(data.folds()):

        if verbose:
            print('#########')
            print('Fold ' + str(fold_i + 1))
        algo.train(trainset)
        predictions = algo.predict(testset)
        mae(predictions)

from surprise import *

def svd(data):
    algo = SVD(verbose=True , n_epochs = 5)

    # Evaluate performances of our algorithm on the dataset.
    perf = evaluate(algo, data, measures=['MAE'])

    #print_perf(perf)

    print("Done...........")
    print("")


if __name__ == '__main__':
    data = Dataset.load_from_folds()
    data.split(n_folds=2)
    svd(data)


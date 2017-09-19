
from four_svd import svd_utils
from four_svd.database import Database

from four_svd.svd import SVD

if __name__ == '__main__':
    # data_trainSet = read_particular_dataset("data//ml-100k/u" + testIndex + ".base", sep="\t")

    files_dir = "../data/ml-100k/"

    # This time, we'll use the built-in reader.


    # folds_files is a list of tuples containing file paths:
    # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
    train_file = files_dir + 'u%d.base'
    test_file = files_dir + 'u%d.test'
    folds_files = [(train_file % i, test_file % i) for i in (1, 2 )] #, 3, 4, 5)]

    data = Database(folds_files)


    #data.split(n_folds=2)
    svd = SVD(n_epochs=1)
    svd_utils.evaluate(svd , data)



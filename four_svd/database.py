import os
from collections import defaultdict
from pathlib import Path

import itertools

''' TrainDatabase is inspired from various  pyhton libraries like scikit-learn when they use dicts to 
deal with sparse matrioes and such sparse matrices always appear in our course
Moreover, you can ask, why use their design and not develop yours ?
Answer to this is simple, This is not DB Design Course, Hence i have not bothered to developed newer DB paradigms
'''

from four_svd.train_db import TrainDatabase


class Database:

    def __init__(self, folds_files=None, sep = "\t"):
        self.sep = sep
        self.fold_files = folds_files

    def actual_ratings_file_foldwise(self):
        for train_file, test_file in self.fold_files:
            with open(train_file) as f:
                actual_train_ratings = [self.parse_line(line) for line in
                               itertools.islice(f, 0, None)]
            with open(test_file) as f:
                actual_test_ratings = [self.parse_line(line) for line in
                               itertools.islice(f, 0, None)]
                yield actual_train_ratings, actual_test_ratings


    def construct_trainset(self, actual_trainset):

        actual2smart_id_users = {}
        actual2smart_id_items = {}

        current_u_index = 0
        current_i_index = 0

        u_dict = defaultdict(list)
        r_dict = defaultdict(list)

        for urid, irid, r in actual_trainset:
            try:
                uid = actual2smart_id_users[urid]
            except KeyError:
                uid = current_u_index
                actual2smart_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = actual2smart_id_items[irid]
            except KeyError:
                iid = current_i_index
                actual2smart_id_items[irid] = current_i_index
                current_i_index += 1

            u_dict[uid].append((iid, r))
            r_dict[iid].append((uid, r))

        total_users = len(u_dict)  # number of users
        total_items = len(r_dict)  # number of items
        total = len(actual_trainset)

        trainset = TrainDatabase(u_dict,
                            r_dict,
                            total_users,
                            total_items,
                          #  total,

                            actual2smart_id_users,
                            actual2smart_id_items)

        return trainset

    def folds(self):
        for actual_trainData, actual_testData in self.actual_ratings_file_foldwise():
            trainset = self.construct_trainset(actual_trainData)
            yield trainset, actual_testData


    def parse_line(self, line):
        line = line.split(self.sep)
        uid, iid, r = (line[i].strip()
                           for i in [0, 1, 2])
        return uid, iid, float(r)



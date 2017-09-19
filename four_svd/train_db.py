import numpy as np
from six import iteritems


##del them

class TrainDatabase:
    def __init__(self, user_ratings, item_ratings,
                 total_users, total_items,
                 #total_ratings,
                 actual2smart_id_users, actual2smart_id_items):

        self.user_ratings = user_ratings
        self.item_ratings = item_ratings
        self.total_users = total_users

        self.total_items = total_items
        #self.n_ratings = total_ratings


        self._actual2smart_id_users = actual2smart_id_users
        self._actual2smart_id_items = actual2smart_id_items
        self._total_mean = None

        self._smart2actual_id_users = None
        self._smart2actual_id_items = None

    def is_user_in_train_set(self, uid):
        return uid in self.user_ratings


    def is_item_in_train_set(self, iid):
        return iid in self.item_ratings


    def to_smart_uid(self, ruid):
        try:
            return self._actual2smart_id_users[ruid]
        except KeyError:
            raise ValueError("")

    # def to_actual_uid(self, iuid):
    #     if self._smart2actual_id_users is None:
    #         self._smart2actual_id_users = {smart: actual for (actual, smart) in
    #                                     iteritems(self._actual2smart_id_users)}
    #     try:
    #         return self._smart2actual_id_users[iuid]
    #     except KeyError:
    #         raise ValueError("")

    def to_smart_iid(self, riid):
        try:
            return self._actual2smart_id_items[riid]
        except KeyError:
            raise ValueError("")

    # def to_actual_iid(self, iiid):
    #     if self._smart2actual_id_items is None:
    #         self._smart2actual_id_items = {smart: actual for (actual, smart) in
    #                                     iteritems(self._actual2smart_id_items)}
    #     try:
    #         return self._smart2actual_id_items[iiid]
    #     except KeyError:
    #         raise ValueError("")

    def all_ratings(self):
        for u, u_ratings in iteritems(self.user_ratings):
            for i, r in u_ratings:
                yield u, i, r

    def get_total_mean_value(self):
        ##Need to find only of all actual Ratings

        if self._total_mean is None:
            self._total_mean = np.mean([r for (_, _, r) in
                                        self.all_ratings()])
        return self._total_mean

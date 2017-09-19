
import numpy as np
from six.moves import range
#from surprise import Dataset


from four_svd import svd_utils

INITIAL_MEAN = 0
INITIAL_STD_DEVIATION = .1



class SVD():
    def __init__(self, n_factors=100, n_epochs=20, learnRate=.005,
                 regularizor=.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        
        self.learnRate_bias_user = self.learnRate_bias_item = learnRate
        self.learnRate_p_user = self.learnRate_q_item = learnRate
        self.regularizor_bias_user = self.regularizor_bias_item = regularizor
        self.regularizor_p_user = self.regularizor_q_item = regularizor

    def train(self, trainset):
        self.bias_user = self.bias_item = None
        self.trainset = trainset
        self.sgd(trainset)

    def sgd(self, trainset):
        global_mean = self.trainset.get_total_mean_value()

        bias_user = np.zeros(trainset.total_users, np.double)
        bias_item = np.zeros(trainset.total_items, np.double)
        
        p_user = np.random.normal( INITIAL_MEAN, INITIAL_STD_DEVIATION,
                              (trainset.total_users, self.n_factors))
        q_item = np.random.normal( INITIAL_MEAN , INITIAL_STD_DEVIATION,
                              (trainset.total_items, self.n_factors))

        '''
        
        
        '''

        for current_epoch in range(self.n_epochs):

            print("Process epoch " + str(current_epoch))
            for u, i, r in trainset.all_ratings():

                dot = np.dot(q_item[i] , p_user[u])
                err = r - (global_mean + bias_user[u] + bias_item[i] + dot)

                bias_user[u] += self.learnRate_bias_user * (err - self.regularizor_bias_user * bias_user[u])
                bias_item[i] += self.learnRate_bias_item * (err - self.regularizor_bias_item * bias_item[i])

                for f in range(self.n_factors):
                    current_p_user = p_user[u, f]
                    current_q_item = q_item[i, f]
                    p_user[u, f] += self.learnRate_p_user * (err * current_q_item - self.regularizor_p_user * current_p_user)
                    q_item[i, f] += self.learnRate_q_item * (err * current_p_user - self.regularizor_q_item * current_q_item)

        self.bias_user = bias_user
        self.bias_item = bias_item
        self.p_user = p_user
        self.q_item = q_item


    def estimate(self, u, i):

        is_user = self.trainset.is_user_in_train_set(u)
        is_item = self.trainset.is_item_in_train_set(i)

        predicted_rating = self.trainset.get_total_mean_value()

        ##Add user Bias
        if is_user:
            predicted_rating += self.bias_user[u]

        ##Add Item Bias
        if is_item:
            predicted_rating += self.bias_item[i]

        ##Woah Both of them exits : let's do better
        if is_user and is_item:
            predicted_rating += np.dot(self.q_item[i], self.p_user[u])

        return predicted_rating

    def predict(self, testset):
        predictions = [self._predict_single(uid,
                                            iid,
                                            r_ui_trans)

                       for (uid, iid, r_ui_trans) in testset]
        return predictions

    def _predict_single(self, uid, iid, r_ui=None):

        # Convert actual ids to smart ids
        try:
            iuid = self.trainset.to_smart_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_smart_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        try:
            predicted_rating = self.estimate(iuid, iiid)
        except ValueError:
            predicted_rating = self.trainset.get_total_mean_value()


        predicted_rating = min(5, predicted_rating)
        predicted_rating = max(1, predicted_rating)

        pred = [uid, iid, float(r_ui), predicted_rating ]

        #print(pred)

        return pred

# if __name__ == '__main__':
#     # data_trainSet = read_particular_dataset("data//ml-100k/u" + testIndex + ".base", sep="\t")
#
#     data = Dataset.load_bias_useriltin('ml-100k')
#     #data.split(n_folds=2)
#     svd = SVD(n_epochs=5)
#     svd_utils.evaluate(svd , data)

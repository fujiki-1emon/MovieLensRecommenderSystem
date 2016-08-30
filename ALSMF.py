import time

import numpy as np
import scipy as sp
import pandas as pd


class ALSMF(object):

    def __init__(self, train_matrix, reg_param, n_factors, n_iter, explicit=True, alpha=0):
        '''
        Parameters
        ----------
        train_matrix (ndarray): user-item matrix for training a model.
        reg_param (float): a reguralization parameter, lambda
        n_factors (int): number of latent factors
        n_iter (int): number of iterations to update user-factor and item-factor matrices
        explicit (boolean): explicit feedback if True, else implicit feedback
        '''
        self.train_matrix = train_matrix
        self.n_users = self.train_matrix.shape[0]
        self.n_items = self.train_matrix.shape[1]
        self.reg_param = reg_param
        self.alpha = alpha
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.explicit = explicit
        self.user_factor_matrix = None
        self.item_factor_matrix = None
        self.prediction = None
        if self.explicit is True:
            self.confidence_matrix = self.train_matrix.copy()
            self.confidence_matrix[self.confidence_matrix > 0] = 1
        else:
            self.confidence_matrix = 1 + self.alpha * self.train_matrix

    def build_model(self):
        '''Training and building a latent factor model with alternating least squares method.'''
        # Initialize the user-factor matrix and the item-factor matrix with random number [0, 3)
        self.user_factor_matrix = 3 * np.random.rand(self.n_users, self.n_factors)
        self.item_factor_matrix = 3 * np.random.rand(self.n_items, self.n_factors)
        # Initialize item-factor matrix by assigning the average rating for that movie as the first row
        self.item_factor_matrix[0, :] = self.train_matrix[:, 0].mean()

        t0 = time.time()
        for i in range(self.n_iter):
            # update user-factor matrix
            self.user_factor_matrix = self.update_matrix(self.item_factor_matrix, mode='user')
            # update item-factor matrix
            self.item_factor_matrix = self.update_matrix(self.user_factor_matrix, mode='item')

        t1 = time.time()
        print('Building the model in %f seconds' % (t1 - t0))

    def update_matrix(self, F, mode='user'):
        '''
        Parameters
        ----------
        F (ndarray): Fixed matrix (i.e. user-factor matrix or item-factor matrix)
        mode (str): Set to 'user' for calculating a user-factor matrix

        Returns
        ----------
        calculated_matrix (ndarray)
        '''
        n_rows = self.n_users if mode == 'user' else self.n_items
        calculated_matrix = np.zeros((n_rows, self.n_factors))
        # Rating matrix (transposed when calculating item-factor matrix)
        R = self.train_matrix if mode == 'user' else self.train_matrix.T
        # f * f dentity matrix (f: n_factors)
        E = np.eye(self.n_factors)

        C = self.confidence_matrix if mode == 'user' else self.confidence_matrix.T

        for i, Ci in enumerate(C):
            # m * m or n * n diagonal matrix (n: n_users, m: n_items)
            D = np.diag(Ci)
            # Number of ratings that the user or the item has
            n = np.count_nonzero(Ci) if np.count_nonzero(Ci) > 0 else 1

            FTF = np.dot(F.T, np.dot(D, F))
            # A is a f * f matrix
            A = FTF + self.reg_param * n * E
            # V is a f * 1 matrix
            V = np.dot(F.T, np.dot(D, R[i].T))
            # Update factor matrix
            calculated_matrix[i] = np.linalg.solve(A, V)

        return calculated_matrix

    def predict(self):
        self.prediction = np.dot(self.user_factor_matrix, self.item_factor_matrix.T)
        return self.prediction

    def recommend_top_n(self, user, top_n):
        '''Recommend top n items for a given user.
        Parameters
        ----------
        user (int)
        top_n (int)

        Returns
        ----------
        top_n_items (ndarray)
        '''
        if self.prediction is None:
            self.predict()
        train_s = pd.Series(self.train_matrix[user, :])
        pred_s = pd.Series(self.prediction[user, :])
        recommend = pred_s[train_s == 0].sort_values(ascending=False)
        top_n_items = recommend.head(top_n).index.get_values()

        return top_n_items

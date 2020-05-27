"""
Classifiers which create an ensemble of models.

pyDataSci, Helper functions for binary classification problems.

Copyright (C) 2020  F. P. Chmiel

Email:francischmiel@hotmail.co.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import warnings
import numpy as np
import itertools


#relevant sklearn imports
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold


# This currently doesn't do anything!!!


#class MeanEnsembler(BaseEstimator, ClassifierMixin):
#    """Creates a mean ensemble of predictions for a binary classification 
#    problem.
#      
#    Parameters:
#    -----------    
#
#    metric : {None or callable}, (default=roc_auc_score)
#        Metric used to evaluate the ensemble if targets are provided.
#    
#    Attributes:
#    -----------
#
#    cv_scores_ : array-like, shape [n_classifiers]
#        cv score of each classifier in the ensemble.
#    """
#    _allowed_metrics = [roc_auc_score, average_precision_score]
#    
#    def __init__(self, models, metric=roc_auc_score):
#        self.models = models
#        self.metric = metric
#    
#    def _reset(self):
#        """Reset prediction dependent state of the scaler."""
#        self.cv_scores_ = None
#
#    def fit(self, X, y, verbose=False, init_rounds=10):
#        """
#        Generates the weights to average the ensemble.
#
#        Parameters:
#        -----------
#        P : {array-like, sparse matrix}, shape [n_samples, n_classifers]
#            The independent predictions of the positive class of each classifer
#            to be used in the ensemble.
#
#        y : array-like, shape [n_samples]
#            The target class of each sample in P.
#        
#        weights : array-like, shape [n_classifiers] (default=None)
#            Weights to apply to each classifier.
#
#        verbose : bool, (default=False)
#            Whether to print the classifier performance to screen.
#
#        init_rounds: int, (default=10)
#            Number of times to train ensemble with randomly initalized weights.
#            Used only if method="optimize".
#        
#        Returns:
#        --------
#        self : object
#        """
#        P, y = check_X_y(P, y, accept_sparse=True)
#
#        # prepare the weights
#        if weights is None:
#            weights = np.ones(P.shape[1])
#            if self.method=="weighted":
#                msg = ("No weights provided, equal weights will be used for"
#                       "each classifier in the ensemble.")
#                warnings.warn(msg, UserWarning)
#        self.weights_ = weights
#
#        if self.method=='optimize':
#            ensemble_score = 0 # higher score is better
#            num_clf = P.shape[1]
#            # check how individual models performs and those with random weights
#            single_model_ws = np.zeros((num_clf, num_clf))
#            np.fill_diagonal(single_model_ws, 1)
#            w_arr = np.c_[single_model_ws, 
#                          np.random.rand((num_clf, init_rounds))]
#            for i in range(num_clf+init_rounds):
#                ws = w_arr[:,i] / np.sum(w_arr[:,i])
#                ps = (P*ws).mean(axis=1)
#                score = self.metric(y, ps)
#                print('Model {0} score: {1:.3f}'.format(i, score))
#                print('Weights: {}'.format(ws))
#                if score>ensemble_score:
#                    self.weights_ = ws
#                    ensemble_score = score
#        return self
#    
#    def predict(self, P, y=None):
#        """
#        Creates the prediction ensemble.
#
#        Parameters:
#        -----------
#        P : {array-like, sparse matrix}, shape [n_samples, n_classifers]
#            The independent predictions of each classifer to be used in the
#            ensemble.
#
#        y : 
#            Ignored
#
#        Returns:
#        --------
#        ensemble_predictions: array-like
#            The predictions of the ensemble of classifiers provided in P.
#        """
#        check_is_fitted(self, 'weights_')
#        P = check_array(P)
#        return (P*self.weights_).mean(axis=1)
    
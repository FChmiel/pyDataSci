import numpy as np

#relevant sklearn imports
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

class Ensembler(BaseEstimator, ClassifierMixin):
    """Creates an (weighted) averaging ensemble of predictions.
      
    Parameters:
    -----------    
    method : str, optional (default='mean')
        Specifies the method to used when creating the ensemble.
        Must be one of 'mean', 'weighted' or 'optimized'.
        If method=='mean' and averaging ensemble is made.
        If method=='weighted' a weighted average ensemble is made, with
        user specified weights. 
        If method=='optimizied'  a weighted average ensemble is made but the 
        weights are optimized using cross-validation.
    metric : {None or sklearn.metric}, (default=roc_auc_score)
        Metric used to evaluate the ensemble if targets are provided.
    
    Attributes:
    -----------
    weights_ : array-like, shape [n_classifiers]
        Weights of each classifer used in the averaging ensemble.
    cv_scores_ : array-like, shape [n_classifiers]
        cv score of each classifier in the ensemble.

    Examples:
    TO WRITE show a 2 model ensemble.

    TO DO:
        - Add an option to add a sklearn model and create a stack by cross-validation.
    """
    def __init__(self,
                 method='mean',
                 metric=roc_auc_score):
        self.method = method
        self.metric = metric
    
    def _reset(self):
        """Reset prediction dependent state of the scaler."""
        self.weights_ = None

    def fit(self, P, y, weights=None):
        """
        Generates the weights to average the ensemble.

        Parameters:
        -----------
        P : {array-like, sparse matrix}, shape [n_samples, n_classifers]
            The independent predictions of each classifer to be used in the
            ensemble.
        y : {array-like}, shape [n_samples]
            The target class of each sample in P.
        """
        self.weights_ = weights
        if self.weights_ is None:
            self.weights_ = np.ones(P.shape[1])
        # optimize weights if required. TODO
        # calculate the cv scores of the ensemble
    
    def predict(self, P, y=None):
        """
        Creates the prediction ensemble.

        Parameters:
        -----------
        P : {array-like, sparse matrix}, shape [n_samples, n_classifers]
            The independent predictions of each classifer to be used in the
            ensemble.
        y : 
            Ignored
        """
        return (P*self.weights_).sum(axis=1)
    
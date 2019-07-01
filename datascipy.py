import numpy as np

#relevant sklearn imports
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

class Ensembler(BaseEstimator, TransformerMixin):
    """Creates an ensemble of predictions.
      
    Parameters:
    -----------    
    predictions : array-like, shape [n_samples, n_classifiers]
                  The predictions of each of the n_classifiers.
    method : str, the method to create the ensemble with. 
             Options include: 'mean', 'weighted' or 'optimized'.
             If 'mean' the ensemble is simply averaged.
             If 'weighted' the weights parameter is used to weight the 
             predictions of each classifier.
             If 'optimizied' the weights are optimized using cross-validation.
    weights : {None or array-like, shape [n_classifiers]}
              Initial weights of each classifier. If None they the initial
              weights are set to an array of ones.
    targets : {None or array-like, shape [n_samples]}
              Target of each training instance.
    metric : None or function from sklearn.metric
             Metric used to evaluate the ensemble if targets is not None.
    
    Attributes:
    -----------
    To Write.

    Examples:
    TO WRITE show a 2 model ensemble.

    TO DO:
        - Add an option to add a sklearn model and create a stack by cross-validation.
    """
    def __init__(self,
                 predictions,
                 method='mean',
                 weights=None,
                 targets=None,
                 metric=roc_auc_score):
        self.ps = predictions
        self.method = method
        self.weights = weights
        self.metric = metric
    
    def _reset(self):
        """Reset prediction dependent state of the scaler."""
        pass
        # self.weights_ = None

    def fit(self):
        pass
    
    def transform(self):
        ensemble_predictions = 1
        return ensemble_predictions
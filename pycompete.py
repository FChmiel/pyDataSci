import warnings
import numpy as np


#relevant sklearn imports
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

class Ensembler(BaseEstimator, ClassifierMixin):
    """Creates an (weighted) averaging ensemble of predictions.
      
    Parameters:
    -----------    
    method : str, optional (default="mean")
        Specifies the method to used when creating the ensemble.
        Must be one of "mean", "weighted" or "optimized".
        If method=="mean" and averaging ensemble is made.
        If method=="weighted" a weighted average ensemble is made, with
        user specified weights. 
        If method=="optimizied"  a weighted average ensemble is made but the 
        weights are optimized using cross-validation.

    metric : {None or callable}, (default=roc_auc_score)
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
        - Add optimizing weights code.
    """
    def __init__(self, method="mean", metric=roc_auc_score):
        self.method = method
        if callable(metric):
            self.metric = metric
        else:
            raise Exception("metric must be callable and consistent with the"
                            "sklearn.metrics module.")
    
    def _reset(self):
        """Reset prediction dependent state of the scaler."""
        self.weights_ = None

    def fit(self, P, y, weights=None, verbose=False):
        """
        Generates the weights to average the ensemble.

        Parameters:
        -----------
        P : {array-like, sparse matrix}, shape [n_samples, n_classifers]
            The independent predictions of each classifer to be used in the
            ensemble.

        y : array-like, shape [n_samples]
            The target class of each sample in P.
        
        weights : array-like, shape [n_classifiers] (default=None)
            Weights to apply to each classifier.

        verbose : bool, (default=False)
            Whether to print the classifier performance to screen.
        
        Returns:
        --------
        self : object
        """
        # prepare the weights
        if weights is None:
            weights = np.ones(P.shape[1])
            if self.method=="weighted":
                msg = ("No weights provided, equal weights will be used for"
                       "each classifier in the ensemble.")
                warnings.warn(msg, UserWarning)
        self.weights_ = weights

        if self.method=='optimize':
            pass
            # optimize weights if required. TODO

        # calculate the cv scores of the ensemble
    
        return self
    
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

        Returns:
        --------
        ensemble_predictions: array-like
            The predictions of the ensemble of classifiers provided in P.
        """
        return (P*self.weights_).sum(axis=1)
    
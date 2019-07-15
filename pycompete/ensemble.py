import warnings
import numpy as np
import itertools


#relevant sklearn imports
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold

class BinaryEnsembler(BaseEstimator, ClassifierMixin):
    """Creates an (weighted) averaging ensemble of predictions for
    a binary classification problem.
      
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
    ---------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    >>> linear_clf = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    >>> linear_ps = linear_clf.predict_proba(X_test)[:,1]
    >>> svm_clf = SVC(probability=True, gamma='auto').fit(X_train, y_train)
    >>> svm_ps = svm_clf.predict_proba(X_test)[:,1]
    >>> ensembler = Ensembler(method='mean')
    >>> P = np.c_[linear_ps, svm_ps]
    >>> ensembler.fit(P, y_test)
    >>> ensembler_predictions = ensembler.predict(P)
    >>> print(ensembler.weights_)

    TO DO:
        - Add optimizing weights code.
    """
    _allowed_methods = ["mean", "weighted", "optimize"]
    
    def __init__(self, method="mean", metric=roc_auc_score):
        self.method = method
        self.metric = metric
    
    def _reset(self):
        """Reset prediction dependent state of the scaler."""
        self.weights_ = None
        self.cv_scores_ = None

    def fit(self, P, y, weights=None, verbose=False, init_rounds=10):
        """
        Generates the weights to average the ensemble.

        Parameters:
        -----------
        P : {array-like, sparse matrix}, shape [n_samples, n_classifers]
            The independent predictions of the positive class of each classifer
            to be used in the ensemble.

        y : array-like, shape [n_samples]
            The target class of each sample in P.
        
        weights : array-like, shape [n_classifiers] (default=None)
            Weights to apply to each classifier.

        verbose : bool, (default=False)
            Whether to print the classifier performance to screen.

        init_rounds: int, (default=10)
            Number of times to train ensemble with randomly initalized weights.
            Used only if method="optimize".
        
        Returns:
        --------
        self : object
        """
        P, y = check_X_y(P, y, accept_sparse=True)

        # prepare the weights
        if weights is None:
            weights = np.ones(P.shape[1])
            if self.method=="weighted":
                msg = ("No weights provided, equal weights will be used for"
                       "each classifier in the ensemble.")
                warnings.warn(msg, UserWarning)
        self.weights_ = weights

        if self.method=='optimize':
            ensemble_score = 0 # higher score is better
            num_clf = P.shape[1]
            # check how individual models performs and those with random weights
            single_model_ws = np.zeros((num_clf, num_clf))
            np.fill_diagonal(single_model_ws, 1)
            w_arr = np.c_[single_model_ws, 
                          np.random.rand((num_clf, init_rounds))]
            for i in range(num_clf+init_rounds):
                ws = w_arr[:,i] / np.sum(w_arr[:,i])
                ps = (P*ws).mean(axis=1)
                score = self.metric(y, ps)
                print('Model {0} score: {1:.3f}'.format(i, score))
                print('Weights: {}'.format(ws))
                if score>ensemble_score:
                    self.weights_ = ws
                    ensemble_score = score
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
        check_is_fitted(self, 'weights_')
        P = check_array(P)
        return (P*self.weights_).mean(axis=1)
    
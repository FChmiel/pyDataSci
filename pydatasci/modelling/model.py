"""
Wrappers for fitting, tuning and predicting modelling. This is useful for rapid
prototyping of models when dealing with highly-structured data.
"""
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

def fit_model(X, y, model=XGBClassifier, params={}, uid=None, 
              cv_splitter=GroupKFold, nsplits=5, evaluate=True):
    """
    Fits a model using given params and CV method and returns the fitted
    classifier. If multiple-fold cross-validation is used each individual 
    model is returned stored in a length k list, where k is the number of
    cross-validation folds.
    
    Parameters:
    -----------
    X : np.array,
        Training data shape (n,m) where n is the number of instances and m the
        number of features.
        
    y : np.array,
        Labels for training data, shape (n,).
    
    model : sklearn.base.BaseEstimater (default=XGBClassifier),
        The classifier to fit. Must be compatible with the sklearn API.
    
    params : dict,
        Parameter dictionary to be passed to the model.
    
    uid : np.array (default=None),
        User ids for each training instance, shape (n,). This is passed to the
        cv_splitter so (for example) patient-wise validation folds can be 
        created.
    
    cv_splitter : cross-validation generator, 
        The cross-validation splitter (consitent with sklearn API).
        
    nsplits : int,
        The number of validation folds to create. Passed to cv_splitter.
    
    evaluate: bool,
        Whether to validate the models performance on out-of-fold data.
        
    Returns:
    --------
    models, sklearn.base.BaseEstimater 
        Returns the fitted model(s) instance.
    """

    X, y = check_X_y(X, y, accept_sparse=True)
    
    splitter = cv_splitter(n_splits=nsplits)

    models = []
    for i, (train_index, test_index) in enumerate(splitter.split(X, y, uid)):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        model = model(**params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        if evaluate:
            train_preds = model.predict_proba(X_train)[:,1]
            train_score = roc_auc_score(y_train, train_preds)
            test_preds = model.predict_proba(X_test)[:,1]
            test_score = roc_auc_score(y_test, test_preds)
            print(f'Training score, Fold {i}: {train_score}')
            print(f'Test score, Fold {i}: {test_score}')
        models.append(model)
    return models

def predict_from_model(model, X):
    """
    Makes predictions using a pre-fitted model. If the provided model is an
    array-like object (i.e., a list of fitted models) the average of each models
    prediction is returned.
    
    Parameters:
    -----------
    model : {sklearn.base.BaseEstimater, array-like},
        A pre-fit model, compatiable with the sklearn API. If the model is a
        list of models the average prediction of each model is returned.

    X : np.array,
        Training data shape (n,m) where n is the number of instances and m the
        number of features.
        
    Returns:
    --------
    preds : np.array,
        The predictions, length (n,) array.
    """
    try:
        check_is_fitted(model, 'feature_importances_')
        preds = model.predict_proba(X)
    except AttributeError:
        # take average prediction of each model
        preds = np.zeros(len(X))
        num_models = len(model)
        for m in model:
            check_is_fitted(model, 'feature_importances_')
            preds += m.predict_proba(X) / num_models        
    return preds
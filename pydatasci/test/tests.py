import unittest
import numpy as np

import pydatasci.modelling.ensemble as ensemble
import pydatasci.modelling.model as modelling
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

class BinaryEnsemblerTestCase(unittest.TestCase):
    """BinaryEnsembler test cases."""
    
    def setUp(self):
        """Creates example targets and predictions array."""

        global targets, predictions, true_averages, ex_weights
        global weighted_average_predictions

        targets = np.array([0,1,1,0,1,1,0,1])

        # each permutation of binary target for 3 classifiers
        predictions = np.array([[1,0,0],
                                [0,1,0],
                                [0,0,1],
                                [1,1,0],
                                [0,1,1],
                                [1,0,1],
                                [0,0,0],
                                [1,1,1]])
        true_averages = np.array([1,1,1,2,2,2,0,3])/3
        
        # weights of each classifier and weighted predictions
        ex_weights= np.array([0.5, 0.2, 0.5])
        weighted_average_predictions = (ex_weights*predictions).mean(axis=1)
    
    def tearDown(self):
        """TearDown"""
        pass
    
    def test_preserves_number_of_rows(self):
        """Verifys that averaging is performed along the columnar axis."""
        ensembler = ensemble.BinaryEnsembler(method="mean")
        ensembler.fit(predictions, targets)
        ensemble_ps = ensembler.predict(predictions)
        self.assertTrue(ensemble_ps.shape[0]==predictions.shape[0])
    
    def test_rows_average_correctly(self):
        """Verifys that the sum of each row is correct when the mean
        method is used."""
        ensembler = ensemble.BinaryEnsembler(method="mean")
        ensembler.fit(predictions, targets)
        ensemble_ps = ensembler.predict(predictions)
        self.assertTrue(all(ensemble_ps==true_averages))

    def test_weights_are_applied(self):
        """Verify correct ensembled predictions when weights applied."""
        ensembler = ensemble.BinaryEnsembler(method="weighted")
        ensembler.fit(predictions, targets, weights=ex_weights)
        ensemble_ps = ensembler.predict(predictions)
        self.assertTrue(all(ensemble_ps==weighted_average_predictions))

    def test_only_accept_numpy_arrays(self):
        """Verifys ValueError exception is raised when a array is not
        provided"""
        with self.assertRaises(ValueError):
            ensembler = ensemble.BinaryEnsembler()
            ensembler.fit(1, targets)
        with self.assertRaises(ValueError):
            ensembler = ensemble.BinaryEnsembler()
            ensembler.fit(predictions, 1)
        with self.assertRaises(ValueError):
            ensembler = ensemble.BinaryEnsembler()
            ensembler.fit("example", targets)
        with self.assertRaises(ValueError):
            ensembler = ensemble.BinaryEnsembler()
            ensembler.fit(predictions, "example")

    def test_warning_raised_if_no_weights_provided(self):
        """Verifys a warning is raised if weights are not provided but
        method=="weighted"."""
        with self.assertWarns(UserWarning):
            ensembler = ensemble.BinaryEnsembler(method="weighted")
            ensembler.fit(predictions, targets)


class PredictFromModelTestCase(unittest.TestCase):
    """predict_from_model test cases."""
    
    def setUp(self):
        """Creates example data and pre-fitted models."""

        global X, y, model

        X, y = load_breast_cancer(return_X_y=True)
        model = LogisticRegression().fit(X,y)

    def tearDown(self):
        """TearDown"""
        pass
    
    def test_single_model_creates_predictions(self):
        """Verifys that an array of the same length as y is produced when 
        a model is provided."""
        preds = modelling.predict_from_model(model, X)
        self.assertTrue(len(preds)==len(y))

    def test_multiple_model_creates_predictions(self):
        """Verifys that an array of the same length as y is produced when 
        a list of models is provided."""
        preds = modelling.predict_from_model([model, model, model], X)
        self.assertTrue(len(preds)==len(y))

class FitModelTestCase(unittest.TestCase):
    """fit_model test cases."""
    
    def setUp(self):
        """Creates example data and pre-fitted models."""

        global X, y, model, n_splits

        X, y = load_breast_cancer(return_X_y=True)
        model = LogisticRegression()
        n_splits = 3

    def tearDown(self):
        """TearDown"""
        pass
    
    def test_multiple_models_are_returned(self):
        """Verifys that the same number of models as n_splits is returned 
        as a list"""
        models = modelling.fit_model(model, X)
        self.assertTrue(len(models)==n_splits)
        
if __name__ == "__main__":
    unittest.main()
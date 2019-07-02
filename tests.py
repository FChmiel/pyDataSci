import unittest
import numpy as np
import pycompete as compete

# To do:
# make the Ensembler test case consistent with sklearn objects.

class EnsemblerTestCase(unittest.TestCase):
    """Ensembler test cases."""
    
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
        weighted_average_predictions = (ex_weights*predictions).sum(axis=1)
    
    def tearDown(self):
        """TearDown"""
        pass
    
    def test_preserves_number_of_rows(self):
        """Verifys that averaging is performed along the columnar axis."""
        ensembler = compete.Ensembler(method='mean')
        ensemble_ps = ensembler.fit_transform(predictions, targets)
        self.assertTrue(ensemble_ps.shape[0]==predictions.shape[0])
        self.assertTrue(ensemble_ps.shape[1]==1)
    
    def test_rows_average_correctly(self):
        """Verifys that the sum of each row is correct when the mean
        method is used."""
        ensembler = compete.Ensembler(method='mean')
        ensemble_ps = ensembler.fit_transform(predictions, targets)
        self.assertTrue(all(ensemble_ps==true_averages))

    def test_weights_are_applied(self):
        """Verify correct ensembled predictions when weights applied."""
        ensembler = compete.Ensembler(method='weighted', weights=ex_weights)
        ensemble_ps = ensembler.fit_transform(predictions, targets)
        self.assertTrue(all(ensemble_ps==weighted_average_predictions))

    def test_only_accept_numpy_arrays(self):
        """Verifys TypeError exception is raised when a numpy array is not
        provided"""
        with self.assertRaises(TypeError):
            ensembler = compete.Ensembler()
            ensembler.fit_transform(1, targets)
        with self.assertRaises(TypeError):
            ensembler = compete.Ensembler()
            ensembler.fit_transform(predictions, 1)
        with self.assertRaises(TypeError):
            ensembler = compete.Ensembler()
            ensembler.fit_transform('example', targets)
        with self.assertRaises(TypeError):
            ensembler = compete.Ensembler()
            ensembler.fit_transform(predictions, 'example')

if __name__ == '__main__':
    unittest.main()
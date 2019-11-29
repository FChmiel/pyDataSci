"""
A class to be used in hyperparameter tuning, such that all models can be stored.
"""
import pandas as pd
import os

class Recorder():
    """Base recorder class"""
   
    def __init__(self, filename, columns):
        self.filename = filename
        self.columns = columns
        self.create_record()
    
    def create_record(self):
        if not os.path.isfile(self.filename):
            record = pd.DataFrame(columns=self.columns)
            record.to_csv(self.filename, index=False) 

    def add_entry(self, entry):
        """
        Parameters:
        -----------
        entry, pd.DataFrame
            Dataframe containing the entries to append to the file.    
        """
        if entry.shape[1]!=len(self.columns):   
            raise Exception('entry columns must equal record columns')
        entry.to_csv(self.filename, index=False, mode='a')
    
    def remove_entry(self):
        # loads the record and removes an entry
        pass

    def get_record(self):
        """
        Loads and returns the record as a pandas DataFrame.
        """
        record = pd.read_csv(self.filename)
        return record


class TrainingRecord(Recorder):
    """
    Recorder for recording information about models created during 
    the hyperparameter tuning process.
    """
    def __init__(self):
        pass

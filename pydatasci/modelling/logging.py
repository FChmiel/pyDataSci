"""
Sets of functions for helping to log the results of experiments
"""


def save_record(run_name, summary, seed=None, fpath=None, number='auto'):
    """
    Saves provided summary and results of a model as a json file.

    Parameters:
    -----------
    run_name : str, 
        The name of the model.

    summary : dict, 
        Dictionary summarizing the model and any associated results.

    seed : int (default=None),
        The random seed used in the model training.

    fpath : str (default=None),
        Path to save the result to. If None, a new folder is created named
        'results' in the current directory.

    number : {int, str} (default='auto'),
        Numerical suffix to the filename. If auto the function checks the 
        specified directory for files of the same name and increments the
        numerical suffix by one if the file exists.
    """

def create_run_summary():
    """
    Creates a .csv which collates a series of single runs.

    Parameters:
    -----------
    """
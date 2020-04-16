"""
Sets of functions for helping to log the results of experiments
"""
import os
import json

def save_record(run_name, summary, fdir='results/', number='auto'):
    """
    Saves provided summary and results of a model as a json file.

    Parameters:
    -----------
    run_name : str, 
        The name of the model.

    summary : dict, 
        Dictionary summarizing the model and any associated results.

    fdir : str (default='results/'),
        Path to save the result to. If not provided, a new folder is created
        named 'results' in the current directory.

    number : {int, str} (default='auto'),
        Numerical suffix to the filename. If auto the function checks the 
        specified directory for files of the same name and increments the
        numerical suffix by one if the file exists.
    """
    
    # create directory if not already existing
    if not os.isdir(fdir):
        try:
            os.mkdir(fdir)
        except OSError:
            print(f'Creation of {fdir} failed.')
    
    # create file name
    if number=='auto':
        number = 0
        fname = fdir + run_name + f'_{number}'
        while os.path.isfile(fname):
            number += 1
            fname = fdir + run_name + f'_{number}'
    else:
        fname = fdir + run_name + f'_{number}'

    # save the file
    json.dump(summary, open( f'{fname}.json', 'w' ) )
   


def create_run_summary():
    """
    Creates a .csv which collates a series of single runs.

    Parameters:
    -----------
    """
#!/usr/bin/env python3
import sys
import os


def load_raw_data(filename: str):
    """Loads raw signal data from a file.

    The function reads a text file, skipping the first line (presumably a header).
    Each subsequent line is expected to contain two space-separated values:
    a signal identifier (string) and a corresponding signal value (float).
    The function stores this data in a dictionary where keys are the signal
    identifiers and values are lists containing a single float representing
    the signal value.

    Args:
        filename: The path to the text file containing the raw signal data.

    Returns:
        A dictionary where keys are signal identifiers (strings) and values
        are lists containing a single float representing the signal value.
    """ 
    signal_data = {}
    with open(filename, 'r') as infile:
        infile.readline()
        
        for line in infile:
            line_data = line.split()
            signal_data[line_data[0]] = [float(line_data[1])]
    
    
    return signal_data       

def compile_dataset(path: os.path):
    """Compiles a dataset by loading and merging data from multiple files in a directory.

    This function iterates through all files in the specified directory. For each file,
    it calls the `load_raw_data` function to load the signal data. The data from each
    file is then merged into a single dictionary. It also extracts the subject identifier
    from the filename (assuming filenames start with 'subjectID_').

    Args:
        path: The path to the directory containing the data files.

    Returns:
        A tuple containing:
          - data: A dictionary where keys are signal identifiers (strings) and values
            are lists of floats, with each float representing a signal value from a
            different file.
          - subjects: A list of subject identifiers (strings) extracted from the filenames.
    """
    data = None
    subjects = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        subjects.append(file.split('_', maxsplit=1)[0])
        if data is None:
            data = load_raw_data(filepath)
        else:
            next_data = load_raw_data(filepath)
            for key, value in next_data.items():
                data[key].append(next_data.get(key)[0])
                
    return data, subjects

def save_dataset(filename: str, data: dict[str, list[float]] , subjects: list[str]):
    """Saves the compiled dataset to a tab-separated file.

    The function writes the dataset to a text file. The first line contains
    a header with 'Probe_ID' followed by the list of subject identifiers.
    Subsequent lines represent the signal data for each probe. Each line starts
    with the probe identifier, followed by the corresponding signal values for
    each subject, separated by tabs.

    Args:
        filename: The path to the file where the dataset will be saved.
        data: A dictionary where keys are probe identifiers (strings) and values
            are lists of floats, representing the signal values for each subject.
        subjects: A list of subject identifiers (strings), corresponding to the
            order of values in the lists within the `data` dictionary.
    """
    with open(filename, 'w') as outfile:
        outfile.write('\t'.join(['Probe_ID']+subjects)+'\n')
        for key, value in data.items():
            
            outfile.write('\t'.join([key]+[str(item) for item in value])+'\n')
            
save_dataset(sys.argv[1], *compile_dataset(sys.argv[2]))
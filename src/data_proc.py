#!/usr/bin/env python3
import sys
import os


def load_raw_data(filename: str):
    
    signal_data = {}
    with open(filename, 'r') as infile:
        infile.readline()
        
        for line in infile:
            line_data = line.split()
            signal_data[line_data[0]] = [float(line_data[1])]
    
    
    return signal_data       

def compile_dataset(path: os.path):
    
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
    with open(filename, 'w') as outfile:
        outfile.write('\t'.join(['Probe_ID']+subjects+['\n']))
        for key, value in data.items():
            
            outfile.write('\t'.join([key]+[str(item) for item in value]+['\n']))
            
save_dataset(sys.argv[1], *compile_dataset(sys.argv[2]))
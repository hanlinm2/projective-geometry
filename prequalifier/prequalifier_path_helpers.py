import glob
from pandas.core.common import flatten
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from torchvision.models import resnet50
import argparse

def load_all_paths(base_path, data_paths):
    all_train_paths = []
    all_val_paths = []
    all_test_paths = []
    
    for data_path in data_paths:
        data_path = base_path + data_path

        train_paths = []
        val_paths = []
        test_paths = []
        
        for path in glob.glob(data_path):
            train_paths.append(glob.glob(path + "/train/*/*"))
            val_paths.append(glob.glob(path + "/val/*/*"))
            test_paths.append(glob.glob(path + "/test/*/*"))

        train_paths = list(flatten(train_paths))
        val_paths = list(flatten(val_paths))
        test_paths = list(flatten(test_paths))

        all_train_paths += train_paths
        all_val_paths += val_paths
        all_test_paths += test_paths

    all_train_paths = list(set(all_train_paths))
    all_val_paths = list(set(all_val_paths))
    all_test_paths = list(set(all_test_paths))

    return all_train_paths, all_val_paths, all_test_paths


def timestamp_paths_from_data_paths(base_path, data_paths):
    recent_timestamp_paths = {}
    for index, data_path in enumerate(data_paths):
        full_path = base_path + data_path
        recent_string = data_path.split("_")[0]
        dataset_string = data_path.split("_")[1]
        key = f"{recent_string}_{dataset_string}"
        paths = []
        for path in glob.glob(full_path):
            paths.append(glob.glob(path + "/test/*/*"))
        paths = list(flatten(paths))
        recent_timestamp_paths[key] = paths
    return recent_timestamp_paths

def load_recent_timestamp_paths(base_path, recent_timestamp_data_paths, category):
    if category == "indoor" or category == "outdoor":
        return timestamp_paths_from_data_paths(base_path, recent_timestamp_data_paths)
        
    else:
        indoor_recent_timestamp_data_paths, outdoor_recent_timestamp_data_paths = recent_timestamp_data_paths
        
        indoor_recent_timestamp_paths = timestamp_paths_from_data_paths(base_path, indoor_recent_timestamp_data_paths)
        outdoor_recent_timestamp_paths = timestamp_paths_from_data_paths(base_path, outdoor_recent_timestamp_data_paths)
        
        combined_recent_timestamp_paths = {}
        
        for key in list(indoor_recent_timestamp_paths.keys()):
            combined_recent_timestamp_paths[key] = indoor_recent_timestamp_paths[key] + outdoor_recent_timestamp_paths[key]
        
        return combined_recent_timestamp_paths
        
            
                
                
            
            
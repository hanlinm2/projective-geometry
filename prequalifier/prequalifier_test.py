import glob
from pandas.core.common import flatten
import torch, torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import argparse

from prequalifier_model import Prequalifier, load_model
from prequalifier_path_helpers import load_all_paths, load_recent_timestamp_paths
from prequalifier_dataset import get_test_dataloaders, get_recent_timestamp_dataloaders
from prequalifier_test_helpers import test, batch_test




if __name__ == "__main__":

    idx_to_class = {0: 'real', 1: 'gen'}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    print(class_to_idx)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default="indoor", help='choose: indoor, outdoor or combined')
    args = parser.parse_args()

    category = args.category
    print("category:", category)

    base_path = "../dataset/"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    indoor_recent_timestamp_paths = ["Recent_SDXL_Indoor", "Recent_Deepfloyd_Indoor", "Recent_Pixart_Indoor", "Recent_Kandinsky_Indoor"]
    outdoor_recent_timestamp_paths = ["Recent_SDXL_Outdoor", "Recent_Deepfloyd_Outdoor", "Recent_Pixart_Outdoor", "Recent_Kandinsky_Outdoor"]
    
    misclassified_indoor_file = "./misclassified_indoor_list.pkl"
    misclassified_outdoor_file = "./misclassified_outdoor_list.pkl"
    misclassified_combined_file = "./misclassified_combined_list.pkl"

    unconfident_indoor_file = "./unconfident_indoor_list.pkl"
    unconfident_outdoor_file = "./unconfident_outdoor_list.pkl"
    unconfident_combined_file = "./unconfident_combined_list.pkl"

    if category == "indoor":
        image_data_paths = ["Kandinsky_Indoor"]
        recent_timestamp_data_paths = indoor_recent_timestamp_paths

        misclassified_image_paths = sorted(pickle.load(open(misclassified_indoor_file, "rb")))
        unconfident_image_paths = sorted(pickle.load(open(unconfident_indoor_file, "rb")))
        
        save_path = "./checkpoints/Prequalifier_indoor.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    elif category == "outdoor":

        image_data_paths = ["Kandinsky_Outdoor"]
        recent_timestamp_data_paths = outdoor_recent_timestamp_paths

        misclassified_image_paths = sorted(pickle.load(open(misclassified_outdoor_file, "rb")))
        unconfident_image_paths = sorted(pickle.load(open(unconfident_outdoor_file, "rb")))
        
        save_path = "./checkpoints/Prequalifier_outdoor.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    elif category == "combined":
        image_data_paths = ["Kandinsky_Indoor", "Kandinsky_Outdoor"]
        recent_timestamp_data_paths = [indoor_recent_timestamp_paths, outdoor_recent_timestamp_paths]
        
        misclassified_image_paths = sorted(pickle.load(open(misclassified_combined_file, "rb")))
        unconfident_image_paths = sorted(pickle.load(open(unconfident_combined_file, "rb")))
        
        save_path = "./checkpoints/Prequalifier_combined.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    train_image_paths, val_image_paths, test_image_paths = load_all_paths(base_path, image_data_paths)
    recent_timestamp_paths = load_recent_timestamp_paths(base_path, recent_timestamp_data_paths, category)
    
    easy_dataloader, unconfident_dataloader, misclassified_dataloader = get_test_dataloaders(test_image_paths, unconfident_image_paths, misclassified_image_paths, class_to_idx)
    
    recent_timestamp_dataloaders = get_recent_timestamp_dataloaders(recent_timestamp_paths, class_to_idx)
   
    test(model, easy_dataloader, save_path, "easy")
    test(model, unconfident_dataloader, save_path, "unconfident")
    test(model, misclassified_dataloader, save_path, "misclassified")
    
    roc_plot_labels = ["SDXL", "Deepfloyd", "PixArt-α", "Kandinsky"]
    roc_plot_colors = ["orange", "red", "blue", "green"]
    plot_name = save_path.split("/")[-1].split(".")[-2]
    roc_save_path = f"./plots/{plot_name}_recent_timestamps"
    
    batch_test(model, recent_timestamp_dataloaders, save_path, roc_plot_labels, roc_plot_colors, roc_save_path)
    
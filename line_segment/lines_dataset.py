import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



class LineSegmentDataset(Dataset):
    def __init__(self, image_paths, image_path_to_lines, class_to_idx, transform = False, number_of_lines = 250):
        self.image_paths = image_paths
        self.transform = transform
        self.number_of_lines = number_of_lines
        self.image_path_to_lines = image_path_to_lines
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        
        flattened_lines = self.image_path_to_lines[image_filepath]
        
        line_disparity = self.number_of_lines - flattened_lines.shape[0] 
        
        if line_disparity > 0:
            # Duplicate points
            sampling_indices = np.random.choice(flattened_lines.shape[0], line_disparity)
            new_points = flattened_lines[sampling_indices, :]
            flattened_lines = np.concatenate((flattened_lines, new_points),axis=0)
        else:
            sampling_indices = np.random.choice(flattened_lines.shape[0], self.number_of_lines)
            flattened_lines = flattened_lines[sampling_indices, :]
        
        flattened_lines = torch.tensor(flattened_lines)

        label = image_filepath.split("/")[-2]
        label = self.class_to_idx[label]
        
        return flattened_lines, label
    
def get_train_dataloaders(train_image_paths, val_image_paths, image_path_to_lines, class_to_idx):
    train_dataset = LineSegmentDataset(train_image_paths, image_path_to_lines, class_to_idx)
    val_dataset = LineSegmentDataset(val_image_paths, image_path_to_lines, class_to_idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size = 256, shuffle = True, num_workers=6)
    
    return train_dataloader, val_dataloader

def get_test_dataloaders(test_image_paths, unconfident_image_paths, misclassified_image_paths, image_path_to_lines, class_to_idx):
    
    easy_image_paths = list(set(test_image_paths) - set(unconfident_image_paths + misclassified_image_paths))
    
    easy_dataset = LineSegmentDataset(easy_image_paths, image_path_to_lines, class_to_idx)
    easy_dataloader = DataLoader(easy_dataset, batch_size=256, shuffle=True)
    
    unconfident_dataset = LineSegmentDataset(unconfident_image_paths + misclassified_image_paths, image_path_to_lines, class_to_idx)
    unconfident_dataloader = DataLoader(unconfident_dataset, batch_size=256, shuffle=True)
    
    misclassified_dataset = LineSegmentDataset(misclassified_image_paths,image_path_to_lines, class_to_idx)
    misclassified_dataloader = DataLoader(misclassified_dataset, batch_size=256, shuffle=True)
    
    return easy_dataloader, unconfident_dataloader, misclassified_dataloader

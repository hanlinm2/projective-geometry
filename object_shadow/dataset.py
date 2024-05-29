import torch
import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image
    
class ShadowObjectDataset(Dataset):
    def __init__(self, shadow_paths, object_paths, class_to_idx, transform = None):
        self.shadow_paths = shadow_paths
        self.object_paths = object_paths
        self.toTensor = torchvision.transforms.ToTensor()
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self. shadow_paths)

    def __getitem__(self, idx):
        shadow_filepath = self.shadow_paths[idx]
        object_filepath = self.object_paths[idx]

        shadow_image = Image.open(shadow_filepath)
        object_image = Image.open(object_filepath)

        shadow_image = self.toTensor(shadow_image)
        object_image = self.toTensor(object_image)

        joined_image = torch.cat([shadow_image, object_image], dim = 0)

        label = shadow_filepath.split("/")[-2]
        label = self.class_to_idx[label]
        return joined_image, label
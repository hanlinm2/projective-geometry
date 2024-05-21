import torchvision.transforms as transforms
import math
import torch
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader


class JPEGCompressionTransform:
    def __init__(self, quality):
        """
        Initialize the transformation with the desired JPEG quality.
        :param quality: JPEG quality level, from 1 (worst) to 95 (best).
        """
        self.quality = quality

    def __call__(self, img):
        """
        Apply JPEG compression to the input image.
        :param img: PIL Image to be compressed.
        :return: Compressed and then decompressed PIL Image.
        """
        # Check if the input is a PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError('Input type should be PIL Image. Got {}'.format(type(img)))

        # Compress and decompress the image using JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)

        return img_jpeg

class BaseDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, is_train = False, blur_prob = 0.1, jpeg_prob = 0.1, set_real = False):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.is_train = is_train
        self.blur_prob = blur_prob
        self.jpeg_prob = jpeg_prob
        self.set_real = set_real
    
    def __len__(self):
        return len(self.image_paths)
    
    # modify get_item based on transform and cv2 = double check perspective_fields architecture
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        
        if self.is_train:
            image = Image.open(image_filepath).convert('RGB')
            image = transforms.Resize((256, 256))(image)
            image = transforms.RandomHorizontalFlip(p = 0.5)(image)
            if random.random() < self.blur_prob:
                sigma = np.random.uniform(0, 3, 1)[0]
                kernel_size = 2*math.ceil(3 * sigma) + 1
                image = transforms.GaussianBlur(kernel_size = kernel_size, sigma = sigma)(image)
            if random.random() < self.jpeg_prob:
                quality = int(np.random.uniform(30, 95, 1)[0])
                image = JPEGCompressionTransform(quality = quality)(image)
            image = transforms.RandomCrop(224)(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            
        else:
            image = Image.open(image_filepath).convert('RGB')
            image = transforms.Resize((256, 256))(image)
            image = transforms.CenterCrop(224)(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)   
                
        if self.set_real:
            label = "real"
        else:
            label = image_filepath.split("/")[-2]
        
        label = self.class_to_idx[label]

        return image, label


def get_test_dataloaders(test_image_paths, unconfident_image_paths, misclassified_image_paths, class_to_idx):
    
    easy_image_paths = list(set(test_image_paths) - set(unconfident_image_paths + misclassified_image_paths))
    
    easy_dataset = BaseDataset(easy_image_paths, class_to_idx, is_train = False)
    easy_dataloader = DataLoader(easy_dataset, batch_size=256, shuffle=True)
    
    unconfident_dataset = BaseDataset(unconfident_image_paths + misclassified_image_paths, class_to_idx, is_train = False)
    unconfident_dataloader = DataLoader(unconfident_dataset, batch_size=256, shuffle=True)
    
    misclassified_dataset = BaseDataset(misclassified_image_paths, class_to_idx, is_train = False)
    misclassified_dataloader = DataLoader(misclassified_dataset, batch_size=256, shuffle=True)
    
    return easy_dataloader, unconfident_dataloader, misclassified_dataloader

    
def get_recent_timestamp_dataloaders(recent_timestamp_paths, class_to_idx):
    recent_timestamp_dataloaders = {}
    for key in list(recent_timestamp_paths.keys()):
        dataset = BaseDataset(recent_timestamp_paths[key], class_to_idx, is_train = False)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        recent_timestamp_dataloaders[key] = dataloader
    return recent_timestamp_dataloaders

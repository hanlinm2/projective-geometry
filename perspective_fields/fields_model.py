import torch
from torchvision.models import resnet50
import torch.nn as nn


class FieldsClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(pretrained = False)

        nr_filters = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(nr_filters, 2)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
    
    
def create_model(target_device):
    model = FieldsClassifier()
    model.to(target_device)
    return model

def load_model(target_device, path_to_checkpoint):
    model = FieldsClassifier()
    try:
        if target_device == "cpu":
            model.load_state_dict(torch.load(path_to_checkpoint, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(path_to_checkpoint))
        print("Successfully Loaded Saved Model")
    except Exception as error:
        print("Failed to load Saved Model")
        print(error)
    model.to(target_device)
    return model
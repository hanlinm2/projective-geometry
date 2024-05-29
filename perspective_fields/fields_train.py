import glob
from pandas.core.common import flatten

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import argparse

from fields_dataset import get_train_dataloaders
from fields_model import create_model


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


def train(model, train_dataloader, val_dataloader, save_path_prefix):
    
    print(f"Training Perspective Fields Classifier: {category.capitalize()} Setting.\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_accuracy = 0.0
    best_epoch = -1
    for epoch in tqdm(range(num_epoch), desc="epochs"):
        model.train()
        loss_epoch = 0.0
        correct = 0
        total = 0
        for train_data in tqdm(train_dataloader, desc="training 1 epoch"):
            inputs, labels = train_data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        accuracy = 100 * correct / total
        train_accuracies.append(accuracy)
        train_losses.append(loss_epoch)
        
        # Validate
        correct = 0
        total = 0
        loss_epoch = 0.0
        model.eval()
        all_predicted = torch.tensor([]).to(device)
        all_labels = torch.tensor([]).to(device)
        all_pred_probs = torch.tensor([]).to(device)
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Validating"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                loss_epoch += loss.item()
                all_predicted = torch.cat((all_predicted, predicted))
                all_labels = torch.cat((all_labels, labels))
                all_pred_probs = torch.cat((all_pred_probs, outputs.data))

        conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
        print(f"Epoch {epoch}:")
        print("Val confusion matrix:")
        print(conf_matrix)
        print(f"{conf_matrix[0].sum().item()} real images, {conf_matrix[1].sum().item()} generated images")
        tn = conf_matrix[0,0]
        tp = conf_matrix[1,1]
        fp = conf_matrix[0,1]
        fn = conf_matrix[1,0]
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")

        save_path = save_path_prefix + f"_{epoch}.pth"
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.2f}")

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
        print(f"Epoch {epoch}: Saving model to {save_path} with val accuracy {accuracy}%")
        print(f"Current Best: epoch{best_epoch}")
        torch.save(model.state_dict(), save_path)
        print()

    print(f"Best Accuracy: {best_accuracy} at epoch {best_epoch}")
    # train and val loss plot
    plt.figure()
    plot_name = save_path_prefix.split("/")[-1]
    plt.plot(train_losses, label = "training loss")
    plt.plot(val_losses, label = "validation loss")
    plt.legend()
    plt.savefig(f"./plots/{plot_name}_losses.png")
    
    # train and val accuracies plot
    plt.figure()
    plot_name = save_path_prefix.split("/")[-1]
    plt.plot(train_accuracies, label = "training accuracies")
    plt.plot(val_accuracies, label = "validation accuracies")
    plt.legend()
    plt.savefig(f"./plots/{plot_name}_accuracies.png")


if __name__ == "__main__":
    num_epoch = 30

    classes = []
    class_to_idx = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default="indoor", help='choose: indoor, outdoor or combined')
    args = parser.parse_args()

    category = args.category
    print("category:", category)

    base_path = "../dataset/"

    if category == "indoor":
        image_data_paths = ["Kandinsky_Indoor"]
    elif category == "outdoor":
        image_data_paths = ["Kandinsky_Outdoor"]
    elif category == "combined":
        image_data_paths = ["Kandinsky_Indoor", "Kandinsky_Outdoor"]

    train_image_paths, val_image_paths, test_image_paths = load_all_paths(base_path, image_data_paths)

    print(f"Image: Loaded {len(train_image_paths)} train, {len(val_image_paths)} val")

    idx_to_class = { 0: 'real', 1: 'gen'}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    print(class_to_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_path_prefix = f"./checkpoints/Fields_{category}"
    
    train_dataloader, val_dataloader = get_train_dataloaders(train_image_paths, val_image_paths, class_to_idx)
    
    model = create_model(target_device = device)

    train(model, train_dataloader, val_dataloader, save_path_prefix)
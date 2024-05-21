import glob
from pandas.core.common import flatten

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import argparse

from fields_dataset import get_test_dataloaders
from fields_model import load_model

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

def test(model, test_dataloader, save_path, test_type):

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_pred_probs = torch.tensor([]).to(device)
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted = torch.cat((all_predicted, predicted))
            all_labels = torch.cat((all_labels, labels))
            all_pred_probs = torch.cat((all_pred_probs, outputs.data))

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(all_labels.cpu(), all_pred_probs[:,1].cpu())
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plot_name = save_path.split("/")[-1].split(".")[-2]
    plt.title(f'ROC: {plot_name}_{test_type}')
    plt.legend(loc="lower right")
    plt.savefig(f"./plots/{plot_name}_{test_type}.png")
    with open(f"./plots/{plot_name}_{test_type}.pkl", 'wb') as f:
        pickle.dump([fpr, tpr, roc_auc], f)

    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(f"{plot_name} {test_type}")
    print("ROC curve area:", roc_auc)
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} real images, {conf_matrix[1].sum().item()} generated images")
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    accuracy = 100 * correct / total
    print(f"Accuracy for {save_path}: {accuracy}")
    print()

    
    
def full_test(model, test_dataloader, mode = "Misclassified", save_to_file = None, threshold = 0.5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_generated_probs = torch.tensor([]).to(device)
    all_highest_probs = torch.tensor([]).to(device)
    # since we're not training, we don't need to calculate the gradients for our outputs

    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="testing"):
            images = images.float().to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            
            probabilities = nn.Softmax(dim = 1)(predictions)
            highest_probs, indices = torch.max(probabilities, dim=1)
            #print(probabilities)
            generated_probabilities = probabilities[:, 1]
#             predicted_labels =  torch.tensor([1 if x >= threshold else 0 for x in generated_probabilities], 
#                                              dtype=torch.int64).to(device)

            predicted_labels = torch.argmax(predictions, dim = 1)
            
            total += labels.size(0)
            
            correct += (predicted_labels == labels).sum().item()
            
            all_predicted = torch.cat((all_predicted, predicted_labels))
            
            all_labels = torch.cat((all_labels, labels))
            
            all_generated_probs = torch.cat((all_generated_probs, generated_probabilities))
            all_highest_probs = torch.cat((all_highest_probs, highest_probs))
    
    print(f"Current threshold: {threshold}")
    print(f"Real images are label 0, on the first row. Generated Images are label 1, on the second row.")
    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    if mode == "Misclassified":
        print(f"Only Misclassified Test Set Confusion Matrix for:")
    elif mode == "Easy":
        print("Easy (No Misclassified or Unconfident) Confusion Matrix:")
    elif mode == "Unconfident":
        print("Only Unconfident Test Set Confusion Matrix:")
    elif mode == "Misclassified and Unconfident":
        print("Both Misclassified and Unconfident Test Set Confusion Matrix:")
    elif mode == "Full":
        print("Full Test Set Confusion Matrix:")
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} real images, {conf_matrix[1].sum().item()} generated images")
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    accuracy = 100 * correct / total
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy}")
    
    fpr, tpr, thresholds = roc_curve(all_labels.cpu(), all_generated_probs.cpu(), pos_label = 1)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {mode} Test Set ')
    plt.legend(loc="lower right")
    plt.show()
    if save_to_file is not None:
        with open(f'final_plots/new/{save_to_file}', 'wb') as f:
            pickle.dump([fpr, tpr, roc_auc], f)
    return all_highest_probs


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

    misclassified_indoor_file = "./misclassified_indoor_list.pkl"
    misclassified_outdoor_file = "./misclassified_outdoor_list.pkl"

    unconfident_indoor_file = "./unconfident_indoor_list.pkl"
    unconfident_outdoor_file = "./unconfident_outdoor_list.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if category == "indoor":
        image_data_paths = ["Kandinsky_Indoor"]

        misclassified_image_list = sorted(pickle.load(open(misclassified_indoor_file, "rb")))
        unconfident_image_list = sorted(pickle.load(open(unconfident_indoor_file, "rb")))
        
        save_path = "./checkpoints/Fields_indoor.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    elif category == "outdoor":

        image_data_paths = ["Kandinsky_Outdoor"]

        misclassified_image_list = sorted(pickle.load(open(misclassified_outdoor_file, "rb")))
        unconfident_image_list = sorted(pickle.load(open(unconfident_outdoor_file, "rb")))
        
        save_path = "./checkpoints/Fields_outdoor.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    elif category == "combined":
        image_data_paths = ["Kandinsky_Indoor", "Kandinsky_Outdoor"]

        misclassified_image_list = sorted(pickle.load(open(misclassified_indoor_file, "rb"))) + \
            sorted(pickle.load(open(misclassified_outdoor_file, "rb")))
        unconfident_image_list = sorted(pickle.load(open(unconfident_indoor_file, "rb"))) + \
            sorted(pickle.load(open(unconfident_outdoor_file, "rb")))
        
        save_path = "./checkpoints/Fields_combined.pt"
        model = load_model(target_device = device, path_to_checkpoint = save_path)

    train_image_paths, val_image_paths, test_image_paths = load_all_paths(base_path, image_data_paths)
    
    easy_dataloader, unconfident_dataloader, misclassified_dataloader = get_test_dataloaders(test_image_paths, unconfident_image_paths, misclassified_image_paths, class_to_idx)
    
    test(model, easy_dataloader, save_path, "easy")
    test(model, unconfident_dataloader, save_path, "unconfident")
    test(model, misclassified_dataloader, save_path, "misclassified")
    
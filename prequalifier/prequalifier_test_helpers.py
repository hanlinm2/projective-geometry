# from dataset import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def evaluate_on_dataloader(model, test_dataloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_generated_probs = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="testing"):
            
            images = images.float().to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            
            probabilities = nn.Softmax(dim = 1)(predictions)

            generated_probabilities = probabilities[:, 1]

            predicted_labels = torch.argmax(predictions, dim = 1)
            
            total += labels.size(0)
            
            correct += (predicted_labels == labels).sum().item()
            
            all_predicted = torch.cat((all_predicted, predicted_labels))
            
            all_labels = torch.cat((all_labels, labels))
            
            all_generated_probs = torch.cat((all_generated_probs, generated_probabilities))
    
    return all_predicted.cpu(), all_labels.cpu(), all_generated_probs.cpu(), correct, total


def compute_roc_curve(all_labels, all_generated_probs):
    fpr, tpr, thresholds = roc_curve(all_labels, all_generated_probs, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_singular_roc_curve(fpr, tpr, roc_auc, save_path, test_type):
    fig = plt.figure()
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
    plt.show()
    plt.savefig(f"./plots/{plot_name}_{test_type}.png")
    with open(f"./plots/{plot_name}_{test_type}.pkl", 'wb') as f:
        pickle.dump([fpr, tpr, roc_auc], f)
    print("ROC curve area:", roc_auc)
    return

def plot_multiple_roc_curves(roc_curves, plot_labels, colors, save_path):
    plt.figure()
    lw = 4
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label = "random (no skill)")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    title = save_path.split("/")[-1]
    plt.title(f'{title}')
    for index, roc_curve in enumerate(roc_curves):
        fpr, tpr, roc_auc = roc_curve
        plot_label = plot_labels[index]
        plt.plot(fpr, tpr, color = colors[index], lw=lw, label=f'{plot_label} (area: %0.2f)' % roc_auc)
    plt.grid(color = 'gray', linestyle = '--', dashes=(7, 7), linewidth = 0.5)
    plt.legend(loc="lower right", fontsize="13")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches="tight")
    return

def display_confusion_matrix(all_labels, all_predicted, save_path, test_type):
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    plot_name = save_path.split("/")[-1].split(".")[-2]
    print(f"{plot_name} {test_type}")
    print(f"Real images are label 0, on the first row. Generated Images are label 1, on the second row.")
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} real images, {conf_matrix[1].sum().item()} generated images")
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    return

def compute_accuracy(correct, total, save_path, test_type):
    accuracy = 100 * (correct/total)
    plot_name = save_path.split("/")[-1].split(".")[-2]
    print(f"Accuracy for {plot_name} {test_type}: {accuracy}")
    return

def test(model, test_dataloader, save_path, test_type):
    all_predicted, all_labels, all_generated_probs, correct, total = evaluate_on_dataloader(model, test_dataloader)
    fpr, tpr, roc_auc = compute_roc_curve(all_labels, all_generated_probs)
    plot_singular_roc_curve(fpr, tpr, roc_auc, save_path, test_type)
    display_confusion_matrix(all_labels, all_predicted, save_path, test_type)
    compute_accuracy(correct, total, save_path, test_type)
    return

def batch_test(model, test_dataloaders, save_path, roc_plot_labels, roc_plot_colors, roc_save_path):
    roc_curves = []
    for key in list(test_dataloaders.keys()):
        test_dataloader = test_dataloaders[key]
        all_predicted, all_labels, all_generated_probs, correct, total = evaluate_on_dataloader(model, test_dataloader)
        fpr, tpr, roc_auc = compute_roc_curve(all_labels, all_generated_probs)
        roc_curves.append((fpr, tpr, roc_auc))
        display_confusion_matrix(all_labels, all_predicted, save_path, test_type = key)
        compute_accuracy(correct, total, save_path, test_type = key)
    plot_multiple_roc_curves(roc_curves, roc_plot_labels, roc_plot_colors, roc_save_path)
    return
        
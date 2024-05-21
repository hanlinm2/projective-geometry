import glob
from pandas.core.common import flatten
import os
import pickle


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


def load_outdoor_problem_paths(base_path):   
    load_all_problem_paths = os.path.join(base_path, "all_outdoor_problem_paths.pkl")
    
    with open(load_all_problem_paths, 'rb') as f:
        all_problem_paths = pickle.load(f)
    
    return all_problem_paths

def remove_problem_paths(list_of_paths, base_path, category):
    if category == "indoor":
        return list_of_paths
    else:
        all_problem_paths = load_outdoor_problem_paths(base_path)
        
        for index, path_list in enumerate(list_of_paths):
            list_of_paths[index] = list(set(path_list) - set(all_problem_paths))
        
        return list_of_paths
    

# def remove_problem_paths(all_train_paths, all_val_paths, all_test_paths, base_path, category):
#     if category == "indoor":
#         return all_train_paths, all_val_paths, all_test_paths
#     else:
#         all_problem_paths = load_outdoor_problem_paths(base_path)
        
#         all_train_paths = list(set(all_train_paths) - set(all_problem_paths))
#         all_val_paths = list(set(all_val_paths) - set(all_problem_paths))
#         all_test_paths = list(set(all_test_paths) - set(all_problem_paths))
        
#         return all_train_paths, all_val_paths, all_test_paths


def load_image_path_to_lines(base_path):
    load_path = os.path.join(base_path, f'image_path_to_lines.pkl')
    
    with open(load_path, 'rb') as f:
        image_path_to_lines = pickle.load(f)
        
    return image_path_to_lines
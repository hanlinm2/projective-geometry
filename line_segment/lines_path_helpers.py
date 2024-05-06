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
    
    load_train_problem_path = os.path.join(base_path, "outdoor_train_problem_paths.pkl")
    load_val_problem_path = os.path.join(base_path, "outdoor_val_problem_paths.pkl")
    load_test_problem_path = os.path.join(base_path, "outdoor_test_problem_paths.pkl")
    
    with open(load_train_problem_path, 'rb') as f:
        train_problem_paths = pickle.load(f)
    
    with open(load_val_problem_path, 'rb') as f:
        val_problem_paths = pickle.load(f)

    with open(load_test_problem_path, 'rb') as f:
        test_problem_paths = pickle.load(f)
    
    return train_problem_paths, val_problem_paths, test_problem_paths


def get_path_counterparts(paths):
    counterparts = []
    for path in paths:
        temp = path.split('/')
        if temp[-2] == "real":
            temp[-2] = "gen"
        else:
            temp[-2] = "real"
        new_path = '/'.join(temp)
        counterparts.append(new_path)
    return counterparts
        

def remove_problem_paths(all_train_paths, all_val_paths, all_test_paths, base_path, category):
    if category == "indoor":
        return all_train_paths, all_val_paths, all_test_paths
    else:
        train_problem_paths, val_problem_paths, test_problem_paths = load_outdoor_problem_paths(base_path)
        all_problem_paths = train_problem_paths + val_problem_paths + test_problem_paths
        all_counterparts = get_path_counterparts(all_problem_paths)
        
        all_train_paths = list(set(all_train_paths) - set(all_problem_paths + all_counterparts))
        all_val_paths = list(set(all_val_paths) - set(all_problem_paths + all_counterparts))
        all_test_paths = list(set(all_test_paths) - set(all_problem_paths + all_counterparts))
        
        return all_train_paths, all_val_paths, all_test_paths


def load_image_path_to_lines(base_path):
    load_path = os.path.join(base_path, f'all_paths_to_lines.pkl')
    with open(load_path, 'rb') as f:
        image_path_to_lines = pickle.load(f)
    return image_path_to_lines
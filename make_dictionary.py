from tqdm import tqdm
import pickle


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not graph.__contains__(start):
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def make_dictionary():
    classes_dict = {}
    classes_list = []
    with open("/Users/mirawelner/Documents/food_word_embedding_validator/data/foodon_pairs/foodonpairs.txt", 'r+') as f:
        next(f)
        for line in f:
            parent = line.split('\t')[1][:-1]
            child = line.split('\t')[0]
            if parent in classes_dict:
                classes_dict[parent].append(child)
            else:
                classes_dict[parent] = [child]
            if parent not in classes_list:
                classes_list.append(parent)
            if child not in classes_list:
                classes_list.append(child)
    dictionary = {}
    for food_class in tqdm(classes_list, leave=False, desc='Generating Dictionary'):
        paths = find_all_paths(classes_dict, 'foodon product type', food_class)
        dictionary[food_class] = min(paths, key=len)
    save_dic(dictionary, "/Users/mirawelner/Documents/food_word_embedding_validator/saved_data/class_dictionary.txt")
    return dictionary


def save_dic(dictionary,File):
    with open(File, "wb") as myFile:
        pickle.dump(dictionary, myFile)
        myFile.close()
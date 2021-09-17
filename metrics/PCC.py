"""
Authors:
    Mira Welner - mewelner@ucdavis.edu

Description:
    Find pearson correlation coefficient

To-do:
"""
from embedding_generators.get_word2vec_embedding import *
from find_hops import get_distance
from scipy.stats import pearsonr
from tqdm import tqdm

classes = []
with open('/saved_data/classes_list.txt', 'r') as file_name:
    for line in file_name:
        currentPlace = line[:-1]
        classes.append(currentPlace)


ontology_data = []
word2vec_data = []
model = Word2Vec.load("/Users/mirawelner/Documents/food_word_embedding_validator/models/word2vec/word2vec.model")
for class1 in tqdm(classes):
    try:
        class_embedding1 = get_word2vec_average(class1, model)
    except KeyError:
        continue
    for class2 in classes[0:3]:
        try:
            class_embedding2 = get_word2vec_average(class2, model)
        except KeyError:
            continue
        word2vec_data.append(get_cosine_similarity(class_embedding1, class_embedding2, model))
        ontology_data.append(get_distance(class1, class2))

corr, _ = pearsonr(ontology_data, word2vec_data)
print('Pearsons correlation: %.3f' % corr)

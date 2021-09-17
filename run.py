"""
Authors:
    Mira Welner - mewelner@ucdavis.edu

Description:
    run validator

To-do:
"""
import os
from os import path

from embedding_generators.get_word2vec_embedding import *
from utils.get_classes_from_pairs import make_classes_from_pairs
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from make_dictionary import make_dictionary
from gensim.models import KeyedVectors
from find_hops import get_distance
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def embedding_test(string1, string2, mod):
    print(mod.wv.n_similarity(string1.lower().split(), string2.lower().split()))


def hops_test(string1, string2, dic):
    print(get_distance(string1, string2, dic))


def store_list(comparison_list, file_name):
    """A helper function to put a list into a file"""
    with open(file_name, 'w') as file_name:
        for item in comparison_list:
            file_name.write('%s\n' % item)


def load_list(filename):
    """A helper function to load a file into a list"""
    comparison_list = []
    with open(filename, 'r') as file_name:
        for line in file_name:
            current_place = line[:-1]
            comparison_list.append(float(current_place))
    return comparison_list


def get_classes():
    """Load classes contained within the specific branch"""
    classes = []
    with open('/Users/mirawelner/Documents/food_word_embedding_validator/saved_data/classes_list.txt', 'r') as f:
        for line in f:
            current_place = line[:-1]
            classes.append(current_place)
    return classes


def make_plots(cosine_distances, hops_distances, branch, hops_metric, t):
    """Plot the hops distances vs cosine distances, display result of used metric"""
    fig, ax = plt.subplots()
    ax.cla()
    ax.scatter(cosine_distances, hops_distances)
    ax.set_ylabel('hops distance')
    ax.set_xlabel('cosine similarity')
    title = 'Hops metric vs cosine similarity for the ' + branch + ' branch\n' \
    ' Pearson Correlation (1-cosine similarity vs hops): ' + str(hops_metric) + 'type: ' + t
    ax.set_title(title, loc='center', wrap=True)
    fig.savefig('/Users/mirawelner/Documents/food_word_embedding_validator/generated_graphs/' + branch + '_' + t + '.png')


def make_histogram(cos, hops, branch):
    sibling_distances = []
    for d in range(len(hops)):
        if hops[d]:
            sibling_distances.append(cos[d])
    fig, ax = plt.subplots()
    ax.hist(sibling_distances, bins='auto')
    mae = mean_absolute_error(sibling_distances, np.ones(len(sibling_distances)))
    ax.set_title("Branch: " + branch + ", mean absolute error: " + str(mae))
    fig.savefig('/Users/mirawelner/Documents/food_word_embedding_validator/generated_graphs/' + branch + '_histogram.png')


def get_similarity(model, metric, branch="foodon product type", override=False):
    hops_distances = [0,0,0,0]
    make_classes_from_pairs(branch)
    classes = get_classes()
    if not os.path.exists('saved_data/embedding_comparison_' + branch + '.txt') or override:
        cosine_distances, errors, hops_distances = get_word2vec_data(classes, model)
        store_list(cosine_distances, 'saved_data/embedding_comparison_' + branch + '.txt')
        store_list(hops_distances[0], 'saved_data/hops_distances_a' + branch + '.txt')
        store_list(hops_distances[1], 'saved_data/hops_distances_b' + branch + '.txt')
        store_list(hops_distances[2], 'saved_data/hops_distances_c' + branch + '.txt')
        store_list(hops_distances[3], 'saved_data/hops_distances_d' + branch + '.txt')

        with open('saved_data/errors_' + branch + '.txt', 'w') as file_name:
            file_name.write(str(errors))
    else:
        cosine_distances = load_list('saved_data/embedding_comparison_' + branch + '.txt')
        hops_distances[0] = load_list('saved_data/hops_distances_a' + branch + '.txt')
        hops_distances[1] = load_list('saved_data/hops_distances_b' + branch + '.txt')
        hops_distances[2] = load_list('saved_data/hops_distances_c' + branch + '.txt')
        hops_distances[3] = load_list('saved_data/hops_distances_d' + branch + '.txt')

        with open('saved_data/errors_' + branch + '.txt', 'r') as file_name:
            errors = file_name.read()
    if metric == 'pcc' or metric == 'pearson':
        hops_a = round(pearsonr(hops_distances[0], cosine_distances)[0], 4)
        hops_b = round(pearsonr(hops_distances[1], cosine_distances)[0], 4)
        hops_c = round(pearsonr(hops_distances[2], cosine_distances)[0], 4)
    make_histogram(cosine_distances, hops_distances[3], branch)
    make_plots(cosine_distances, hops_distances[0], branch, hops_a, 'a')
    make_plots(cosine_distances, hops_distances[1], branch, hops_b, 'b')
    make_plots(cosine_distances, hops_distances[2], branch, hops_c, 'c')
    error = str(round(int(errors)/(int(errors)+len(hops_distances[0])), 4))
    nodes = str(len(hops_distances[0]))
    return {'Branch': branch, 'metric a': hops_a, 'metric b': hops_b, 'metric c': hops_c, 'nodes w/o embeddings': error, 'nodes':nodes}


if not path.isfile('/Users/mirawelner/Documents/food_word_embedding_validator/saved_data/class_dictionary.txt'):
    make_dictionary()
model = KeyedVectors.load_word2vec_format('/Users/mirawelner/Documents/food_word_embedding_validator/models/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

df = pd.DataFrame(columns=['Branch', 'metric a', 'metric b', 'metric c', 'nodes w/o embeddings', 'nodes'])
df = df.append(get_similarity(model, 'pcc', 'cow milk cheese'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'nut food product'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'fruit food product'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'beef food product'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'spice or herb'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'cereal grain food product', True), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'mollusc food product'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'avian egg food product'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'candy food product'), ignore_index=True)
df = df.append(get_similarity(model, 'pcc', 'cheese food product'), ignore_index=True)
df.to_html('./data_frame.html')
print(df)



import regex as re
from gensim.models import KeyedVectors
from find_hops import get_distance
import pickle
from tqdm import tqdm


def make_word2vec_model():
    model = KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    model.save("../models/word2vec/word2vec.model")


def load_dictionary(file):
    with open(file, "rb") as myFile:
        dictionary_to_save = pickle.load(myFile)
        myFile.close()
        return dictionary_to_save


def get_word2vec_embedding(word, model):
    vector = model.wv[word.lower()]
    return vector


def get_cosine_similarity(class1, class2, model):
    # https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.n_similarity.html
    return model.wv.n_similarity(class1.lower().split(), class2.lower().split())


def get_word2vec_data(classes, model):
    """Gets a list of embeddings representing similarity between each class"""
    dictionary = load_dictionary("/Users/mirawelner/Documents/food_word_embedding_validator/saved_data"
                                 "/class_dictionary.txt")
    classes_not_in_model = 0
    word2vec_data = []
    hops_a = []
    hops_b = []
    hops_c = []
    hops_d = []

    count = 0
    for class1 in tqdm(classes):
        count += 1
        for class2 in classes:
            try:
                word2vec_data.append(get_cosine_similarity(re.sub(r'[()]', '', class1), re.sub(r'[()]', '', class2), model))
                hops_a.append(get_distance(class1, class2, dictionary)[0])
                hops_b.append(get_distance(class1, class2, dictionary)[1])
                hops_c.append(get_distance(class1, class2, dictionary)[2])
                hops_d.append(get_distance(class1, class2, dictionary)[3])
            except KeyError:
                #print(re.sub(r'[()]', '', class1), re.sub(r'[()]', '', class2))
                classes_not_in_model += 1
    hops = [hops_a, hops_b, hops_c, hops_d]
    return word2vec_data, classes_not_in_model, hops

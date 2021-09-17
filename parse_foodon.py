"""
Authors:
    Jason Youn - jyoun@ucdavis.edu
    Tarini Naravane - tnaravane@ucdavis.edu

Description:
    Parse FoodOn.

To-do:
"""
# standard imports
import logging as log
import os
import sys
import random

# third party imports
import pandas as pd
import networkx as nx

# local imports
from utils.config_parser import ConfigParser
from utils.utilities import file_exists, save_pkl, load_pkl
from utils.set_logging import set_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '/'))


class ParseFoodOn:
    """
    Class for parsing FoodOn.
    """

    def __init__(self, config_filepath):
        """
        Class initializer.

        Inputs:
            config_filepath: (str) Configuration filepath.
        """
        self.configparser = ConfigParser(config_filepath)

        # read configuration file
        self.filepath = self.configparser.getstr('filepath')
        self.full_ontology_pkl = self.configparser.getstr('full_ontology_pkl')
        self.candidate_ontology_pkl = self.configparser.getstr('candidate_ontology_pkl')
        self.skeleton_and_entities_pkl = self.configparser.getstr('skeleton_and_entities_pkl')
        self.overwrite_pkl = self.configparser.getbool('overwrite_pickle_flag')
        self.outputFoodOn = self.configparser.getstr('outputFoodOn')

        self.num_seeds = self.configparser.getint('num_seeds')
        self.num_min_extracted_entities = self.configparser.getint('num_min_extracted_entities')

        # generate pairs from csv file
        self.pd_food_on_pairs = self.generate_pairs()
        self.all_classes, self.all_entities = self.get_classes_and_entities()
        self.food_on_graph, self.graph_dict, self.graph_dict_flip = self.generate_graph()

    def generate_graph(self):
        graph_dict = {k: v for v, k in enumerate(self.all_classes)}
        graph_dict_flip = {v: k for v, k in enumerate(self.all_classes)}

        G = nx.DiGraph()

        for _, row in self.pd_food_on_pairs.iterrows():
            if row['Parent'] in self.all_classes and row['Child'] in self.all_classes:
                node_from = graph_dict[row['Parent']]
                node_to = graph_dict[row['Child']]
                G.add_edge(node_from, node_to)

        return G, graph_dict, graph_dict_flip

    def get_classes_and_entities(self):
        all_classes = self.pd_food_on_pairs['Parent'].tolist()
        all_classes = list(set(all_classes))
        all_classes.sort()
        log.debug('Found %d classes.', len(all_classes))

        child = self.pd_food_on_pairs['Child'].tolist()
        child = list(set(child))
        child.sort()
        all_entities = [c for c in child if c not in all_classes]
        log.debug('Found %d entities.', len(all_entities))

        return all_classes, all_entities

    def generate_pairs(self):
        log.info('Generating pairs of FoodOn.')

        if file_exists(self.outputFoodOn) and not self.overwrite_pkl:
            log.info('Using pre-generated pairs file.')
            return pd.read_csv(self.outputFoodOn, sep='\t')

        # 1.Read specified columns from FoodON.csv file
        food_on = pd.read_csv(self.filepath, usecols=['Class ID', 'Parents', 'Preferred Label'])

        # 2.Create dictionary of URI and ClassLabel
        labels_tmp = food_on[["Class ID", "Preferred Label"]].copy()
        self.labels = labels_tmp.set_index('Class ID')['Preferred Label'].to_dict()

        # 3.Create data frame with columns - child and all its' parents
        foodon_orig_df = (food_on[["Class ID", "Parents"]].copy()).rename(columns={'Class ID': 'Child'})

        # 4.Split above DF into pairs of Child-Parent
        pairs = []
        for _, row in foodon_orig_df.iterrows():
            parents = str(row['Parents'])
            parentList = parents.split("|")
            for pClass in parentList:
                child = str(row['Child'])
                pairs.append([child, pClass])
        food_on_df = pd.DataFrame(pairs, columns=['Child', 'Parent'])
        food_on_df = self.filter_ontology(food_on_df, 'http://purl.obolibrary.org/obo/FOODON_00001872')
        food_on_df = self.get_subtree(food_on_df, 'http://purl.obolibrary.org/obo/FOODON_00001002')

        # In food_on_df, replace URI by label
        for idx, pair in food_on_df.iterrows():
            pair['Child'] = self.labels[pair['Child']]
            if pair['Parent'] in self.labels:
                pair['Parent'] = self.labels[pair['Parent']]

        food_on_df.drop_duplicates(inplace=True, ignore_index=True)
        food_on_df.to_csv(self.outputFoodOn, sep='\t', index=False)

        return food_on_df

    def filter_ontology(self, dfObj, classname):
        # Remove class and its children from the ontology.
        # Works only if the children are leaf nodes.
        indexNames = dfObj[dfObj['Parent'] == classname].index
        dfObj.drop(indexNames, inplace=True)
        indexNames = dfObj[dfObj['Child'] == classname].index
        dfObj.drop(indexNames, inplace=True)

        return dfObj

    def get_subtree(self, df, rootclass):
        subtree_df, nextlevelclasses = self.traverse_next_level(
            df, ['http://purl.obolibrary.org/obo/FOODON_00001002'])

        while (len(nextlevelclasses) > 0):
            pairs_df, nextlevelclasses = self.traverse_next_level(df, nextlevelclasses)
            subtree_df = pd.concat([subtree_df, pairs_df], ignore_index=True)

        return subtree_df

    def traverse_next_level(self, df, classnames):
        nextlevel = []
        subtree_pairs = []
        for parent in classnames:
            selectedPairs = df[df['Parent'] == parent]
            for idex, pair in selectedPairs.iterrows():
                subtree_pairs.append([pair['Child'], pair['Parent']])
                ifparent = df[df['Parent'] == pair['Child']]  # Check if it is a leaf node
                if ifparent.empty != True:
                    nextlevel.append(pair['Child'])

        subtreeDF = pd.DataFrame(subtree_pairs, columns=['Child', 'Parent'])

        return (subtreeDF, nextlevel)

    def get_all_classes_dict(self):
        """
        Get all candidate classes.
        """
        log.info('Generating dictionary of all classes.')

        if file_exists(self.full_ontology_pkl) and not self.overwrite_pkl:
            log.info('Using pre-generated full classes dictionary file.')
            return load_pkl(self.full_ontology_pkl)

        full_classes_dict = {}
        for class_label in self.all_classes:
            pd_match = self.pd_food_on_pairs[self.pd_food_on_pairs['Parent'] == class_label]
            children = pd_match['Child'].tolist()
            children_entities = [c for c in children if c in self.all_entities]

            node_from = self.graph_dict['foodon product type']
            node_to = self.graph_dict[class_label]

            paths = []
            if class_label == 'foodon product type':
                paths.append(tuple(['foodon product type']))
            else:
                for path in nx.all_simple_paths(self.food_on_graph, source=node_from, target=node_to):
                    translated_path = [self.graph_dict_flip[p] for p in path]
                    paths.append(tuple(translated_path[::-1]))

            full_classes_dict[class_label] = (paths, children_entities)

        save_pkl(full_classes_dict, self.full_ontology_pkl)

        return full_classes_dict

    def get_candidate_classes(self):
        """
        Get all candidate classes.
        """
        log.info('Generating dictionary of candidate classes.')

        if file_exists(self.candidate_ontology_pkl) and not self.overwrite_pkl:
            log.info('Using pre-generated candidate classes dictionary file: %s',
                     self.candidate_ontology_pkl)
            return load_pkl(self.candidate_ontology_pkl)

        candidate_classes_dict = {}
        for class_label in self.all_classes:
            pd_match = self.pd_food_on_pairs[self.pd_food_on_pairs['Parent'] == class_label]
            children = pd_match['Child'].tolist()
            children_entities = [c for c in children if c in self.all_entities]

            if len(children_entities) > 0:
                node_from = self.graph_dict['foodon product type']
                node_to = self.graph_dict[class_label]

                paths = []
                if class_label == 'foodon product type':
                    paths.append(tuple(['foodon product type']))
                else:
                    for path in nx.all_simple_paths(self.food_on_graph, source=node_from, target=node_to):
                        translated_path = [self.graph_dict_flip[p] for p in path]
                        paths.append(tuple(translated_path[::-1]))

                candidate_classes_dict[class_label] = (paths, children_entities)

        log.info('Found %d candidate classes out of %d all classes.',
                 len(candidate_classes_dict.keys()), len(self.all_classes))

        save_pkl(candidate_classes_dict, self.candidate_ontology_pkl)

        return candidate_classes_dict

    def get_seeded_skeleton(self, candidate_classes_dict):
        log.info('Generating dictionary of skeleton candidate classes.')

        if file_exists(self.skeleton_and_entities_pkl) and not self.overwrite_pkl:
            log.info('Using pickled skeleton file: %s', self.skeleton_and_entities_pkl)
            return load_pkl(self.skeleton_and_entities_pkl)

        skeleton_candidate_classes_dict = {}
        candidate_entities = []
        for candidate_class in candidate_classes_dict.keys():
            entities = candidate_classes_dict[candidate_class][1]

            if len(entities) <= self.num_seeds:
                temp_num_seeds = len(entities) - self.num_min_extracted_entities

                if temp_num_seeds > 0:
                    seeds = random.sample(entities, temp_num_seeds)
                    candidate_entities.extend(list(set(entities) - set(seeds)))
                else:
                    seeds = entities.copy()
            else:
                seeds = random.sample(entities, self.num_seeds)
                candidate_entities.extend(list(set(entities) - set(seeds)))

            skeleton_candidate_classes_dict[candidate_class] = (
                candidate_classes_dict[candidate_class][0],
                seeds)

        candidate_entities = list(set(candidate_entities))
        candidate_entities.sort()

        log.info('Found %d candidate entities to populate out of %d all entities.',
                 len(candidate_entities), len(self.all_entities))

        return_value = (skeleton_candidate_classes_dict, candidate_entities)
        save_pkl(return_value, self.skeleton_and_entities_pkl)

        return return_value


if __name__ == '__main__':
    # set log, parse args, and read configuration
    set_logging()

    # parse FoodOn
    parse_foodon = ParseFoodOn('config/foodon_parse.ini')
    all_classes_dict = parse_foodon.get_all_classes_dict()
    candidate_classes_dict = parse_foodon.get_candidate_classes()
    (skeleton_candidate_classes_dict, candidate_entities) = parse_foodon.get_seeded_skeleton(
        candidate_classes_dict)

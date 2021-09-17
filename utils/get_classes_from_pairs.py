def make_classes_from_pairs(branch_root_node):
    pairs_file = open("/Users/mirawelner/Documents/food_word_embedding_validator/data/foodon_pairs/foodonpairs.txt", "r")
    classes_file = open("/Users/mirawelner/Documents/food_word_embedding_validator/saved_data/classes_list.txt", "w")
    sub_classes = []
    for line in pairs_file:
        values = line.split('\t')
        if values[1][:-1] == branch_root_node or values[1][:-1] in sub_classes:
            sub_classes.append(values[0])
            classes_file.write(values[0] + '\n')
    pairs_file.close()
    classes_file.close()

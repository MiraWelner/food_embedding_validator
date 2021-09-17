import tensorflow as tf
BERT_PATH = '../models/bert_model'


with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph("/Users/mirawelner/Documents/food_word_embedding_validator/models/bert_model/bert_model.ckpt.meta")
    saver.restore(sess, "/Users/mirawelner/Documents/food_word_embedding_validator/models/bert_model/bert_model.ckpt")

ckpt_reader = tf.train.load_checkpoint("/Users/mirawelner/Documents/food_word_embedding_validator/models/bert_model/bert_model.ckpt")
vars_list = tf.train.list_variables("/Users/mirawelner/Documents/food_word_embedding_validator/models/bert_model/bert_model.ckpt")
embeddings = ckpt_reader.get_tensor("bert/embeddings/word_embeddings")

def get_bert_embedding(raw_word):
    word = raw_word.lower()
    for index, line in open("/Users/mirawelner/Documents/food_word_embedding_validator/models/bert_model/vocab.txt", "r"):
        if line == word:
            return embeddings.at(index)




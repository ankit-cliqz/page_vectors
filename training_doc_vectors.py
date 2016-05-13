#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import re

filename= "queries_filtered_sample_100k.txt"
#filename= "queries_filtered_sample.txt"
sentences = []

print "Caching documents to a list in memory."
for uid, line in enumerate(open(filename)):
    sentences.append(TaggedDocument(line.split(), ['S_{}'.format(uid)]))

_alpha, _min_alpha, _passes = (0.045, 0.001, 25)
alpha_delta = (_alpha - _min_alpha) / _passes
_num_worker_cores = 35


print "Initializing Model ... "
model = Doc2Vec(alpha=_alpha, min_alpha=_min_alpha, workers=_num_worker_cores)
print "Model Initialized. Now building vocabulary ... "
model.build_vocab(sentences)
print "Vocabulary building complete."


print "Starting to train a doc2vec model..... "
for epoch in range(_passes):
    print "EPOCH : {}, Alpha:{} ".format(epoch+1, model.alpha)
    model.train(sentences)
    model.alpha -= alpha_delta
    model.min_alpha = model.alpha
    print "Finished training epoch: {}".format(epoch+1)

print "Saving Trained Model on Disk ... "
# store the model to mmap-able files
model.save('queries_full.doc2vec')
print "Model Saved Successfully!"

#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import re

filename= "queries_filtered_sample_100k.txt"
sentences = []

for uid, line in enumerate(open(filename)):
    sentences.append(TaggedDocument(line.split(), ['SENT_{}'.format(uid)]))

print "Initializing Model ... "
model = Doc2Vec(alpha=0.025, min_alpha=0.025, workers=35)
print "Model Initialized. Now building vocabulary ... "
model.build_vocab(sentences)
print "Vocabulary building complete."

print "Starting to train a doc2vec model..... "

for epoch in range(10):
    print "EPOCH : {}, Alpha:{} ".format(epoch+1, model.alpha)
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    print "Finished training epoch: {}".format(epoch+1)

print "Saving Trained Model on Disk ... "
# store the model to mmap-able files
model.save('queries_sample.doc2vec')
print "Model Saved Successfully!"

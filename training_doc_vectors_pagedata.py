#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ujson as json
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

class LabeledTaggedDocument(object):
    """ This is a TaggedDocument iterator class for streaming the data directly from a file on disk
     without loading into memory """

    def __init__(self, filename):
       self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            tokens = line.split("\t")
            urlid = json.loads(tokens[0])
            data= json.loads(tokens[1])
            doc = ' '.join(data['top_n_q']) + " " + data['title'] + " " + data['desc']
            yield TaggedDocument(doc.split(), [urlid])

docs_it = LabeledTaggedDocument("topq-title-desc.txt")

_alpha, _min_alpha, _passes = (0.020, 0.001, 20)
alpha_delta = (_alpha - _min_alpha) / _passes
_num_worker_cores = 35
_window_size = 8

print "Initializing Model ... "
model = Doc2Vec(alpha=_alpha, min_alpha=_min_alpha, window=_window_size, workers=_num_worker_cores)

print "Model Initialized. Now building vocabulary ... "
model.build_vocab(docs_it)
print "Vocabulary building complete."

print "Starting to train a doc2vec model..... "
for epoch in range(_passes):
    print "EPOCH : {}, Alpha:{} ".format(epoch+1, model.alpha)
    model.train(docs_it)
    model.alpha -= alpha_delta
    model.min_alpha = model.alpha
    print "Finished training epoch: {}".format(epoch+1)

print "Saving Trained Model on Disk ... "
# store the model to mmap-able files
model.save('pages.doc2vec')
print "Model Saved Successfully!"

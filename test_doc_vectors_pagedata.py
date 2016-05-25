#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
filename= "topq-title-desc.txt"
import ujson as json

sentence_dict= {}
for uid, line in enumerate(open(filename)):
    tokens = line.split("\t")
    urlid = json.loads(tokens[0])
    data = json.loads(tokens[1])
    doc = ' '.join(data['top_n_q']) + " " + data['title']
    sentence_dict[urlid] = doc.strip()


print "Loading Doc2Vec Model"
model = Doc2Vec.load('pages.doc2vec')
print "Model Loaded Successfully!"

sample_query = "giacometti wikipedia italiano"
print "Sample Query: {}".format(sample_query)

tokens = sample_query.lower().split()
dv = model.infer_vector(tokens)     # note: may want to use many more steps than default

#print "Document Vector:"
#print dv

sims = model.docvecs.most_similar(positive=[dv])
print "\nList of similar Documents:>> "
for x, y in sims:
    print (x, sentence_dict[x], y)

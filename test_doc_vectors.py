#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
filename= "queries_filtered_sample_100k.txt"

sentence_dict= {}
for uid, line in enumerate(open(filename)):
    # sentences.append(LabeledSentence(words=line.split(), labels=['SENT_%s' %       uid]))
    sentence_dict['SENT_{}'.format(uid)] = line.strip()


print "Loading Doc2Vec Model"
model = Doc2Vec.load('queries.doc2vec')
print "Model Loaded Successfully!"

sentence_sample = "russ ballard dream on mp3 download"
print "Document Sample: {}".format(sentence_sample)

tokens = sentence_sample.lower().split()
dv = model.infer_vector(tokens)     # note: may want to use many more steps than default

#print "Document Vector:"
#print dv

sims = model.docvecs.most_similar(positive=[dv])
print "\nList of similar Documents:>> "
for x, y in sims:
    print (sentence_dict[x], y)

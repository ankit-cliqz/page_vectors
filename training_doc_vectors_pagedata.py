#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ujson as json
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import pykeyvi
import os


print "------------------------------------------------------------"
print "                      Doc2Vec Training                      "
print "------------------------------------------------------------"
# Util Functions
def create_dir(dir_name):
    """ Creates a new directory recursively if not already present. """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def convert_line_unicode(line):
    """ UTF-8 encodes a string (if not already) and returns the encoded string"""
    try:
        if not isinstance(line, unicode):
            line = line.encode("utf-8")
    except:
        pass
    return line

# NUM_TRAINING_DOCS = 1 * 100000000 # Training  -- Final Mode
NUM_TRAINING_DOCS = 20 * 1000  # Training --  Testing Mode

training_data_file = "/raid/ankit/doc2vec/data/splits/snippet_and_pagemodel"
output_data_path = "/raid/ankit/doc2vec/out_s_p_test"
create_dir(output_data_path)
docvecs_process_input_file = "docvecs_input.txt"
docvecs_process_input_keyvi_index_file = "docvecs_urlid_url.kv"
doc2vec_trained_model = 'pages_with_spaces.doc2vec'

# Runner Flags
compile_keyvi_index = True  # Set this to True if you wish to recompile keyvi index for the input data.
run_training = True        # Set this to True, to run Docevec training and save model.

class LabeledTaggedDocument(object):
    """ This is a TaggedDocument iterator class for streaming the data directly from a file on disk
     without loading into memory """

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        i = 0
        for line in open(self.filename):
            tokens = line.split("\t")
            urlid = json.loads(tokens[0])
            data = json.loads(tokens[1])
            if not len(data['top_n_q']) == 0 and (not data['title'] == "") and (not data['desc'] == ""):
                i += 1
                if i == NUM_TRAINING_DOCS:
                    break
                doc = ' '.join(data['top_n_q']) + " " + data['title'] + " " + data['desc']
                yield TaggedDocument(doc.split(), [urlid])

if run_training:
    docs_it = LabeledTaggedDocument(training_data_file)

    _alpha, _min_alpha, _passes = (0.020, 0.001, 20)
    alpha_delta = (_alpha - _min_alpha) / _passes
    _num_worker_cores = 35
    _window_size = 8
    _vec_dim = 100
    print "\nInitializing Model ... "
    model = Doc2Vec(alpha=_alpha, min_alpha=_min_alpha, window=_window_size, workers=_num_worker_cores, size=_vec_dim)

    print "Model Initialized. Now building vocabulary ... "
    model.build_vocab(docs_it)
    print "Vocabulary building complete."

    print "\nStarting to train a doc2vec model..... "
    for epoch in range(_passes):
        print "EPOCH : {}, Alpha:{} ".format(epoch + 1, model.alpha)
        model.train(docs_it)
        model.alpha -= alpha_delta
        model.min_alpha = model.alpha
        print "Finished training epoch: {}".format(epoch + 1)

    print "\nSaving Trained Model on Disk ... "
    # store the model to mmap-able files
    model.save("{}/{}".format(output_data_path, doc2vec_trained_model))
    print "Model Saved Successfully!"

if compile_keyvi_index:
    print "\nSaving the data in a separate file and building keyvi index ....."
    fw = open("{}/{}".format(output_data_path, docvecs_process_input_file), "w")
    keyvi_index_compiler = pykeyvi.JsonDictionaryCompiler()
    with open(training_data_file) as fo:
        i = 0
        for line in fo:
            tokens = line.split("\t")
            urlid = json.loads(tokens[0])
            data = json.loads(tokens[1])
            if not len(data['top_n_q']) == 0 and (not data['title'] == "") and (not data['desc'] == ""):
                i += 1
                if i == NUM_TRAINING_DOCS:
                    break
                fw.write("{}\t{}\t{}\n".format(urlid, data['url'].encode("utf-8"), ','.join(data['top_n_q']).encode("utf-8")))
                #fw.write("{}\t{}\n".format(urlid, data['url'].encode("utf-8")))
                try:
                    keyvi_index_compiler.Add(str(urlid), json.dumps(data['url']))

                except Exception as e:
                    print "Exception for urlid: {} : {}".format(urlid, e)
    fw.close()
    print "Finished: Saving the data in a seperate file."
    print "\nCompiling Keyvi Index < url:id - url for training data >... "
    keyvi_index_compiler.Compile()
    print "Finished Compileing Keyvi. Now writing keyvi index to file... "
    keyvi_index_compiler.WriteToFile("{}/{}".format(output_data_path, docvecs_process_input_keyvi_index_file))
    print "Finished writing keyvi file!"



print "\nAll operations Finished! :) :) :)"

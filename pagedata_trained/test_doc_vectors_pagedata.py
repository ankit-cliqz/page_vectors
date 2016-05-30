#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim.models.doc2vec import Doc2Vec
import pykeyvi

docvecs_process_input_keyvi_index_file = "docvecs_urlid_url.kv"
output_data_path = "/raid/ankit/doc2vec/out_s_p_1M"
doc2vec_trained_model = 'pages_with_spaces.doc2vec'
_alpha, _min_alpha, _passes = (0.020, 0.001, 20)

print "Loading keyvi dictionaries ..."
keyvi_dict=pykeyvi.Dictionary("{}/{}".format(output_data_path, docvecs_process_input_keyvi_index_file))
print "Finished Loading key-vi Dictionary."

print "Loading Doc2Vec Model ... "
model = Doc2Vec.load("{}/{}".format(output_data_path, doc2vec_trained_model))
print "Model Loaded Successfully!"


def get_similar_urls(sample_query, nearest_num):
    tokens = sample_query.lower().split()
    dv = model.infer_vector(tokens, alpha=_alpha, min_alpha=_min_alpha, steps=_passes)     # note: may want to use many more steps than default
    sims = model.docvecs.most_similar(positive=[dv],  topn=nearest_num)
    for url_id, distance in sims:
        url = ""
        for m in keyvi_dict.Get(str(url_id)):
            url = m.GetValueAsString()
        print "{}\t{}\t{}".format(url_id, url, distance)

def main():
    print "\nSimilar URLS for Queries - Doc2Vec Retrieval Interface [All URL's]"
    print "---------------------------------------------------------------------------"
    flag = "True"
    while (flag == "True"):
        testquery = raw_input("Enter Query: ")
        nearest_num = raw_input("Number of similar queries: ")
        if nearest_num == 0 or nearest_num == "":
            nearest_num = 10
        nearest_num = int(nearest_num)
        if not testquery.strip() =="":
            # Return and Print the Top 10 nearest Queries to the Original Query
            print "\nCandidate Nearest URLS [TOP "+str(nearest_num)+"]: "
            get_similar_urls(testquery, nearest_num)

            user_input = raw_input("\nDo you wish to continue again? (Type 'no' to quit): ")
            if user_input == "no":
                print "\nGoodbye!"
                break
            else:
                print "\n"
                continue

if __name__=="__main__":
    main()
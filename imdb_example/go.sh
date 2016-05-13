#!/usr/bin/env bash

# Download and Prepares IMDB DATASET for Doc2vec
if [ ! -f "aclImdb/alldata-id.txt" ]
then
    if [ ! -d "aclImdb" ]
    then
        if [ ! -f "aclImdb_v1.tar.gz" ]
        then
          wget --quiet http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
        fi
      tar xf aclImdb_v1.tar.gz
    fi

  export LC_ALL=C
  for j in train/pos train/neg test/pos test/neg train/unsup; do
    rm temp
    for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done

    awk '{print tolower($0);}' < temp | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' -e 's/\;/ \; /g' -e 's/\:/ \: /g' > temp-norm

    mv temp-norm aclImdb/$j/norm.txt
  done
  mv aclImdb/train/pos/norm.txt aclImdb/train-pos.txt
  mv aclImdb/train/neg/norm.txt aclImdb/train-neg.txt
  mv aclImdb/test/pos/norm.txt aclImdb/test-pos.txt
  mv aclImdb/test/neg/norm.txt aclImdb/test-neg.txt
  mv aclImdb/train/unsup/norm.txt aclImdb/train-unsup.txt

  cat aclImdb/train-pos.txt aclImdb/train-neg.txt aclImdb/test-pos.txt aclImdb/test-neg.txt aclImdb/train-unsup.txt > aclImdb/alldata.txt
  awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < aclImdb/alldata.txt > aclImdb/alldata-id.txt
fi
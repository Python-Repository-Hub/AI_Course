#!/bin/bash

dataPath=./dataSplit
fairseq-preprocess \
  --source-lang en --target-lang fr \
  --tokenizer moses \
  --trainpref $dataPath/train \
  --validpref $dataPath/dev \
  --destdir data-bin/tutorial-NMT

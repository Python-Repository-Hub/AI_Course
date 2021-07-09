#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

savePath=./checkpoints/tutorial-NMT
binPath=./data-bin/tutorial-NMT
fairseq-train $binPath \
  --save-dir $savePath --keep-best-checkpoints 1 --max-epoch 20 --no-progress-bar \
  --lr 0.0005 --optimizer adam --clip-norm 0.1 \
  --dropout 0.1 --max-tokens 3000 \
  --arch lstm \
  --encoder-embed-dim 128 --encoder-bidirectional \
  --decoder-embed-dim 128 --decoder-attention True

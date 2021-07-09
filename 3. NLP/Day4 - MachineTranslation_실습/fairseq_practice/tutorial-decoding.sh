export CUDA_VISIBLE_DEVICES=0
binPath=data-bin/tutorial-NMT
model=checkpoints/tutorial-NMT/checkpoint_best.pt

fairseq-interactive \
	--path $model $binPath \
	--beam 5 --source-lang en --target-lang fr \
	--tokenizer moses

# ai-539-nlp-group
Group Project for Natural Language Processing 

# Environment Sourcing:
install conda or miniconda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
> conda env create -f environment_cuda.yml  # if you have a gpu
> 
> conda env create -f environment_cpu.yml   # if you do not have a gpu
> 
> conda activate ai_539_group


# Fairseq with pruning - added support for pruning any transformer model in fairseq. 

To install fairseq v0.10.2 with pruning, clone this repository:

> cd fairseq
> 
> pip install --editable ./

### Additional parameters introduced: 

1. pruning_interval (how long to wait before we do next pruning)
2. prune_start_step (what training step we want to start pruning)
3. prune_type (magnitude or random)
4. target_sparsity
5. prune_embedding - by default it is false
6. num_pruning_steps

### Files changed:

1. fairseq/checkpoint_utils.py
2. fairseq/models/fairseq_model.py
3. fairseq/optim/fairseq_optimizer.py
4. fairseq/optim/fp16_optimizer.py
5. fairseq/options.py
6. fairseq/tasks/fairseq_task.py
7. fairseq/trainer.py


### Sample command: 
> CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ../cnn-dailymail/cnn_dm-bin     --restore-file $BART_PATH     --max-tokens $MAX_TOKENS     --task translation     --source-lang source --target-lang target     --truncate-source     --layernorm-embedding     --share-all-embeddings     --share-decoder-input-output-embed     --required-batch-size-multiple 1     --arch bart_base     --criterion label_smoothed_cross_entropy     --label-smoothing 0.1     --dropout 0.1 --attention-dropout 0.1     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08     --clip-norm 0.1 --save-interval-updates 2000    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES     --fp16 --update-freq $UPDATE_FREQ     --skip-invalid-size-inputs-valid-test     --find-unused-parameters     --keep-interval-updates 1     --no-epoch-checkpoints --ddp-backend=no_c10d --reset-dataloader --prune_start_step 7000 --target_sparsity 0.5 --pruning_interval 7000 --num_pruning_steps 30 --prune_type magnitude --max-update 20000 --max-epoch 20


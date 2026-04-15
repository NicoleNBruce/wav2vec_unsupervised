#!/bin/bash

# This script runs the GANS training of  unsupervised wav2vec pipeline

# Wav2Vec Unsupervised Pipeline Runner
# This script runs the entire fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

source utils.sh

#=========================== GANS training and preparation ==============================
train_gans(){
   local step_name="train_gans"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   # export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="/$DIR_PATH:$PYTHONPATH"


   if is_completed "$step_name"; then
        log "Skipping gans training  (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "$step_name"
   

    PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
     --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
     dataset.batch_size=12 \
        dataset.num_workers=0 \
        dataset.required_batch_size_multiple=1 \
        dataset.valid_subset=valid \
        dataset.disable_validation=False \
     optimization.max_update=30000 \
     dataset.validate_interval_updates=1000 \
     checkpoint.save_interval_updates=1000 \
     common.log_interval=25 \
     common.seed=0 \
     model.code_penalty=2 \
     model.gradient_penalty=1.5 \
     model.smoothness_weight=1.0 \
     +optimizer.groups.generator.optimizer.lr="[0.00008]" \
     +optimizer.groups.discriminator.optimizer.lr="[0.00001]" \
    ~optimizer.groups.generator.optimizer.amsgrad \
    ~optimizer.groups.discriminator.optimizer.amsgrad \
    2>&1 | tee $RESULTS_DIR/training1.log

    

   if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "gans trained successfully"
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}


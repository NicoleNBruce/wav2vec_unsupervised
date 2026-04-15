#!/bin/bash

# This script runs the entire evaluation of the fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs

source utils.sh

MODEL_PATH=$1 # the model should be a .pt file 



# ==================== HELPER FUNCTIONS ====================

# Create directories if they don't exist
#============================ model evaLuation =======================

transcription_gans_viterbi(){

   export HYDRA_FULL_ERROR=1
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH
#    

#evaluating the GANS models for validation phones
python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
 --config-name viterbi fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
  fairseq.task.data="'$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled'" \
  fairseq.common_eval.path="'$MODEL_PATH'" \
  fairseq.dataset.gen_subset=test \
  results_path="'$GANS_OUTPUT_PHONES'" \
  fairseq.task.text_data="'$TEXT_OUTPUT/phones/'" \
  fairseq.dataset.batch_size=1 \
  fairseq.dataset.num_workers=0 \
  fairseq.dataset.required_batch_size_multiple=1
}




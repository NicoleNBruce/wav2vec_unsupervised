#!/bin/bash

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs

# Main directories
#.... directories to add to root.......
# Directory where launcher scripts live (workspace path)
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Installation root (where fairseq_, venv, kenlm, rVADfast are created)
DIR_PATH="${DIR_PATH:-$HOME/wav2vec_unsupervised}" # the root directory of the installed project artifacts
DATA_ROOT="$DIR_PATH/data" # a folder that stores all the data generated from pipeline
FAIRSEQ_ROOT="$DIR_PATH/fairseq_" # the root directory of the fairseq repository
KENLM_ROOT="$DIR_PATH/kenlm/build/bin"  # Path to KenLM installation
VENV_PATH="$DIR_PATH/venv"    # Path to virtual environment (optional)
RVAD_ROOT="$DIR_PATH/rVADfast/src/rVADfast" # the root directory of the rVADfast repository

GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"



# ==================== HELPER FUNCTIONS ====================

#fairseq file paths with slight changes made 
SPEECHPROCS="$DIR_PATH/rVADfast/src/rVADfast/speechproc/speechproc.py"
PREPARE_AUDIO="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh"
ADD_SELF_LOOP_SIMPLE="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/add-self-loop-simple.cc"
OPENFST_PATH="$DIR_PATH/fairseq/examples/speech_recognition/kaldi/kaldi_initializer.py"


# Arguments/variables
NEW_SAMPLE_PCT=0.5
MIN_PHONES=3
NEW_BATCH_SIZE=32
PHONEMIZER="G2P"
LANG="en"

#models 
FASTTEXT_LIB_MODEL="$DIR_PATH/lid_model/lid.176.bin"  # the path to the language identification model
MODEL="$DIR_PATH/pre-trained/wav2vec_vox_new.pt" # the path to the pre-trained wav2vec model for audio feature extraction

# Dataset specifics
DATASET_NAME="timit"

# Output directories (will be created if they don't exist)
MANIFEST_DIR="$DATA_ROOT/manifests" # the directory that stores the manifest files for the audio dataset
NONSIL_AUDIO="$DATA_ROOT/processed_audio/" #the directory that stores the audio files with silence removed 
MANIFEST_NONSIL_DIR="$DATA_ROOT/manifests_nonsil" #the directory that stores the manifest files foe audio dataset with silence removed
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET_NAME"  #stores the output of audio processing, the psuedophonemes(cluster IDs), Audio features
RESULTS_DIR="$DATA_ROOT/results/$DATASET_NAME" # Stores all the training information of the gans
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET_NAME" # stores the progress checkpoint file which keeps track of processes implemented 
LOG_DIR="$DATA_ROOT/logs/$DATASET_NAME" #stores the pipeline logs 
TEXT_OUTPUT="$DATA_ROOT/text" # stores the processes output from the prepared text function 


# Checkpoint file to track progress
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"


# Log message with timestamp
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_DIR/pipeline.log"
}

# Check if a step has been completed
is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:COMPLETED$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Check if a step is in progress (for recovery after crash)
is_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:IN_PROGRESS$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Mark a step as completed
mark_completed() {
    local step="$1"
    echo "$step:COMPLETED" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as completed"
}

# Mark a step as in progress
mark_in_progress() {
    local step="$1"
    # First remove any existing in-progress markers for this step
    if [ -f "$CHECKPOINT_FILE" ]; then
        sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
    fi
    echo "$step:IN_PROGRESS" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as in progress"
}

setup_path() {
    export HYDRA_FULL_ERROR=1
    export LD_LIBRARY_PATH="${KALDI_ROOT}/src/lib:${KENLM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
}


# Activate virtual environment if provided

activate_venv() {
    if [ -n "$VENV_PATH" ]; then
        log "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    fi
}


# Create directories if they don't exist
create_dirs() {
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR" "$MANIFEST_NONSIL_DIR" \
             "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$TEXT_OUTPUT" "$GANS_OUTPUT_PHONES"
}





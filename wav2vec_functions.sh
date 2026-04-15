#!/bin/bash

# Wav2Vec Unsupervised Pipeline Runner
# This script runs the entire fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

TRAIN_DATASETS=$1 #/path/to/unlabelled/train_audio_data 
VAL_DATASETS=$2 #/path/to/unlabelled/validation_audio_data 
TEST_DATASETS=$3
UNLABELLED_TEXT=$4 #/path/to/unlabelled_text_file 

source utils.sh


# ==================== HELPER FUNCTIONS ====================


# updating code in SPEECHPROCS to return outputs that are compatible with code change in vads.py
fixing_sflux() {
    TARGET_FILE="$SPEECHPROCS"

    # Check if the file exists
    if [ -f "$TARGET_FILE" ]; then
        echo "Updating sflux() to return two values in $TARGET_FILE..."
        
        # Find and modify the return statement inside sflux()
        sed -i '/def sflux/,/return/ s/^ *return .*/    return s_flatness, n_frames/' "$TARGET_FILE"
        
        # Confirm the fix by printing the modified return statement
        echo "Updated return statement in sflux():"
        grep "return " "$TARGET_FILE"
        
        echo "Fix applied successfully!"
    else
        echo "Error: $TARGET_FILE not found!"
        exit 1
    fi
}


#update is done in add-self-loop-simple.cc to replace std::endl with "\n" since std::endl is not compatible with pykaldi installation for text preprocessing
replace_std_endl() {
    local input_file="$1"
    
    if [[ ! -f "$input_file" ]]; then
        echo "Error: File '$input_file' not found!"
        return 1
    fi

    # Use sed to replace std::endl with \n and save the output
    sed -i 's/std::endl/"\\n"/g' "$input_file"

    echo "Replacement done in '$input_file'"
}


# Update sample_pct in the file 'prepare_audio'-- the variable measures the amount of audio dataset
#to us in generating k-mean clusters 
update_sample_pct(){
# This regex matches '--sample-pct' followed by any whitespace and a number (integer or decimal)
# and replaces it with '--sample-pct' followed by the new value.
    sed -i.bak -E "s/(--sample-pct[[:space:]]+)[0-9]*\.?[0-9]+/\1${NEW_SAMPLE_PCT}/g" $PREPARE_AUDIO
    echo "Updated '--sample-pct' to ${NEW_SAMPLE_PCT} in 'prepare_audio'. Backup saved as 'prepare_audio.bak'."

}

# Update batch_size in the file 'prepare_audio'
update_batch_size()
{
    sed -i.bak -E "s/(--batch-size[[:space:]]+)[0-9]+/\1${NEW_BATCH_SIZE}/g" $PREPARE_AUDIO
    echo "Updated '--batch-size' to ${NEW_BATCH_SIZE} in 'prepare_audio'. Backup saved as 'prepare_audio.bak'."

}



# ==================== MAIN STEPS ====================

# Step 1: Create data manifests
# Step 1: Create data manifests - Modified Functions

create_manifests_train() {

    local step_name="create_manifests_train" 

    if is_completed "$step_name"; then
        log "Skipping train manifest creation (already completed)"
        return 0
    fi

    log "Creating TRAIN data manifest..."
    mark_in_progress "$step_name"

    # Ensure the validation file potentially created here is removed or ignored later
    # Run the script to create train.tsv (and an empty valid.tsv)
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TRAIN_DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent 0 # Force 100% to train.tsv

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        # Optional: remove the empty valid.tsv if you want clarity
        # rm -f "$MANIFEST_DIR/valid.tsv"
        mark_completed "$step_name"
        log "TRAIN manifest creation completed successfully"
    else
        log "ERROR: TRAIN manifest creation failed"
        exit 1
    fi
}

create_manifests_val() {

    local step_name="create_manifests_val" 

    if is_completed "$step_name"; then
        log "Skipping validation manifest creation (already completed)"
        return 0
    fi

    log "Creating VALIDATION data manifest..."
    mark_in_progress "$step_name"

    # Create a temporary directory for the validation manifest generation
    local TEMP_VAL_DIR
    TEMP_VAL_DIR=$(mktemp -d "$MANIFEST_DIR/val_manifest.XXXXXX")
    log "Using temporary directory for validation manifest: $TEMP_VAL_DIR"

    # Generate manifests in the temporary directory (will create empty train.tsv and desired valid.tsv)
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$VAL_DATASETS" \
        --dest "$TEMP_VAL_DIR" \
        --ext wav \
        --valid-percent 1.0 # Force 100% to valid.tsv

    local python_exit_code=$?

    if [ $python_exit_code -eq 0 ]; then
        # Move the generated valid.tsv to the main manifest directory, overwriting the empty one if it exists
        if [ -f "$TEMP_VAL_DIR/valid.tsv" ]; then
            mv "$TEMP_VAL_DIR/valid.tsv" "$MANIFEST_DIR/valid.tsv"
            log "Moved validation manifest to $MANIFEST_DIR/valid.tsv"
            mark_completed "$step_name"
            log "VALIDATION manifest creation completed successfully"
        else
             log "ERROR: Expected valid.tsv not found in temporary directory $TEMP_VAL_DIR"
             rm -rf "$TEMP_VAL_DIR" # Clean up temp dir
             exit 1
        fi
    else
        log "ERROR: VALIDATION manifest creation failed (Python script error)"
        # No need to mark completed
    fi

    # Clean up the temporary directory
    rm -rf "$TEMP_VAL_DIR"
    log "Cleaned up temporary directory $TEMP_VAL_DIR"
}

create_manifests_test() {

    local step_name="create_manifests_test" 

    if is_completed "$step_name"; then
        log "Skipping train manifest creation (already completed)"
        return 0
    fi
    
    log "Creating Test data manifest..."
    mark_in_progress "$step_name"
    MANIFEST_TEST_DIR=$DATA_ROOT/manifest_test
    
    mkdir -p $MANIFEST_TEST_DIR

    # Ensure the validation file potentially created here is removed or ignored later
    # Run the script to create train.tsv (and an empty valid.tsv)
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TEST_DATASETS" \
        --dest "$MANIFEST_TEST_DIR" \
        --ext wav \
        --valid-percent 0 # Force 100% to train.tsv

    # Check if the command was successful
    cp -r $MANIFEST_TEST_DIR/train.tsv $MANIFEST_NONSIL_DIR/test.tsv
    rm -rf $MANIFEST_TEST_DIR
    if [ $? -eq 0 ]; then
        # Optional: remove the empty valid.tsv if you want clarity
        # rm -f "$MANIFEST_DIR/valid.tsv"
        mark_completed "$step_name"
        log "TEST manifest creation completed successfully"
    else
        log "ERROR: TEST manifest creation failed"
        exit 1
    fi
}

# --- In your main function ---
# Step 2: create vads files out of the audios 
create_rVADfast() { 
    
    local step_name="create_rVADfast"
    # # fixing certain code errors in the rvads  
    fixing_sflux #this script changes the sflux function to return both ft and n_frames

    if is_completed "$step_name"; then
        log "Skipping audio silence removal (already completed)"
        return 0
    fi
    
    
    log "removing silence from audios"
    mark_in_progress "$step_name"
    python "$SCRIPT_ROOT/vads.py" -r $RVAD_ROOT < "$MANIFEST_DIR/train.tsv" > "$MANIFEST_DIR/train.vads"
    python "$SCRIPT_ROOT/vads.py" -r $RVAD_ROOT < "$MANIFEST_DIR/valid.tsv" > "$MANIFEST_DIR/valid.vads"
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "silence removed successfully"
    else
        log "ERROR: silence removal  failed"
        exit 1
    fi
}

# Step 3: Remove silence from audios with vads files 
remove_silence() {

    local step_name="remove_silence"

   if is_completed "$step_name"; then
        log "Skipping audio silence removal1 (already completed)"
        return 0
    fi
    
    
    log "removing silence from audios1"
    mark_in_progress "$step_name"

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" --tsv "$MANIFEST_DIR/train.tsv" --vads "$MANIFEST_DIR/train.vads" --out "$NONSIL_AUDIO/train"
    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" --tsv "$MANIFEST_DIR/valid.tsv" --vads "$MANIFEST_DIR/valid.vads" --out "$NONSIL_AUDIO/val"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "silence1 removed successfully"
    else
        log "ERROR: silence removal  failed"
        exit 1
    fi

}

#Step 4: create new manifest files for train and validation set with no silence 
create_manifests_nonsil_train() {
    local step_name="create_manifests_nonsil_train"
    if is_completed "$step_name"; then
        log "Skipping nonsil manifest creation (already completed)"
        return 0
    fi
    
    log "Creating data manifests..."
    mark_in_progress "$step_name"
    
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/train" \
        --dest "$MANIFEST_NONSIL_DIR" \
        --ext wav \
        --valid-percent 0 

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "nonsil Manifest creation completed successfully"
    else
        log "ERROR: nonsil Manifest creation failed"
        exit 1
    fi
}

create_manifests_nonsil_val() {
    local step_name="create_manifests_nonsil_val"
    
    if is_completed "$step_name"; then
        log "Skipping nonsil validation manifest creation (already completed)"
        return 0
    fi
    
    log "Creating nonsil validation manifests..."
    mark_in_progress "$step_name"

    local TEMP_VAL_DIR
    TEMP_VAL_DIR=$(mktemp -d "$MANIFEST_NONSIL_DIR/val_manifest.XXXXXX")
    log "Using temporary directory for validation manifest: $TEMP_VAL_DIR"

    # Run the manifest creation and capture exit code immediately
    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/val" \
        --dest "$TEMP_VAL_DIR" \
        --ext wav \
        --valid-percent 1.0
    local python_exit_code=$?

    # Process results
    if [ $python_exit_code -eq 0 ]; then
        if [ -f "$TEMP_VAL_DIR/valid.tsv" ]; then
            mkdir -p "$MANIFEST_NONSIL_DIR"
            mv "$TEMP_VAL_DIR/valid.tsv" "$MANIFEST_NONSIL_DIR/valid.tsv"
            log "Moved validation manifest to $MANIFEST_NONSIL_DIR/valid.tsv"
            mark_completed "$step_name"
            log "VALIDATION manifest creation completed successfully"
        else
            log "ERROR: Expected valid.tsv not found in temporary directory $TEMP_VAL_DIR"
            rm -rf "$TEMP_VAL_DIR"
            exit 1
        fi
    else
        log "ERROR: VALIDATION manifest creation failed (Python exit code: $python_exit_code)"
        rm -rf "$TEMP_VAL_DIR"
        exit 1
    fi

    rm -rf "$TEMP_VAL_DIR"
}

#Step 5: Prepare audio file
prepare_audio() {

   local step_name="prepare_audio"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   # export KALDI_ROOT=$KALDI_ROOT
   # export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"

   update_sample_pct #personal scripts added to change sample_pct variable in prepare_audio.sh
   update_batch_size #personal scripts added to change batch_size variable in prepare_audio.sh  


   export KENLM_ROOT="$KENLM_ROOT"


   if is_completed "$step_name"; then
        log "Skipping audio preparation (already completed)"
        return 0
    fi
    
    log "audio preparation"
    mark_in_progress "$step_name"

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh" "$MANIFEST_NONSIL_DIR" $CLUSTERING_DIR $MODEL 512 14


    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "audio preparation successfully"
    else
        log "ERROR: audio preparation  failed"
        exit 1
    fi

}

#======================Text preparation =================================
# unsupervised/wav2vec-U/libri_dataset/librispeech-lm-norm_4k.txt
prepare_text() {
   local step_name="prepare_text"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   # export KALDI_ROOT=$KALDI_ROOT
   # export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"

   if is_completed "$step_name"; then
        log "Skipping text preparation (already completed)"
        return 0
    fi

    log "text preparation."
    mark_in_progress "$step_name"
    replace_std_endl $ADD_SELF_LOOP_SIMPLE  # this replaces the fixes error caused by the old script std::endl with \n
    # Ensure text prep can be safely re-run after partial failures.
    rm -rf "$TEXT_OUTPUT"
    mkdir -p "$TEXT_OUTPUT"

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh" "$LANG" "$UNLABELLED_TEXT" "$TEXT_OUTPUT" "$MIN_PHONES" "$PHONEMIZER" "$FASTTEXT_LIB_MODEL" 0.25
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "text preparation successfully"
    else
        log "ERROR: text preparation  failed"
        exit 1
    fi

}

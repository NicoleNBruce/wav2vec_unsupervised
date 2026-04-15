#!/bin/bash
set -e
source utils.sh
activate_venv

source_dir="$MANIFEST_NONSIL_DIR"
tgt_dir="$CLUSTERING_DIR"
model="$MODEL"
layer=14
dim=512

for split in test; do
    echo "Processing $split..."
    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py" "$source_dir" --split $split \
      --save-dir "$tgt_dir" --checkpoint "$model" --layer $layer

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py" "$tgt_dir" \
      --checkpoint "$model" --path "$tgt_dir/CLUS128" --split $split

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py" "$tgt_dir" --split $split \
      --save-dir "$tgt_dir/precompute_pca$dim" --pca-path "$tgt_dir/pca/${dim}_pca" --batch-size 32

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py" "$tgt_dir/precompute_pca$dim" \
      --cluster-dir "$tgt_dir/CLUS128" --split $split --save-dir "$tgt_dir/precompute_pca${dim}_cls128_mean" --pooling mean

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py" "$tgt_dir/precompute_pca${dim}_cls128_mean" \
      --save-dir "$tgt_dir/precompute_pca${dim}_cls128_mean_pooled" --split $split

    echo "Finished $split!"
done

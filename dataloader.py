import glob
import io
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


SRC_GLOB = "librispeech_dataset/data/test-*.parquet"
OUT_ROOT = "prepared_data"
TARGET_SR = 16000


def decode_audio_field(context_obj):
    if isinstance(context_obj, dict):
        if context_obj.get("array") is not None:
            arr = np.asarray(context_obj["array"], dtype=np.float32)
            sr = int(context_obj.get("sampling_rate", TARGET_SR))
            return arr, sr

        if context_obj.get("bytes") is not None:
            arr, sr = sf.read(io.BytesIO(context_obj["bytes"]), dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)

        if context_obj.get("path"):
            arr, sr = sf.read(context_obj["path"], dtype="float32")
            return np.asarray(arr, dtype=np.float32), int(sr)

    raise ValueError("Unsupported context audio format")


def to_mono_16k(arr, sr):
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        arr = resample_poly(arr, TARGET_SR, sr).astype(np.float32)
    return arr


def main():
    parquet_files = sorted(glob.glob(SRC_GLOB))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for pattern: {SRC_GLOB}")

    tables = [pq.read_table(p) for p in parquet_files]
    table = pa.concat_tables(tables)
    rows = table.to_pylist()

    random.seed(42)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    split_rows = {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }

    train_dir = os.path.join(OUT_ROOT, "train_wav")
    val_dir = os.path.join(OUT_ROOT, "val_wav")
    test_dir = os.path.join(OUT_ROOT, "test_wav")
    targets = {"train": train_dir, "val": val_dir, "test": test_dir}

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    texts = []
    written = {"train": 0, "val": 0, "test": 0}

    for split_name, rows_in_split in split_rows.items():
        out_dir = targets[split_name]
        for i, row in enumerate(rows_in_split):
            try:
                arr, sr = decode_audio_field(row["context"])
            except Exception:
                continue

            arr = to_mono_16k(np.asarray(arr, dtype=np.float32), sr)
            wav_path = os.path.join(out_dir, f"{split_name}_{i:06d}.wav")
            sf.write(wav_path, arr, TARGET_SR)
            written[split_name] += 1

            answer = row.get("answer")
            instruction = row.get("instruction")
            if isinstance(answer, str) and answer.strip():
                texts.append(answer.strip())
            elif isinstance(instruction, str) and instruction.strip():
                texts.append(instruction.strip())

    text_path = os.path.join(OUT_ROOT, "unlabeled_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    print("Done.")
    print("train:", train_dir)
    print("val:", val_dir)
    print("test:", test_dir)
    print("text:", text_path)
    print("written_counts:", written)

    if sum(written.values()) == 0:
        raise RuntimeError("No wav files were written. Check parquet schema and audio payload fields.")


if __name__ == "__main__":
    main()

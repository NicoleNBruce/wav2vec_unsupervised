import os

tsv_path = '/home/nikki/wav2vec_unsupervised/data/clustering/timit/precompute_pca512_cls128_mean_pooled/test.tsv'
timit_dir = '/mnt/c/Users/anoth/OneDrive - Ashesi University/Grad School (ICS)/NLP/wav2vec_unsupervised/timit/data'
output_path = '/mnt/c/Users/anoth/OneDrive - Ashesi University/Grad School (ICS)/NLP/wav2vec_unsupervised/test_ground_truth.txt'

# 1. Map all .PHN files in the TIMIT directory
phn_files = {}
for root, dirs, files in os.walk(timit_dir):
    for f in files:
        if f.upper().endswith('.PHN'):
            basename = f.upper().replace('.PHN', '')
            speaker = os.path.basename(root).upper()
            # Store with key "SPEAKER/BASENAME" for quick lookup
            phn_files[f"{speaker}/{basename}"] = os.path.join(root, f)

# 2. Read the valid.tsv file to get exact processing order
try:
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    print(f"Cannot find {tsv_path}")
    exit(1)

root_audio_dir = lines[0] # first line is root dir
gt_lines = []
missing = 0

for line in lines[1:]:
    if not line.strip(): continue
    audio_path = line.split('\t')[0] # e.g., FMLD0/SX385.WAV.wav
    
    # Parse speaker and file name to match TIMIT structure
    parts = audio_path.split('/')
    if len(parts) >= 2:
        speaker = parts[-2].upper()
    else:
        speaker = ""
    # Strip extensions (can be .WAV or .WAV.wav)
    basename = parts[-1].upper().replace('.WAV', '')
    
    key = f"{speaker}/{basename}"
    mapped_phn = phn_files.get(key)
    
    # Fallback to pure basename search if speaker folder mismatched
    if not mapped_phn:
        for k, v in phn_files.items():
            if k.endswith(f"/{basename}"):
                mapped_phn = v
                break
                
    if mapped_phn:
        with open(mapped_phn, 'r', encoding='utf-8') as pf:
            phones = []
            for pline in pf.read().splitlines():
                p = pline.strip().split()
                if len(p) == 3:
                    phone = p[2]
                    # Strip standard silence symbols
                    if phone not in ['h#', 'pau', 'epi', 'q']: 
                        # To match standard FAIRSEQ/TIMIT formatting, convert to uppercase 
                        # or keep lowercase depending on `valid.txt`. 
                        # `valid.txt` showed: ER R R D R M IY ER IY (Uppercase!)
                        phones.append(phone.upper())
        gt_lines.append(" ".join(phones))
    else:
        print(f"Warning: .PHN not found for {audio_path}")
        gt_lines.append("")
        missing += 1

# 3. Write out the valid_ground_truth.txt exactly matching the predictions order
with open(output_path, 'w', encoding='utf-8') as f:
    for gl in gt_lines:
        f.write(gl + "\n")
        
print(f"Successfully extracted {len(gt_lines)} ground truth lines to {output_path}.")
if missing > 0:
    print(f"Missed mapping for {missing} files.")

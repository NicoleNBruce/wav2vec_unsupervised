import os
import shutil
import random

test_dir = 'timit/data/TEST'
val_dir = 'timit/data/VAL'

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Each speaker is usually inside DRx directory
dialects = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
speakers_moved = 0
for dr in dialects:
    dr_path = os.path.join(test_dir, dr)
    
    speakers = [s for s in os.listdir(dr_path) if os.path.isdir(os.path.join(dr_path, s))]
    
    # Move ~15% of speakers to Validation
    num_val = max(1, int(len(speakers) * 0.15))
    
    # Sort for deterministic splitting if wanted, or random sample
    # Let's just sort and take the first few to be reproducible
    speakers.sort()
    val_speakers = speakers[:num_val]
    
    for s in val_speakers:
        src = os.path.join(dr_path, s)
        dest = os.path.join(val_dir, dr, s)
        
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        speakers_moved += 1
        
print(f"Splitting complete! Moved {speakers_moved} speakers from TEST to VAL.")

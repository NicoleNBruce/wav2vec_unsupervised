import sys

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_per(predictions_file, ground_truth_file):
    with open(predictions_file, 'r', encoding='utf-8') as f_pred:
        # Each phoneme is isolated by spaces in valid.txt, so we split them into arrays
        preds = [line.strip().split() for line in f_pred]
        
    with open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
        truths = [line.strip().split() for line in f_gt]
        
    if len(preds) != len(truths):
        print(f"Warning: Number of predictions ({len(preds)}) does not match ground truths ({len(truths)}).")
        print("PER calculation will proceed for overlapping pairs.")
        
    total_distance = 0
    total_length = 0
    
    for pred, truth in zip(preds, truths):
        total_distance += levenshtein_distance(pred, truth)
        total_length += len(truth)
        
    if total_length == 0:
        print("Error: Ground truth file has 0 valid phonemes.")
        return 0.0
        
    per = (total_distance / total_length) * 100
    return per

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_per.py <predictions.txt> <ground_truth.txt>")
        print("Example: python calculate_per.py valid.txt valid_ground_truth.txt")
        sys.exit(1)
        
    pred_file = sys.argv[1]
    gt_file = sys.argv[2]
    
    per = calculate_per(pred_file, gt_file)
    print(f"Phoneme Error Rate (PER): {per:.2f}%")
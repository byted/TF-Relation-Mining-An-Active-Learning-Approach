import sys
import csv, json
# from collections import Set

if len(sys.argv) < 4:
    print('Usage: python {} mapping predictions expanded_testset'.format(sys.argv[0]))
    sys.exit()

with open(sys.argv[1]) as f:
    mapping = json.load(f)

with open(sys.argv[2]) as f:
    predictions = f.read().strip().split('\n')[1:]

with open(sys.argv[3]) as f:
    testdata = f.readlines()

by_original_sent = {}
for idx, line in enumerate(predictions):
    prediction = float(predictions[idx].split('\t')[3])  # take proba for label 1
    mapping_val = mapping[str(idx)]
    if mapping_val not in by_original_sent:
        by_original_sent[mapping_val] = []
    by_original_sent[mapping_val].append((idx, prediction))

best_by_sent = {orig_idx:sorted(candidates, key=lambda x: float(x[1]), reverse=True)[0] for orig_idx, candidates in by_original_sent.items()}

with open(sys.argv[3] + '.collapsed', 'w') as f:
    for orig_idx, (best_expanded_idx, _) in best_by_sent.items():
        f.write(testdata[best_expanded_idx])


import sys, argparse
import csv, json
import gene_find
from copy import deepcopy

parser = argparse.ArgumentParser(description='Generate all possible gene combinations in Top2500 (if ambiguous)')
parser.add_argument('--top2500', '-t', type=str, required=True,
                help='Path to Top2500 CSV file')
parser.add_argument('--bidirectional', action='store_true',
                help='Also genereate reversed relations')
args = parser.parse_args()

with open(args.top2500) as f:
    reader = csv.DictReader(f)
    top = [l for l in reader]

def does_overlap(range1, range2):
    return len(set(range1).intersection(set(range2))) > 0

new_lines = []
mapping = {}
for i, line in enumerate(top):
    gene1, gene2 = line['gene1'], line['gene2']
    sentence = line['sentence']

    g1s = gene_find.find(gene1, sentence)
    g2s = gene_find.find(gene2, sentence)

    # if sentence.startswith('However, HOXB13 downregulated the expression'):
    #     import ipdb; ipdb.set_trace()
    for g1 in g1s:
        for g2 in g2s:
            if does_overlap(range(*g1.span()), range(*g2.span())):
                continue
            new_lines_i = len(new_lines)
            new_l = deepcopy(line)
            new_l['gene1_char_start'], new_l['gene1_char_end'] = g1.span()
            new_l['gene2_char_start'], new_l['gene2_char_end'] = g2.span()
            new_lines.append(new_l)
            mapping[new_lines_i] = i  # save index of original line

            if args.bidirectional:
                new_lines_i = len(new_lines)
                new_l = deepcopy(line)
                new_l['gene1'], new_l['gene2'] = new_l['gene2'], new_l['gene1']
                new_l['gene1_char_start'], new_l['gene1_char_end'] = g2.span()
                new_l['gene2_char_start'], new_l['gene2_char_end'] = g1.span()
                new_lines.append(new_l)
                mapping[new_lines_i] = i  # save index of original line

with open(args.top2500 + ('.bidirectional' if args.bidirectional else '') + '.expanded', 'w') as f:
    writer = csv.DictWriter(f, new_lines[0].keys())
    writer.writeheader()
    for l in new_lines:
        writer.writerow(l)
print(len(new_lines))
print(len(mapping))

with open(args.top2500 + ('.bidirectional' if args.bidirectional else '') + '.expanded-mapping', 'w') as f:
    json.dump(mapping, f)
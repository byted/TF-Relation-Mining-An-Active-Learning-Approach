import sys, re
from collections import namedtuple
from scipy import stats
import Top2500Parser

def parse_result_file(filepath):
    with open(filepath, 'r') as inf:
        lines = inf.read().split('\n')

    Row = namedtuple('Row', ['id', 'prediction', 'confidence', 'label'])
    rows = [Row(idx, *[float(cell) for cell in line.split('\t')]) for idx, line in enumerate(lines) if line != '']
    return rows 

def sort_by_confidence(rows):
    return sorted(rows, key=lambda t: t.confidence)

def get_negative_predictions_ids(rows):
    return [r.id for r in rows if int(r.prediction) == 0]

def save_results(outfilepath, rows):
    with open(outfilepath, 'w') as outf:
        for row in rows:
            outf.write('{}\t{}\t{}\t{}\n'.format(*row))

def check_multiple_instances_of_genes_in_sentence(list_of_sentence_ids):
    corpus = Top2500Parser.CorpusParser('../resources/tf_extraction_paper_thomas/TF.GO_enriched2.PMID-jSRE_v2_PD.both.sorted.csv')
    counter = 0
    for idx in list_of_sentence_ids:
        row = corpus.data[idx]
        sent, gene1, gene2 = row['Sentence'], row['Gene_mention1'], row['Gene_mention2']
        if len(re.findall(re.escape(gene1), sent)) > 1 or len(re.findall(re.escape(gene2), sent)) > 1:
           counter += 1
    return counter

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print('Usage: python {} predictions outfile'.format(sys.argv[0]))
        print('Compare predictions of classifier to gold standard')
        print('This script assumes that the test set is sorted by confidence (DESC).')
        sys.exit()
    predictions, outfile = sys.argv[1:3]

    rows = parse_result_file(predictions)
    negs = get_negative_predictions_ids(rows)
    rows = sort_by_confidence(rows)
    old_ranking = list(range(len(rows)))
    new_ranking = [r.id for r in rows]
    print(len(old_ranking), len(new_ranking))
    corr, _ = stats.spearmanr(old_ranking, new_ranking)
    print('ranking correlation:', corr)
    print('num of negative predictions: {} ({}%)'.format(len(negs), 100*len(negs)/2500))
    print('\tnum of sentences containing more than 1 occurance of the genes: {}'.format(check_multiple_instances_of_genes_in_sentence(negs)))
    print('negative predictions:')
    print(negs)
    save_results(outfile, rows)
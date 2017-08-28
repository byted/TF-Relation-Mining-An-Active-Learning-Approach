from libact.base.dataset import Dataset, import_jsre
from libact.models.jsre import JSRE
from active_learn import subsample

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import sys, os, argparse, statistics, datetime, time, json
import numpy as np

parser = argparse.ArgumentParser(description='Simple CV with JSRE')
parser.add_argument('--jsre_data', '-d', type=str,
                help='Path to JSRE compatible file')
parser.add_argument('--random_seed', '-r', type=int, default=None,
                help='Random seed used for splitting.')
parser.add_argument('--folds', '-f', type=int, default=5,
                help='Number of folds to run.')
parser.add_argument('--jsre', type=str, required=True, default=None,
                help='Path to JSRE source code')
parser.add_argument('--suffix', type=str, default='cv_eval',
                help='Text that will be appended to the output directory name')
parser.add_argument('--base', '-b', type=str, default=None,
                help='Path to JSRE training file to use as additional training data for each fold.')
parser.add_argument('--testset_size', type=int, default=None,
                help='Specify size of testset. Draws a testset of this size --folds times')
parser.add_argument('--subsample-base', type=int, required=False,
                help='Subsample the given number of instances from the base set in each fold')

args = parser.parse_args()

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
X, y = import_jsre(args.jsre_data).format_sklearn()
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
instances_per_fold = int(len(X)/args.folds)

if args.base is not None:
    X_base, y_base = import_jsre(args.base).format_jsre()
    print('Dataset: {} instances + additional {} from base data'.format(len(X), len(X_base)))
else:
    X_base, y_base = [], []
    print('Dataset: {} instances; {} folds => {} train size; {} test size'.format(
        len(X), args.folds, len(X)-instances_per_fold, instances_per_fold))

if args.testset_size is None:
    print('{} folds => {}+{}={} train size; {} test size'.format(
        args.folds, len(X)-instances_per_fold, len(X_base),
        (len(X)+len(X_base))-instances_per_fold, instances_per_fold))
else:
    if args.base is not None:
        print('{} rounds => {}+{}={} train size; {} test size'.format(
            args.folds, len(X)-args.testset_size, len(X_base), (len(X)+len(X_base))-args.testset_size, args.testset_size))
    else:
        print('{} rounds => {} train size; {} test size'.format(
            args.folds, len(X)-args.testset_size, args.testset_size))

scores = []

for i, (trn_idx, tst_idx) in enumerate(skf.split(X, y)):
    X_base_fold, y_base_fold = X_base, y_base
    if args.subsample_base is not None:
        X_base_fold, y_base_fold = subsample(X_base, y_base, args.subsample_base, rnd_state=args.random_seed)

    if args.testset_size is not None:  # random redrawing folds
        if args.testset_size == len(X):  # full AL set as testset
            X_trn, X_tst, y_trn, y_tst = np.array([]), X, np.array([]), y
        else:
            X_trn, X_tst, y_trn, y_tst = \
                train_test_split(X, y, test_size=args.testset_size, stratify=y, random_state=args.random_seed)
        trn_ds = Dataset([[i] for i in X_base_fold] + X_trn.tolist(), list(y_base_fold) + y_trn.tolist())
        tst_ds = Dataset(X_tst, y_tst)
    else:  # normal k-fold CV
        if args.base is not None:
            trn_ds = Dataset(
                [[i] for i in X_base_fold] + X[trn_idx].tolist(), 
                list(y_base_fold) + y[trn_idx].tolist())
        else:
            trn_ds = Dataset(X[trn_idx].tolist(), y[trn_idx].tolist())
        tst_ds = Dataset(X[tst_idx].tolist(), y[tst_idx].tolist())
    
    print('⚙\tFold {}... train size: {}, test size: {}'.format(i, len(trn_ds), len(tst_ds)))
    model = JSRE(args.jsre)
    model.train(trn_ds)
    scores.append(1 - model.score(tst_ds))

stats = {
    'metadata': {
        'commands': ' '.join(sys.argv),
    },
    'avg': sum(scores) / len(scores),
    'min': min(scores),
    'max': max(scores),
    'stdev': statistics.stdev(scores),
    'scores': scores
}
print(json.dumps(stats, indent=2))
with open('{}__{}_results.json'.format(timestamp, args.suffix), 'w') as f:
    json.dump(stats, f, indent=2)
print('Done 👏')

import sys, os, json, time, datetime, statistics, argparse, shutil
import random
from tqdm import tqdm
import numpy as np
import copy

from libact.base.dataset import Dataset, import_jsre
from libact.labelers.jsre_labeler import JSRELabeler
from libact.models.jsre import JSRE
from libact.query_strategies import UncertaintySampling
from libact.query_strategies.multiclass import EER
from libact.query_strategies import ActiveLearningByLearning
from libact.query_strategies import RandomSampling
from libact.models import LogisticRegression
from libact.query_strategies import QueryByCommittee

from sklearn.model_selection import train_test_split

available_query_strategies = {
    'random': lambda trn_ds, model: RandomSampling(trn_ds),
    'us_LC': lambda trn_ds, model: UncertaintySampling(trn_ds, method='lc', model=model),
    'us_SM': lambda trn_ds, model: UncertaintySampling(trn_ds, method='sm', model=model),
    'us_ENTROPY': lambda trn_ds, model: UncertaintySampling(trn_ds, method='entropy', model=model),
    'eer_log': lambda trn_ds, model: EER(trn_ds, model=model, loss='log'),
    'eer_01': lambda trn_ds, model: EER(trn_ds, model=model, loss='01'),
    'eer_log_SMPL': lambda trn_ds, model: EER(trn_ds, model=model, loss='log', random_sampling=200),
    'eer_01_SMPL': lambda trn_ds, model: EER(trn_ds, model=model, loss='01', random_sampling=200),
    'qbc_KL-3': lambda trn_ds, models: QueryByCommittee(trn_ds, disagreement='kl_divergence', models=models),
    'albl-3': lambda trn_ds, model, query_strategies, T=100: ActiveLearningByLearning(trn_ds, T=T, query_strategies=query_strategies, model=model)
}

def subsample(X, y, sample_size, rnd_state=42):
    X_sample, _, y_sample, _ = \
    train_test_split(X, y, train_size=sample_size, stratify=y, random_state=rnd_state)
    return X_sample, y_sample

def subsample_libact(data, sample_size, rnd_state=42):
    X, y = zip(*data.get_entries())
    return Dataset(*subsample(X, y, sample_size, rnd_state=rnd_state))

def split_train_test(base_data, al_data, pool_size, n_labeled, rnd_state=42):
    try:
        X_base, y_base = zip(*base_data.get_entries())
    except AttributeError:
        X_base, y_base = [], []
    X_al, y_al = zip(*al_data.get_entries())

    if len(X_base) == 0:  # take n_labeled from al_data set
        X_initial, X_al, y_initial, y_al = \
            train_test_split(X_al, y_al, test_size=len(X_al)-n_labeled, random_state=rnd_state)
    elif  n_labeled < len(X_base):
        _, X_initial, _, y_initial = \
            train_test_split(X_base, y_base, test_size=n_labeled, random_state=rnd_state)
    else:
        X_initial, y_initial = X_base, y_base
    
    test_size = len(X_al) - pool_size
    X_pool, X_test, y_pool, y_test = \
        train_test_split(X_al, y_al, test_size=test_size, random_state=rnd_state)

    # include labeled base data in pool
    trn_X = np.concatenate([X_initial, X_pool])
    trn_y = np.concatenate([y_initial, [None] * len(y_pool)])
    trn_ds = Dataset(trn_X, trn_y) 
    tst_ds = Dataset(X_test, y_test)
    complete_labeled_trn_y = np.concatenate([y_initial, y_pool])
    fully_labeled_trn_ds = Dataset(trn_X, complete_labeled_trn_y)
    initial_pool_size = len(y_pool)

    return trn_ds, tst_ds, initial_pool_size, fully_labeled_trn_ds

def run(trn_ds, tst_ds, lbr, model, qs, quota, fn_prefix):
    E_in, E_out = [], []
    durations = []

    for i in tqdm(range(quota)):
        start_time = time.time()
        ask_id = qs.make_query()

        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)
        durations.append(time.time() - start_time)

        model.train(trn_ds)
        E_in.append(1 - model.score(trn_ds))
        E_out.append(1 - model.score(tst_ds))


    with open('{}.results.json'.format(fn_prefix), 'w') as f:
        json.dump({'E_in': E_in, 'E_out': E_out, 'durations': durations}, f, indent=2)

    return E_in, E_out, durations

def print_info(trn_ds, tst_ds, fold_nbr, quota):
    _, labeled = trn_ds.format_jsre()
    print('\tFold {}'.format(fold_nbr))
    print('\t\tCreated data with following counts:')
    print('\t\t\tinitially labeled trainset size:\t{}'.format(len(labeled)))
    print('\t\t\tinitially unlabeled pool size:\t{}'.format(len(trn_ds)-len(labeled)))
    print('\t\t\ttestset size {}'.format(len(tst_ds)))
    print('\t\t\trounds: {}'.format(quota))

def calc_stats(observations_per_fold):
    '''Calculates min/max/avg/stdev from a list of list where each inner lists represents a fold run'''
    stats = []
    for i, _ in enumerate(observations_per_fold[0]):
        across_folds = [observations_per_fold[j][i] for j in range(len(observations_per_fold))]
        if len(across_folds) == 1:
            stdev = -1
        else:
            stdev = statistics.stdev(across_folds)
        stats.append({
            'avg': sum(across_folds) / len(observations_per_fold),
            'min': min(across_folds),
            'max': max(across_folds),
            'stdev': stdev
        })
    return stats


def main():
    parser = argparse.ArgumentParser(description='CV evaluation of AL with JSRE')
    parser.add_argument('--base_data', '-bd', type=str,
                    help='Path to JSRE compatible training file used for initial training')
    parser.add_argument('--al_data', '-ad', type=str, required=True,
                    help='Path to JSRE compatible training file used for AL and evaluation')
    parser.add_argument('--poolsize', '-ps', type=float, default='200',
                    help='Absolute size of the pool that is used for AL. Rest is used as test set \
                          - if value is smaller than 1 it is interpreted as a percentage.')
    parser.add_argument('--rounds', '-r', type=int,
                    help='Number of times to draw from the pool. Defaults to poolsize of not explicitly set.')
    parser.add_argument('--start_label_count', '-s', type=int,
                    help='Number of samples that are initially labeled.')
    parser.add_argument('--folds', '-f', type=int, default=5,
                    help='Number of folds to run.')
    parser.add_argument('--query_strategies', required=True, nargs='*', choices=sorted(available_query_strategies.keys()))
    parser.add_argument('--jsre', type=str, required=True, default=None,
                    help='Path to JSRE source code')
    parser.add_argument('--suffix', type=str, default='cv_eval',
                    help='Text that will be appended to the output directory name')
    parser.add_argument('--random-seeds', '-rs', required=False, nargs='+',
                    help='Random seeds for each fold')
    parser.add_argument('--subsample-base', type=int, required=False,
                    help='Subsample the given number of instances from the base set in each fold')

    args = parser.parse_args()

    if args.base_data is None:
        base_data = ([],[])
    else:
        base_data = import_jsre(args.base_data)
    al_data = import_jsre(args.al_data)
    pool_size = int(len(al_data[0]) * args.poolsize) if args.poolsize < 1 else int(args.poolsize)
    quota = args.rounds or pool_size
    folds = args.folds
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    parent_folder = '{}/{}__{}/'.format(os.getcwd(), timestamp, args.suffix)
    jsre_path = args.jsre
    subsample_base_size = args.subsample_base
    if subsample_base_size is not None and subsample_base_size >= len(base_data.data):
        print('Subsample size of {} is larger than the base dataset ({} instances)\
         - this makes no sense'.format(subsample_base_size, len(base_data.data)))
    if args.start_label_count is None:
        n_labeled = len(base_data.data)
    else:
        n_labeled = args.start_label_count
    
    if args.random_seeds is None:
        random_seeds = [random.randint(0, 10000) for _ in range(folds)] 
    else:
        random_seeds = [int(i) for i in args.random_seeds]
        print('Using random seeds from command line')
        if len(random_seeds) != folds:
            print('{} random seeds for {} folds given - mismatch'.format(len(random_seeds), folds))
            sys.exit()
    print('Random seeds: {}'.format(random_seeds))

    query_strategies = {k:v for k,v in available_query_strategies.items() if k in args.query_strategies}

    try:
        print('Base dataset: {} instances'.format(len(base_data.data)))
    except AttributeError:
        print('Base dataset: n/a')
    print('AL dataset: {} instances'.format(len(al_data.data)))

    try:
        os.mkdir(parent_folder)
    except FileExistsError:
        print('ERROR: could not create folder {}'.format(parent_folder))
        sys.exit()

    if args.base_data is not None:
        shutil.copy2(args.base_data, parent_folder)
    shutil.copy2(args.al_data, parent_folder)

    with open('{}/config.txt'.format(parent_folder), 'w') as f:
        f.write('command:\t' + ' '.join(sys.argv) + '\n')
        f.write('random seeds per fold:\t' + str(random_seeds) + '\n')
    print('Created {} to collect data for this run and initialized with metadata'.format(parent_folder))

    for qs_name, qs_constructor in query_strategies.items():
        print('Run query strategy "{}"'.format(qs_name))
        E_ins, E_outs = [], []
        durations = []
        for fold_nbr in range(folds):
            fold_fn_prefix = '{}/{}_fold-{}'.format(parent_folder, qs_name, fold_nbr)

            if subsample_base_size is None:
                this_fold_base_data = base_data
            else:
                print('\tSubsampling activated: subsample {} instances from {}'.format(subsample_base_size, args.base_data))
                this_fold_base_data = subsample_libact(base_data, subsample_base_size)
                lines = ['{}\t{}'.format(lbl, feat) for feat, lbl in zip(*this_fold_base_data.format_jsre())]
                with open('{}/fold-{}-subsample'.format(parent_folder, fold_nbr), 'w') as f:
                    f.write('\n'.join(lines))

            # Load data for fold
            trn_ds, tst_ds, initial_pool_size, fully_labeled_trn_ds = \
                split_train_test(this_fold_base_data, al_data, pool_size, n_labeled, rnd_state=random_seeds[fold_nbr])

            # Write (debug) stats
            print_info(trn_ds, tst_ds, fold_nbr, quota)

            labeler = JSRELabeler(fully_labeled_trn_ds)
            model = JSRE(jsre_path)

            if qs_name.startswith('qbc'):
                qs = qs_constructor(trn_ds, [
                    JSRE(jsre_path),  # default C,
                    JSRE(jsre_path, C=0.01),  # small C
                    JSRE(jsre_path, C=1)  # big C
                ])
            elif qs_name.startswith('albl'):
                qs = qs_constructor(trn_ds, JSRE(jsre_path), [
                    UncertaintySampling(trn_ds, model=JSRE(jsre_path)),
                    UncertaintySampling(trn_ds, model=JSRE(jsre_path, C=0.01)),
                    UncertaintySampling(trn_ds, model=JSRE(jsre_path, C=2))
                ], T=quota)
            else:
                qs = qs_constructor(trn_ds, model=JSRE(jsre_path))

            E_in, E_out, fold_durations = run(trn_ds, tst_ds, labeler, model, qs, quota, fold_fn_prefix)
            E_ins.append(E_in)
            E_outs.append(E_out)
            durations.append(fold_durations)

            print('\t\tWrote fold results to "{}.results.json"'.format(fold_fn_prefix))

        with open('{}/{}_results_raw.json'.format(parent_folder, qs_name), 'w') as f:
            json.dump({
                'E_ins': E_ins,
                'E_outs': E_outs,
            }, f, indent=2)

        with open('{}/{}_results.json'.format(parent_folder, qs_name), 'w') as f:
            json.dump({
                'E_ins': calc_stats(E_ins),
                'E_outs': calc_stats(E_outs),
                'durations': calc_stats(durations)
            }, f, indent=2)
        print('DONE with {}\n\n'.format(qs_name))

if __name__ == '__main__':
    main()
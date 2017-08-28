from libact.base.dataset import Dataset, import_jsre
from libact.models.jsre import JSRE

import sys, os, argparse, statistics, datetime, time, json

parser = argparse.ArgumentParser(description='Simple CV with JSRE')
parser.add_argument('--jsre_data', '-d', type=str,
                help='Path to JSRE compatible file')
parser.add_argument('--folds', '-f', type=int, default=5,
                help='Number of folds to run.')
parser.add_argument('--jsre', type=str, required=True, default=None,
                help='Path to JSRE source code')
parser.add_argument('--suffix', type=str, default='train_time',
                help='Text that will be appended to the output directory name')
parser.add_argument('--train_sizes', '-ts', type=int, nargs='+', required=True,
				help='List of sizes to train')

args = parser.parse_args()

X, y = import_jsre(args.jsre_data).format_sklearn()
sizes = args.train_sizes
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

print('Dataset: {} instances; measure train time for sizes: {}, repeat {} times per size'.format(
	len(X), sizes, args.folds))

measures = {s: [] for s in sizes}

model = JSRE(args.jsre)
for size in sizes:
	trn_ds = Dataset(X[:size], y[:size])
	for i in range(args.folds):
		print('⚙ Measure time for size {} ({}. round...)'.format(size, i))
		start = time.time()
		model.train(trn_ds)
		end = time.time()
		duration = end - start
		print('⚙ ... took {} seconds'.format(duration))
		measures[size].append(duration)

stats = {size: {
	'avg': sum(values) / len(values),
    'min': min(values),
    'max': max(values),
    'stdev': statistics.stdev(values),
    'scores': values
} for size, values in measures.items()}

out = {
	'metadata': {
		'commands': ' '.join(sys.argv),
	},
	'results': stats
}

print(json.dumps(out, indent=2))
with open('{}__{}_results.json'.format(timestamp, args.suffix), 'w') as f:
	json.dump(out, f, indent=2)
print('Done 👏')

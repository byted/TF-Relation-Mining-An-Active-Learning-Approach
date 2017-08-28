import sys, os, re, json, statistics
from active_learn import calc_stats

path = sys.argv[1]

stats_files = [f for f in os.listdir(path) if re.match(r'.*?_results\.json', f)]

print('Found {} stats files'.format(stats_files))

prefix = re.search(r'(.*?).results\.json', stats_files[0]).group(1)
outfile_name = prefix + 'results_merged.json'

values = []
metadatas = []
for stat in [json.load(open(os.path.join(path, fn), 'r')) for fn in stats_files]:
	values.extend(stat['scores'])
	metadatas.append(stat['metadata'])

with open(os.path.join(path, outfile_name), 'w') as f:
    json.dump({
	'metadatas': metadatas,
    'avg': sum(values) / len(values),
    'min': min(values),
    'max': max(values),
    'stdev': statistics.stdev(values),
    'scores': values
}, f, indent=2)


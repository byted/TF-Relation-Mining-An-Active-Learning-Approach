import sys, os, re, json
from active_learn import calc_stats

path = sys.argv[1]

fold_stats_files = [f for f in os.listdir(path) if re.match(r'.*?_fold-\d+\.results\.json', f)]

print('Found {} fold stats files'.format(fold_stats_files))

prefix = re.search(r'(.*?)_fold-\d+\.results\.json', fold_stats_files[0]).group(1)
outfile_name = prefix + '_results.json'

E_ins, E_outs = [], []
for stat in [json.load(open(os.path.join(path, fn), 'r')) for fn in fold_stats_files]:
	E_ins.append(stat['E_in'])
	E_outs.append(stat['E_out'])


with open(os.path.join(path, outfile_name), 'w') as f:
    json.dump({
        'E_ins': calc_stats(E_ins),
        'E_outs': calc_stats(E_outs),
    }, f, indent=2)


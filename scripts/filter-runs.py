# filter runs by some combination of keys
# something that TB should do but doesnt
import argparse
import simplejson as json
import os
import os.path as osp
import re
import itertools
from collections import defaultdict

def regex_parts(winners, losers):
    "Return parts that match at least one winner, but no loser."
    wholes = {'^' + w + '$'  for w in winners}
    parts = {d for w in wholes for p in subparts(w) for d in dotify(p)}
    return wholes | {p for p in parts if not matches(p, losers)}

def subparts(word, N=4):
    "Return a set of subparts of word: consecutive characters up to length N (default 4)."
    return set(word[i:i+n+1] for i in range(len(word)) for n in range(N))

def dotify(part):
    "Return all ways to replace a subset of chars in part with '.'."
    choices = map(replacements, part)
    return {cat(chars) for chars in itertools.product(*choices)}

def replacements(c): return c if c in '^$' else c + '.'

cat = ''.join

def findregex(winners, losers, k=4):
    "Find a regex that matches all winners but no losers (sets of strings)."
    # Make a pool of regex parts, then pick from them to cover winners.
    # On each iteration, add the 'best' part to 'solution',
    # remove winners covered by best, and keep in 'pool' only parts
    # that still match some winner.
    pool = regex_parts(winners, losers)
    solution = []
    def score(part): return k * len(matches(part, winners)) - len(part)
    while winners:
        best = max(pool, key=score)
        solution.append(best)
        winners = winners - matches(best, winners)
        pool = {r for r in pool if matches(r, winners)}
    return OR(solution)

def matches(regex, strings):
    "Return a set of all the strings that are matched by regex."
    return {s for s in strings if re.search(regex, s)}

OR = '|'.join # Join a sequence of strings with '|' between them

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--args", type=str, nargs='+')

args=parser.parse_args()
path=args.path
filter_args=dict(y.split('=') for y in args.args)
# TODO recursively collect runs from path
#for root, dirs, files in os.walk(path):
#    dirs
#
runs = [d for d in os.listdir(path) if osp.isdir(osp.join(path, d))]
print('path={}'.format(path))
matching=[]
for run in runs:
    conf_file = osp.join(path, run,'config.json')
    if os.path.exists(conf_file):
        with open(conf_file, 'rt') as fh:
            try:
                config = json.load(fh)
                if all(str(config.get(k))==str(v) for k,v in filter_args.items()):
                    print(run)
                    matching.append(run)
            except json.JSONDecodeError:
                pass
print(f'{len(matching)} runs matching {filter_args}')
print('regex={}'.format(OR(matching)))
mset=set(matching)
print('regex={}'.format(findregex(mset, set(runs).difference(mset))))

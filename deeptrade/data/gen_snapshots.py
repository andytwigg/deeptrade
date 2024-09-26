from deeptrade.envs.bookgen import BookGenerator
import simplejson as json
import os, re, argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from deeptrade.envs.book import OrderBook
from deeptrade.envs.shadow_book import ShadowOrderBook
import gzip

"""
generates snapshot files for use with l2bookgen.py
"""
# TODO only support one of minmatches, minprice, mintime
# change to --step_type [time, vol, price] --step_val 1 [float] 
# TODO change from num matches to vol (cumulative on either side)


parser = argparse.ArgumentParser()
parser.add_argument('product_id')
parser.add_argument('step_type', choices=['time','match','price'], type=str)
parser.add_argument('step_val', type=float)
parser.add_argument('--depth', default=20, type=int)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()
product_id = args.product_id
path = os.environ['DEEPTRADE_DATA']
curr_path = os.path.join(path, 'gdax_book', product_id, 'split_daily')
out_path = os.path.join(path, 'gdax_book', product_id, 'snapshots', f'{args.step_type}_{args.step_val}')
if not os.path.exists(out_path): os.makedirs(out_path)

NUM_LVLS=args.depth
def playback(f, stop):
    print(f'[start] {f}')
    def do_dump(book, matches):
        book.sanity_check()
        #bid_p, bid_q = book.get_bids(NUM_LVLS)
        #ask_p, ask_q = book.get_asks(NUM_LVLS)
        snapshot = {
            'time': book.time,
            'sequence': book.seq,
            'matches': matches,
            'bids': book.get_bids(NUM_LVLS),
            'asks': book.get_asks(NUM_LVLS),
            'stats': book.stats,
            #'bids': (bid_p, bid_q, [book.lim_qtys.get(p,0) for p in bid_p], [book.cancel_qtys.get(p,0) for p in bid_p]),
            #'asks': (ask_p, ask_q, [book.lim_qtys.get(p,0) for p in ask_p], [book.cancel_qtys.get(p,0) for p in ask_p]),
        }
        if not args.dryrun:
            print(json.dumps(snapshot, use_decimal=True), file=fh)
        book.reset_qty_counters()

    t0={}; seq0={}
    duration={}; nseqs={}
    fname = os.path.basename(f)
    # use the shadow order book so that maker order seqs are preserved across snapshots
    # TODO move book patching code to regular OrderBook
    bookfn = lambda: ShadowOrderBook(product_id, fill_type='optimistic', match_type='optimistic')
    bookgen = BookGenerator([f], product_id, bookfn, args.step_type, args.step_val)
    i=0
    with gzip.open(os.path.join(out_path, fname), 'wt') as fh:
        for i, (book, matches) in enumerate(bookgen.replay()):
            if book.valid():
                do_dump(book, matches)
            if stop.is_set(): return

        bookgen.stop_replay()
    print(f'[done] {f} len={i}')# duration={duration} nseqs={nseqs}')


files = filter(lambda f: f.endswith('.gz'), os.listdir(curr_path))
#files = filter(lambda f: re.match(r'^\d+.gz$', f), files)
files = sorted(files, key=lambda f: f.split('.')[0])
files = [os.path.join(curr_path, f) for f in files]
print(f'processing {len(files)} files with args {vars(args)}')
manager = multiprocessing.Manager()
with ProcessPoolExecutor() as pool:
    fs = []
    stop = manager.Event()
    for i, f in enumerate(files):
        fs.append(pool.submit(playback, files[i], stop))
    try:
        for future in as_completed(fs):
            future.result()
    except KeyboardInterrupt:
        print('shutting down')
        stop.set()
        pool.shutdown(wait=True)
    except Exception as exc:
        import traceback, sys
        print(f'[ERROR] {exc}')
        print(traceback.format_exc())

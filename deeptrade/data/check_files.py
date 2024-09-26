import os, re, argparse
from deeptrade.envs.bookgen import BookGenerator
from deeptrade.envs.shadow_book import ShadowOrderBook
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument('product_id', default='BTC-USD')
parser.add_argument('--step_type', type=str, default='match')
parser.add_argument('--step_val', type=float, default=1.0)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--shadow', action='store_true')
args=parser.parse_args()
product_id = args.product_id
step_type=args.step_type
step_val=args.step_val
seed=args.seed
path = os.environ['DEEPTRADE_DATA']
curr_path = os.path.join(path, 'gdax_book', args.product_id, 'split_daily')


def check(f):
    fname = os.path.basename(f)
    try:
        bookfn = lambda: ShadowOrderBook(product_id, fill_type='realistic', match_type='realistic')
        #bookfn = lambda: OrderBook(product_id)
        bookgen = BookGenerator([f], product_id, bookfn, step_type, step_val, seed=seed)
        n=0
        start_time = 0
        start_seq = 0
        for n, (book,matches) in enumerate(bookgen.replay()):
            #book=books[product_id]
            if n==0:
                start_time = book.time
                start_seq = book.seq
            book.sanity_check()
        nseqs = book.seq - start_seq
        ntime = book.time - start_time
        nmatches = len(matches)
        if n>1:
            print(f'[OK] {fname} len={n} nseqs={nseqs} nmatches={nmatches} duration={ntime:.2f}s')
        else:
            print(f'[ERROR] {fname} too short len={n}')
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(f'[ERROR] {fname} err={e}')

files = filter(lambda f: f.endswith('.gz'), os.listdir(curr_path))
files = sorted(files, key=lambda f: int(f.split('.')[0].replace('-','')))
files = [os.path.join(curr_path, f) for f in files]
print(f'path={curr_path} processing {len(files)} files with args {vars(args)}')
with ProcessPoolExecutor() as pool:
    try:
        fs=[]
        for f in files:
            fs.append(pool.submit(check, f))

        #n=len(files)
        #for i in range(n-1):
        #    fs.append(pool.submit(playback, files[i:]))
        for future in as_completed(fs):
            try:
                future.result()
            except Exception as exc:
                import traceback
                print(f'[ERROR] {exc}')
                print(traceback.format_exc())
    except KeyboardInterrupt:
        print('interrupted...')

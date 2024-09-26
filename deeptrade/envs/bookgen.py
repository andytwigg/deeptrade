import simplejson as json
import os
import gzip
import ciso8601
import random
from deeptrade.envs.book import OrderBook
from deeptrade.envs.shadow_book import REALISTIC_FILL, REALISTIC_MATCH
from deeptrade.envs.shadow_book import ShadowOrderBook

class SeqgapError(Exception):
    pass

def file_data_iterator(files, random_order=True):
    n = len(files)
    i = random.randint(0, n-1) if random_order else 0
    while i<len(files):
        f = files[i]
        with gzip.open(f, 'rt') as fh:
            print(f'opening file {f}')
            for line in fh:
                seq, msg = line.split(maxsplit=1)
                msg = json.loads(msg)
                # TODO parse time in split_files or preprocessing
                msg['time'] = ciso8601.parse_datetime(msg['time']).timestamp()
                msg['sequence'] = int(msg['sequence'])
                yield i, int(seq), msg
        i+=1
    return

class BookGenerator:
    def __init__(self, files, product_id, bookfn, step_type, step_val, seed=123):
        self.product_id = product_id
        self.step_type = step_type
        self.step_val = step_val

        self.files = files
        self.need_reset = True
        self.bookfn = bookfn
        self.MAX_SEQ_GAP=1000  # if we see a seqgap larger than this size, don't try to recover
        self.MAX_PRICE_GAP=1.0  # if we see a price change larger than this size between valid msgs, don't try to recover
        random.seed(seed)
        self.reset()

    def reset(self):
        self.curr_seq = 0
        self.valid = False
        self.book = self.bookfn()
        self.data = file_data_iterator(self.files, random_order=True)
        self.curr_msg = None
        self.cur_file = None
        self.need_reset = False

    def replay(self):
        if self.need_reset:
            self.reset()
        data = self.data
        book = self.book
        matches = []
        self.last_cur_file = self.cur_file
        last_booktime = -1
        last_bookprice = -1
        next_t = None
        try:
            """
            this is annoyingly complex, it does 2 things which could be implemented as wrappers:
            1) allows the caller to stop consuming the generator and continue by calling replay()
            2) deals with episodes that span files => deals with seqgap errors when files arent contiguous            
            """
            while True:
                msg = None
                if self.curr_msg:
                    msg = self.curr_msg
                    self.curr_msg = None
                else:
                    file, seq, msg = next(data)
                    self.cur_file = file
                    self.curr_msg = msg

                # ignore any sequence numbers before or equal to the current one
                if msg['sequence'] <= self.curr_seq:
                    #print(str(msg)[:200])
                    #print('backwards seq: curr_seq={} msg_seq={}'.format(self.curr_seq, msg['sequence']))
                    self.curr_msg = None
                    continue
                if self.valid and msg['sequence'] != (self.curr_seq+1):
                    # we have become invalid
                    print('invalid at msg_seq={}, curr_seq={}'.format(msg['sequence'], self.curr_seq))
                    self.valid = False
                if msg['type'] == 'snapshot':
                    if not self.valid:
                        # we have become valid again
                        # if gap is small enough, try to resync
                        seqgap = msg['sequence'] - book.seq
                        if last_booktime>0 and seqgap>self.MAX_SEQ_GAP:
                            raise SeqgapError
                        self.valid = True
                    else:
                        self.curr_msg = None
                        continue # skip this msg so we can process next one with same seq

                self.curr_seq = msg['sequence']
                self.curr_time = msg['time']

                if self.valid:
                    if msg['type']=='match':
                        msg['maker_order_seq'] = book.open_orders[msg['maker_order_id']]['seq']
                        matches.append(msg)
                    book.update(msg)

                    if book.valid():
                        bid, _ = book.get_bid()
                        ask, _ = book.get_ask()
                        #spread = float(ask-bid)
                        if (book.time-last_booktime)>0:# and spread<=0.01:
                            if (self.step_type=='match' and len(matches)>=self.step_val) \
                                or (self.step_type=='price' and abs(book.price()-last_bookprice)>=self.step_val) \
                                or (self.step_type=='time' and (book.time-last_booktime)>=self.step_val):
                                # step
                                last_booktime = book.time
                                last_bookprice = book.price()
                                next_t = None
                                yield book, matches
                                matches = []
                    #if msg['type']=='match':
                    #    msg['maker_order_seq'] = book.open_orders[msg['maker_order_id']]['seq']
                    #    matches.append(msg)
                    #book.update(msg)
                    self.curr_msg = None # mark this msg as processed
        except SeqgapError as e:
            # underlying generator still okay
            self.need_reset = False
        except StopIteration as e:
            # underlying generator has finished
            self.need_reset = True

    def stop_replay(self):
        pass
        # if episode spanned > 1 file, reset (to avoid linearly walking all files)
        #if self.last_cur_file and (self.cur_file != self.last_cur_file):
        #    self.need_reset = True

    def close(self):
        pass

def get_bookgen(path, product_id, step_type, step_val, fill_type=REALISTIC_FILL, match_type=REALISTIC_MATCH, seed=123, lastn=0, exclude_lastn=None, shadowbook=True):
    curr_path = os.path.join(path, 'gdax_book', product_id, 'split_daily')
    assert os.path.exists(curr_path), '{} not found'.format(curr_path)
    files = filter(lambda f: f.endswith('.gz'), os.listdir(curr_path))
    files = sorted(files, key=lambda f: int(f.split('.')[0].replace('-','')))
    files = [os.path.join(curr_path, f) for f in files][-lastn:]
    print(files, exclude_lastn)
    if exclude_lastn:
        files = files[:-exclude_lastn]
    assert len(files)>0, curr_path
    if shadowbook:
        bookfn = lambda: ShadowOrderBook(product_id, fill_type=fill_type, match_type=match_type)
    else:
        bookfn = lambda: OrderBook(product_id)#, fill_type=fill_type, match_type=match_type)
    return BookGenerator(files, product_id, bookfn, step_type, step_val, seed=seed)

if __name__ == '__main__':
    from time import time
    from datetime import datetime
    from os import environ
    import argparse
    from decimal import Decimal
    parser = argparse.ArgumentParser()
    parser.add_argument('product_id', default='BTC-USD')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--step_type', type=str)
    parser.add_argument('--step_val', type=float)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--files', type=str) # file regex
    args = parser.parse_args()
    path = environ['DEEPTRADE_DATA']
    print(f'playback DEEPTRADE_DATA={path} {vars(args)}')
    files = args.files
    if files:
        import glob
        curr_path = os.path.join(path, 'gdax_book', args.product_id, 'split_daily')
        files = glob.glob(curr_path+'/'+args.files)
        files = sorted(files, key=lambda f: int(f.split('/')[-1].split('.')[0].replace('-','')))
        #files = [os.path.join(curr_path, f) for f in files]
        for f in files: print(f)
        bookfn = lambda: ShadowOrderBook(args.product_id, fill_type='realistic', match_type='optimistic')
        bookgen= BookGenerator(files, args.product_id, bookfn, args.step_type, args.step_val, seed=args.seed)
    else:
        bookgen = get_bookgen(path, args.product_id, args.step_type, args.step_val, seed=args.seed, shadowbook=False)
    while True:
        print('===== START EPISODE =====')
        t=time(); start_time=0; start_seq=0; nmatches=0
        for l, (book, matches) in enumerate(bookgen.replay()):
            if start_time==0:
                start_time = book.time
                start_seq = book.seq
            book.sanity_check()
            nmatches+=len(matches)
            isotime=datetime.utcfromtimestamp(book.time).isoformat()
            price = book.price() or 0
            # . The side field indicates the maker order side. 
            bidvol = sum([Decimal(m['size']) for m in matches if m['side']=='buy']) # * price
            askvol = sum([Decimal(m['size']) for m in matches if m['side']=='sell']) # * price
            if l>0:
                print('-- {} -- ({:.0f} fps) avg_step_t={:.2f}s'.format(l, l/(time()-t), (book.time-start_time)/l))
                print(f'{isotime} p={price:.3f} bidvol={bidvol:.2f} askvol={askvol:.2f} book={book}\t nmatches={nmatches}')
#            if l%1000==0:
#                print(f'{l} {isotime} seq={book.seq} nmatches={nmatches} elapsed={(book.time-start_time)/86400:.3f} days speedup={(book.time-start_time)/(time()-t):.1f} fps={l/(time()-t):.1f} p={book.price()} book={book} step_duration={(book.time-start_time)/(l+1):.1f}s')
        print(f'END EPISODE: seq={book.seq} nseqs={book.seq-start_seq} ep_duration={(book.time-start_time):.3f}s speedup={(book.time-start_time)/(time()-t):.1f} book={book}')
        bookgen.stop_replay()

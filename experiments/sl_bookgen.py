import simplejson as json
import os
import re
import gzip
import random
from decimal import Decimal
from deeptrade.envs.l2book import L2ShadowOrderBook
from deeptrade.envs.bookgen import SeqgapError

# replay from a random set of files
# each file begins with a snapshot msg
def random_data_iterator(files):
    # choose random file f, play until end of file, move to next file f+1, play until end, ...
    i = random.randint(0, len(files)-1)
    while i<len(files):
        f = files[i]
        #print(f'opening file {f}')
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                msg = json.loads(line, use_decimal=True)
                msg['sequence'] = int(msg.get('sequence') or msg['seq'])
                msg['time'] = float(msg['time'])
                yield i, msg['sequence'], msg
        i+=1

def linear_data_iterator(files):
    for i,f in enumerate(files):
        #print(f'opening file {f}')
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                msg = json.loads(line, use_decimal=True)
                msg['sequence'] = int(msg.get('sequence') or msg['seq'])
                msg['time'] = float(msg['time'])
                currseq=msg['sequence']
                yield i, msg['sequence'], msg


class SLBookGenerator:
    def __init__(self, files, product_id, max_time_gap, max_price_gap, seed=123):
        self.currencies = [product_id]
        self.n_currencies = len(self.currencies)
        self.files = files
        self.need_reset = True
        random.seed(seed)
        self.max_time_gap = max_time_gap
        self.max_price_gap = max_price_gap

    def reset(self):
        self.curr_time = 0
        self.curr_seq = 0
        self.valid = False
        product_id = self.currencies[0]
        self.book = L2ShadowOrderBook(product_id)
        self.data = linear_data_iterator(self.files)  # random_data_iterator(self.files)
        self.curr_msg = None
        self.cur_file = None
        self.need_reset = False

    def replay(self):
        if self.need_reset:
            self.reset()
        data = self.data
        book = self.book
        self.last_cur_file = self.cur_file
        self.curr_time = 0
        self.curr_seq = 0
        self.curr_price = 0
        product_id = self.currencies[0]
        while True:
            try:
                msg = None
                if self.curr_msg:
                    msg = self.curr_msg
                    self.curr_msg = None
                else:
                    file, seq, msg = next(data)
                    self.last_cur_file = self.cur_file
                    self.cur_file = file

                assert msg['sequence']>self.curr_seq
                msg_price = Decimal(msg['bid']+msg['ask'])/2  # msg['price']
                file_changed = self.last_cur_file!=self.cur_file
                # only seqgap possible on file change
                if file_changed and self.curr_time>0:
                    time_gap = msg['time'] - self.curr_time
                    price_gap = msg_price - self.curr_price
                    if time_gap>self.max_time_gap and abs(price_gap)>self.max_price_gap:
                        #print(f'msg_price={msg_price} price_gap={price_gap} time_gap={time_gap}, msg_time={msg["time"]}')
                        raise SeqgapError
                self.curr_seq = msg['sequence']
                self.curr_time = msg['time']
                self.curr_price = msg_price
                yield msg
                self.curr_msg = None # mark this msg as processed
            except SeqgapError as e:
                # underlying generator still okay
                self.need_reset = False
                raise StopIteration
            except StopIteration as e:
                # underlying generator has finished
                self.need_reset = True
                raise e

    def stop_replay(self):
        # if episode spanned > 1 file, reset (to avoid linearly walking all files)
        if self.last_cur_file and (self.cur_file != self.last_cur_file):
            self.need_reset = True

    def close(self):
        pass


def get_slrandombook_gen(path, product_id, mintime, minprice, seed=123, lastn=500, exclude_lastn=None):
    # use files[-lastn:-exclude_lastn]
    #assert minprice==0 # for now
    curr_path = os.path.join(path, 'gdax_book', product_id, 'snapshots', f't_{mintime}_p_{minprice}')
    files = filter(lambda f: f.endswith('.gz'), os.listdir(curr_path))
    files = filter(lambda f: re.match(r'^\d+.gz$', f), files)
    files = sorted(files, key=lambda f: int(f.split('.')[0]))
    files = [os.path.join(curr_path, f) for f in files][-lastn:]
    if exclude_lastn and exclude_lastn>0:
        files = files[:-exclude_lastn]
    print(f'bookgen[{product_id}]: got {len(files)} files from {curr_path}')

    max_time_gap = 4*mintime# if minprice==0 else 100
    max_price_gap = 4*minprice  # if mintime>0 we can have price changes>minprice
    book_gen = SLBookGenerator(files, product_id, max_time_gap, max_price_gap, seed=seed)
    return book_gen


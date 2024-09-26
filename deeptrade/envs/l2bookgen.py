import simplejson as json
import os
import re
import gzip
import random
import numpy as np
from decimal import Decimal
from deeptrade.envs.l2book import L2ShadowOrderBook
from deeptrade.envs.bookgen import SeqgapError
from deeptrade.envs.shadow_book import REALISTIC_FILL, REALISTIC_MATCH
from collections import defaultdict


def syn_data_iterator(curr='BTC-USD'):
    # https://pdfs.semanticscholar.org/1a49/99c918c6206cd9804c48f7dce1bac6ec5b4a.pdf
    # p_t = p_t-1 + b_t-1 + k*eps_t
    # b_t = alpha*b_t-1 + v_t
    # alpha, k constant
    # eps, v ~ normal(0,1)
    # z_t = exp(p_t/R)
    # R = max(p_t)-min(p_t) over sliding window
    n = 10000
    p, beta = np.zeros(n), np.zeros(n)
    alpha = 0.9
    k = 3.0
    for i in range(1, n):
        beta[i] = alpha * beta[i - 1] + np.random.normal(0, 1)
    for i in range(1, n):
        p[i] = p[i - 1] + beta[i - 1] + k * np.random.normal(0, 1)
    p = 10+10*np.exp(p / (max(p) - min(p)))

    time_step = 60
    nlvls = 11
    time, seq = 0, 0
    tradeid = 1
    price = Decimal(p[seq]).quantize(PRICE_QUANTIZE)
    last_price = price
    while seq<n-1:
        time += time_step
        seq += 1
        price = Decimal(p[seq]).quantize(PRICE_QUANTIZE)
        pstep = Decimal('0.1')
        bids = [price-pstep*(i+1) for i in range(nlvls)][::-1]
        bidvols = [1.0 for i in range(nlvls)][::-1]
        asks = [price+pstep*(i+1) for i in range(nlvls)]
        askvols = [1.0 for i in range(nlvls)]
        matches = []
        if seq>1:
            m_size = max(Decimal(0), np.random.normal(0.1,0.1))
            m_side = 'buy' if price<last_price else 'sell'
            if m_side=='buy':
                m_price = last_price-pstep
            else:
                m_price = last_price+pstep
            matches.append({
                'type': 'match',
                'trade_id': tradeid,
                'side': m_side,
                'size': m_size,
                'price': m_price,
                'product_id': curr,
                'sequence': seq, # sequence of the match
                'maker_order_seq': seq, # sequence of the maker order
                'time': time,
            })
            tradeid += 1

        msg = {
            'curr': curr,
            'time': time,
            'sequence': seq,
            'matches': matches,
            'bids': [bids, bidvols],
            'asks': [asks, askvols],
            'bid': [bids[0], bidvols[0]],
            'ask': [asks[0], askvols[0]],
            'price': price,
        }
        last_price = price
        yield 0, int(seq), msg


def file_data_iterator(files, random_order=True):
    n = len(files)
    i = random.randint(0, n-1) if random_order else 0
    while i<len(files):
        f = files[i]
        print(f)
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                msg = json.loads(line, use_decimal=True)
                msg['sequence'] = int(msg.get('sequence') or msg['seq'])
                msg['time'] = float(msg['time'])
                yield i, msg['sequence'], msg
        i+=1
    return

PRICE_QUANTIZE = Decimal('0.01')
QTY_QUANTIZE = Decimal('0.0001')

class L2BookGenerator:
    def __init__(self, files, product_id, bookfn, max_time_gap, max_price_gap, seed=123, book_noise=0.01):
        self.product_id = product_id
        self.files = files
        self.need_reset = True
        random.seed(seed)
        self.max_time_gap = max_time_gap
        self.max_price_gap = max_price_gap
        self.book_noise = book_noise  # add random noise to prices
        self.bookfn = bookfn

    def reset(self):
        self.curr_time = 0
        self.curr_seq = 0
        self.valid = False
        self.book = self.bookfn()
        self.data = file_data_iterator(self.files, random_order=True)
        self.curr_msg = None
        self.cur_file = None
        self.need_reset = False
        self.last_price_noise = Decimal(0)

    def add_noise(self, msg, noise_std=0.01):
        # add noise to msg prices + qtys
        # 1) shift entire book by price delta
        # 2) add multiplicative noise to qty
        price_noise = Decimal(np.random.normal(0, noise_std)).quantize(PRICE_QUANTIZE)
        qty_noise = Decimal(np.random.normal(1, noise_std))
        # updates nonshadow orders from an L2 snapshot msg
        for match in msg['matches']:
            match['size'] = Decimal(match['size']) * Decimal(np.random.normal(1, noise_std))
            match['price'] = Decimal(match['price']) + self.last_price_noise

        msg['bids'][0] = [Decimal(x) + price_noise for x in msg['bids'][0]]
        msg['bids'][1] = [Decimal(x) * Decimal(np.random.normal(1, noise_std)) for x in msg['bids'][1]]
        msg['bid'] = [msg['bids'][0][-1], msg['bids'][1][-1]]

        msg['asks'][0] = [Decimal(x) + price_noise for x in msg['asks'][0]]
        msg['asks'][1] = [Decimal(x) * Decimal(np.random.normal(1, noise_std)) for x in msg['asks'][1]]
        msg['ask'] = [msg['asks'][0][0], msg['asks'][1][0]]

        msg['price'] += price_noise
        msg['time'] += np.random.random()
        self.last_price_noise = price_noise
        return msg

    def replay(self):
        if self.need_reset:
            self.reset()
        data = self.data
        book = self.book
        self.last_cur_file = self.cur_file
        self.curr_time = 0
        self.curr_seq = 0
        self.curr_price = 0
        product_id = self.product_id
        try:
            while True:
                msg = None
                if self.curr_msg:
                    msg = self.curr_msg
                    self.curr_msg = None
                else:
                    file, seq, msg = next(data)
                    self.last_cur_file = self.cur_file
                    self.cur_file = file
                    self.curr_msg = msg

                #if self.book_noise>0:
                #    msg = self.add_noise(msg, noise_std=self.book_noise)

                assert msg['sequence']>=self.curr_seq, 'ERROR: msgseq={} curr_seq={} msg={}'.format(msg['sequence'], self.curr_seq, msg)
                file_changed = self.last_cur_file!=self.cur_file
                time_gap = msg['time'] - self.curr_time
                #msg_price=Decimal(msg['bids'][0][-1] + msg['asks'][0][0])/2
                #price_gap = msg_price-self.curr_price
                if self.curr_time>0 and time_gap>self.max_time_gap:
                    #print('price_gap={} time_gap={} curr_time={} msg_time={}'.format(price_gap, time_gap,self.curr_time, msg['time']))
                    raise SeqgapError
                self.curr_seq = msg['sequence']
                self.curr_time = msg['time']
                #self.curr_price = msg_price
                book.update_from_l2snapshot(msg)
                self.curr_msg = None # mark this msg as processed
                yield book, msg['matches']

        except SeqgapError as e:
            # underlying generator still okay
            self.need_reset = False
        except StopIteration as e:
            # underlying generator has finished
            self.need_reset = True

    def stop_replay(self):
        pass
        #if self.resample:
        #if episode spanned > 1 file, reset (to avoid linearly walking all files)
        #if self.last_cur_file and (self.cur_file != self.last_cur_file):
        #self.need_reset = True

    def close(self):
        pass

def get_l2bookgen(path, product_id, step_type, step_val, fill_type=REALISTIC_FILL, match_type=REALISTIC_MATCH, seed=123, lastn=0, exclude_lastn=None, book_noise=0.0):
    # use files[-lastn:-exclude_lastn]
    #assert minmatches==0 # for now
    curr_path = os.path.join(path, 'gdax_book', product_id, 'snapshots', f'{step_type}_{step_val}')
    assert os.path.exists(curr_path), '{} not found'.format(curr_path)
    files = filter(lambda f: f.endswith('.gz'), os.listdir(curr_path))
    #files = filter(lambda f: re.match(r'^\d+.gz$', f), files)
    files = sorted(files, key=lambda f: int(f.split('.')[0].replace('-','')))
    files = [os.path.join(curr_path, f) for f in files][-lastn:]
    if exclude_lastn and exclude_lastn>0:
        files = files[:-exclude_lastn]
    print(f'bookgen[{product_id}] seed={seed}: got {len(files)} files from {curr_path}')
    assert len(files)>0
    max_time_gap = 3600#4*mintime if minprice==0 else 100
    max_price_gap = 4  # if mintime>0 we can have price changes>minprice
    bookfn = lambda: L2ShadowOrderBook(product_id, fill_type=fill_type, match_type=match_type)
    return L2BookGenerator(files, product_id, bookfn, max_time_gap, max_price_gap, seed=seed, book_noise=book_noise)


if __name__ == '__main__':
    from time import time
    from collections import deque
    from datetime import datetime
    from os import environ
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('curr', default='BTC-USD')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--step_type', type=str)
    parser.add_argument('--step_val', type=float)
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    path = environ['DEEPTRADE_DATA']
    print(f'playback DEEPTRADE_DATA={path} {vars(args)}')
    bookgen = get_l2bookgen(path, args.curr, args.step_type, args.step_val, seed=args.seed)
    #with open('bookgen_out.csv', 'wt') as fh:
    #fh.write('time,bid,bidvol,ask,askvol,nmatches\n')
    from decimal import Decimal
    EPSILON=Decimal('1e-6')
    while True:
        print('===== START EPISODE =====')
        t=time(); start_time=0; start_seq=0; nmatches=0
        stats = deque(maxlen=100)
        for l, (book, matches) in enumerate(bookgen.replay()):
            if start_time==0:
                start_time = book.time
                start_seq = book.seq
            book.sanity_check()
            stats.append(book.stats.copy())
            nmatches+=len(matches)
            isotime=datetime.utcfromtimestamp(book.time).isoformat()
            price = book.price() or 0
            # . The side field indicates the maker order side.
            bidvol = sum([Decimal(m['size']) for m in matches if m['side']=='buy']) # * price
            askvol = sum([Decimal(m['size']) for m in matches if m['side']=='sell']) # * price
            if l>0:
                print('-- {} -- ({:.0f} fps) avg_step_t={:.2f}s'.format(l, l/(time()-t), (book.time-start_time)/l))
                print(f'{isotime} p={price:.3f} bidvol={bidvol:.2f} askvol={askvol:.2f} book={book}\t nmatches={nmatches}')

                imbalance = lambda x,y: (x-y)/(x+y+EPSILON)
                bs_buy, bs_sell = book.stats['buy'], book.stats['sell']
                db = lambda k: stats[-1]['buy'][k]-stats[-2]['buy'][k]
                ds = lambda k: stats[-1]['sell'][k]-stats[-2]['sell'][k]
                
                flow_imbalance_buy = float((db('vl')-db('vc')-db('vm'))/(db('vl')+db('vc')+db('vm')+EPSILON))
                flow_imbalance_sell = float((ds('vl')-ds('vc')-ds('vm'))/(ds('vl')+ds('vc')+ds('vm')+EPSILON))
                vol_imbalance = imbalance(db('vm'), ds('vm'))
                print('flow_imb_buy={:.3f}\t flow_imb_sell={:.3f}\t vol_imb={:.3f} stats={}'.format(flow_imbalance_buy, flow_imbalance_sell, vol_imbalance, book.stats))

            #isotime=datetime.fromtimestamp(book.time).isoformat()
            #bid,bidvol=book.get_bid()
            #ask,askvol=book.get_ask()
            #fh.write(f'{book.time-start_time:.3f},{bid},{bidvol},{ask},{askvol},{nmatches}\n')
            #if l%1000==0 or args.verbose:
            #    print(f'{l} {isotime} seq={book.seq} nmatches={nmatches} elapsed={(book.time-start_time)/86400:.3f} days speedup={(book.time-start_time)/(time()-t):.1f} fps={l/(time()-t):.1f} p={book.price()} book={book} step_duration={(book.time-start_time)/(l+1):.1f}s')
        bookgen.stop_replay()


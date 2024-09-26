
from deeptrade.envs.l2bookgen import get_l2bookgen
from deeptrade.envs.bookgen import get_bookgen
from os import environ
import numpy as np
from decimal import Decimal
from collections import deque
import csv

path = environ.get('DEEPTRADE_DATA', '/Users/atwigg/work/differentiable/deeptrade-data/')
product_id = 'BTC-USD'

LVLRANGE = 5
nlvls = 11
state_w = 2
nstack=10
nahead=1
tau_p=0.001
p_sample=0.005
seed=1

np.random.seed(seed)

def pprint_book(book, k=10):
    print(book)
    bid,bidvol=book.get_bid()
    ask,askvol=book.get_ask()
    print('========== bidvol={} askvol={} time={} seq={} '.format(sum(book.bids.values()), sum(book.asks.values()), book.time, book.seq))
    for p, q in reversed(list(zip(*book.get_asks(k)))):
        print('{:.2f}: {}'.format(p, q))
    print('----- spread={} price={:.2f} last_match={:.2f}'.format(float(ask-bid),float(book.price()), float(book.last_match or 0)))
    for p, q in reversed(list(zip(*book.get_bids(k)))):
        print('{:.2f}: {}'.format(p, q))

#from deeptrade.utils import book2agglvls
def _sumsorted(bins, x, weights, side='right'):
    assert len(x) == len(weights)
    y = np.searchsorted(bins, x, side)  # assume min(x)>= bins[0]
    if side == 'right':
        y = np.maximum(0, y - 1)
    return np.bincount(y, weights=weights, minlength=len(bins))

def aggregate_qtys(lvls, prices, qtys):
    # aggregate prices and qtys into n lvls
    n=len(lvls)
    #print('lvls=',lvls)
    #print('pces=',prices)
    #print('qtys=',qtys)
    X= _sumsorted(lvls, prices, qtys)[:n]
    #print('X   =',X)
    return X

def book2agglvls(book, w=10, n=11):
    # absolute price differences
    wdec = Decimal(w)
    bidp, bidsz = book.get_bids_p(wdec)
    askp, asksz = book.get_asks_p(wdec)
    bidp = np.asarray(bidp).astype(float)[::-1]
    bidsz = np.asarray(bidsz).astype(float)[::-1]
    askp = np.asarray(askp).astype(float)
    asksz = np.asarray(asksz).astype(float)
    bid,ask = bidp[0], askp[0]
    mid = (bid+ask)/2
    lvls = np.linspace(0, w, n)
    #lvls = np.geomspace(1,100,n)/100 - 1
    #print(pprint_book(book))
    agg_bids = aggregate_qtys(lvls, np.maximum(0, bid-bidp), bidsz)
    agg_asks = aggregate_qtys(lvls, np.maximum(0, askp-ask), asksz)
    #agg_bids = np.cumsum(agg_bids)
    #agg_asks = np.cumsum(agg_asks)
    return agg_bids, agg_asks, lvls

def feats(book, matches, last_bid, last_ask):
    bid, bidvol = book.get_bid()
    ask, askvol = book.get_ask()
    mid = float(bid + ask) / 2
    spread = float(ask-bid)
    bid_change = bid-last_bid if last_bid>0 else 0
    ask_change = ask-last_ask if last_ask>0 else 0
#    cndl_o = float(matches[0]['price']) - mid if len(matches) > 0 else 0
#    cndl_h = max([float(o['price']) for o in matches]) - mid if len(matches) > 0 else 0
#    cndl_l = min([float(o['price']) for o in matches]) - mid if len(matches) > 0 else 0
#    cndl_c = float(matches[-1]['price']) - mid if len(matches) > 0 else 0
    cndl_v_bid = sum([float(o['size']) for o in matches if o['side'] == 'buy'])
    cndl_v_ask = sum([float(o['size']) for o in matches if o['side'] == 'sell'])
    feats=[
        # market feats
        bid_change, ask_change, spread,
#        cndl_o, cndl_h, cndl_l, cndl_c,
        cndl_v_bid, cndl_v_ask,
    ]
    return feats

def levelstate(book, matches, last_bid, last_ask):
    k=nlvls
    bid_p, bid_sz = book.get_bids(k)
    ask_p, ask_sz = book.get_asks(k)
    bid_sz = np.asarray(bid_sz).astype(float)[::-1]
    bid_p = np.asarray(bid_p).astype(float)[::-1]
    ask_sz = np.asarray(ask_sz).astype(float)
    ask_p = np.asarray(ask_p).astype(float)
    bid, ask = bid_p[0], ask_p[0]
    S = np.zeros((3, 2*k))
    # level qtys
    S[0, :k] = -ask_sz/10
    S[0, k:] = bid_sz/10
    # level prices
    S[1, :k] = ask_p-ask
    S[1, k:] = bid-bid_p
    f = feats(book, matches, last_bid, last_ask)
    S[2, :len(f)] = f
    return np.reshape(S,-1) #np.clip(np.reshape(S, -1), -10, 10)

def microstate(book, matches, last_bid, last_ask):
    bidlvls, asklvls, lvls = book2agglvls(book, n=nlvls, w=state_w)
    bid, bidvol = book.get_bid()
    ask, askvol = book.get_ask()
    mid = float(bid + ask) / 2
    S = np.zeros((3, nlvls))
    S[0,:] = -np.cumsum(asklvls)[::-1]/100
    S[1,:] = np.cumsum(bidlvls)/100
    f = feats(book, matches, last_bid, last_ask)
    S[2,:len(f)] = f
    return np.clip(np.reshape(S, -1), -10, 10)


bookgen = get_l2bookgen(
    path=path,
    product_id=product_id,
    minmatches=0,
    mintime=10.0,
    minprice=0.0,
    seed=seed,
    lastn=0,
    exclude_lastn=0,
    #booktype='normal',
    skewed_sampling=False
)

w_len = nstack+nahead
states = deque(maxlen=w_len)
bids, asks = deque(maxlen=w_len), deque(maxlen=w_len)
prices = deque(maxlen=w_len)
Y = []
nsteps = int(1e5)
print(f'collecting {nsteps} rollouts path={path}')
step=0
with open('states.data', 'w') as outfile:
    tsv_writer = csv.writer(outfile, delimiter='\t')
    while len(Y)<nsteps:
        for book, matchlist in bookgen.replay():
            price = float(book.price())
            last_bid=bids[-1] if len(bids)>0 else -1
            last_ask=asks[-1] if len(asks)>0 else -1
            state = microstate(book, matchlist, last_bid, last_ask)
            states.append(state)

            # is interesting?
            if len(prices)>=w_len:
                y = (prices[-1]-prices[-nahead-1])/prices[-nahead-1]
                #y = to_sgn_target(y)
                y = 2 if y>tau_p else 0 if y<-tau_p else 1
                if y==0 or y==2:
                    sample=True
                else:
                    sample=np.random.random()<p_sample
                if sample:
                    x=np.asarray(states)[:nstack].copy()
                    tsv_writer.writerow([x,y])
                    #X.append(x)
                    Y.append(y)

            bid, _ = book.get_bid()
            ask, _ = book.get_ask()
            prices.append(price)
            bids.append(bid)
            asks.append(ask)
            if step%10000==0:
                print(f'{len(Y)} / {nsteps}', np.bincount(Y))
                print(book)
            if len(Y)>nsteps:
                break
            step+=1
    
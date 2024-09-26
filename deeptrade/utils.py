import numpy as np
from decimal import Decimal


def value_to_returns(V, additive=True):
    V = np.asarray(V)
    if additive:
        return V[1:]-V[:-1]
    else:
        return V[1:]/V[:-1] - 1

def matched_profit(ma, va, mb, vb):
    m = min(ma, mb)
    pa = va / ma if ma > 0 else 0
    pb = vb / mb if mb > 0 else 0
    return float(m * (pa - pb)) # profit from paired matches

def input_yesno(prompt):
    choices = ['yes', 'no']
    resp=''
    while resp not in choices:
        resp = input(f'{prompt} ({"/".join(choices)}) ').lower().strip()
    return not choices.index(resp)

def merge_iterators(X,Y):
    x=None; y=None
    try:
        x = next(X)
        y = next(Y)
        while True:
            if x<y:
                yield x
                x=next(X)
            else:
                yield y
                y=next(Y)
    except StopIteration:
        for x in X:
            yield x
        for y in Y:
            yield y


def prefix_dict(d, prefix):
    return {f'{prefix}{k}':v for k,v in d.items()}

def mm_opt_profit(matches, min_pdelta=0.01, force_match_all=False):
    bids = sorted([[float(m['price']),float(m['size'])] for m in matches if m['side'] == 'buy'])
    asks = sorted([[float(m['price']),float(m['size'])] for m in matches if m['side'] == 'sell'], reverse=True)
    assert len(bids)+len(asks)==len(matches)
    profit, qty = 0, 0
    # greedy max matching
    # work from outside to inside, matching qtys
    b,a=0,0  # cur bid,ask ix
    while b<len(bids) and a<len(asks):
        ma,mb=asks[a],bids[b]
        q = min(ma[1], mb[1])
        pdelta = ma[0]-mb[0]
        p = q*pdelta
        if force_match_all or (p>0 and pdelta>min_pdelta):
            mb[1] -= q
            ma[1] -= q
            if mb[1] <= 0: b += 1
            if ma[1] <= 0: a += 1
            profit += p
            qty += q
        else:
            break
    #assert profit>=0
    #assert qty>=0
    return profit#, qty

def ensure_decimal(x):
    return x if isinstance(x, Decimal) else Decimal(x)

def assert_decimal(x):
    assert isinstance(x,Decimal)

def _sumsorted(bins, x, weights, side='right'):
    assert len(x) == len(weights)
    y = np.searchsorted(bins, x, side)  # assume min(x)>= bins[0]
    if side == 'right':
        y = np.maximum(0, y - 1)
    return np.bincount(y, weights=weights, minlength=len(bins))

def aggregate_qtys(lvls, prices, qtys):
    # aggregate prices and qtys into n lvls
    n=len(lvls)
    return _sumsorted(lvls, prices, qtys)[:n]

def book2agglvls1(book, w=1, n=11):
    # quantize prices first
    bidp, bidsz = book.get_bids(n)
    askp, asksz = book.get_asks(n)
    bidp = np.round(np.asarray(bidp).astype(float)[::-1], 1)
    bidp # QUANTIZE
    bidsz = np.asarray(bidsz).astype(float)[::-1]
    askp = np.round(np.asarray(askp).astype(float), 1)
    asksz = np.asarray(asksz).astype(float)

    bid,ask = bidp[0], askp[0]
    mid = (bid+ask)/2
    lvls = np.linspace(0, w, n)
    #lvls = np.geomspace(1,100,n)/100 - 1
    agg_bids = aggregate_qtys(lvls, np.maximum(0, mid-bidp), bidsz)
    agg_asks = aggregate_qtys(lvls, np.maximum(0, askp-mid), asksz)
    #agg_bids = np.cumsum(agg_bids)
    #agg_asks = np.cumsum(agg_asks)
    return agg_bids, agg_asks, lvls


def book2agglvls(book, w=1, n=11):
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
    agg_bids = aggregate_qtys(lvls, np.maximum(0, mid-bidp), bidsz)
    agg_asks = aggregate_qtys(lvls, np.maximum(0, askp-mid), asksz)
    #agg_bids = np.cumsum(agg_bids)
    #agg_asks = np.cumsum(agg_asks)
    return agg_bids, agg_asks, lvls

def book2agglvls_rel(book, w=1.001, n=50):
    # relative price differences
    wdec = Decimal(w)*book.price()
    bidp, bidsz = book.get_bids_p(wdec)
    askp, asksz = book.get_asks_p(wdec)
    bidp = np.asarray(bidp).astype(float)[::-1]
    bidsz = np.asarray(bidsz).astype(float)[::-1]
    askp = np.asarray(askp).astype(float)
    asksz = np.asarray(asksz).astype(float)
    bid,ask = bidp[0], askp[0]
    mid = (bid+ask)/2
    assert w>1, w
    lvls = np.linspace(1, w, n)
    agg_bids = aggregate_qtys(lvls, np.maximum(1., mid/bidp), bidsz)
    agg_asks = aggregate_qtys(lvls, np.maximum(1., askp/mid), asksz) # or 2-lvls, mid/askp
    #agg_bids = np.cumsum(agg_bids)
    #agg_asks = np.cumsum(agg_asks)
    return agg_bids, agg_asks, lvls*mid

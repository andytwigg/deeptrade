from sortedcontainers import SortedDict
from decimal import Decimal
from abc import abstractmethod
from collections import defaultdict
from deeptrade.utils import ensure_decimal, assert_decimal
import sys
import simplejson as json

MKT_IMPACT_THRESH = 0.005

class BookError(Exception):
    pass

def pprint_book(book, my_orders=None, k=10, file=None):
    file = file or sys.stdout
    bid,bidvol=book.get_bids(1)
    ask,askvol=book.get_asks(1)
    print(book, file=file)
    orders = defaultdict(list)
    my_orders = my_orders or {}
    for o in my_orders.values():
        # max 1 order/lvl
        if 'id' in o:
            pos = book.position(o.get('id'))
            orders[o['price']].append(pos)

    def _orderstr(norders, lvl_q, o_list):
        s = ['-']*(1+int(lvl_q)) #norders# sf*float(lvl_q))
        if o_list:
            for pos in o_list:
                s[min(len(s) - 1, int(pos))] = '*'
        return ''.join(s)

    for p, q in reversed(list(zip(*book.get_asks(k)))):
        norders = len(book.asks.get(p,[]))
        o_str = '*' if p in orders else ''
        #o_str = _orderstr(norders,q,orders.get(p))
        print('{:.4f}\t {:.2f} {}'.format(q,p,o_str), file=file)
    print('--- spread={:.2f} ---'.format(ask[0]-bid[0]), file=file)
    for p, q in reversed(list(zip(*book.get_bids(k)))):
        norders = len(book.bids.get(p,[]))
        o_str = '*' if p in orders else ''
        #o_str = _orderstr(norders,q,orders.get(p))
        print('{:.4f}\t {:.2f} {}'.format(q,p,o_str), file=file)


class OrderBook:

    def __init__(self, curr):
        self.verbose=False
        self.curr = curr
        self.verbose=0
        self.clear()

    def limit_order(self, side, price, size, **kwargs):
        raise NotImplementedError

    def valid(self):
        return len(self.bids)>0 and len(self.asks)>0

    def clear(self):
        self.last_match = None
        self.open_orders = {}
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.bids_sz = SortedDict()
        self.asks_sz = SortedDict()
        self.stats = {
            'buy': {'vm':0, 'vl':0, 'vc':0, 'nm':0, 'nl':0, 'nc':0}, # vol/num mkt, lim, cncl vols
            'sell': {'vm':0, 'vl':0, 'vc':0, 'nm':0, 'nl':0, 'nc':0}, # vol/num mkt, lim, cncl vols
        }
        self.reset_qty_counters()
        self._reset_book({'time':0, 'bids':[], 'asks':[], 'sequence':0})

    def reset_qty_counters(self):
        self.cancel_qtys = defaultdict(float) # per level
        self.lim_qtys = defaultdict(float) # per level

    def qty_at_price(self, price):
        return self.bids_sz.get(price) or self.asks_sz.get(price) or 0

    def get_bid(self):
        return self.bids_sz.items()[-1]  # self.bids.peekitem(-1)[0]

    def get_ask(self):
        return self.asks_sz.items()[0]  # asks.peekitem(0)[0]

    # bids/asks aggregated by price methods
    # top k bids
    def get_bids(self, k=None):
        k = k or 0
        return self.bids_sz.keys()[-k:], self.bids_sz.values()[-k:]

    # top k asks
    def get_asks(self, k=None):
        k = k or len(self.asks)
        return self.asks_sz.keys()[:k], self.asks_sz.values()[:k]

    # bid-p ... bid
    def get_bids_p(self, p):
        ix = self.bids_sz.bisect_left(self.bids_sz.keys()[-1]-p)
        return self.bids_sz.keys()[ix:], self.bids_sz.values()[ix:]

    # bid ... ask+p
    def get_asks_p(self, p):
        ix = self.asks_sz.bisect(self.asks_sz.keys()[0]+p)
        return self.asks_sz.keys()[:ix], self.asks_sz.values()[:ix]

    # order book methods
    #####
    def market_order(self, side, qty, use_base_currency, dryrun=False, unmatched='ignore'):
        assert isinstance(qty, Decimal)
        v, q, matches = self._matching_orders(side, qty, use_base_currency, unmatched)
        # warn if market impact large
        if matches:
            mid_before = self.price()
            mid_after = matches[-1]['price']
            impact = (mid_after-mid_before)/mid_before
            if abs(impact) > MKT_IMPACT_THRESH:
                if self.verbose>0:
                    print(f'WARNING: market_order {side} {qty} has large impact={impact:.4f}: mid_before={mid_before} mid_after={mid_after}')

        if not dryrun:
            self.remove_matches(matches)
        # FIXME put back in
        #self.stats[side]['vm'] += qty
        #self.stats[side]['nm'] += 1
        return v, q

    def _matching_orders(self, taker_side, qty, use_base_currency, unmatched='ignore'):
        open_orders = self.open_orders
        def orders_iterator(orders):
            for p, o_list in orders:
                for oid in o_list:
                    o = open_orders[oid]
                    yield p, o['seq'], o

        items = reversed(self.bids.items()) if taker_side=='sell' else self.asks.items()
        orders = orders_iterator(items)
        assert qty>0
        qrem = qty # desired quantity
        matches = []
        for price, o_seq, o in orders:
            if qrem<=0: break # exhausted match qty
            size = o['size']
            if use_base_currency:
                raise NotImplementedError('ubc=True needs checking')
                q = min(size, qrem/price)
                qrem -= q*price
            else:
                q = min(size, qrem)
                qrem -= q
            matches.append({'id': o['id'], 'side': o['side'], 'price': price, 'size': q, 'new_size': size - q})

        v, q = 0, 0
        for match in matches:
            v += match['price'] * match['size']
            q += match['size']

        if qrem>0:
            if unmatched=='error':
                raise ValueError(f'ERROR: could not satisfy whole order=(side={taker_side} q={qty} ubc={use_base_currency}); qrem={qrem}')
            elif unmatched=='force':
                if self.verbose>0:
                    print(f'WARNING: forcing match of remaining qty={qrem} at price={price}: order=(side={taker_side} q={qty} ubc={use_base_currency})')
                # price is the worst price matched at
                q += qrem
                v += (price*qrem)
                assert q==qty
            else:
                if self.verbose>0:
                    print(f'WARNING: ignoring unmatched qty: order=(side={taker_side} q={qty} ubc={use_base_currency}); qrem={qrem}')

        return v,q,matches

    def position(self, order_id):
        order = self.open_orders.get(order_id)
        if order:
            orders, _ = self._getorders(order['side'], order['price'])
            ix = orders.index(order_id)
            pos = sum(self.open_orders[o]['size'] for o in orders[:ix])
            return pos
        else:
            return 0

    def price(self):
        if len(self.bids_sz)>0 and len(self.asks_sz)>0:
            return (self.bids_sz.keys()[-1]+self.asks_sz.keys()[0])/2
        else:
            return None

    def bookhash(self):
        bid, bidvol = self.bids_sz.peekitem(-1)
        ask, askvol = self.asks_sz.peekitem(0)
        return hash((bid, bidvol, ask, askvol))

    def snapshot(self):
        return {
            'type': 'snapshot',
            'last_match': self.last_match,
            'time': self.time,
            'sequence': self.seq,
            'bids': [(p,self.open_orders[o]['size'],o) for p,orders in self.bids.items() for o in orders],
            'asks': [(p,self.open_orders[o]['size'],o) for p,orders in self.asks.items() for o in orders],
        }

    def __repr__(self):
        bid, bidvol = self.get_bids(1) if len(self.bids)>0 else ([-1],[0])
        ask, askvol = self.get_asks(1) if len(self.asks)>0 else ([-1],[0])
        return f'Book: t={self.time} seq={self.seq} spread={ask[0]-bid[0]:.2f} bid={bidvol[0]:.4f} @ {bid[0]:.2f} ask={askvol[0]:.4f} @ {ask[0]:.2f}'

    def as_dict(self):
        bid, bidvol = self.get_bids(1) if len(self.bids)>0 else ([-1],[0])
        ask, askvol = self.get_asks(1) if len(self.asks)>0 else ([-1],[0])
        return {'bid': bid[0], 'ask': ask[0], 'bidvol': bidvol[0], 'askvol': askvol[0], 'seq': self.seq, 'time': self.time, 'spread': ask[0]-bid[0]}

    def _check_order(self, order):
        assert order['id'] not in self.open_orders, f'order {order} already in open_orders'
        if order['side']=='buy':
            if len(self.asks)>0:
                fill = self.asks.peekitem(0)[0]
                assert order['price'] < fill, f'ERROR: order {order} price must be < fill={fill}, book={self}'
        else:
            if len(self.bids)>0:
                fill = self.bids.peekitem(-1)[0]
                assert order['price'] > fill, f'ERROR: order {order} price must be > fill={fill}, book={self}'

    def _add(self, order):
        order = {
            'id': order.get('order_id') or order['id'],
            'side': order['side'],
            'price': Decimal(order['price']),
            'size': Decimal(order.get('size') or order['remaining_size']),
            'seq': order.get('seq') or self.seq,
        }
        self._check_order(order)
        orders, orders_sz = self._getorders(order['side'], order['price'])
        orders.append(order['id'])
        orders_sz[order['price']]+=order['size']
        self.open_orders[order['id']]=order
        self.stats[order['side']]['vl'] += order['size']
        self.stats[order['side']]['nl'] += 1

    def _update(self, order_id, new_size):
        if order_id not in self.open_orders:
            return
        new_size = ensure_decimal(new_size)
        assert new_size >= 0, new_size
        order = self.open_orders.get(order_id)
        price, size = order['price'], order['size']
        orders, orders_sz = self._getorders(order['side'], price)
        # update agg sizes
        orders_sz[price]-=(size-new_size)
        if orders_sz[price] <= 0: del orders_sz[price]
        if new_size == 0:
            # FIXME make this O(logn) not O(n) n=#orders at price
            orders.remove(order_id)
            del self.open_orders[order_id]
            self._setorders(order['side'], price, orders)
        else:
            order['size'] = new_size

    def cancel_order(self, order_id):
        if order_id in self.open_orders:
            o = self.open_orders[order_id]
            self.stats[o['side']]['vc'] += o['size']
            self.stats[o['side']]['nc'] += 1
            self._remove(order_id)

    def _remove(self, order_id):
        self._update(order_id, 0)

    def _done(self, msg):
        if msg['reason'] == 'canceled':
            self.cancel_order(msg['order_id'])
        elif msg['reason'] == 'filled':
            self._remove(msg['order_id'])
        else:
            raise ValueError(f'unknown reason: {msg}')

    def _match(self, order):
        """
        The aggressor or taker order is the one executing immediately after being received and the maker
        order is a resting order on the book. The side field indicates the maker order side. If the side is
        sell this indicates the maker was a sell order and the match is considered an up-tick.
        """
        size = Decimal(order['size'])
        price = Decimal(order['price'])
        side = order['side'] # MAKER SIDE
        assert size>0
        self.last_match = price
        maker_order = self.open_orders[order['maker_order_id']]
        self.stats[side]['vm'] += size
        self.stats[side]['nm'] += 1
        # UPDATE the book here rather than waiting for the subsequent done messages
        if side=='buy':
            bids = self.bids.get(price)
            if not bids:
                return
            assert bids[0] == maker_order['id']
            # update/remove maker_order_id
            new_size = maker_order['size'] - size
            self._update(maker_order['id'], new_size)
        else:
            asks = self.asks.get(price)
            if not asks:
                return
            assert asks[0] == maker_order['id']
            # update/remove maker_order_id
            new_size = maker_order['size'] - size
            self._update(maker_order['id'], new_size)

    def _change(self, order):
        oid = order.get('order_id') or order['id']
        self._update(oid, order.get('new_size'))

    def _setorders(self, side, price, orders):
        assert side in ['buy', 'sell']
        if side == 'buy':
            d = self.bids
        elif side == 'sell':
            d = self.asks

        if not orders:
            del d[price]
        else:
            d[price] = orders

    def _getorders(self, side, price):
        assert side in ['buy', 'sell']
        if side == 'buy':
            X, X_sz = self.bids, self.bids_sz
        elif side == 'sell':
            X, X_sz = self.asks, self.asks_sz
        if price not in X_sz:
            X_sz[price]=0
        if price not in X:
            X[price]=[]
        return X[price], X_sz

    ### update methods from level3/snapshot messages
    ########
    def _reset_book(self, msg):
        # resets book from a snapshot msg
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.bids_sz = SortedDict()
        self.asks_sz = SortedDict()
        self.open_orders = {}
        self.last_match = msg.get('last_match')
        self.time = msg['time']
        self.seq = msg['sequence']
        # assert: all price, size are Decimal
        for bid in msg['bids']: # decreasing price [(price, size, id)]
            self._add({
                'id': bid[2],
                'side': 'buy',
                'price': bid[0],
                'size': bid[1],
            })
        for ask in msg['asks']: # increasing price
            self._add({
                'id': ask[2],
                'side': 'sell',
                'price': ask[0],
                'size': ask[1],
            })

    def sanity_check(self):
        assert len(self.asks)>0, f'ERROR: no asks, book={self}'
        assert len(self.bids)>0, f'ERROR: no bids, book={self}'
        if len(self.asks)>0 and len(self.bids)>0:
            b = self.bids.peekitem(-1)[0]
            a = self.asks.peekitem(0)[0]
            assert b<a, f'ERROR: bid={b} >= ask={a}, book={self}'
            assert a>0 and b>0, f'ERROR: negative bid={b} ask={a}, book={self}'
            # check agg bid/ask lvls match unagg
            assert self.bids_sz.peekitem(-1)[0] == b
            assert self.asks_sz.peekitem(-0)[0] == a

    def update(self, msg):
        try:
            if 'sequence' in msg:
                assert msg['sequence']>0
                assert self.seq <= msg['sequence'], f'ERROR: self.seq={self.seq} msg_seq={msg["sequence"]}'
            msg_type = msg['type']
            if msg_type == 'snapshot':
                self._reset_book(msg)
            elif msg_type == 'open':
                self._add(msg)
            elif msg_type == 'done':
                self._done(msg)
            elif msg_type == 'match':
                self._match(msg)
            elif msg_type == 'change':
                self._change(msg)

            self.seq = msg['sequence']
            self.time = msg['time']
        except AssertionError as err:
            print(err)
            raise BookError(f'BookError: error={err} processing msg={json.dumps(msg)[:512]}')

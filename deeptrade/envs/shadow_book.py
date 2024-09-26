from sortedcontainers import SortedDict
from decimal import Decimal
from uuid import uuid1 as uuid  # uuid4
from deeptrade.envs.book import OrderBook
from deeptrade.utils import ensure_decimal
import numpy as np

# FILL TYPES
OPTIMISTIC_FILL='optimistic'
PESSIMISTIC_FILL='pessimistic'
REALISTIC_FILL='realistic'
# MATCH TYPES
OPTIMISTIC_MATCH='optimistic'
REALISTIC_MATCH='realistic'

def other_side(side): return 'buy' if side=='sell' else 'sell'

class ShadowOrderBook(OrderBook):

    def __init__(self, curr, fill_type, match_type):
        super().__init__(curr)
        #print('created shadow order book with fill_type={} match_type={}'.format(fill_type, match_type))
        self.fill_type=fill_type
        self.match_type=match_type
        #assert self.fill_type in [OPTIMISTIC_FILL, REALISTIC_FILL, PESSIMISTIC_FILL]
        assert self.match_type in [OPTIMISTIC_MATCH, REALISTIC_MATCH]

    """
    implements a 'shadow' order book supporting shadow orders placed alongside real orders replayed from history
    idea is to simulate matches of hypothetical orders without disturbing the historical message flow
    """
    def clear(self):
        self.seq=0
        self.time=0
        self.clear_shadow()
        super().clear()

    def clear_shadow(self):
        # remove all shadow orders
        self.shadow_bids = SortedDict()
        self.shadow_asks = SortedDict()
        self.shadow_ids = set()

    def _reset_book(self, msg):
        seqgap=msg['sequence']-self.seq
        timegap=msg['time']-self.time
        nshadow = self.n_shadow_orders()
        if self.verbose:
            print('reset_book: nshadow_orders={} currseq={} msgseq={} seqgap={} booktime={} msgtime={} timegap={}'\
                  .format(nshadow, self.seq, msg['sequence'], seqgap, self.time, msg['time'],timegap))

        #if nshadow==0:
        #    return super()._reset_book(msg)

        # does not clear shadow msgs or open_orders - use clear() for that
        # keep seqs consistent with current state
        # compute sequence of cancel, change, open msgs to go from current book to snapshot
        bids = set(m[2] for m in msg['bids'])
        asks = set(m[2] for m in msg['asks'])
        msgseq, msgtime = msg['sequence'], msg['time']
        to_cancel, to_change, to_open = [], [], []

        # cancel msgs
        existing_oids = set(self.open_orders.keys())
        for oid in existing_oids.difference(bids.union(asks)):
            to_cancel.append({'type': 'done', 'reason': 'canceled', 'order_id': oid, 'sequence': msgseq, 'time': msgtime})

        for bid in msg['bids']:
            price, size, oid = Decimal(bid[0]), Decimal(bid[1]), bid[2]
            existing = self.open_orders.get(oid, None)
            if existing:
                if existing['size'] != size:
                    to_change.append({'type': 'change', 'order_id': oid, 'new_size': size, 'sequence': msgseq, 'time': msgtime})
            else:
                to_open.append({'type': 'open', 'order_id': oid, 'side': 'buy', 'price': price, 'size': size, 'sequence': msgseq, 'time': msgtime})

        for ask in msg['asks']:
            price, size, oid = Decimal(ask[0]), Decimal(ask[1]), ask[2]
            existing = self.open_orders.get(oid, None)
            if existing:
                if existing['size'] != size:
                    to_change.append({'type': 'change', 'order_id': oid, 'new_size': size, 'sequence': msgseq, 'time': msgtime})
            else:
                to_open.append({'type': 'open', 'order_id': oid, 'side': 'sell', 'price': price, 'size': size, 'sequence': msgseq, 'time': msgtime})

        # apply msgs
        #print(f'soft_reset @ {msg["sequence"]} n_shadow={len(self.shadow_ids)} n_cancel={len(to_cancel)} n_change={len(to_change)} n_open={len(to_open)}')
        for m in to_cancel:
            self.update(m)
        for m in to_change:
            self.update(m)
        for m in to_open:
            self.update(m)

        self.last_match = msg.get('last_match')
        self.time = msg['time']
        self.seq = msg['sequence']
        assert isinstance(self.seq, int)

    def n_shadow_orders(self):
        return len(self.shadow_ids)

    def position(self, order_id):
        o = self.open_orders.get(order_id)
        if o is None:
            return 0
        else:
            return o['q_head'] if o['shadow'] else super().position(order_id)

    def limit_order(self, side, price, size, shadow=False, callback=None):
        def _gen_orderid():
            return str(uuid())
        assert size>0
        # shadow: if True, this order will be cancelled once there are no other nonshadow orders at this price level
        # callback: a function to call once the order is cancelled or matched
        order_id = _gen_orderid()
        order = {
            'id': order_id,
            'side': side,
            'price': price,
            'size': size,
            'callback': callback,
            'shadow': shadow,
        }
        self._add(order)
        return order_id

    def remove_matches(self, matches):
        for match in matches:
            oid = match['id']
            self.last_match = match['price']
            callback = self.open_orders[oid].get('callback')
            if callback:
                callback(match)
            self._update(oid, match['new_size'])

    def _add(self, order):
        order = {
            'id': order.get('order_id') or order['id'],
            'side': order['side'],
            'price': Decimal(order['price']),
            'size': Decimal(order.get('size') or order['remaining_size']),
            'shadow': order.get('shadow', False),
            'callback': order.get('callback'),
            'seq': order.get('seq') or self.seq,
            'q_head': self.qty_at_price(order['price']),
        }
        self._check_order(order)
        orders, orders_sz = self._getorders(order['side'], order['price'], order['shadow'])
        orders.append(order['id'])
        if order['shadow']:
            self.shadow_ids.add(order['id'])
        if orders_sz: orders_sz[order['price']]+=order['size']
        self.open_orders[order['id']]=order
        self.stats[order['side']]['vl'] += order['size']
        self.stats[order['side']]['nl'] += 1

    def _getorders(self, side, price, shadow=False, create_new=True):
        if not shadow:
            return super()._getorders(side, price)
        else:
            if side == 'buy':
                d = self.shadow_bids
            elif side == 'sell':
                d = self.shadow_asks
            else:
                raise ValueError(side)
            if price not in d:
                if create_new:
                    d[price]=[]
                else:
                    return [], None
            return d[price], None

    def _setorders(self, side, price, orders, shadow=False):
        if not shadow:
            super()._setorders(side, price, orders)
        else:
            if side == 'buy':
                d = self.shadow_bids
            elif side == 'sell':
                d = self.shadow_asks
            if orders is None or len(orders)==0:
                del d[price]
                # this may push a shadow level into the fill
            else:
                d[price] = orders

    def _remove(self, order_id):
        order = self.open_orders.get(order_id)
        if order:
            self._update(order_id, 0)
            callback = order.get('callback')
            if callback:
                callback({'id': order_id, 'type': 'cancel'}) #, reason=REASON_CANCELLED)

    def _update(self, order_id, new_size):
        if order_id not in self.open_orders:
            return
        new_size = ensure_decimal(new_size)
        assert new_size>=0, new_size
        order = self.open_orders.get(order_id)
        price, size = order['price'], order['size']
        dq = size - new_size

        # update qhead for position for shadows
        # FIXME this is a lot of work for each non-shadow update
        shadows, _ = self._getorders(order['side'], price, shadow=True, create_new=False)
        for oid in shadows:
            # update q_head
            o = self.open_orders[oid]
            if order['seq']<o['seq']:  # TODO <= ?
                o['q_head'] -= dq
                assert o['q_head'] >= 0, f'dq={dq} o={o} order={order}'

        orders, orders_sz = self._getorders(order['side'], price, order['shadow'])
        if orders_sz:
            orders_sz[price] -= dq
            if orders_sz[price] <= 0: del orders_sz[price]
        if new_size == 0:
            # TODO make this O(logn) not O(n) n=#orders at price
            # by using doubly linked list and pointers into list from open_orders
            orders.remove(order_id)
            del self.open_orders[order_id]
            if order['shadow']:
                self.shadow_ids.remove(order_id)
            self._setorders(order['side'], price, orders, order['shadow'])
        else:
            order['size'] = new_size

    def _shadow_match(self, order, maker_order_seq):
        # order['side'] is maker side
        open_orders = self.open_orders
        def orders_iterator(orders):
            for p, o_list in orders:
                for oid in o_list:
                    o = open_orders[oid]
                    yield p, o

        # if no bids/asks then skip
        if len(self.bids_sz)==0 or len(self.asks_sz)==0:
            return
        bid, _ = self.get_bid()
        ask, _ = self.get_ask()
        size = Decimal(order['size'])
        price = Decimal(order['price'])
        side = order['side']
        # match all shadow orders upto maker_order seq and size
        shadowitems = reversed(self.shadow_bids.items()) if side=='buy' else self.shadow_asks.items()
        shadow_matches = []
        qrem=size
        for p, o in orders_iterator(shadowitems):
            assert o['side']==side
            assert o['size']>0
            #if (side=='buy' and p>=ask) or (side=='sell' and p<=bid):
            #    continue # o was a stop limit order; not yet been hit so ignore it
            if qrem <= 0:
                break  # exhausted match msg qty
            if (side=='buy' and p<price) or (side=='sell' and p>price):
                break  # no more shadow orders will be matched by this match
            assert (p>=price if side=='buy' else p<=price)
            # we have a match if p is strictly better than price
            # or if p==price and we are ahead of maker_order in the queue
            # use seq numbers for queue position if p==price
            # fill_type=optimistic: orders are filled when price touches order (orders jump to front of queue)
            # fill_type=pessimistic: orders are filled when price moves through order (orders stay at back of queue)
            # fill_type=realistic: orders are filled in order they are entered
            #assert isinstance(o['seq'], int)
            #assert isinstance(maker_order_seq, int)
            #if self.fill_type==REALISTIC_FILL and p==price:
            #    print(o['seq'], maker_order_seq, o['seq']-maker_order_seq)
            #    lvlorders, _ = self._getorders(side, price)
            #    print('{} orders at {}'.format(len(lvlorders), price))
            #    for oo in lvlorders:
            #        ox = self.open_orders[oo]
            #        print(ox['seq'], ox['size'])
            
            if (p!=price or
                self.fill_type==OPTIMISTIC_FILL or
                (self.fill_type==REALISTIC_FILL and o['seq'] <= maker_order_seq) or
                (self.fill_type=='stochastic' and np.random.random()<0.01)#(qrem/o['initial_queue']))
            ):
                # ---(qty_ahead)--- o --- maker_order ---
                # match_type=optimistic: match the entire order, no partial match
                # match_type=realistic: only match the real qty, allow partial matches
                q = o['size'] if self.match_type==OPTIMISTIC_MATCH else min(qrem, o['size'])
                qrem -= q
                shadow_matches.append({
                    'type': 'match',
                    'id': o['id'],
                    'side': o['side'],
                    'price': o['price'],
                    'size': q,
                    'new_size': o['size']-q,
                    'seq': self.seq,
                    'time': self.time,
                })

        self.remove_matches(shadow_matches)

    def _match(self, order):
        maker_order_seq = self.open_orders[order['maker_order_id']].get('seq')
        #super()._match(order) # match against nonshadow orders incl maker_order
        self._shadow_match(order, maker_order_seq=maker_order_seq)
        super()._match(order)

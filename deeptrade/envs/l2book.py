from sortedcontainers import SortedDict
from decimal import Decimal
from deeptrade.envs.shadow_book import ShadowOrderBook
import numpy.random as random
from collections import defaultdict

REMOVE_ADVERSE_SELECTION=False

class L2ShadowOrderBook(ShadowOrderBook):
    # the only thing we really need to keep track of are the positions of our orders.
    # the book is useful as a state, but only in aggregated form
    # so maybe separate out the logic of order maintenance from book maintenance
    # eg in live trading if there is a problem, we can simply refresh the book and this updates our order position statistics

    def position(self, order_id):
        return 0
        # until updating q_head is implemented
        #o = self.open_orders.get(order_id)
        #return o['q_head'] if o else 0

    def _match(self, order):
        self._shadow_match(order, maker_order_seq=order['maker_order_seq'])

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
        #self._check_order(order)
        orders, orders_sz = self._getorders(order['side'], order['price'], order['shadow'])
        orders.append(order['id'])
        if order['shadow']:
            self.shadow_ids.add(order['id'])
        if orders_sz: orders_sz[order['price']]+=order['size']
        self.open_orders[order['id']]=order

    def update_from_l2snapshot(self, msg):
        # updates nonshadow orders from an L2 snapshot msg
        #match_qtys = defaultdict(int)
        for m in msg['matches']:
            #print('MATCH:',m)
            self._match(m)
            #print(m)
            #match_qtys[Decimal(m['price'])]+=Decimal(m['size'])

        # now update book
        self.bids.clear()#SortedDict()
        self.asks.clear()# = SortedDict()
        self.bids_sz.clear()# = SortedDict()
        self.asks_sz.clear()# = SortedDict()

        self.open_orders = {oid:self.open_orders[oid] for oid in self.shadow_ids}
        id_=0
        # price, new_qty, lim_qty, cancel_qty
        for p,q in zip(*msg['bids']):
            self._add({
                'id': id_,
                'side': 'buy',
                'price': p,
                'size': q,
                'shadow': False,
            })
            id_+=1

        for p,q in zip(*msg['asks']):
            self._add({
                'id': id_,
                'side': 'sell',
                'price': p,
                'size': q,
                'shadow': False,
            })
            id_+=1

        # FIXME annoying to write it here
        # the problem is that shadowbook is being abused by this l2book; we just want to write in the new data, not use the add etc methods
        #self.stats = msg['stats']
        self.time = msg['time']
        self.seq = msg['sequence']
        assert isinstance(msg['sequence'], int)

        # put back any shadow orders
        # handle adverse selection by cancelling any shadow orders that are now invalid
        if REMOVE_ADVERSE_SELECTION:
            bid = self.bids.keys()[-1]
            ask = self.asks.keys()[0]
            to_cancel=[]
            for oid in self.shadow_ids:
                o=self.open_orders[oid]
                if ((o['side']=='buy' and o['price']>=ask) or (o['side']=='sell' and o['price']<=bid)):
                    #print('adversely selected order: {} {} {} {}'.format(oid, o['side'], o['price'], o['size']))
                    #print('book:',self)
                    to_cancel.append(oid)
            for oid in to_cancel:
                self.cancel_order(oid)

        
        

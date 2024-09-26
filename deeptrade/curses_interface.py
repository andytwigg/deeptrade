import curses
from datetime import datetime
import logging
import math
from time import time
from decimal import Decimal
from collections import defaultdict
import numpy as np
from deeptrade.utils import book2agglvls

class CursesDisplay:

    def __init__(self, title, enabled=True):
        self.enabled = enabled
        self.title = title
        if not self.enabled:
            return
        self.logger = logging.getLogger('trader-logger')
        self.stdscr = curses.initscr()
        self.width=100
        self.padsize = 100
        self.pad = curses.newpad(self.padsize, self.width)
        self.timestamp = ""
        self.last_update = 0
        self.portfolio = None
        curses.start_color()
        curses.noecho()
        curses.cbreak()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_RED)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
        self.stdscr.keypad(True)
        self.pad.addstr(1, 0, "waiting...")

    def pause(self):
        if self.enabled:
            self.stdscr.getch()

    def update(self, env):
        info_dict = env.get_info()#detail=True)

        def pprint_book(pad, book, k=10, agg=0.1):
            bid, bidvol = book.get_bid()
            ask, askvol = book.get_ask()
            pad.addstr('{} {} {}\n'.format(book, bid, ask))
            orders = defaultdict(list)
            for o in env.orders.values():
                # max 1 order/lvl
                if 'id' in o:
                    pos = book.position(o.get('id'))
                    orders[o['price']].append(pos)

            def _orderstr(lvl_q, o_list):
                posfn=lambda x:int(math.sqrt(1+x))
                s = ['-']*(1+posfn(lvl_q))
                if o_list:
                    for pos in o_list:
                        s[posfn(pos)]='*'
                return ''.join(s)

            for p, q in reversed(list(zip(*book.get_asks(k)))):
                o_str = _orderstr(q,orders.get(p))
                c = curses.color_pair(2) if p in orders else curses.color_pair(3)
                pad.addstr('{:5.2f}\t {:.2f}  {}\n'.format(q,p,o_str), c)
            pad.addstr('--- spread={:.2f} --- \n'.format(ask-bid))
            for p, q in reversed(list(zip(*book.get_bids(k)))):
                o_str = _orderstr(q,orders.get(p))
                c = curses.color_pair(1) if p in orders else curses.color_pair(3)
                pad.addstr('{:5.2f}\t {:.2f}  {}\n'.format(q,p,o_str), c)

        def update_books():
            self.pad.addstr('--- BOOKS ---\n')
            #for book in env.books.values():
            pprint_book(self.pad, env.book, agg=False)

        def update_orders():
            book = env.book
            curr_time=book.time
            self.pad.addstr('--- {} OPEN ORDERS ---\n'.format(len(env.orders)))
            p, ticksize = book.price(), Decimal('0.01')

            def order_feats(o):
                oid = o.get('id')
                ahead = float(book.position(oid))
                #behind = float(book.qty_at_price(o['price'])) - ahead
                #initial_ahead = float(env.orders[oid]['initial_queue'])
                #progress = ahead / initial_ahead if initial_ahead > 0 else 0
                #position = ahead / (ahead + behind) if ahead > 0 else 0
                #dist = o['price'] - p
                age=max(0, env.ep_len-o['created_at']) #curr_time-o.get('created_at')
                return float(o['size']), ahead, age

            for o in sorted(env.orders.values(), key=lambda o:abs(p-o['price']))[:10]:
                size, ahead, age = order_feats(o)
                self.pad.addstr("{} {:.4f} @ {:.3f} age={:d} ahead={:.2f} id={} time:{}\n"\
                                .format(o['side'].upper(), size, float(o.get('price')), age, ahead, o.get('id'), o.get('created_at')))

        def update_fills():
            avg_fill_time = np.mean([f['time_since_created'] for f in env.fills])
            self.pad.addstr("--- RECENT FILLS ({} total) avg_fill_time={:.2f} ---\n".format(len(env.fills), avg_fill_time))
            for fill in env.fills[-5:]:
                self.pad.addstr("{}\t {}\t {:.4f}@{:.3f}\t seq={}\n"\
                                .format(fill.get('type'), fill.get('side'), float(fill.get('size')), float(fill.get('price')), fill.get('seq')))

        def update_vols():
            # matches not just against our orders
            bidvol=sum(env.bidmatch_vol[-10:])
            askvol=sum(env.askmatch_vol[-10:])
            vol_imb=(bidvol-askvol)/(bidvol+askvol) if (bidvol+askvol)>0 else 0
            self.pad.addstr("--- VOLUMES (last 10 steps)---\n")
            self.pad.addstr("bidvol={:.3f} askvol={:.3f} flow_imbalance={:.3f}\n".format(bidvol,askvol,vol_imb))

        def update_accounts():
            product_id = env.product_id
            if auth_client:
                if not self.portfolio or (time() - self.last_update)>1:
                    accounts = auth_client.get_accounts()
                    self.portfolio = {acc['currency']: float(acc['balance']) for acc in accounts}
                    self.last_update = time()
                self.pad.addstr("[auth] account: {}={:.5f} USD={:.5f}".format(product_id, self.portfolio[product_id], self.portfolio['USD']))
            self.pad.addstr("[env] account: {}={:.5f} USD={:.5f}\n".format(product_id, env.portfolio[0], env.portfolio[-1]))

        def update_info_dict():
            for k, v in info_dict.items():
                if isinstance(v, float):
                    self.pad.addstr('{}:\t {:.4f}\n'.format(k, v), self.color(v > 0))
                else:
                    self.pad.addstr('{}:\t {}\n'.format(k, v))

        if not self.enabled:
            return
        auth_client = env.client or None
        self.pad.erase()
        #self.pad.resize(self.padsize, self.width)
        # write to the pad
        book = env.book
        dt = datetime.utcfromtimestamp(book.time)
        self.pad.addstr(0, 0, f'{self.title}\t time={dt.isoformat()}\thour={dt.hour} day={dt.weekday()} \n')

        update_accounts()
        update_info_dict()
        update_books()
        update_vols()
        update_fills()
        update_orders()

        h, w = self.stdscr.getmaxyx()
        self.pad.refresh(0,0,0,0,h-1,w-1)

    def color(self, expr):
        if expr:
            return curses.color_pair(1)
        else:
            return curses.color_pair(2)

    def close(self):
        if not self.enabled:
            return
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()

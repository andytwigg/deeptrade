import numpy as np
import sys
import warnings
import random
import gym
from gym import spaces
from decimal import Decimal
from pprint import pprint
from collections import defaultdict, deque

from deeptrade.envs.book import pprint_book
from deeptrade.envs.states import get_statefn
from deeptrade.utils import ensure_decimal

REASON_MATCHED = 'matched'
REASON_CANCELLED = 'cancelled'

PRICE_QUANTIZE = Decimal('0.01')
QTY_QUANTIZE = Decimal('0.00001')
EPSILON = Decimal('1e-6')
EPSILON_FLOAT = 1e-6

SIDE_BUY = 'buy'
SIDE_SELL = 'sell'

class LimitEnv(gym.Env):
    def __init__(self, data,
                 episode_len=100,
                 fee=0,
                 start_fee=0,
                 start_fn=1,
                 done_fn=1,
                 reward_fn='pnl',
                 state_fn='macro',
                 max_inventory=1,
                 end_on_target=True,
                 burnin=2,
                 rew_eta=0.8,
                 ob_scale=1.0,
                 ob_noise=0,
                 allow_negative=True,
                 report_detail=False,
                 verbose=False,
                 **kwargs):
        self.verbose = verbose
        self.product_id = data.product_id
        self.episode_len = episode_len
        self.data = data
        self.anneal_config = {}
        self.fee = Decimal(str(fee))
        if start_fee!=-1:
            self.anneal_config['fee'] = {'start': start_fee, 'end': fee, 'schedule': 'linear', 'type': Decimal}
        self.anneal_params(1.0)
        self.done_fn = done_fn
        self.reward_fn = reward_fn
        self.start_fn = start_fn
        self.state_class = get_statefn(state_fn, env=self, scale=ob_scale, noise=ob_noise, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.state_class.shape(), dtype=np.float32)
        self.action_space = None
        self.allow_negative = allow_negative
        self.end_on_target = end_on_target
        self.MAX_INVENTORY = max_inventory
        self.burnin = burnin
        self.rew_eta = rew_eta
        self.report_detail = report_detail # whether to report info with each call to step
        self.portfolio = None
        self.time = -1
        self._seed = 0
        self.client = None
        self.book = None
        self.orders = {}

    def anneal_params(self, frac):
        # frac = frac of training remaining: 1->0
        for p, c in self.anneal_config.items():
            old_val = getattr(self, p)
            sch = c.get('schedule')
            if sch=='linear':
                new_val = c['start'] + (1-frac)*(c['end']-c['start'])
            elif sch=='quadratic':
                new_val = c['start'] + (1-frac**2)*(c['end']-c['start'])
            else:
                raise ValueError(sch)
            if 'type' in c:
                new_val = c['type'](new_val)
            setattr(self, p, new_val)
            if self.verbose:
                print(f'annealing {p} with frac={frac:.4f}: {old_val:.5f} -> {new_val:.5f}')
        # any sanity checks
        assert self.fee>=0

    def close(self):
        self.data.close()

    def render(self, mode='human', file=None):
        file = file or sys.stdout
        info = self.get_info()
        book = self.book
        if mode=='ansi' or mode=='human':
            print(f'#### LimitEnv step={self.ep_len} time={book.time} seq={book.seq} ###', file=file)
            pprint(info, stream=file)
            pprint_book(book, self.orders, file=file)
            print('orders:', file=file)
            for o in sorted(self.orders.values(), key=lambda o:o['price']):
                print(o, file=file)
        else:
            raise NotImplementedError

    def get_info(self, detail=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = self.book
            bid, bidvol = book.get_bid()
            ask, askvol = book.get_ask()
            spread = float(ask-bid)
            V = self.value_history
            price_hist = np.asarray(self.mid_history)
            realized_pnls = np.asarray(self.realized_pnls)
            realized_deltap = np.asarray(self.realized_deltap)
            fill_rate = 0 if self.stats['num_limit_orders_placed']==0 else self.stats['num_limit_orders_filled']/self.stats['num_limit_orders_placed']

            inv = float(self.portfolio[0])
            info = {
                'time': book.time,
                'seq': book.seq,
                'inv': inv,
                'map': np.mean(np.abs(self.stat_history['inv'])),
                'usd': float(self.portfolio[1]),
                'apv': V[-1]-V[0],
                'avg_fill_time': np.nanmean([f['time_since_created'] for f in self.fills]),
                'avg_fill_rate': fill_rate,
                'n_realized': len(realized_pnls),
                'n_unrealized': len(self.unrealized_asks)+len(self.unrealized_bids),
                'deltap_rpnl': np.nanmean(realized_deltap),
                'mean_rpnl': np.nanmean(realized_pnls),
                'winrate_rpnl': np.nanmean(realized_pnls>0),
                'fee': float(self.fee),
                'fees_paid': float(self.total_fees),
                'ep_len': self.ep_len,
                'lim_buy_pending': sum([float(o['size']) for o in self.orders.values() if o['side']==SIDE_BUY]),
                'lim_sell_pending': sum([float(o['size']) for o in self.orders.values() if o['side']==SIDE_SELL]),
                'ep_duration': book.time-self.start_time,
                'price': price_hist[-1],
                'price_delta': price_hist[-1]-price_hist[0],
                'spread': spread,
                'book_imbalance': float((bidvol-askvol)/(bidvol+askvol+EPSILON)),
            }
            info.update(self.stats)
            #info.update(prefix_dict(book.as_dict(), 'book_'))

        return info

    def _clear_stats(self):
        self.time = 0
        self.stats = {
            'time': 0,
            'num_limit_orders_placed': 0,
            'num_limit_orders_filled': 0,
            'lim_buy_qty': 0,
            'lim_sell_qty': 0,
            'mkt_buy_qty': 0,
            'mkt_sell_qty': 0,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'deltap_rpnl': 0,
            'avg_up': 0,
        }
        self.stats_prev = self.stats.copy()
        #self.book_stats = deque(maxlen=10)
        self.total_fees = 0
        self.mid_history = []
        self.bid_history = []
        self.ask_history =[]
        self.bidvol_history = []
        self.askvol_history =[]
        self.mpm_history = []
        self.value_history = []
        self.bidmatch_vol, self.askmatch_vol = [], []
        self.potential_history = [0.0]
        self.last_matched_profit = 0
        self.stat_history = defaultdict(list)
        self.book = None
        self.matches = None
        self.mm_opt_profit = 0

    def reset(self):
        self.done = False
        self.orders = {}
        self.fills = []
        self.unrealized_bids, self.unrealized_asks, = deque(), deque()
        self.realized_pnls, self.realized_deltap = [], []
        self.realized_pnl = Decimal(0)
        self.ep_len = 0

        if self.book:
            self.book.clear_shadow()

        self.state_class.reset()

        # start state
        start_usd=100000
        #if self.allow_negative:
        #    start_0=Decimal(np.random.randint(-100, 100))/Decimal(100)*self.MAX_INVENTORY
        #    self.portfolio = [Decimal(start_0), Decimal(start_usd)]            
        #else:
        #    start_0=Decimal(np.random.randint(0, 100))/Decimal(100)*self.MAX_INVENTORY
        #    self.portfolio = [Decimal(start_0), Decimal(start_usd)]
        self.portfolio = [Decimal(0), Decimal(start_usd)]

        # burnin
        success = False
        while not success:
            try:
                self._clear_stats()
                self.data_gen = self.data.replay()
                for _ in range(2+np.random.randint(self.burnin)):
                    self._step_books()
                success = True
            except StopIteration:
                self.data.stop_replay()

        self.start_time = self.time
        self.stats['time'] = self.time
        assert self.book.n_shadow_orders() == 0, self.book.n_shadow_orders()
        # call any user-defined reset code
        self._reset()
        return self._state()

    def _reset(self):
        pass

    def _step_books(self):
        # any limit order fills are handled by callbacks during this step
        self.book, self.matches = next(self.data_gen)
        book = self.book
        matches = self.matches
        book.sanity_check()
 
        bid, bidvol = book.get_bid()
        ask, askvol = book.get_ask()
        bid, ask, bidvol, askvol = float(bid), float(ask), float(bidvol), float(askvol)
        mid = (bid + ask) / 2

        self.time = book.time
        self.value_history.append(self.get_value())
        self.mpm_history.append(mid - self.mid_history[-1] if len(self.mid_history) > 0 else 0.0)
        self.mid_history.append(mid)
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        self.bidvol_history.append(bidvol)
        self.askvol_history.append(askvol)
        #self.book_stats.append(book.stats.copy())

        self.bidmatch_vol.append(sum([float(o['size']) for o in matches if o['side'] == SIDE_BUY]))
        self.askmatch_vol.append(sum([float(o['size']) for o in matches if o['side'] == SIDE_SELL]))

    def end_episode(self):
        self.cancel_all()
        if self.end_on_target:
            self.clear_inventory()
        self.data.stop_replay()

    def clear_inventory(self):
        # WARNIGN: this may fail if we have active limit orders
        # execute market orders
        dq = self.portfolio[0]
        if dq>0:
            q=self._market_order(SIDE_SELL, dq)
            assert q==dq, 'sell error q={} dq={}'.format(q,dq)
        elif dq<0:
            q=self._market_order(SIDE_BUY, -dq)
            assert q==-dq, 'buy error q={} dq={}'.format(q,-dq)

    def _market_order(self, side, qty):
        # fees applied to cash in both buy and sell
        qty = ensure_decimal(qty)#.quantize(QTY_QUANTIZE)
        if qty<=0:
            return 0.0
        book = self.book
        inventory = self.portfolio[0]
        v = 0
        q = 0
        if side == SIDE_BUY and inventory<self.MAX_INVENTORY:
            # buy qty of curr (NB ubc=False)
            #if not self.allow_negative:
            #    qty = min(qty, self.portfolio[-1]/float(book.price()))  # estimate of cost
            v, q = book.market_order(SIDE_BUY, qty, False, dryrun=True, unmatched='force')
            assert q==qty
            self.portfolio[0] += q  #*(1-self.fee)
            self.portfolio[-1] -= v*(1+self.fee) # /(1-self.fee)  # apply fee to cash to keep order sizes same
            self.stats['mkt_buy_qty'] += float(q)
        elif side == SIDE_SELL and inventory>-self.MAX_INVENTORY:
            # sell qty of curr
            if not self.allow_negative:
                qty = min(qty, self.portfolio[0])
                if qty<=0:
                    return 0.0
            v, q = book.market_order(SIDE_SELL, qty, False, dryrun=True, unmatched='force')
            assert q==qty
            self.portfolio[0] -= q
            self.portfolio[-1] += v*(1-self.fee)
            self.stats['mkt_sell_qty'] += float(q)
        # record the fill
        if q>0:
            # avg price for order = v/q
            fee = v*self.fee
            fill = {
                'side': side,
                'size': q,
                'price': v/q,
                'seq': book.seq,
                'time': book.time,
                'type': 'market',
                'fee': fee,
                'time_since_created': 0,
            }
            self._process_fill(fill)
        return q

    def _limit_order(self, side, price, qty, stop=None, stop_price=None):
        if self.verbose:
            print(f'limit order: {side} {qty}@{price}')
        qty = ensure_decimal(qty)#.quantize(QTY_QUANTIZE)  # turned off qty quantization to help develop envs
        price = ensure_decimal(price).quantize(PRICE_QUANTIZE)
        inventory = self.portfolio[0]
        if side==SIDE_SELL and not self.allow_negative:
            qty = min(qty, inventory) # TODO make work when we can have multiple orders open per side
        if qty<=0:
            return
        #if (side==SIDE_BUY and price < self.book.get_ask()[0] and inventory<self.MAX_INVENTORY) or \
        #   (side==SIDE_SELL and price > self.book.get_bid()[0] and inventory>-self.MAX_INVENTORY):
        if (side==SIDE_BUY and inventory<self.MAX_INVENTORY) or \
           (side==SIDE_SELL and inventory>-self.MAX_INVENTORY):
            oid = self.book.limit_order(side, price, qty, shadow=True, callback=self._callback)
        else:
            if abs(inventory)!=self.MAX_INVENTORY:# elif self.verbose:
                print(f'WARNING: invalid order: {side} {qty}@{price} curr_inv={inventory} book={self.book}')
            return None
        
        assert oid is not None
        # order successfully placed
        self.orders[oid] = {
            'type': 'limit',
            'id': oid,
            'cur': 0,
            'side': side,
            'price': price,
            'size': qty,
            'initial_queue': self.book.qty_at_price(price),
            'seq': self.book.seq,
            'created_at': self.ep_len #book.time,
        }
        self.stats['num_limit_orders_placed'] += 1
        #print('placed order: ', self.orders[oid])
        return oid

    def _on_fill(self, fill):
        pass

    def _process_fill(self, fill):
        if self.verbose:
           print('FILL: {} {:.4f} @ {:.3f}'.format(fill['side'], fill['size'], fill['price']))
        self.fills.append(fill)
        # update rp/up
        unrealized_bids, unrealized_asks = self.unrealized_bids, self.unrealized_asks
        q, p, fee, side = fill['size'], fill['price'], fill.get('fee', 0), fill['side']
        # any fee is realized immediately
        self.total_fees += fee
        # side == maker side, so side==buy => matching against unrealized asks
        unrealized = unrealized_asks if side==SIDE_BUY else unrealized_bids
        delta_rpnl = 0
        while len(unrealized) > 0 and q > 0:
            qu, pu = unrealized[0]
            dq = min(q, qu)
            q -= dq
            if dq >= qu:
                unrealized.popleft()  # remove elemnt 0
            else:
                unrealized[0][0] = qu-dq
            # we have realized dq
            delta_p = pu-p if side==SIDE_BUY else p-pu
            delta_rpnl += dq*delta_p
            self.realized_deltap.append(float(delta_p))

        self.realized_pnl += delta_rpnl
        self.realized_pnls.append(float(delta_rpnl))

        # any remainder is unrealized on the other side
        if q > 0:
            assert len(unrealized) == 0
            unrealized_other = unrealized_asks if side==SIDE_SELL else unrealized_bids
            unrealized_other.append([q, p])

        # user callback
        self._on_fill(fill)

    def _pnl(self):
        # returns realized, unrealized pnl
        book = self.book
        mid = book.price()
        upnla = sum([(x[1]-mid)*x[0] for x in self.unrealized_asks])
        upnlb = sum([(x[1]-mid)*x[0] for x in self.unrealized_bids])
        unrealized_pnl = upnla - upnlb
        realized_pnl = float(self.realized_pnl - self.total_fees)
        # average price-mid of unrealized fills
        avg_up = 0
        if len(self.unrealized_bids)>0 or len(self.unrealized_asks)>0:
            avg_up = (upnla + upnlb)/sum([x[0] for x in self.unrealized_asks+self.unrealized_bids])
        return realized_pnl, unrealized_pnl, avg_up

    def _callback(self, msg):
        order_id, type = msg['id'], msg['type']
        order = self.orders[order_id]
        c = order['cur']
        assert c==0
        if type=='match' and order['type']=='limit':
            new_size = msg['new_size']
            matched_size = msg['size']# order['size'] - new_size
            assert 0 < matched_size <= order['size'], 'ERROR: matched_size={} order_size={}'.format(matched_size, order['size'])
            v = msg['price'] * matched_size
            if order['side'] == SIDE_BUY: #maker side
                self.portfolio[c] += matched_size
                self.portfolio[-1] -= v
                self.stats['lim_buy_qty'] += float(matched_size)
                self.stats['num_limit_orders_filled'] += 1
            else:
                self.portfolio[c] -= matched_size
                self.portfolio[-1] += v
                self.stats['lim_sell_qty'] += float(matched_size)
                self.stats['num_limit_orders_filled'] += 1

            # record the fill
            fill = {
                'cur': c,
                'id': order_id,
                'side': order['side'],
                'size': matched_size,
                'price': msg['price'],
                'seq': msg['seq'],
                'time': msg['time'],
                'type': 'limit',
                'time_since_created': self.ep_len-order['created_at'],
                #'fee': self.fee_limit,
            }
            self._process_fill(fill)

            # update our open order
            order['size'] = new_size
            #print(f'callback: oid={order_id}, new_size={new_size}')
            if new_size <= 0:
                del self.orders[order_id]
            assert order_id not in self.orders
        elif type == 'match' and order['type'] == 'stop':
            self._market_order(msg['side'], msg['qty'])
        elif type == 'cancel':
            del self.orders[order_id]
        else:
            raise ValueError(f'limitenv._callback: unknown callback msg type: {msg}')

    def _state(self):
        return self.state_class.state()

    def _act(self, action):
        raise NotImplementedError

    def _done(self):
        if self.done_fn==1:
            done = self.ep_len > self.episode_len
        elif self.done_fn==2:
            ep_duration = self.time - self.start_time
            done = ep_duration > self.episode_len
        return done

    def _reward(self):
        price_hist = self.mid_history
        inv =  float(self.portfolio[0])
        delta_mid = price_hist[-1] - price_hist[-2] if len(price_hist)>1 else 0
        V = self.value_history

        def rew_log_pnl():
            return np.log(V[-1]) - np.log(V[-2]) # * t/(t+1))

        def rew_pnl():
            return V[-1]-V[-2] # * t/(t+1))

        def rew_realized_pnl():
            # only reward realized pnl changes; sparser than rew_pnl
            delta_pnl = self.stats['realized_pnl'] - self.stats_prev['realized_pnl']
            return delta_pnl - self.rew_eta*inv**2

        def rew_pnl_damped():
            # https://github.com/tspooner/rl_markets/blob/master/src/environment/base.cpp#L180
            # eta should be ~ 0.5-0.8
            pnl_step = V[-1]-V[-2]
            theta = delta_mid*inv
            reward = pnl_step - self.rew_eta * max(0, theta)
            return reward

        def rew_pnl_pos():
            # punish holding a position
            # eta should be ~ 0.01-0.1
            return (V[-1]-V[-2]) - self.rew_eta * inv**2

        def rew_potential():
            # potential-based reward shaping doesn't alter optimal strategy
            # eta should be ~ 0.95-1.0
            potential = -inv**2# + 0.1*self.stats['unrealized_pnl']
            prev_pot = self.potential_history[-1]
            rew = (V[-1]-V[-2]) - prev_pot + self.rew_eta * potential
            self.potential_history.append(potential)
            return rew

        def rew_somekey(k):
            return self.stats[k] - self.stats_prev[k]

        if self.reward_fn=='pnl':
            reward = rew_pnl()
        elif self.reward_fn=='rpnl':
            reward = rew_realized_pnl()
        elif self.reward_fn=='log_pnl':
            reward = rew_log_pnl()
        elif self.reward_fn=='pnl_damped':
            reward = rew_pnl_damped()
        elif self.reward_fn=='pnl_pos':
            reward = rew_pnl_pos()
        elif self.reward_fn=='potential':
            reward = rew_potential()
        elif self.reward_fn.startswith('_'):
            reward = rew_somekey(self.reward_fn[1:])
        else:
            raise ValueError(self.reward_fn)

        if self.verbose: print(f'{self.reward_fn} rew={reward:.5f}')
        if np.isnan(reward) or not np.isfinite(reward):
            reward = 0.0

        # bonus to encourage trading, to avoid getting stuck in local optima of noop
        #bonus = self.stats['lim_buy_qty'] + self.stats['lim_sell_qty'] - self.stats_prev['lim_buy_qty'] - self.stats_prev['lim_sell_qty']
        #reward += 0.01*bonus

        return reward

    def step(self, action):
        if self.verbose:
            print(f'STEP {self.ep_len}\t ACTION {action}')
        assert not self.done

        # execute action
        if action is not None:
            self._act(action)

        # advance the book generator
        # will result in callbacks on match etc
        try:
            self._step_books()
            self.ep_len += 1
            self.done |= self._done()
        except StopIteration:
            #if self.verbose:
            print('[ENV] StopIteration ep_len={} book={}'.format(self.ep_len, self.book))
            self.done = True

        if self.done:
            self.end_episode()

        # update stats
        # TODO combine stat_history/stats to tidy up
        self.stat_history['inv'].append(float(self.portfolio[0]))
        self.stat_history['lim_buy_pending'].append(sum([float(o['size']) for o in self.orders.values() if o['side']==SIDE_BUY]))
        self.stat_history['lim_sell_pending'].append(sum([float(o['size']) for o in self.orders.values() if o['side']==SIDE_SELL]))
        for k in self.stats:
            self.stat_history[k].append(self.stats[k])

        rpnl, upnl, avg_up = self._pnl()
        self.stats['realized_pnl']=float(rpnl)
        self.stats['unrealized_pnl']=float(upnl)
        self.stats['avg_up']=float(avg_up)
        self.stats['deltap_rpnl']=np.nanmean(self.realized_deltap) if self.realized_deltap else 0.0
        self.stats['time']=self.time

        reward = self._reward()
        #mbidvol, maskvol=self.bidmatch_vol[-1], self.askmatch_vol[-1]
        info = {
            'step': self.ep_len,
            'time': self.book.time,
            'seq': self.book.seq,
            'inv': float(self.portfolio[0]),
            'usd': float(self.portfolio[1]),
            'price': float(self.mid_history[-1]),
            'bid': float(self.bid_history[-1]),
            'ask': float(self.ask_history[-1]),
            #'matchvol': self.bidmatch_vol[-1]-self.askmatch_vol[-1],
            #'matchvolimb': (mbidvol-maskvol)/(mbidvol+maskvol+1e-6),
        }
        if self.done or self.report_detail:
            info.update(self.get_info())

        # aux targets
        # for now, all auxiliary tasks are assumed to be 3-way multiclass and optimized with xent loss
        sgn_target = lambda x: 1+int(np.sign(x))
        price_hist = self.mid_history
        # aux tasks
        spread = (self.ask_history[-1]-self.bid_history[-1])
        spread_last = (self.ask_history[-2]-self.bid_history[-2])
        aux_targets = {
            'spread': sgn_target(spread-spread_last),
            'ask': sgn_target(self.ask_history[-1]-self.ask_history[-2]),
            'bid': sgn_target(self.bid_history[-1]-self.bid_history[-2]),
            'price': sgn_target(price_hist[-1]-price_hist[-2]),
        }
        #for k in ['lim_buy_qty','lim_sell_qty','realized_pnl','unrealized_pnl','inv']:
        #    #print(k,self.stat_history[k])
        #    if len(self.stat_history[k])<2:
        #        aux_targets[k]=1 # sign -1,0,1 => 0,1,2
        #    else:
        #        aux_targets[k] = sgn_target(self.stat_history[k][-1]-self.stat_history[k][-2])
        info['aux'] = aux_targets
        state = self._state()
        self.stats_prev = self.stats.copy()
        return state, reward, self.done, info

    def get_value(self):
        #return self.portfolio @ self._prices() #
        return float(self.portfolio[0]*self.book.price() + self.portfolio[1])

    def check_orders_age(self):
        self.cancel_by_fn(lambda o: (self.book.time-o.get('created_at', 0))>o.get('max_age',-1) and o.get('max_age',-1)>0)

    def cancel_orders(self, order_id_list):
        for order_id in order_id_list:
            self.cancel_order(order_id)

    def cancel_order(self, order_id):
        if order_id in self.orders:
            if self.verbose:
                print('canceling {}'.format(self.orders[order_id]))
            #cur = self.orders[order_id]['cur']
            book = self.book
            book.cancel_order(order_id) # should result in a callback

    def cancel_all(self):
        orders = list(self.orders.keys())
        for order_id in orders:
            self.cancel_order(order_id)
        assert len(self.orders)==0
        #for book in self.books.values():
        assert self.book.n_shadow_orders()==0

    def cancel_by_fn(self, cancel_fn):
        orders_to_trim = list(filter(cancel_fn, self.orders.values()))
        for o in orders_to_trim:
            self.cancel_order(o['id'])
        return orders_to_trim

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

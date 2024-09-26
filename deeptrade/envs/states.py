#######
# Different state representations for limitenv
#######

from abc import abstractmethod
import numpy as np
from decimal import Decimal
from deeptrade.utils import book2agglvls, book2agglvls_rel
from datetime import datetime

N_STACK=100
N_STATES=22
N_INDICATOR_FEATS=11

EPSILON_FLOAT = 1e-6
EPSILON = Decimal('1e-6')

def get_statefn(state_fn, env, **kwargs):
    if state_fn=='candle':
        return CandleState(env, **kwargs)
    elif state_fn == 'microstack':
        return MicroStackedState(env, **kwargs)
    elif state_fn == 'tiny':
        return TinyState(env, **kwargs)
    elif state_fn == 'level':
        return LevelState(env, **kwargs)
    elif state_fn == 'micro':
        return MicroState(env, **kwargs)
    elif state_fn == 'microcum':
        return MicroCumState(env, **kwargs)
    elif state_fn == 'price2d':
        return Price2DState(env, **kwargs)
    else:
        raise ValueError('unknown state fn={}'.format(state_fn))

# base state class, extend it to add other state components
class State:
    def __init__(self, env, scale=False, noise=0, **kwargs):
        self.scale=scale
        self.env=env
        self.noise=noise

    # called at end of episode to clear any history
    def reset(self):
        pass

    def shape(self):
        return (N_STATES,)

    def state(self):
        env = self.env
        # market features
        bidhist, bidvolhist = env.bid_history, env.bidvol_history
        askhist, askvolhist = env.ask_history, env.askvol_history
        ask, ask_1 = askhist[-1], askhist[-2]
        bid, bid_1 = bidhist[-1], bidhist[-2]
        mid = (ask+bid)/2.
        mid_1 = (ask_1+bid_1)/2.

        # encode a single order
        def order_feats(o):
            age = max(0, env.ep_len-o['created_at'])
            rp = mid
            # rp = bid if side==sell else ask
            return 1, rp-float(o['price']), np.log(age)
#            return float(o['size']/env.qty), rp-float(o['price']), np.log(age)

        myask = [o for o in env.orders.values() if o['side'] == 'sell']
        mybid = [o for o in env.orders.values() if o['side'] == 'buy']
        # TODO report sum of sizes
        lim_a1, lim_a2, lim_a3 = order_feats(myask[0]) if myask else (0,0,0)
        lim_b1, lim_b2, lim_b3 = order_feats(mybid[0]) if mybid else (0,0,0)
        #stats = env.book_stats
        delta_stats = lambda k: env.stats[k]-env.stats_prev[k]
        #db = lambda k: stats[-1]['buy'][k]-stats[-2]['buy'][k]
        #ds = lambda k: stats[-1]['sell'][k]-stats[-2]['sell'][k]
        #flow_imbalance_buy = float((db('vl')-db('vc')-db('vm'))/(db('vl')+db('vc')+db('vm')+EPSILON))
        #flow_imbalance_sell = float((ds('vl')-ds('vc')-ds('vm'))/(ds('vl')+ds('vc')+ds('vm')+EPSILON))
        #imbalance = lambda x,y: (x-y)/(x+y+1e-6)
        #vol_imbalance = imbalance(np.sum(env.bidmatch_vol[-10:]), np.sum(env.askmatch_vol[-10:]))
        tick_inv=100
        dt = datetime.fromtimestamp(env.book.time)
        # some mavgs
        mha = np.asarray(env.mid_history[-50:])

        assert N_STATES >= 21
        S = np.zeros(N_STATES)
        S[:21] = [
            # market feats
            #flow_imbalance_buy,
            #flow_imbalance_sell,
            #np.sum(env.bidmatch_vol[-10:]),
            #-np.sum(env.askmatch_vol[-10:]),
            env.bidmatch_vol[-1],
            -env.askmatch_vol[-1],
            #min(10,delta_stats('time')), # useful for vol ticks
            (ask-bid)*tick_inv, # normalized spread
            mid-np.mean(mha[-10:]),
            mid-np.mean(mha[-20:]),
            mid-np.mean(mha[-50:]),
            1000*(np.log(mid)-np.log(mid_1)), # midchange
            1000*(np.log(bid)-np.log(bid_1)), # bidchange
            1000*(np.log(ask)-np.log(ask_1)), # askchange
            dt.hour/24, #dt.weekday(),
            # agent feats
            float(env.portfolio[0]) / env.MAX_INVENTORY, # position
            delta_stats('lim_buy_qty') / float(env.qty), # buy qty filled last step
            delta_stats('lim_sell_qty') / float(env.qty), # ask qty filled last step
            # limit order feats
            lim_a1, lim_a2, lim_a3, # do we have an active lim buy order
            lim_b1, lim_b2, lim_b3, # do we have an active lim sell order
            delta_stats('realized_pnl'),
            #delta_stats('unrealized_pnl'),
            env.stats['unrealized_pnl'],
            #env.stats['avg_up'],
        ]
        #print('state:', S)
        #matched_last_step=',delta_stats('lim_buy_qty'),delta_stats('lim_sell_qty'))
        #print('state: num env orders=', len(env.orders))
        if self.noise>0:
            S += np.random.normal(0, self.noise, S.shape)
        return S


class TinyState(State):
    def shape(self):
        return (N_STATES,1)
    def state(self):
        return np.reshape(super().state(), (-1,1))

def rpad(x,n,value=0):
    return np.pad(x,(value,n-len(x)), 'constant')

class LevelState(State):
     # states aligned to levels only
    def shape(self):
        return (N_STATES*3,1)

    def state(self):
        k=N_STATES//2
        assert N_STATES%2==0
        bid_p, bid_sz = self.env.book.get_bids(k)
        ask_p, ask_sz = self.env.book.get_asks(k)
        bid_sz = np.asarray(bid_sz).astype(float)[::-1]
        bid_p = np.asarray(bid_p).astype(float)[::-1]
        ask_sz = np.asarray(ask_sz).astype(float)
        ask_p = np.asarray(ask_p).astype(float)
        bid, ask = bid_p[0], ask_p[0]

        S = np.zeros((3, N_STATES))
        na,nb = len(ask_sz), len(bid_sz)
        S[0, :na] = ask_sz #np.log(1+np.cumsum(ask_sz))
        S[0, k:k+na] = (ask_p-ask)
        S[1, :nb] = -bid_sz #np.log(1+np.cumsum(bid_sz))
        S[1, k:k+nb] = (bid_p-bid)
        # base feats
        S[2, :] = super().state()
        return np.expand_dims(np.reshape(S, -1),-1)


class MicroState(State):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cum = False

    def shape(self):
        return (N_STATES*2,)

    def state(self):
        k = N_STATES//2
        assert N_STATES%2==0
        mid=float(self.env.book.price())
        w = mid*0.005
        S = np.zeros((2, N_STATES))
        bidlvls, asklvls, lvls = book2agglvls(self.env.book, w, k)
        #S[0,:]=(np.cumsum(bidlvls) - np.cumsum(asklvls)) / (np.cumsum(bidlvls) + np.cumsum(asklvls))
        if self.cum:
            S[0, :k] = np.log(1+np.cumsum(asklvls))
            S[0, k:] = np.log(1+np.cumsum(bidlvls))
        else:
            S[0, :k] = -asklvls/10
            S[0, k:] = bidlvls/10
        S[1, :] = super().state()
        return np.reshape(S, -1)


class MicroCumState(MicroState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cum = True

class MicroStackedState(State):
    # as micro but last k steps, all states centered on current price

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cum = False

    def shape(self):
        return (N_STACK, N_STATES*4)

    def reset(self):
        self.step=0
        self.stack=np.zeros(self.shape())
        self.price_hist=np.zeros(N_STACK)

    def state(self):
        k = N_STATES
        w = 5
        S = np.zeros((4, N_STATES))
        bidlvls, asklvls, lvls = book2agglvls(self.env.book, w, k)
        S[0, :] = -asklvls[::-1]/10
        S[1, :] = bidlvls/10
        #S[2, :] = super().state()

        p = float(self.env.book.price())

        def shift(x,offset):
            if offset>0:
                x[:-offset] = x[offset:]
                x[-offset:]=x[-1]
            elif offset<0:
                offset=-offset
                x[offset:] = x[:-offset]
                x[:offset]=x[0]

        # update historical states with offset
        """
        if self.step>0:
            offset=int((p-self.price_hist[-1])/(w/k))
            j=2*k
            if offset>0:
                self.stack[:,:-offset] = self.stack[:,offset:]
                self.stack[:,-offset:]=0
            elif offset<0:
                offset=-offset
                self.stack[:,offset:] = self.stack[:,:-offset]
                self.stack[:,:offset]=0
        """

        self.stack[:-1,...]=self.stack[1:,...]
        self.stack[-1]=S.reshape(-1)
        self.price_hist[:-1]=self.price_hist[1:]
        self.price_hist[-1]=p
        self.step += 1

        # return shifted copy
        S = self.stack.copy()
        # quantize before and after prices
        p1 = np.round(p/(w/k),1)
        p2 = np.round(self.price_hist/(w/k),1)
        offsets = p1-p2
        for i,p in enumerate(self.price_hist):
            shift(S[i], int(offsets[i]))
            # draw price
            #o = int(np.clip((self.price_hist[-1]-p)/(w/k), -k//2, k//2))
            #S[i,-o-(k//2)] = 1
        # unshifted copy
        S[:,N_STATES*2:] = self.stack[:,:N_STATES*2]

        return S




class CandleState(State):
    YRANGE=100
    SCALE=0.1
    def shape(self):
        return (N_STACK, 2*self.YRANGE)#(2*self.YRANGE, N_STACK)

    def state(self):
        env = self.env
        S = np.zeros((N_STACK, 2*self.YRANGE))
        bid, bidsz = env.book.get_bid()
        ask, asksz = env.book.get_ask()
        spread = ask-bid
        mid = float((ask+bid)/2)
        p2y = lambda p: self.YRANGE+int(np.round(np.clip((p-mid)/self.SCALE, -self.YRANGE, self.YRANGE),1))
        fill_imbalance = (bidsz-asksz)/(bidsz+asksz)
        mid_change = env.mid_history[-1] - env.mid_history[-2] if len(env.mid_history)>1 else 0.0

        candles = env.inds.candle_arr()[-N_STACK:, :]
        # time, low, high, open, close, vol
        #-> high-low, close-open, curr_price-open, vol
        for i,cndl in enumerate(candles[::-1]):
            t,l,h,o,c,v,vb,va = cndl
            assert l<=h
            S[-i-1, p2y(l):1+p2y(h)] = -1
            direction=np.sign(c-o)
            if direction==-1:
                tmp=o; o=c; c=tmp
            S[-i-1, p2y(o):1+p2y(c)] = 1#direction
            S[-i-1, 0] = vb
            S[-i-1, 1] = va
            S[-i-1, 2] = (h-l)
            S[-i-1, 3] = (o-c)

        #S[0, :ncandles] = candles[:,2] - candles[:,1] # high-low
        #S[1, :ncandles] = candles[:,4] - candles[:,3] # close-open
        #S[2, :ncandles] = env.mid_history[-1] - candles[:,3] # price_now-open
        #S[3, :ncandles] = candles[:,5] # vol
        return S#np.reshape(S, (2*self.YRANGE, N_STACK))


class Price2DState(State):
    # not using candles

    YRANGE=50
    SCALE=1
    def shape(self):
        return (2*self.YRANGE, N_STACK, 1)
    def state(self):
        env = self.env
        bid, bidsz = env.book.get_bid()
        ask, asksz = env.book.get_ask()
        mid = float((ask+bid)/2)
        p2y = lambda p: self.YRANGE+int(np.clip((p-mid)/self.SCALE, -self.YRANGE, self.YRANGE))  # FIXME np.round not int
        bidhist, bidvolhist, bidmatchvol = env.bid_history, env.bidvol_history, env.bidmatch_vol
        askhist, askvolhist, askmatchvol = env.ask_history, env.askvol_history, env.askmatch_vol
        S = np.zeros((N_STACK, 2*self.YRANGE))
        for i in range(min(len(bidhist), N_STACK)):
            j=-i-1
            pb,pa=p2y(bidhist[j]), p2y(askhist[j])
            imb=(bidvolhist[j]-askvolhist[j])/(bidvolhist[j]+askvolhist[j]+1e-6)
            S[j, pb:pa+1] = np.sign(imb)
            #S[i, min(po,pc):max(po,pc)+1] = 1#np.clip(cndl[5]/100,0,1)
            #S[j, 0] = np.clip(bidmatchvol[j]/100,0,1)
            #S[j, 1] = -np.clip(askmatchvol[j]/100,0,1)
            #S[j, 2] = pa-pb
        S[-1,self.YRANGE]=-1 # marker
        return np.expand_dims(S.T, -1)



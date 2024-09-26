import numpy as np
from deeptrade.envs.limitenv import LimitEnv, PRICE_QUANTIZE, QTY_QUANTIZE, SIDE_BUY, SIDE_SELL
from gym import spaces
from decimal import Decimal
import random

EPSILON=1e-6
EPSILON_DEC=Decimal('1e-6')
ORDER_REPLACE_QTY=Decimal('0.01')
TICK=Decimal('0.01')

# helper functions

def ensure_quote(env, side, price, qty, keep_closer=False):
    price = price.quantize(PRICE_QUANTIZE)
    qty = qty.quantize(QTY_QUANTIZE)
    orders = [o for o in env.orders.values() if o['side']==side]
    #print(orders, env.portfolio)
    #print(side, price, qty)
    existing_qty = 0
    o_at_price = []
    for o in orders:
        if (keep_closer and o['price'] <= price and side == SIDE_SELL) or \
        (keep_closer and o['price'] >= price and side == SIDE_BUY) or \
        (not keep_closer and o['price']==price):
            existing_qty += o['size']
            o_at_price.append(o['id'])
        else:
            env.cancel_order(o['id'])
    dq = qty-existing_qty
    if existing_qty==0 or abs(dq) >= ORDER_REPLACE_QTY:
        env.cancel_orders(o_at_price)
        env._limit_order(side, price, qty)

def ensure_quote_simple(env, side, price, qty):
    price = price.quantize(PRICE_QUANTIZE)
    already = any(o for o in env.orders.values() if o['side']==side and o['price']==price)
    if not already:
        qty = qty.quantize(QTY_QUANTIZE)
        env._limit_order(side, price, qty)


def other_side(side): return SIDE_BUY if side==SIDE_SELL else SIDE_SELL

def cancel_side(env, side):
    env.cancel_by_fn(lambda o: o['side']==side)

def place_ladder(env, side, start_price, alpha, depth, qty):
    cancel_side(env, side)
    for i in range(depth):
        env._limit_order(side, start_price+(alpha*i), qty)

def place_ladder_exp(env, side, start_price, alpha, depth, qty):
    cancel_side(env, side)
    for i in range(depth):
        env._limit_order(side, start_price*(alpha**i), qty)


# example env showing callbacks
class ExampleEnv(LimitEnv):

    def __init__(self, *env_args, **env_kw_args):
        # you must specify a valid action_space
        self.action_space = spaces.Discrete(4) # ...
        # set any env-specific variables here

    def _reset(self):
        # called on new episode
        pass

    def _done(self):
        # called at each step
        # return True if end of episode eg timeout or goal reached
        return False

    def _reward(self):
        # return reward at this time step
        return 0.0

    def _act(self, action):
        # called at each step with the action selected by the policy (from self.action_space)
        # execute whatever actions are needed in response to this
        bid, bidvol = self.book.get_bid()
        oid = self._limit_order(side=SIDE_BUY, price=bid-1, qty=Decimal(1))

    def _on_fill(self, fill):
        # called when an existing order is filled. Might be a partial fill.
        # do something in response to this if needed
        pass


def unflatten(x, ncats):
    return [(x // (1 << i)) % 2 for i in range(ncats)]


"""
there are several styles of environment and how to place/maintain orders.
- positional vs nonpositional:
In positional envs, actions are placing orders to achieve a desired position.
In nonpositional envs: actions are placing orders with a given qty. Can have multiple actions correposnding to different qtys.

- holding vs nonholding actions: market orders are executed immediately, so this applies to how to place limit orders.
holding actions mean that an action has to be continuously held in order for the order to remain active
nonholding actions mean that one action places the order, and another action cancels it, and there needs to be a 'pass/noop' action

the intuition is that using noholding actions to place limit orders away from the fill make it easier to learn a policy
since the action doesn't need to be adjusted each time the price moves.

there is also a maintain spread argument: if true, don't let the agent place orders inside the spread



"""


def move_to_position(env, target, urgency=0):
    inv = env.portfolio[0]
    #urgency : 0=mkt order, 1=bid, 2=bid+tick,..
    dq = Decimal(target-inv)
    ask, _ = env.book.get_ask()
    bid, _ = env.book.get_bid()
    #mp = (ask-bid)/2
    if dq>0: # buy
        #env.cancel_all()
        cancel_side(env, SIDE_SELL)
        if urgency==0:
            cancel_side(env, SIDE_BUY)
            env._market_order(SIDE_BUY, dq)
        elif urgency==1:
            ensure_quote(env, SIDE_BUY, bid, dq)
        elif urgency==2:
            ensure_quote(env, SIDE_BUY, ask-TICK, dq)
    elif dq<0: # sell
        #env.cancel_all()
        cancel_side(env, SIDE_BUY)
        if urgency==0:
            cancel_side(env, SIDE_SELL)
            env._market_order(SIDE_SELL, -dq)
        elif urgency==1:
            ensure_quote(env, SIDE_SELL, ask, -dq)
        elif urgency==2:
            ensure_quote(env, SIDE_SELL, bid+TICK, -dq)
    else:
        env.cancel_all()

                    
def market_move_to_position(env, target):
    inv = env.portfolio[0]
    dq = Decimal(target-inv)
    if dq>0: # buy
        env._market_order(SIDE_BUY, dq)
    elif dq<0: # sell
        env._market_order(SIDE_SELL, -dq)

class MomentumEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(1) # ignore ti

    def _act(self, action):
        k=2
        tau=0
        if len(self.mid_history)>k:
            mom = self.mid_history[-1]-self.mid_history[-k]
            if mom>tau:
                market_move_to_position(self, self.qty)
            elif mom<-tau:
               market_move_to_position(self, -self.qty)


from deeptrade.utils import book2agglvls
class ImbalanceEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(1)

    def _act(self, action):
        book = self.book
        bid, bidsz = book.get_bid()
        ask, asksz = book.get_ask()
        spread = float(ask - bid)
        inv = self.portfolio[0]
        b_imbalance = (bidsz - asksz) / (bidsz + asksz) if (bidsz + asksz) > 0 else 0
        tau = 0.5
        if spread <= 0.1:
            if b_imbalance > tau: # buy signal
                ensure_quote(self, SIDE_BUY, bid, self.qty-inv)
            elif b_imbalance < -tau: # sell signal
                ensure_quote(self, SIDE_SELL, ask, self.qty+inv)
            #else:
            #    move_to_position(self, 0, urgency=urgency)
        else:
            self.cancel_all()
        #    move_to_position(self, 0, urgency=0)


class MOEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(3)

    def _act(self, action):
        if action==0:
            pass
        elif action==1:
            self._market_order(SIDE_BUY, self.qty)
        elif action==2:
            self._market_order(SIDE_SELL, self.qty)

class LOLadderEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(5) #[2,2,2])
        self.ladder_depth = 10

    def _act(self, action):
        price_step=Decimal('0.1')
        bid, bidvol = self.book.get_bid()
        ask, askvol = self.book.get_ask()

        if action==0:
            buy_act,sell_act,clear_act=0,0,0
        elif action==1:
            buy_act,sell_act,clear_act=0,1,0
        elif action==2:
            buy_act,sell_act,clear_act=1,0,0
        elif action==3:
            buy_act,sell_act,clear_act=1,1,0
        elif action==4:
            buy_act,sell_act,clear_act=0,0,1

        if clear_act==1:
            self.clear_inventory()

        if buy_act==1:
            place_ladder(self, SIDE_BUY, bid-price_step, -price_step, self.ladder_depth, self.qty)
        else:
            cancel_side(self, SIDE_BUY)

        if sell_act==1:
            place_ladder(self, SIDE_SELL, ask+price_step, price_step, self.ladder_depth, self.qty)
        else:
            cancel_side(self, SIDE_SELL)


class SkewedLOEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(5)
        self.ladder_offset= Decimal('0.001')

    def _act(self, action):
        book = self.book
        bid, bidvol = book.get_bid()
        ask, askvol = book.get_ask()
        ref = (bid+ask)/2

        if action==0:
            # no change
            pass
        elif action==1:
            # cancel + clear
            self.cancel_all()
            self.clear_inventory()
        elif action==2:
            # symmetric
            self.cancel_all()
            ensure_quote_simple(self, SIDE_BUY, bid*(1-self.ladder_offset), self.qty)
            ensure_quote_simple(self, SIDE_SELL, ask*(1+self.ladder_offset), self.qty)
        elif action==3:
            # downtrend
            self.cancel_all()
            ensure_quote_simple(self, SIDE_BUY, bid*(1-self.ladder_offset), self.qty)
            ensure_quote_simple(self, SIDE_SELL, ask, self.qty)
        elif action==4:
            # uptrend
            self.cancel_all()
            ensure_quote_simple(self, SIDE_BUY, bid, self.qty)
            ensure_quote_simple(self, SIDE_SELL, ask*(1+self.ladder_offset), self.qty)


class BaseEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.MultiDiscrete([3,3])

    def _act(self, action):
        buy_action, sell_action = action
        # {pass, cancel, quote} for each side
        # TODO add actions for qty
        # TODO try flattening multidiscrete
        
        if buy_action==1:
            cancel_side(self, SIDE_BUY)
        elif buy_action==2:
            bid, _ = self.book.get_bid()
            ensure_quote(self, SIDE_BUY, bid, self.qty)
            
        if sell_action==1:
            cancel_side(self, SIDE_SELL)
        elif sell_action==2:
            ask, _ = self.book.get_ask()
            ensure_quote(self, SIDE_SELL, ask, self.qty)
            
                
class BaseFlatEnv(BaseEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(10)

    def _act(self, action):
        super()._act((action%3, action//3))


class BaseEnvWithQty(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.MultiDiscrete([10,10])

    def _act(self, action):
        buy_action, sell_action = action
        # {pass, cancel, quote} for each side

        if buy_action==1:
            cancel_side(self, SIDE_BUY)
        elif buy_action>1:
            bid, _ = self.book.get_bid()
            qty = self.qty*(buy_action-1)
            ensure_quote(self, SIDE_BUY, bid, qty)

        if sell_action==1:
            cancel_side(self, SIDE_SELL)
        elif sell_action>1:
            ask, _ = self.book.get_ask()
            qty = self.qty*(sell_action-1)
            ensure_quote(self, SIDE_SELL, ask, qty)


class PositionalEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.npositions=2
        self.nurgency=3
        self.target_positions = [Decimal(x) for x in np.linspace(-1,1,self.npositions*2+1)]
        self.action_space = spaces.MultiDiscrete([self.npositions*2+1,self.nurgency])

    def _act(self, action):
        position, urgency = action
        if urgency==2:
            pass
        else:
            move_to_position(self, self.target_positions[position], urgency)


class PositionalMOEnv(PositionalEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.action_space = spaces.Discrete(self.npositions*2+1)

    def _act(self, action):
        super()._act((action, 0))


class PositionalLOEnv(PositionalEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.action_space = spaces.Discrete(self.npositions*2+3)
        self.pass_act = self.npositions*2+1
        self.cancel_act= self.pass_act+1

    def _act(self, action):
        if action==self.pass_act:
            pass
        elif action==self.cancel_act:
            self.cancel_all()
        else:
            move_to_position(self, self.target_positions[action], urgency=1)

#POSITION X URGENCY 0,1,2 (0R 0,2)

class PositionalEnv1(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.Discrete(3)

    def _act(self, action):
        #if action==0:
        #    pass
        if action==0:
            move_to_position(self, 0, urgency=1)
        elif action==1:
            move_to_position(self, self.qty, urgency=1)
        elif action==2:
            move_to_position(self, -self.qty, urgency=1)
        #elif action==3:
        #    move_to_position(self, self.qty, urgency=2)
        #elif action==4:
        #    move_to_position(self, -self.qty, urgency=2)
        #elif action==5:
        #    move_to_position(self, -1, urgency=0)


class LOEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, book_aligned_level=False, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.book_aligned_level = book_aligned_level
        self.action_space = spaces.MultiDiscrete([2,2])

    def _act(self, action):
        act_buy, act_sell = action
        bid, _ = self.book.get_bid()
        ask, _ = self.book.get_ask()
        inv = self.portfolio[0]
        tau = 1 # tick multiple for passive orders

        if act_buy==0:
            cancel_side(self, SIDE_BUY)
        else:
            lvl = act_buy - 1
            if self.book_aligned_level:
                book_lvls, _ = self.book.get_bids(5) # reversed
                ensure_quote(self, SIDE_BUY, book_lvls[-lvl-1], self.qty)
            else:
                ensure_quote(self, SIDE_BUY, bid - tau*TICK*lvl, self.qty)

        if act_sell==0:
            cancel_side(self, SIDE_SELL)
        else:
            lvl = act_sell - 1
            if self.book_aligned_level:
                book_lvls, _ = self.book.get_asks(5)
                ensure_quote(self, SIDE_SELL, book_lvls[lvl], self.qty)
            else:
                ensure_quote(self, SIDE_SELL, ask + tau*TICK*lvl, self.qty)

class LOLevelEnv(LOEnv):
    # LOEnv·where·the·order·levels·are·aligned·with·current·book·levels$
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, act_qty=act_qty, book_aligned_level=True, **env_kw_args)

   
class StopLOEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, book_aligned_level=False, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.book_aligned_level = book_aligned_level
        self.action_space = spaces.Discrete(6)

    def _act(self, action):
        book = self.book
        bid, _ = book.get_bid()
        ask, _ = book.get_ask()
        inv = self.portfolio[0]

        if action==0:
            pass
        elif action==1:
            self.cancel_all()
        elif action==2:
            ensure_quote(self, SIDE_BUY, bid, self.qty)
        elif action==3:
            ensure_quote(self, SIDE_BUY, ask, self.qty) # stop limit @ (ask,ask)
        elif action==4:
            ensure_quote(self, SIDE_SELL, ask, self.qty)
        elif action==5:
            ensure_quote(self, SIDE_SELL, bid, self.qty) # stop limit @ (bid,bid)
            

class MOLOEnv(LimitEnv):
    def __init__(self, *env_args, act_qty=0.1, **env_kw_args):
        super().__init__(*env_args, **env_kw_args)
        self.qty = Decimal(str(act_qty))
        self.action_space = spaces.MultiDiscrete([4,4])

    def _act(self, action):
        #if action is None(action)==1 : return # NOOP
        # A = {cancel_all, aggressive, passive_0, passive_1} for each side
        book = self.book
        act_buy, act_sell = action
        bid, _ = book.get_bid()
        ask, _ = book.get_ask()
        inv = self.portfolio[0]
        tau = 2 # tick multiple for passive orders

        if act_buy==0:
            cancel_side(self, SIDE_BUY)
        elif act_buy==1:
            cancel_side(self, SIDE_BUY)
            self._market_order(SIDE_BUY, self.qty)
        else:
            lvl = act_buy - 2
            ensure_quote(self, SIDE_BUY, bid - tau*TICK*lvl, self.qty)

        if act_sell==0:
            cancel_side(self, SIDE_SELL)
        elif act_sell==1:
            cancel_side(self, SIDE_SELL)
            self._market_order(SIDE_SELL, self.qty)
        else:
            lvl = act_sell - 2
            ensure_quote(self, SIDE_SELL, ask + tau*TICK*lvl, self.qty)

def make_tradeenv(env_id, env_args, seed, training):
    import os
    data_path=os.environ.get('DEEPTRADE_DATA')
    if not data_path:
        raise ValueError('env variable DEEPTRADE_DATA not set')

    product_id = env_args.get('product_id')
    neval=env_args.get('neval',10)
    ntrain=env_args.get('ntrain',0)
    step_type=env_args.get('step_type')
    step_val=env_args.get('step_val')
    book_noise=env_args.get('book_noise',0.0)
    #print('setting up envid={}, level={}, data_path='.format(env_name, level, data_path))

    if training:
        # train/eval overlap mode
        if env_args.get('eval_overlap', False):
            exclude_lastn=0
        else:
            exclude_lastn=neval
        lastn=ntrain
    else:
        exclude_lastn=0
        lastn=neval

    # create book generator
    if env_args['level']==3:
        from deeptrade.envs.bookgen import get_bookgen
        assert book_noise==0, 'book_noise not supported for L3 env'
        datagen = get_bookgen(
            path=data_path,
            product_id=product_id,
            step_type=step_type,
            step_val=step_val,
            fill_type=env_args['fill_type'],
            match_type=env_args['match_type'],
            seed=seed,
            lastn=lastn,
            exclude_lastn=exclude_lastn,
        )
    elif env_args['level']==2:
        from deeptrade.envs.l2bookgen import get_l2bookgen
        datagen = get_l2bookgen(
            path=data_path,
            product_id=product_id,
            step_type=step_type,
            step_val=step_val,
            fill_type=env_args['fill_type'],
            match_type=env_args['match_type'],
            seed=seed,
            lastn=lastn,
            exclude_lastn=exclude_lastn,
            book_noise=book_noise,
        )
    else:
        raise ValueError('unknown level {}, must be 2 or 3'.format(env_args['level']))

    # create env
    known_envs = {
        'imbalance': ImbalanceEnv,
        'momentum': MomentumEnv,
        'base': BaseEnv,
        'mo': MOEnv,
        'molo': MOLOEnv,
        'lo': LOEnv,
        'lolevel': LOLevelEnv,
        'loladder': LOLadderEnv,
        'loskewed': SkewedLOEnv,
        'position1': PositionalEnv1,
        'position': PositionalEnv,
        'positionmo': PositionalMOEnv,
        'positionlo': PositionalLOEnv,
        'stoplo': StopLOEnv,
    }

    if env_id not in known_envs:
        raise ValueError(f'unknown env {env_id}, known_envs={list(known_envs.keys())}')
    env = known_envs[env_id](datagen, **env_args)
    env.seed(seed)

    return env

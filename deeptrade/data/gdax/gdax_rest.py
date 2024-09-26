
import pandas as pd
from datetime import datetime, timedelta
from deeptrade.data.restapi import RestAPI

class GDAX(RestAPI):
    def __init__(self, base_url='https://api.pro.coinbase.com/'):
        super().__init__(base_url)

    def products(self):
        return self._call(self.base_url+'products')

    def candles(self, prod, granularity=60, start_time=None, end_time=None):
        call_fn = lambda start, end: self._call(self.base_url+'products/{}/candles'.format(prod), \
               params={'granularity':granularity, 'start': start.isoformat(), 'end': end.isoformat()})
        delta = timedelta(seconds=granularity*300)
        return self._paginate(call_fn, start_time, end_time, delta, s=f'{self.base_url}::{prod}')


if __name__ == '__main__':
    import os
    gdax = GDAX()
    base_currency = 'USD'
    path=os.environ['DEEPTRADE_DATA']
    period= 60
    prods = [x['id'] for x in gdax.products() if x['quote_currency']==base_currency]
    print(prods)
    for prod in prods:
        print('pulling prod={} from gdax'.format(prod))
        results = gdax.candles(prod, period, start_time=datetime(2020,1,1), end_time=datetime.today())
        df = pd.DataFrame.from_dict(results)
        df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
        df.to_pickle(os.path.join(path,'candles','gdax',f'{prod}-{period}.pkl'))

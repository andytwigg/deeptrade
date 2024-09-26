from time import sleep
import requests

# pull GDAX historic price data
class RestAPI:

    def __init__(self, base_url, max_rps=3, max_retries=10):
        self.base_url = base_url
        self.backoff=1/max_rps
        self.max_retries=max_retries

    def _call(self, url, params=None):
        tries=0
        while tries<self.max_retries:
            try:
                r = requests.get(url, params=params)
                if r.status_code==200:
                    result = r.json()
                    if len(result)==0:
                        print("WARNING: empty response")
                    return result  # TODO catch json errors
                elif r.status_code==429:
                    print(f'received {r.text}, backing off..backoff={self.backoff}')
                    sleep(1)
                    self.backoff+=0.1
                elif r.status_code==404:
                    # missing data
                    return []
                else:
                    print(f'unknown response code={r.status_code} {r.text}')
            except Exception as e:
                if r:
                    print(r.text)
                print('ERROR: {}'.format(e))
                raise(e)
            #sleep(self.backoff)
            tries+=1
        raise Exception('exceeded max_retries={} with url={} params={}'.format(self.max_retries, url, params))

    def _paginate(self, call_fn, start_time, end_time, delta, s=''):
        curr_time = start_time
        results = []
        while curr_time < end_time:
            result = call_fn(curr_time, curr_time+delta)
            curr_time = curr_time+delta
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
            print(f'{s} time={curr_time} nresults={len(results)}')
            sleep(self.backoff)
        return results

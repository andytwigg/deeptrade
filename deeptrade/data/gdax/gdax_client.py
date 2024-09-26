import argparse
import json
import signal
import sys
import time
import logging
import base64
import hmac
from websocket import create_connection
from websocket._exceptions import WebSocketBadStatusException
from os import path
import hashlib

from cbpro import PublicClient, AuthenticatedClient
from deeptrade.data.log import Log, StdoutLog, RotatingLog
from deeptrade.data.websocket import WebSocketClient, WebSocketClientStats

logger = logging.getLogger(__name__)


class GdaxFeed(WebSocketClient):

    def __init__(self, sandbox=False, products="BTC-USD", channels="full", log_level="", auth=False, api_key="",
                 api_secret="", api_passphrase="", log=None):
        if sandbox:
            url = "wss://ws-feed-public.sandbox.pro.coinbase.com"
            api_url = "https://api-public.sandbox.pro.coinbase.com"
        else:
            url = "wss://ws-feed.pro.coinbase.com"
            api_url = "https://api.pro.coinbase.com"

        super(GdaxFeed, self).__init__(url, products, channels, log_level,
                                           auth, api_key, api_secret,
                                           api_passphrase, log, 'GDAX_L3')
        assert len(self.products) == 1, 'only pass 1 product_id, got {}'.format(self.products)
        self.product_id = self.products[0]
        if self.auth:
            self._client = AuthenticatedClient(api_key, api_secret, api_passphrase,
                                               api_url=api_url)
        else:
            self._client = PublicClient(api_url=api_url)
        logger.info('Starting GdaxFeed for {}, auth={}'.format(self.product_id, self.auth))

    def on_open(self):
        super(GdaxFeed, self).on_open()
        self._sequence = -1

    def on_close(self):
        super(GdaxFeed, self).on_close()
        if self.log:
            self.log.close()

    def start_connection(self):
        if 'heartbeat' not in self.channels:
            self.channels.append('heartbeat')

        if self.url[-1] == "/":
            self.url = self.url[:-1]

        sub_params = {'type': 'subscribe', 'product_ids': self.products, 'channels': self.channels}

        if self.auth:
            timestamp = str(time.time())
            message = timestamp + 'GET' + '/users/self/verify'
            message = message.encode('ascii')
            hmac_key = base64.b64decode(self.api_secret)
            signature = hmac.new(hmac_key, message, hashlib.sha256)
            signature_b64 = base64.b64encode(signature.digest()).decode('utf-8').rstrip('\n')
            sub_params['signature'] = signature_b64
            sub_params['key'] = self.api_key
            sub_params['passphrase'] = self.api_passphrase
            sub_params['timestamp'] = timestamp

        self.ws = None
        while self.ws is None:
            try:
                logger.info('{} starting WS connection: url={} params={}'.format(self.products, self.url, sub_params))
                self.ws = create_connection(self.url)
                self.ws.send(json.dumps(sub_params))
            except WebSocketBadStatusException as e:
                logger.error('{} got error {} on start_connection'.format(self.products, e))
                time.sleep(self.reconnect_interval)  # to avoid maxing api connection rate
            except Exception as e:
                logger.error('{} got error {} on start_connection'.format(self.products, e))
            time.sleep(self.reconnect_interval)
        self.reset_book()

    def unsubscribe(self):
        sub_params = {'type': 'unsubscribe', 'channels': self.channels}
        try:
            self.ws.send(json.dumps(sub_params))
        except ConnectionError as e:
            logger.error('{} got error {} on start_connection'.format(self.products, e))

    def data_handler(self, msg):
        if msg['type'] == 'error':
            logger.error(msg)
            return
        if msg['type'] == 'heartbeat':
            logger.info('{} {}'.format(self.products, msg))
            return
        if 'sequence' in msg:
            sequence = msg['sequence']
            if self.log:
                self.log.log(json.dumps(msg), msg['sequence'])
            if sequence <= self._sequence:
                return
            if sequence > self._sequence + 1:
                self.on_sequence_gap(self._sequence, sequence)
                return
            self._sequence = sequence

    def on_sequence_gap(self, gap_start, gap_end):
        logger.error('{} seq={} missing seqs {}-{}'.format(self.products, self._sequence, gap_start, gap_end))
        self.reset_book()

    def reset_book(self):
        retries=0
        max_retries=10
        logger.info('{} seq={} reset_book()'.format(self.products, self._sequence))
        msg = {}
        while 'sequence' not in msg:
            t = self._client.get_time()['iso']  # time not included in snapshot response
            msg = self._client.get_product_order_book(product_id=self.product_id, level=3)
            msg['time'] = t
            if 'sequence' not in msg:
                logger.error("reset_book(): bad msg in reset_book; sequence not in msg:{}".format(msg))
                time.sleep(1)
                retries+=1
                if retries>max_retries:
                    logger.error("exceeded max_retries: exiting")
                    self.stop()
                    return {}
        logger.info('{} seq={} got new snapshot with seq={}'.format(self.products, self._sequence, msg['sequence']))
        msg['type'] = 'snapshot'
        if self.log:
            self.log.log(json.dumps(msg), msg['sequence'])
        self._sequence = msg['sequence']
        return msg

    def limit_order(self, side, price, size):
        if self.auth:
            self._client.place_limit_order(self.product_id, side, str(price), str(size))
        else:
            logger.error('limit_order(): trying to send limit_order with cbpro without authentication')

    def cancel_all(self):
        if self.auth:
            self._client.cancel_all(self.product_id)
        else:
            logger.error('cancel_all(): trying to send cancel_all with cbpro without authentication')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='log type: stdout, rotating')
    parser.add_argument('--loglevel', help='logging type: INFO, DEBUG, WARNING, ...', default='INFO')
    parser.add_argument('-o', '--output_dir', help='output dir for rotating log', default='data/book/')
    parser.add_argument('--rotate_freq', help='rotate freq for rotating log', default=1e8, type=float)
    parser.add_argument('-p', '--product_id', help='product_id', required=True, nargs='+')
    args = parser.parse_args()
    print(args)

    if args.log == 'stdout':
        log = StdoutLog()
    elif args.log == 'rotating':
        log = RotatingLog(path.join(args.output_dir, args.product_id), log_rotate_freq=int(args.rotate_freq))
    else:
        log = None

    client = GdaxFeed(
        log=log,
        log_level=args.loglevel,
        products=args.product_id,
    )

    shutdown = False
    def signal_handler(signal, frame):
        global shutdown
        shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    client.start()
    while not shutdown and client.is_running():
        time.sleep(1)
    client.stop()

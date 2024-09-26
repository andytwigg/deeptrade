# deeptrade/websocket.py
# original author: Daniel Paquin
# template modified/generalized by Krishnan Srinivasan
#
# Creates template object to receive messages from multiple data WebSocket feeds

import json
import logging
import os
import time
import sys
import socket
import requests
from datetime import datetime
from collections import defaultdict

from websocket import create_connection, WebSocketConnectionClosedException
from threading import Thread, Timer, Event

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

WEBHOOK =  os.getenv('WEBHOOK', 'https://hooks.slack.com/services/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')

class WebSocketClient(object):

    def __init__(self, url="", products="", channels="", log_level="",
                 auth=False, api_key="", api_secret="", api_passphrase="",
                 log=None, name=""):
        """Initializes a WebSocketClient
        :param url: websocket url to connect to
        :param products: list of multiple (or str for single) data symbols to subscribe to
        :param channels: list of multiple (or str for single) channels to subscribe to
        :param api_key: ...
        :param api_secret: ...
        :param api_passphrase: ...
        """
        self.url = url
        if not isinstance(products, list):
            products = [products]

        if not isinstance(channels, list):
            channels = [channels]

        self.products = products
        self.channels = channels
        self.error = None
        self.ws = None
        self.thread = None

        logging.basicConfig(stream=sys.stdout, level=log_level if log_level else logging.INFO)

        # auth params
        self.auth = auth
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase

        # logging
        self.log = log

        # event flags for maintaining/ending connection
        self.connected = Event()
        self.disconnect_called = Event()
        self.reconnect_required = Event()

        # timers for keeping connection alive
        self.ping_interval = 30
        self.connection_timeout = 10
        self.reconnect_interval = 1
        self.last_ping_time = -1
        self.connection_timer = None

    def start(self):
        def _go():
            self._connect()
            self._listen()
            self._disconnect()
            self.on_close()

        self.on_open()
        self.thread = Thread(target=_go)
        self.thread.start()

    def is_running(self):
        return self.thread.is_alive()

    def _connect(self):
        self.start_connection()
        logger.info('[{}] _connect(): started connection at {}'.format(self.products, datetime.utcnow().isoformat()))
        self.connected.set()
        self._start_timers()
        self.last_ping_time = time.time()

    def start_connection(self):
        # overwrite this method based on data, as connection params vary
        payload = {'type': 'subscribe'}
        self.ws = create_connection(self.url)
        self.ws.send(json.dumps(payload))

    def _send_ping(self):
        # override if ping api is different
        self.ws.ping("keepalive")

    def _stop_timers(self):
        if self.connection_timer:
            self.connection_timer.cancel()

    def _start_timers(self):
        self._stop_timers()
        if self.disconnect_called.is_set():
            return

        # Automatically reconnect if we didnt receive data
        self.connection_timer = Timer(self.connection_timeout, self._connection_timed_out)
        self.connection_timer.start()

    def _connection_timed_out(self):
        logger.error("[{}] connection timed out: issuing reconnect at {}".format(self.products, datetime.utcnow().isoformat()))
        self.reconnect()

    def _listen(self):
        while self.connected.is_set() or self.reconnect_required.is_set():
            if not self.disconnect_called.is_set():
                if self.reconnect_required.is_set():
                    self._reconnect()
                    continue
                try:
                    if time.time()-self.last_ping_time>self.ping_interval:
                        #logger.info('[{}] _send_ping(): sending ping at {}'.format(self.products, datetime.utcnow().isoformat()))
                        self._send_ping()
                        self.last_ping_time = time.time()
                    data = None
                    data = self.ws.recv()
                    msg = json.loads(data)
                except ValueError as e:
                    self.on_error(e, data=data) # dont reconnect, let timeout handle this
                except WebSocketConnectionClosedException as e:
                    self.on_error(e, data=data)
                    if not self.disconnect_called.is_set():
                        self.reconnect_required.set()
                except Exception as e:
                    self.on_error(e, data=data)
                else:
                    self.on_message(msg)

    def unsubscribe(self):
        # placeholder with unsubscribe logic, specific to data
        pass

    def _disconnect(self):
        # disconnects ws
        self._stop_timers()
        try:
            if self.ws:
                self.unsubscribe()
                self.ws.close()
        except WebSocketConnectionClosedException as e:
            logger.error('{} Error while disconnecting: {}'.format(self.products, e))
            pass

    def stop(self):
        # just exits from _listen
        logger.info('{} stop() called; shutting down {}'.format(self.products, self.url))
        self.disconnect_called.set()
        self.connected.clear()
        self.reconnect_required.clear()
        # wait for _listen to finish
        self.thread.join()

    def _reconnect(self):
        # reconnect logic
        self._disconnect()  # stops timers
        logger.info("{} reconnecting in {} seconds".format(self.products, self.reconnect_interval))
        time.sleep(self.reconnect_interval)  # to avoid maxing api connection rate
        self._connect()  # starts timers
        self.reconnect_required.clear()

    def reconnect(self):
        logger.info("{} setting reconnect flag at {}".format(self.products, datetime.utcnow().isoformat()))
        self.reconnect_required.set()

    def on_open(self):
        # placeholder function, in case anything needs to be done before open
        logger.info('{} opening connection url={}'.format(self.products, self.url))

    def on_close(self):
        # placeholder function, in case anything needs to be done after close
        logger.info('{} closing connection url={}'.format(self.products, self.url))

    def on_message(self, msg):
        self._stop_timers()
        self.data_handler(msg)
        self._start_timers()

    def data_handler(self, msg):
        # overwrite this method to properly handle logging incoming messages
        logger.debug('on_message: msg={} time={}'.format(msg, datetime.now()))

    def on_error(self, e, data=None):
        logger.error("{} error={}, data={}".format(self.products, e, data))


class WebSocketClientStats(WebSocketClient):

    def __init__(self, url="", products="", channels="", log_level="",
                 auth=False, api_key="", api_secret="", api_passphrase="",
                 log=None, name=""):
        super(WebSocketClientStats, self).__init__(url, products, channels,
                log_level, auth, api_key, api_secret, api_passphrase, log, name)
        self.name = '{}:{}'.format(socket.getfqdn(), name)
        self.full_name = '{} ({})'.format(self.name, self.products)
        self.reconnects = 0
        self.timeouts = 0
        self.msgs, self.msgs_since = 0, 0
        self.errors = 0
        self.init_time = datetime.utcnow()
        self.last_reconnect = self.init_time
        self.last_log = self.init_time
        self.msgcounts = defaultdict(int)

    def check_log(self):
        now = datetime.utcnow()
        if (now - self.last_log).seconds >= 60*5:
            self.log_info(self.get_stats())
            self.last_log = now
            self.msgs_since = self.msgs

    def get_stats(self):
        msg_count = self.msgs - self.msgs_since
        now_iso = datetime.utcnow().isoformat()
        stats = {
            'products': self.products,
            'time': now_iso,
            'total_messages':self.msgs,
            'msgs_since': msg_count,
            'reconnects':self.reconnects,
            'time_since_reconnect': int((datetime.utcnow()-self.last_reconnect).total_seconds()),
            'timeouts': self.timeouts,
            'errors': self.errors,
            'msgcount': dict(self.msgcounts),
        }
        return stats

    def log_info(self, msg):
        logger.info('{}: {}'.format(self.full_name, msg))
        if WEBHOOK:
            msg = {'text': json.dumps(msg)}
            msg['username'] = self.name
            try:
                response = requests.post(
                    WEBHOOK, data=json.dumps(msg),
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                logger.error('{} error={} posting to webhook, response={}'.format(self.full_name, e, response.text))
            except requests.exceptions.RequestException as e:
                logger.info('{} error={} posting to webhook'.format(self.full_name, e))
            except Exception as e:
                logger.info('{} error={} posting to webhook'.format(self.full_name, e))

    def _reconnect(self):
        super(WebSocketClientStats, self)._reconnect()
        self.last_reconnect = datetime.utcnow()
        self.reconnects += 1

    def on_message(self, msg):
        super(WebSocketClientStats, self).on_message(msg)
        self.msgs += 1
        if 'type' in msg:
            self.msgcounts[msg['type']] += 1
        self.check_log()

    def on_error(self, e, data=None):
        super(WebSocketClientStats, self).on_error(e, data)
        self.errors += 1
        stats = self.get_stats()
        stats['reason'] = 'Error: {}'.format(e)
        self.log_info(stats)

    def _connection_timed_out(self):
        super(WebSocketClientStats, self)._connection_timed_out()
        self.timeouts += 1

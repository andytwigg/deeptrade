import os
from os import path
import gzip


class Log:
    def log(self, msg, seq=None):
        pass

    def close(self):
        pass


class StdoutLog(Log):
    def log(self, msg, seq=None):
        if seq:
            print('[{}] {}'.format(seq,msg))
        else:
            print(msg)


class RotatingLog(Log):
    ROTATE_MSGTYPE='snapshot'

    def __init__(self, log_dir, log_rotate_freq=1e7):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_rotate_freq = int(log_rotate_freq)
        self.curr_file = None
        self.seq_ = -1
        self.last_seq_ = -1

    def log(self, msg, seq=None):
        if seq is None:
            seq = self.seq_+1

        self.seq_ = seq
        if (self.seq_ - self.last_seq_) >= self.log_rotate_freq or self.last_seq_==-1:
            if self.curr_file is not None:
                self.curr_file.close()
            filename = path.join(self.log_dir, '{}.log.gz'.format(self.seq_))
            self.curr_file = gzip.open(filename, 'wt')
            print('[RotatingLog] seq={}, rotating to file {}'.format(self.seq_, filename))
            self.last_seq_ = self.seq_
        print(msg, file=self.curr_file)

    def close(self):
        if self.curr_file:
            self.curr_file.close()
        print('[RotatingLog] seq={}, closed'.format(self.seq_))


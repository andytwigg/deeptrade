# a weird love child of BSD split (-p) and GNU split (--filter gzip $FILE)
# usage example: gunzip -c ../sorted.gz | python3 split_files.py
import gzip
import sys
from datetime import date
import os
import ujson as json
import ciso8601

"""
generates *DAILY* split files from raw files:
-each output file has exactly 1 snapshot msg at the start followed by msgs with contiguous seq numbers
-new files are created when either there is a sequence gap or a file exceeds a max split_size
-removing duplicates
-creating larger files for contiguous seq numbers
-removing redundant snapshot msgs
"""

PATTERN='snapshot'
curr_fh=None
infile=sys.argv[1]
outpath=sys.argv[2]

print("split_files_daily: reading from {}".format(infile))
with gzip.open(infile, 'rt') as in_fh:
    # as long as seq is contiguous, split only on split size
    try:
        currseq=0
        file_size=0
        curr_date=''
        valid=False
        #book = OrderBook()
        for line in in_fh:
            seq, msg = line.split(maxsplit=1)
            seq = int(seq)
            assert seq>=currseq, 'seq={} curr_seq={} msg={}'.format(seq,currseq,msg)
            snapshot_msg = (PATTERN in msg)
            #if snapshot_msg:
            #    print('SNAPSHOT: seq={} curr_date={} wasvalid={}'.format(seq, curr_date, valid))
            # alternatively just write out all msgs and let bookgen try to process them and catch
            # invalidbook exception if failed to apply
            if seq>currseq+1:
                valid=False
            was_valid=valid  # needs to come after seqgap eg 1,2,3(valid),6[snapshot],...
            if snapshot_msg:
                valid=True

            if valid:
                msg = json.loads(msg)
                msg['time_'] = ciso8601.parse_datetime(msg['time']).timestamp()
                msgdate = msg['time'][:10]  # YYYY-MM-DD
                # start new file when: 1) we have a snapshot msg and 2) date changed
                start_new = snapshot_msg and msgdate!=curr_date
                if start_new:
                    if curr_fh: curr_fh.close()
                    fname = os.path.join(outpath, 'split_daily', '{}.gz'.format(msgdate))
                    curr_fh = gzip.open(fname, 'wt', compresslevel=1)
                    print(curr_fh.name)
                    file_size=0
                    curr_date=msgdate
                # write out all valid msgs
                # except filter out unnecessary snapshots
                # ie those where was_valid=True
                if start_new or not (snapshot_msg and was_valid):
                    if snapshot_msg:
                        print('SNAPSHOT: msgtime={} seq={} curr_date={}'.format(msg['time'], seq, curr_date))
                    curr_fh.write(line)
                    file_size+=1
            currseq=seq
    finally:
        if curr_fh:
            curr_fh.close()

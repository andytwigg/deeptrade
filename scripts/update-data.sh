
#! /bin/bash

if [ -z "$DEEPTRADE_DATA" ]; then echo "DEEPTRADE_DATA variable not found"; exit 1; fi
if [ -z "$DEEPTRADE_HOME" ]; then echo "DEEPTRADE_HOME variable not found"; exit 1; fi
DEEPTRADE_TMP=~/tmp/

echo "UPDATING $1 to $DEEPTRADE_DATA"

echo '[STAGE 1] sync s3 data'
cd $DEEPTRADE_DATA
mkdir -p gdax_book/$1
cd gdax_book/$1
#aws s3 sync s3://deeptrade.data/gdax_book/$1/ . --exclude 'split/*' --exclude 'seq/*' --exclude 'snapshots/*'

echo '[STAGE 2] adding seqs'
rm -rf seq
mkdir -p seq
export LC_ALL=C
ls [0-9]*.log.gz -v -1 | tail -n 100 | xargs -P12 -I {} sh -c 'LC_ALL=C zgrep -vw "heartbeat\|last_match\|subscriptions" {} | jq ".sequence,." -c | paste -d " " - - | LC_ALL=C  sort -n -k1 -S 4G -T ~/tmp/ | gzip --fast > seq/seq-{}'

echo '[STAGE 3] merge + dedupe'
#ulimit -n 64000
cmd="LC_ALL=C  sort --batch-size 1000 -mn -k1 -S 16G"
for input in `ls seq/seq-* -v -1`; do
    cmd="$cmd <(gunzip -c '$input')"
done
eval "$cmd" | LC_ALL=C uniq | LC_ALL=C gzip --fast > "$1.gz"

# remove all seq data
#rm -rf seq

echo '[STAGE 4] split'
rm -rf split_daily
mkdir -p split_daily
LC_ALL=C LANG=C PYTHONPATH=$DEEPTRADE_HOME python $DEEPTRADE_HOME/deeptrade/data/split_files_daily.py "$1.gz" .

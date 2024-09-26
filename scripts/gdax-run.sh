#!/bin/bash
# https://spin.atomicobject.com/2017/08/24/start-stop-bash-background-process/
trap "exit" INT TERM ERR
trap "kill -SIGINT 0" EXIT

PRODUCTS="BTC-USD
ETH-USD
LTC-USD
BCH-USD"

OUTPATH=${OUTPATH:-"../deeptrade-data/gdax_book"}
LOG=${LOG:-"rotating"}

for PROD in $PRODUCTS;
do
	echo "$PROD --> $OUTPATH"
	DIRECTORY="$OUTPATH$PROD"
	CMD="python3 -u deeptrade/data/gdax/gdax_client.py -o $OUTPATH -p $PROD --log $LOG"
	./scripts/run_loop.sh $CMD &

done
wait


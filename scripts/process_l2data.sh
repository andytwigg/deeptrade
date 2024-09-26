
# split on first snapshot message
ls -1v *.log.gz | xargs -P`nproc` -I% sh -c 'echo %; zcat % | csplit -f % - /snapshot/'

# remove any empties
find . -size 0 -print -delete

# we have ungzipped output, gzip it
ls -1v | xargs -P`nproc` -I% sh -c 'echo %; gzip %'

# delete remaining unzipped if wanted
# rm *.log.gz


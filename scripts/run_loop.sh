#! /bin/bash

while :
do
    echo [$@] RUNNING...
    eval $@
    echo [$@] EXITED...
    echo -e "subject:run_loop notification\n\n host=`hostname` command=$@ exited" | sendmail andy.twigg@gmail.com
    sleep 2
done
#!/bin/bash
# conda activate xingtian
rm -f run.log
xt_main -f $1 &>$2
# /bin/bash /home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/kill.sh
wait
echo "complete."

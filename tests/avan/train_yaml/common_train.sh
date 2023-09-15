#!/bin/bash
list=$(ls $1)
for l in $list
do
  echo "$l"
  /home/xys/xingtian-ppo-v1/train_yaml/single_train.sh $1/$l
  echo "complete $l."
  wait

done

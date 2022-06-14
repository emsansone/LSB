#!/bin/bash

batch=5

gamma=1.0
factor=20
block=20
maxv=2
skip=100
burnin=500
iters=10000
filename=$PWD'/../UAI/UAI_BN/BN_1.uai' # 100 vars
python3 runner.py --batch $batch --gamma $gamma --factor $factor --block $block --maxv $maxv --skip $skip --burnin $burnin --iters $iters --filename $filename &


gamma=1.0
factor=20
block=20
maxv=2
skip=100
burnin=500
iters=10000
filename=$PWD'/../UAI/UAI_BN/BN_8.uai' # 100 vars
python3 runner.py --batch $batch --gamma $gamma --factor $factor --block $block --maxv $maxv --skip $skip --burnin $burnin --iters $iters --filename $filename &

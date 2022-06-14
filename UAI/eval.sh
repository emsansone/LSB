#!/bin/bash

batch=5

gamma=1.0
factor=20
block=20
maxv=2
burn_in=200
iters=10000
lags=400
filename=$PWD'/../UAI/UAI_BN/BN_1.uai' # 100 vars
python3 eval.py --batch $batch --gamma $gamma --factor $factor --block $block --maxv $maxv --filename $filename --burn_in $burn_in --iters $iters --lags $lags &

gamma=1.0
factor=20
block=20
maxv=2
burn_in=200
iters=10000
lags=400
filename=$PWD'/../UAI/UAI_BN/BN_8.uai' # 100 vars
python3 eval.py --batch $batch --gamma $gamma --factor $factor --block $block --maxv $maxv --filename $filename --burn_in $burn_in --iters $iters --lags $lags &

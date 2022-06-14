#!/bin/bash
rm -r __pycache__

cuda=0
batch=30
n=30

# model 1
lamda=0.0
mu=1.0
sigma=3.0
hidden=10
maxv=2
burn_in=1000
iters=30000
lags=500
python3 eval.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --burn_in $burn_in --iters $iters --lags $lags

#---------------------------------

# model 2
lamda=0.0
mu=3.0
sigma=3.0
burn_in=400
lags=400
python3 eval.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --burn_in $burn_in --iters $iters --lags $lags

#---------------------------------

# model 3
lamda=1.0
mu=1.0
sigma=3.0
burn_in=1000
lags=500
python3 eval.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --burn_in $burn_in --iters $iters --lags $lags

#---------------------------------

# model 4
lamda=1.0
mu=3.0
sigma=3.0
burn_in=400
lags=100
python3 eval.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --burn_in $burn_in --iters $iters --lags $lags

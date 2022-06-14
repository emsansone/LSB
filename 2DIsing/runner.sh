#!/bin/bash
rm -r __pycache__
rm -r exp*

cuda=0
batch=30
n=30

lr=1e-2

# model 1
lamda=0.0
mu=1.0
sigma=3.0
hidden=10
maxv=2
skip=100
burnin=2000 
iters=30000
python3 runner.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --skip $skip --burnin $burnin --iters $iters --lr $lr

#---------------------------------

# model 2
lamda=0.0
mu=3.0
sigma=3.0
burnin=800
python3 runner.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --skip $skip --burnin $burnin --iters $iters --lr $lr

#---------------------------------

# model 3
lamda=1.0
mu=1.0
sigma=3.0
burnin=2000 
python3 runner.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --skip $skip --burnin $burnin --iters $iters --lr $lr

#---------------------------------

# model 4
lamda=1.0
mu=3.0
sigma=3.0
burnin=800
python3 runner.py --cuda $cuda --batch $batch --n $n --lamda $lamda --mu $mu --sigma $sigma --hidden $hidden --maxv $maxv --skip $skip --burnin $burnin --iters $iters --lr $lr

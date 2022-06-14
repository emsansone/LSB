# Readme

Official implementation of the paper "LSB: Local Self-Balancinng MCMC in Discrete Spaces" accepted for presentation at ICML 2022.

## Prerequisites
Checking/Installing prerequisite libraries:

> python 3.8.5 \
> numpy 1.19.5 \
> statsmodels 0.13.0 \
> pgmpy 0.1.16 \
> torch 1.8.1

Other libraries include matplotlib, seaborn, pandas, tqdm and tensorflow_probability.

## Experiments on 2D Ising

To run the simulations, run the bash script `runner.sh`

To evaluate the sampler, run the script `eval.sh`


## Experiments on RBM

Code adapted from [Gibbs-With-Gradients, ICML 2021](https://github.com/wgrathwohl/GWG_release).
To run the simulations, run the bash script `rbm_sample.sh`

## Experiments on UAI

Data can be collected from [link](http://sli.ics.uci.edu/~ihler/uai-data/).
To run the simulations, follow the same procedure for Ising


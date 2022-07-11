# Readme

Official implementation of the paper [LSB: Local Self-Balancinng MCMC in Discrete Spaces](https://proceedings.mlr.press/v162/sansone22a.html) accepted for presentation at ICML 2022.


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

## Experiments on UAI Data

Data can be collected from [link](http://sli.ics.uci.edu/~ihler/uai-data/).
To run the simulations, follow the same procedure for experiments on Ising

## Citation

Please cite our paper as:

```
@inproceedings{sansone2022lsb,
	title = {{LSB}: Local Self-Balancing {MCMC} in Discrete Spaces},
	author = {Sansone, Emanuele},
	booktitle = {Proceedings of the 39th International Conference on Machine Learning},
	pages = {19205--19220},
	year = {2022},
}
```


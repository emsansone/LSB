import argparse
import os
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='First set of experiments for 2D Ising model')

parser.add_argument('--cuda', action='store', required=True, type=int, default=0, help='0 - CPU, 1 - GPU')
parser.add_argument('--batch', action='store', required=True, type=int, default=10, help='Batch size')
parser.add_argument('--n', action='store', required=True, type=int, default=100, help='Grid size')
parser.add_argument('--lamda', action='store', required=True, type=float, default=1.0, help='Lambda')
parser.add_argument('--mu', action='store', required=True, type=float, default=3.0, help='Mu')
parser.add_argument('--sigma', action='store', required=True, type=float, default=3.0, help='Sigma')
parser.add_argument('--hidden', action='store', required=True, type=int, default=10, help='Number of hidden neurons')
parser.add_argument('--maxv', action='store', required=True, type=int, default=2, help='Max value')
parser.add_argument('--burn_in', action='store', required=True, type=int, default=1000, help='Burn-in iterations')
parser.add_argument('--iters', action='store', required=True, type=int, default=30000, help='Iterations')
parser.add_argument('--lags', action='store', required=True, type=int, default=500, help='Maximum lag for autocorrelation')

'''
Main function
'''
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    from statsmodels.tsa.stattools import acf

    from sampler import *
    from self_sampler import *

    args = parser.parse_args()

    dtype = torch.float
    if args.cuda == 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    batch = args.batch
    n = args.n
    lamda = args.lamda
    mu = args.mu
    sigma = args.sigma
    limits = (-6.,6.) # min and max values for alpha
    hidden = args.hidden
    max_value = args.maxv

    burn_in_iters = args.burn_in
    iters = args.iters
    min_lags = args.lags

    name = 'exp_ba{}_n{}_la{}_mu{}_si{}'.format(batch, n, lamda, mu, sigma)
    path = os.path.join('.', name)
    name = 'ma{}'.format(max_value)
    path = os.path.join(path, name)

    print('Experiment {}'.format(path))

    font_size = 24
    marker_size = 16

    print('Burn-in Analysis')
    summary = torch.load(os.path.join(path,'Summary.pt'))
    summary_barker = summary['barker'][:, :burn_in_iters + 1]
    summary_sqrt = summary['sqrt'][:, :burn_in_iters + 1]
    summary_min = summary['min'][:, :burn_in_iters + 1]
    summary_max = summary['max'][:, :burn_in_iters + 1]
    summary_lsb1 = summary['lsb1'][:, :burn_in_iters + 1]
    summary_lsb2 = summary['lsb2'][:, :burn_in_iters + 1]


    df = pd.DataFrame(summary_barker.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='b')
    df = pd.DataFrame(summary_sqrt.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='c')
    df = pd.DataFrame(summary_min.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='g')
    df = pd.DataFrame(summary_max.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='k')
    df = pd.DataFrame(summary_lsb1.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='o', markersize=marker_size, markevery=burn_in_iters//10)
    df = pd.DataFrame(summary_lsb2.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='s', markersize=marker_size, markevery=burn_in_iters//10)

    #plt.legend(prop={'size': font_size})
    plt.title('Traceplot (Burn-in)', fontsize=font_size)
    plt.xlabel('Target Evaluations', fontsize=font_size)
    plt.ylabel('Summary statistics', fontsize=font_size)
    plt.savefig(os.path.join(path, 'Burn-in.png'))
    plt.close()

    print('Mixing Analysis: Traceplots')
    summary = torch.load(os.path.join(path,'Summary.pt'))
    summary_barker = summary['barker'][:, burn_in_iters + 1:]
    summary_sqrt = summary['sqrt'][:, burn_in_iters + 1:]
    summary_min = summary['min'][:, burn_in_iters + 1:]
    summary_max = summary['max'][:, burn_in_iters + 1:]
    summary_lsb1 = summary['lsb1'][:, burn_in_iters + 1:]
    summary_lsb2 = summary['lsb2'][:, burn_in_iters + 1:]

    plt.subplot(231)
    df = pd.DataFrame(summary_barker.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='b')
    plt.xlabel('Target Evaluations', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\frac{t}{1+t}$', fontsize='xx-large')
    plt.subplot(232)
    df = pd.DataFrame(summary_sqrt.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='c')
    plt.xlabel('Target Evaluations', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\sqrt{t}$', fontsize='xx-large')
    plt.subplot(233)
    df = pd.DataFrame(summary_min.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='g')
    plt.xlabel('Target Evaluations', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\min\{1,t\}$', fontsize='xx-large')
    plt.subplot(234)
    df = pd.DataFrame(summary_max.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='k')
    plt.xlabel('Target Evaluations', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\max\{1,t\}$', fontsize='xx-large')
    plt.subplot(235)
    df = pd.DataFrame(summary_lsb1.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='o', markevery=iters//10)
    plt.xlabel('Target Evaluations', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title('LSB 1', fontsize='xx-large')
    plt.subplot(236)
    df = pd.DataFrame(summary_lsb2.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='s', markevery=iters//10)
    plt.xlabel('Target Evaluations', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title('LSB 2', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'Traceplots.png'))
    plt.close()

    print('Mixing Analysis: Autocorrelation Functions')
    lags = min(min_lags, iters)
    auto_corr_func_barker = torch.zeros((batch, lags + 1))
    auto_corr_func_sqrt = torch.zeros((batch, lags + 1))
    auto_corr_func_min = torch.zeros((batch, lags + 1))
    auto_corr_func_max = torch.zeros((batch, lags + 1))
    auto_corr_func_lsb1 = torch.zeros((batch, lags + 1))
    auto_corr_func_lsb2 = torch.zeros((batch, lags + 1))
    for i in range(batch):
        auto_corr_func_barker[i, :] = torch.tensor(acf(summary_barker[i,:], nlags=lags))
        auto_corr_func_sqrt[i, :] = torch.tensor(acf(summary_sqrt[i,:], nlags=lags))
        auto_corr_func_min[i, :] = torch.tensor(acf(summary_min[i,:], nlags=lags))
        auto_corr_func_max[i, :] = torch.tensor(acf(summary_max[i,:], nlags=lags))
        auto_corr_func_lsb1[i, :] = torch.tensor(acf(summary_lsb1[i,:], nlags=lags))
        auto_corr_func_lsb2[i, :] = torch.tensor(acf(summary_lsb2[i,:], nlags=lags))

    df = pd.DataFrame(auto_corr_func_barker.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='b')
    df = pd.DataFrame(auto_corr_func_sqrt.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='c')
    df = pd.DataFrame(auto_corr_func_min.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='g')
    df = pd.DataFrame(auto_corr_func_max.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='k')
    df = pd.DataFrame(auto_corr_func_lsb1.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='o', markersize=marker_size, markevery=lags//10)
    df = pd.DataFrame(auto_corr_func_lsb2.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='s', markersize=marker_size, markevery=lags//10)

    #plt.legend(prop={'size': font_size})
    plt.title('Autocorrelation function', fontsize=font_size)
    plt.xlabel('Target Evaluations', fontsize=font_size)
    plt.ylabel('ACF', fontsize=font_size)
    plt.savefig(os.path.join(path, 'Autocorrelation.png'))
    plt.close()

    accept = torch.load(os.path.join(path,'Accept.pt'))
    accept_barker = accept['barker']
    accept_sqrt= accept['sqrt']
    accept_min = accept['min']
    accept_max = accept['max']
    accept_lsb1 = accept['lsb1']
    accept_lsb2 = accept['lsb2']

    from ess import *

    filename = 'Results.csv'
    with open(filename, 'a') as myfile:
        mean, std = effective_sample_size(auto_corr_func_barker.numpy())
        row = '{},{},{},{},{},{},{},{},barker,{:.4f},{:.4f}\n'.format(batch, n, lamda, mu, sigma, \
                                                    hidden, max_value, \
                                                    torch.mean(accept_barker), mean, std)
        myfile.write(row)
        mean, std = effective_sample_size(auto_corr_func_sqrt.numpy())
        row = '{},{},{},{},{},{},{},{},sqrt,{:.4f},{:.4f}\n'.format(batch, n, lamda, mu, sigma, \
                                                    hidden, max_value, \
                                                    torch.mean(accept_sqrt), mean, std)
        myfile.write(row)
        mean, std = effective_sample_size(auto_corr_func_min.numpy())
        row = '{},{},{},{},{},{},{},{},min,{:.4f},{:.4f}\n'.format(batch, n, lamda, mu, sigma, \
                                                    hidden, max_value, \
                                                    torch.mean(accept_min), mean, std)
        myfile.write(row)
        mean, std = effective_sample_size(auto_corr_func_max.numpy())
        row = '{},{},{},{},{},{},{},{},max,{:.4f},{:.4f}\n'.format(batch, n, lamda, mu, sigma, \
                                                    hidden, max_value, \
                                                    torch.mean(accept_max), mean, std)
        myfile.write(row)
        mean, std = effective_sample_size(auto_corr_func_lsb1.numpy())
        row = '{},{},{},{},{},{},{},{},lsb1,{:.4f},{:.4f}\n'.format(batch, n, lamda, mu, sigma, \
                                                    hidden, max_value, \
                                                    torch.mean(accept_lsb1), mean, std)
        myfile.write(row)
        mean, std = effective_sample_size(auto_corr_func_lsb2.numpy())
        row = '{},{},{},{},{},{},{},{},lsb2,{:.4f},{:.4f}\n'.format(batch, n, lamda, mu, sigma, \
                                                    hidden, max_value, \
                                                    torch.mean(accept_lsb2), mean, std)
        myfile.write(row)


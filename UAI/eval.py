import argparse
import os
from datetime import datetime
import multiprocess as mp
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='First set of experiments for 2D Ising model')

parser.add_argument('--batch', action='store', required=True, type=int, default=10, help='Batch size')
parser.add_argument('--gamma', action='store', required=True, type=float, default=1., help='Gamma (scalar for mononic regularizer')
parser.add_argument('--factor', action='store', required=True, type=int, default=20, help='Number of blocks')
parser.add_argument('--block', action='store', required=True, type=int, default=20, help='Block size')
parser.add_argument('--maxv', action='store', required=True, type=int, default=2, help='Max value')
parser.add_argument('--filename', action='store', required=True, help='Filename for the dataset')
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

    batch = args.batch
    gamma = args.gamma
    factor = args.factor
    block = args.block
    max_value = args.maxv
    filename = args.filename

    burn_in_iters = args.burn_in
    iters = args.iters
    min_lags = args.lags

    name = (filename.split('/')[-1] + '_ba{}').format(batch)
    path = os.path.join('.', name)
    name = 'ga{}_fa{}_bl{}_ma{}'.format(gamma, factor, block, max_value)
    path = os.path.join(path, name)

    print('Experiment {}'.format(path))

    font_size = 24
    marker_size = 14

    print('Loading dataset')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    ##### DEFINE DISTRIBUTION TO SAMPLE FROM (CURRENTLY ONLY BINARY)
    from pgmpy.readwrite import UAIReader
    from pgmpy.models import MarkovModel
    reader = UAIReader(filename)
    model = reader.get_model()
    model.check_model()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Final Loading Time =", current_time)

    from reader import EvidReader
    evidence = EvidReader(filename + '.evid').get_evidence()

    print('Burn-in Analysis')
    summary = torch.load(os.path.join(path,'Summary_burnin.pt'))
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

    # plt.legend(prop={'size': font_size})
    plt.title('Traceplot (Burn-in)', fontsize=font_size)
    plt.xlabel('Markov Chain Time', fontsize=font_size)
    plt.ylabel('Summary statistics', fontsize=font_size)
    plt.savefig(os.path.join(path, 'Burn-in.png'))
    plt.close()

    print('Mixing Analysis: Traceplots')
    summary = torch.load(os.path.join(path,'Summary.pt'))
    summary_barker = summary['barker'][:, :iters + 1]
    summary_sqrt = summary['sqrt'][:, :iters + 1]
    summary_min = summary['min'][:, :iters + 1]
    summary_max = summary['max'][:, :iters + 1]
    summary_lsb1 = summary['lsb1'][:, :iters + 1]
    summary_lsb2 = summary['lsb2'][:, :iters + 1]

    plt.subplot(231)
    df = pd.DataFrame(summary_barker.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='b')
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\frac{t}{1+t}$', fontsize='xx-large')
    plt.subplot(232)
    df = pd.DataFrame(summary_sqrt.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='c')
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\sqrt{t}$', fontsize='xx-large')
    plt.subplot(233)
    df = pd.DataFrame(summary_min.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='g')
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\min\{1,t\}$', fontsize='xx-large')
    plt.subplot(234)
    df = pd.DataFrame(summary_max.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='k')
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title(r'$\max\{1,t\}$', fontsize='xx-large')
    plt.subplot(235)
    df = pd.DataFrame(summary_lsb1.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='o', markevery=iters//10)
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title('LSB 1', fontsize='xx-large')
    plt.subplot(236)
    df = pd.DataFrame(summary_lsb2.detach().numpy()).melt()
    sns.lineplot(x="variable", y="value", data=df, color='r', marker='s', markevery=iters//10)
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.ylabel('Summary statistics', fontsize='xx-large')
    plt.title('LSB 2', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'Traceplots.png'))
    plt.close()

    lags = min(min_lags, iters)
    
    print('Mixing Analysis: Autocorrelation Functions')
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

    # plt.legend(prop={'size': font_size})
    plt.title('Autocorrelation function', fontsize=font_size)
    plt.xlabel('Markov Chain Time', fontsize=font_size)
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
    from partition import *

    summary = torch.load(os.path.join(path,'Samples.pt'))
    sample_barker = summary['barker']
    sample_sqrt = summary['sqrt']
    sample_min = summary['min']
    sample_max = summary['max']
    sample_lsb1 = summary['lsb1']
    sample_lsb2 = summary['lsb2']

    filename = 'Results.csv'
    with open(filename, 'a') as myfile:
        with mp.Pool(processes=batch) as pool:
            z = pool.starmap(log_partition, zip(sample_barker, [model for _ in range(batch)], [evidence for _ in range(batch)]))
        mean_z, std_z = np.mean(z), np.std(z)
        mean, std = effective_sample_size(auto_corr_func_barker.numpy())
        row = '{},{},{},{},{},{},barker,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(batch, \
                                                    gamma, factor, block, max_value, \
                                                    torch.mean(accept_barker), mean, std, mean_z, std_z)
        myfile.write(row)
        with mp.Pool(processes=batch) as pool:
            z = pool.starmap(log_partition, zip(sample_sqrt, [model for _ in range(batch)], [evidence for _ in range(batch)]))
        mean_z, std_z = np.mean(z), np.std(z)
        mean, std = effective_sample_size(auto_corr_func_sqrt.numpy())
        row = '{},{},{},{},{},{},sqrt,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(batch, \
                                                    gamma, factor, block, max_value, \
                                                    torch.mean(accept_sqrt), mean, std, mean_z, std_z)
        myfile.write(row)
        with mp.Pool(processes=batch) as pool:
            z = pool.starmap(log_partition, zip(sample_min, [model for _ in range(batch)], [evidence for _ in range(batch)]))
        mean_z, std_z = np.mean(z), np.std(z)
        mean, std = effective_sample_size(auto_corr_func_min.numpy())
        row = '{},{},{},{},{},{},min,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(batch, \
                                                    gamma, factor, block, max_value, \
                                                    torch.mean(accept_min), mean, std, mean_z, std_z)
        myfile.write(row)
        with mp.Pool(processes=batch) as pool:
            z = pool.starmap(log_partition, zip(sample_max, [model for _ in range(batch)], [evidence for _ in range(batch)]))
        mean_z, std_z = np.mean(z), np.std(z)
        mean, std = effective_sample_size(auto_corr_func_max.numpy())
        row = '{},{},{},{},{},{},max,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(batch, \
                                                    gamma, factor, block, max_value, \
                                                    torch.mean(accept_max), mean, std, mean_z, std_z)
        myfile.write(row)
        with mp.Pool(processes=batch) as pool:
            z = pool.starmap(log_partition, zip(sample_lsb1, [model for _ in range(batch)], [evidence for _ in range(batch)]))
        mean_z, std_z = np.mean(z), np.std(z)
        mean, std = effective_sample_size(auto_corr_func_lsb1.numpy())
        row = '{},{},{},{},{},{},lsb1,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(batch, \
                                                    gamma, factor, block, max_value, \
                                                    torch.mean(accept_lsb1), mean, std, mean_z, std_z)
        myfile.write(row)
        with mp.Pool(processes=batch) as pool:
            z = pool.starmap(log_partition, zip(sample_lsb2, [model for _ in range(batch)], [evidence for _ in range(batch)]))
        mean_z, std_z = np.mean(z), np.std(z)
        mean, std = effective_sample_size(auto_corr_func_lsb2.numpy())
        row = '{},{},{},{},{},{},lsb2,{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(batch, \
                                                    gamma, factor, block, max_value, \
                                                    torch.mean(accept_lsb2), mean, std, mean_z, std_z)
        myfile.write(row)


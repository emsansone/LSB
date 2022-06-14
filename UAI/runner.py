import argparse
import os
import numpy as np
import random
from datetime import datetime

parser = argparse.ArgumentParser(description='Experiments on UAI benchmark datasets (only Markov Nets for now)')

parser.add_argument('--batch', action='store', required=True, type=int, default=10, help='Batch size')
parser.add_argument('--gamma', action='store', required=True, type=float, default=1., help='Gamma (scalar for mononic regularizer')
parser.add_argument('--factor', action='store', required=True, type=int, default=20, help='Number of blocks')
parser.add_argument('--block', action='store', required=True, type=int, default=20, help='Block size')
parser.add_argument('--maxv', action='store', required=True, type=int, default=2, help='Max value')
parser.add_argument('--skip', action='store', required=True, type=int, default=1000, help='Skip printing')
parser.add_argument('--burnin', action='store', required=False, type=int, default=1000, help='Burn-in iterations')
parser.add_argument('--iters', action='store', required=False, type=int, default=1000, help='Sampling iterations')
parser.add_argument('--filename', action='store', required=True, help='Filename for the dataset')

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

    skip = args.skip
    burn_in_iters = args.burnin
    iters = args.iters

    font_size = 20

    print('Experiment on ', filename.split('/')[-1])

    name = (filename.split('/')[-1] + '_ba{}').format(batch)
    path = os.path.join('.', name)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    name = 'ga{}_fa{}_bl{}_ma{}'.format(gamma, factor, block, max_value)
    path = os.path.join(path, name)
    if os.path.isdir(path) == False:
        os.mkdir(path)

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

    ##### GET INITIAL STATE
    start_x = []
    for i in range(batch):
        start_x.append({})
        for j in reader.variables:
            start_x[i][j] = 0

    ##### GET EVIDENCE AND SET THE STATE ACCORDINGLY
    from reader import EvidReader
    evidence = EvidReader(filename + '.evid').get_evidence()
    for i in range(batch):
        for j in start_x[i].keys():
            if j in evidence.keys():
                start_x[i][j] = evidence[j]

    ##### DEFINE SAMPLERS
    samp_barker = LBS(batch=batch, func='barker', start_x=start_x, model=model)
    samp_sqrt = LBS(batch=batch, func='sqrt', start_x=start_x, model=model)
    samp_min = LBS(batch=batch, func='min', start_x=start_x, model=model)
    samp_max = LBS(batch=batch, func='max', start_x=start_x, model=model)
    samp_lsb1 = LSB(batch=batch, func='lsb1', lr=1e-4, start_x=start_x, gamma=gamma, factor=factor, block=block, max_value=max_value, model=model, evidence=evidence)
    samp_lsb2 = LSB(batch=batch, func='lsb2', lr=1e-4, start_x=start_x, gamma=gamma, factor=factor, block=block, max_value=max_value, model=model, evidence=evidence)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time (Burn-in) =", current_time)

    ##### BURN-IN
    batch = batch // 2 + batch % 2
    summary_barker = torch.zeros((batch, burn_in_iters + 1))
    summary_sqrt = torch.zeros((batch, burn_in_iters + 1))
    summary_min = torch.zeros((batch, burn_in_iters + 1))
    summary_max = torch.zeros((batch, burn_in_iters + 1))
    summary_lsb1 = torch.zeros((batch, burn_in_iters + 1))
    summary_lsb2 = torch.zeros((batch, burn_in_iters + 1))
    for b in range(batch):
        summary_barker[b, 0] = np.sum([samp_barker.x[b][var] for var in reader.variables])
        summary_sqrt[b, 0] = np.sum([samp_sqrt.x[b][var] for var in reader.variables])
        summary_min[b, 0] = np.sum([samp_min.x[b][var] for var in reader.variables])
        summary_max[b, 0] = np.sum([samp_max.x[b][var] for var in reader.variables])
        summary_lsb1[b, 0] = np.sum([samp_lsb1.x[b][var] for var in reader.variables])
        summary_lsb2[b, 0] = np.sum([samp_lsb2.x[b][var] for var in reader.variables])

    objective1 = torch.zeros((burn_in_iters,))
    objective2 = torch.zeros((burn_in_iters,))

    for i in range(burn_in_iters):
        samp_barker.sample()
        samp_sqrt.sample()
        samp_min.sample()
        samp_max.sample()
        samp_lsb1.sample_burn_in()
        samp_lsb2.sample_burn_in()
        samp_lsb2.sample_burn_in()
        obj1 = samp_lsb1.obj
        obj2 = samp_lsb2.obj

        for b in range(batch):
            summary_barker[b, i + 1] = np.sum([samp_barker.x[b][var] for var in reader.variables])
            summary_sqrt[b, i + 1] = np.sum([samp_sqrt.x[b][var] for var in reader.variables])
            summary_min[b, i + 1] = np.sum([samp_min.x[b][var] for var in reader.variables])
            summary_max[b, i + 1] = np.sum([samp_max.x[b][var] for var in reader.variables])
            summary_lsb1[b, i + 1] = np.sum([samp_lsb1.x[b][var] for var in reader.variables])
            summary_lsb2[b, i + 1] = np.sum([samp_lsb2.x[b][var] for var in reader.variables])

        if obj1 is not None:
            objective1[i] = obj1.detach()
        if obj2 is not None:
            objective2[i] = obj2.detach()

        if i % skip == 0:
            print('Theta 1 {}'.format(samp_lsb1.theta))
            print('Iteration {} Objective 1 {} Objective 2 {}'.format(i, obj1.detach().item(), obj2.detach().item()))

            plt.plot(objective1)
            plt.xlabel('Markov Chain Time', fontsize='xx-large')
            plt.title('Objective function for LSB 1')
            plt.savefig(os.path.join(path, 'Objective1.png'))
            plt.close()
            plt.plot(objective2)
            plt.xlabel('Markov Chain Time', fontsize='xx-large')
            plt.title('Objective function for LSB 2')
            plt.savefig(os.path.join(path, 'Objective2.png'))
            plt.close()

            data = torch.arange(start = 0., end = samp_lsb1.max_value + 0.1, step = 0.1)
            output = samp_barker.func(data).detach().cpu()
            plt.plot(data.detach().cpu(), output, 'b-', label=r'$\frac{t}{1+t}$')
            output = samp_sqrt.func(data).detach().cpu()
            plt.plot(data.detach().cpu(), output, 'c-', label=r'$\sqrt{t}$')
            output = samp_min.func(data).detach().cpu()
            plt.plot(data.detach().cpu(), output, 'g-', label=r'$\min\{1,t\}$')
            output = samp_max.func(data).detach().cpu()
            plt.plot(data.detach().cpu(), output, 'k-', label=r'$\max\{1,t\}$')
            output = samp_lsb1.func(data).detach().cpu()
            plt.plot(data.detach().cpu(), output, 'ro-', label='LSB 1')
            output = samp_lsb2.func(data).detach().cpu()
            plt.plot(data.detach().cpu(), output, 'rs-', label='LSB 2')
            plt.title('Balancing functions')
            plt.legend(prop={'size': font_size}, ncol=2)
            plt.savefig(os.path.join(path, 'Functions_i{}.png'.format(i)))
            plt.close()

    state_dict = samp_lsb2.mono.state_dict()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Final Time (Burn-in) =", current_time)

    plt.plot(objective1)
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.title('Objective function for LSB 1')
    plt.savefig(os.path.join(path, 'Objective1.png'))
    plt.close()
    plt.plot(objective2)
    plt.xlabel('Markov Chain Time', fontsize='xx-large')
    plt.title('Objective function for LSB 2')
    plt.savefig(os.path.join(path, 'Objective2.png'))
    plt.close()

    data = torch.arange(start = 0., end = samp_lsb1.max_value + 0.1, step = 0.1)
    output = samp_barker.func(data).detach().cpu()
    plt.plot(data.detach().cpu(), output, 'b-', label=r'$\frac{t}{1+t}$')
    output = samp_sqrt.func(data).detach().cpu()
    plt.plot(data.detach().cpu(), output, 'c-', label=r'$\sqrt{t}$')
    output = samp_min.func(data).detach().cpu()
    plt.plot(data.detach().cpu(), output, 'g-', label=r'$\min\{1,t\}$')
    output = samp_max.func(data).detach().cpu()
    plt.plot(data.detach().cpu(), output, 'k-', label=r'$\max\{1,t\}$')
    output = samp_lsb1.func(data).detach().cpu()
    plt.plot(data.detach().cpu(), output, 'ro-', label='LSB 1')
    output = samp_lsb2.func(data).detach().cpu()
    plt.plot(data.detach().cpu(), output, 'rs-', label='LSB 2')
    plt.title('Balancing functions')
    plt.legend(prop={'size': font_size}, ncol=2)
    plt.savefig(os.path.join(path, 'Functions.png'))
    plt.close()

    summary = {
        'barker': summary_barker,
        'sqrt': summary_sqrt,
        'min': summary_min,
        'max': summary_max,
        'lsb1': summary_lsb1,
        'lsb2': summary_lsb2
    }

    torch.save(summary, os.path.join(path,'Summary_burnin.pt'))

    batch = args.batch
    ##### DEFINE SAMPLERS
    start_x = samp_sqrt.x
    samp_barker = LBS(batch=batch, func='barker', start_x=start_x, model=model)
    samp_sqrt = LBS(batch=batch, func='sqrt', start_x=start_x, model=model)
    samp_min = LBS(batch=batch, func='min', start_x=start_x, model=model)
    samp_max = LBS(batch=batch, func='max', start_x=start_x, model=model)
    theta = samp_lsb1.theta
    samp_lsb1 = LSB(batch=batch, func='lsb1', lr=1e-4, start_x=start_x, theta=theta, gamma=gamma, factor=factor, block=block, max_value=max_value, model=model, evidence=evidence, burn_in=False)
    theta = state_dict
    samp_lsb2 = LSB(batch=batch, func='lsb2', lr=1e-4, start_x=start_x, theta=theta, gamma=gamma, factor=factor, block=block, max_value=max_value, model=model, evidence=evidence, burn_in=False)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time (Mixing) =", current_time)

    ##### SAMPLING
    summary_barker = torch.zeros((batch, iters + 1))
    summary_sqrt = torch.zeros((batch, iters + 1))
    summary_min = torch.zeros((batch, iters + 1))
    summary_max = torch.zeros((batch, iters + 1))
    summary_lsb1 = torch.zeros((batch, iters + 1))
    summary_lsb2 = torch.zeros((batch, iters + 1))
    for b in range(batch):
        summary_barker[b, 0] = np.sum([samp_barker.x[b][var] for var in reader.variables])
        summary_sqrt[b, 0] = np.sum([samp_sqrt.x[b][var] for var in reader.variables])
        summary_min[b, 0] = np.sum([samp_min.x[b][var] for var in reader.variables])
        summary_max[b, 0] = np.sum([samp_max.x[b][var] for var in reader.variables])
        summary_lsb1[b, 0] = np.sum([samp_lsb1.x[b][var] for var in reader.variables])
        summary_lsb2[b, 0] = np.sum([samp_lsb2.x[b][var] for var in reader.variables])
    accept_barker = torch.zeros((batch, iters))
    accept_sqrt= torch.zeros((batch, iters))
    accept_min = torch.zeros((batch, iters))
    accept_max = torch.zeros((batch, iters))
    accept_lsb1 = torch.zeros((batch, iters))
    accept_lsb2 = torch.zeros((batch, iters))

    sample_barker = []
    sample_sqrt = []
    sample_min = []
    sample_max = []
    sample_lsb1 = []
    sample_lsb2 = []
    for i in range(batch):
        sample_barker.append([])
        sample_sqrt.append([])
        sample_min.append([])
        sample_max.append([])
        sample_lsb1.append([])
        sample_lsb2.append([])

    for i in range(iters):
        samp_barker.sample()
        samp_sqrt.sample()
        samp_min.sample()
        samp_max.sample()
        samp_lsb1.sample()
        samp_lsb2.sample()
        for b in range(batch):
            summary_barker[b, i + 1] = np.sum([samp_barker.x[b][var] for var in reader.variables])
            summary_sqrt[b, i + 1] = np.sum([samp_sqrt.x[b][var] for var in reader.variables])
            summary_min[b, i + 1] = np.sum([samp_min.x[b][var] for var in reader.variables])
            summary_max[b, i + 1] = np.sum([samp_max.x[b][var] for var in reader.variables])
            summary_lsb1[b, i + 1] = np.sum([samp_lsb1.x[b][var] for var in reader.variables])
            summary_lsb2[b, i + 1] = np.sum([samp_lsb2.x[b][var] for var in reader.variables])
        accept_barker[:, i] = samp_barker.accept
        accept_sqrt[:, i] = samp_sqrt.accept
        accept_min[:, i] = samp_min.accept
        accept_max[:, i] = samp_max.accept
        accept_lsb1[:, i] = samp_lsb1.accept.detach()
        accept_lsb2[:, i] = samp_lsb2.accept.detach()

        if i % skip == 0:
            print('Iteration {}'.format(i))
            for b in range(batch):
                sample_barker[b].append(copy.deepcopy(samp_barker.x[b]))
                sample_sqrt[b].append(copy.deepcopy(samp_sqrt.x[b]))
                sample_min[b].append(copy.deepcopy(samp_min.x[b]))
                sample_max[b].append(copy.deepcopy(samp_max.x[b]))
                sample_lsb1[b].append(copy.deepcopy(samp_lsb1.x[b]))
                sample_lsb2[b].append(copy.deepcopy(samp_lsb2.x[b]))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Final Time (Mixing) =", current_time)

    summary_new = {
        'barker': summary_barker,
        'sqrt': summary_sqrt,
        'min': summary_min,
        'max': summary_max,
        'lsb1': summary_lsb1,
        'lsb2': summary_lsb2
    }

    torch.save(summary_new, os.path.join(path,'Summary.pt'))

    accept = {
        'barker': accept_barker,
        'sqrt': accept_sqrt,
        'min': accept_min,
        'max': accept_max,
        'lsb1': accept_lsb1,
        'lsb2': accept_lsb2
    }

    torch.save(accept, os.path.join(path,'Accept.pt'))

    samples = {
        'barker': sample_barker,
        'sqrt': sample_sqrt,
        'min': sample_min,
        'max': sample_max,
        'lsb1': sample_lsb1,
        'lsb2': sample_lsb2
    }

    torch.save(samples, os.path.join(path,'Samples.pt'))

import argparse
import os

seed=12345678

parser = argparse.ArgumentParser(description='First set of experiments for 2D Ising model')

parser.add_argument('--cuda', action='store', required=True, type=int, default=0, help='0 - CPU, 1 - GPU')
parser.add_argument('--batch', action='store', required=True, type=int, default=10, help='Batch size')
parser.add_argument('--n', action='store', required=True, type=int, default=100, help='Grid size')
parser.add_argument('--lamda', action='store', required=True, type=float, default=1.0, help='Lambda')
parser.add_argument('--mu', action='store', required=True, type=float, default=3.0, help='Mu')
parser.add_argument('--sigma', action='store', required=True, type=float, default=3.0, help='Sigma')
parser.add_argument('--lr', action='store', required=True, type=float, default=1e-2, help='Learning rate')
parser.add_argument('--hidden', action='store', required=True, type=int, default=1., help='Number of hidden neurons')
parser.add_argument('--maxv', action='store', required=True, type=int, default=2, help='Max value')
parser.add_argument('--skip', action='store', required=True, type=int, default=1000, help='Skip printing')
parser.add_argument('--burnin', action='store', required=False, type=int, default=1000, help='Burn-in iterations')
parser.add_argument('--iters', action='store', required=False, type=int, default=1000, help='Sampling iterations')


'''
Main function
'''
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    from statsmodels.tsa.stattools import acf

    from sampler import *
    from self_sampler import *

    torch.manual_seed(seed)

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
    max_value = args.maxv
    hidden = args.hidden
    lr = args.lr

    skip = args.skip
    burn_in_iters = args.burnin
    iters = args.iters

    name = 'exp_ba{}_n{}_la{}_mu{}_si{}'.format(batch, n, lamda, mu, sigma)
    path = os.path.join('.', name)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    name = 'ma{}'.format(max_value)
    path = os.path.join(path, name)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    
    ##### DEFINE ALPHA TENSOR
    obj = mu * torch.ones((batch, n, n))
    i = torch.arange(n)
    j = torch.arange(n)
    obj[:, i < torch.tensor(n // 4), :] = - torch.tensor(mu)
    obj[:, i > torch.tensor(n - (n // 4) - 1), :] = - torch.tensor(mu)
    obj[:, :, j < torch.tensor(n // 4)] = - torch.tensor(mu)
    obj[:, :, j > torch.tensor(n - (n // 4) - 1)] = - torch.tensor(mu)
    alpha = obj + 2 * sigma * torch.rand((batch, n, n)) - torch.tensor(sigma)

    ##### INITIALIZE VARIABLES
    start_x = torch.randint(2, (batch, n, n), device=device)*2 - 1

    ##### DEFINE SAMPLERS
    samp_barker = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='barker', start_x=start_x, device=device)
    samp_sqrt = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='sqrt', start_x=start_x, device=device)
    samp_min = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='min', start_x=start_x, device=device)
    samp_max = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='max', start_x=start_x, device=device)
    samp_lsb1 = LSBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='lsb1', lr=lr, start_x=start_x, hidden=hidden, max_value=max_value, device=device)
    samp_lsb2 = LSBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='lsb2', lr=lr, start_x=start_x, hidden=hidden, max_value=max_value, device=device)

    ##### BURN-IN
    # batch = batch // 2 + batch % 2
    summary_barker = torch.zeros((batch, burn_in_iters + iters + 1))
    summary_barker[:, 0] = torch.sum(samp_barker.x[:batch, :, :], [1, 2])
    summary_sqrt = torch.zeros((batch, burn_in_iters + iters + 1))
    summary_sqrt[:, 0] = torch.sum(samp_sqrt.x[:batch, :, :], [1, 2])
    summary_min = torch.zeros((batch, burn_in_iters + iters + 1))
    summary_min[:, 0] = torch.sum(samp_min.x[:batch, :, :], [1, 2])
    summary_max = torch.zeros((batch, burn_in_iters + iters + 1))
    summary_max[:, 0] = torch.sum(samp_max.x[:batch, :, :], [1, 2])
    summary_lsb1 = torch.zeros((batch, burn_in_iters + iters + 1))
    summary_lsb1[:, 0] = torch.sum(samp_lsb1.x[:batch, :, :], [1, 2])
    summary_lsb2 = torch.zeros((batch, burn_in_iters + iters + 1))
    summary_lsb2[:, 0] = torch.sum(samp_lsb2.x[:batch, :, :], [1, 2])

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

        summary_barker[:, i + 1] = torch.sum(samp_barker.x[:batch, :, :], [1, 2])
        summary_sqrt[:, i + 1] = torch.sum(samp_sqrt.x[:batch, :, :], [1, 2])
        summary_min[:, i + 1] = torch.sum(samp_min.x[:batch, :, :], [1, 2])
        summary_max[:, i + 1] = torch.sum(samp_max.x[:batch, :, :], [1, 2])
        summary_lsb1[:, i + 1] = torch.sum(samp_lsb1.x[:batch, :, :], [1, 2])
        summary_lsb2[:, i + 1] = torch.sum(samp_lsb2.x[:batch, :, :], [1, 2])

        if obj1 is not None:
            objective1[i] = obj1.detach()
        if obj2 is not None:
            objective2[i] = obj2.detach()

        if i % skip == 0:
            print('Iteration {} Objective1 {} Theta1 {}'.format(i, obj1.detach().item(), samp_lsb1.theta.detach()))
            plt.rcParams["axes.grid"] = False
            plt.subplot(231)
            img = plt.imshow(samp_barker.x[0,:,:].cpu())
            img.set_cmap('gray')
            plt.title(r'$\frac{t}{1+t}$')
            plt.subplot(232)
            img = plt.imshow(samp_sqrt.x[0,:,:].cpu())
            img.set_cmap('gray')
            plt.title(r'$\sqrt{t}$')
            plt.subplot(233)
            img = plt.imshow(samp_min.x[0,:,:].cpu())
            img.set_cmap('gray')
            plt.title(r'$\min\{1,t\}$')
            plt.subplot(234)
            img = plt.imshow(samp_max.x[0,:,:].cpu())
            img.set_cmap('gray')
            plt.title(r'$\max\{1,t\}$')
            plt.subplot(235)
            img = plt.imshow(samp_lsb1.x[0,:,:].cpu())
            img.set_cmap('gray')
            plt.title('LSB 1')
            plt.subplot(236)
            img = plt.imshow(samp_lsb2.x[0,:,:].cpu())
            img.set_cmap('gray')
            plt.title('LSB 2')
            plt.tight_layout()
            plt.savefig(os.path.join(path, 'burn_in_i{}.png'.format(i)))
            plt.close()

            # Analysis of the learnt function
            data = torch.arange(start = 0., end = samp_lsb1.max_value + 0.1, step = 0.1, device=device)
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
            plt.legend()
            plt.savefig(os.path.join(path, 'Functions_i{}.png'.format(i)))
            plt.close()

    state_dict = samp_lsb2.model.state_dict()

    plt.plot(objective1)
    plt.xlabel('Iteration')
    plt.title('Objective function for LSB 1')
    plt.savefig(os.path.join(path, 'Objective1.png'))
    plt.close()
    plt.plot(objective2)
    plt.xlabel('Iteration')
    plt.title('Objective function for LSB 2')
    plt.savefig(os.path.join(path, 'Objective2.png'))
    plt.close()

    data = torch.arange(start = 0., end = samp_lsb1.max_value + 0.1, step = 0.1, device=device)
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
    plt.legend()
    plt.savefig(os.path.join(path, 'Functions.png'))
    plt.close()

    batch = args.batch
    ##### DEFINE SAMPLERS
    start_x = samp_barker.x
    samp_barker = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='barker', start_x=start_x, device=device)
    start_x = samp_sqrt.x
    samp_sqrt = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='sqrt', start_x=start_x, device=device)
    start_x = samp_min.x
    samp_min = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='min', start_x=start_x, device=device)
    start_x = samp_max.x
    samp_max = LBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='max', start_x=start_x, device=device)
    start_x = samp_lsb1.x
    theta = samp_lsb1.theta
    samp_lsb1 = LSBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='lsb1', lr=lr, start_x=start_x, theta=theta, hidden=hidden, max_value=max_value, device=device, burn_in=False)
    start_x = samp_lsb2.x
    theta = state_dict
    samp_lsb2 = LSBS(batch=batch, n=n, lamda=lamda, alpha=alpha, func='lsb2', lr=lr, start_x=start_x, theta=theta, hidden=hidden, max_value=max_value, device=device, burn_in=False)

    ##### SAMPLING
    accept_barker = torch.zeros((batch, iters))
    accept_sqrt= torch.zeros((batch, iters))
    accept_min = torch.zeros((batch, iters))
    accept_max = torch.zeros((batch, iters))
    accept_lsb1 = torch.zeros((batch, iters))
    accept_lsb2 = torch.zeros((batch, iters))

    for i in range(iters):
        samp_barker.sample()
        samp_sqrt.sample()
        samp_min.sample()
        samp_max.sample()
        samp_lsb1.sample()
        samp_lsb2.sample()
        summary_barker[:, burn_in_iters + i + 1] = torch.sum(samp_barker.x, [1, 2])
        summary_sqrt[:, burn_in_iters + i + 1] = torch.sum(samp_sqrt.x, [1, 2])
        summary_min[:, burn_in_iters + i + 1] = torch.sum(samp_min.x, [1, 2])
        summary_max[:, burn_in_iters + i + 1] = torch.sum(samp_max.x, [1, 2])
        summary_lsb1[:, burn_in_iters + i + 1] = torch.sum(samp_lsb1.x, [1, 2])
        summary_lsb2[:, burn_in_iters + i + 1] = torch.sum(samp_lsb2.x, [1, 2])
        accept_barker[:, i] = samp_barker.accept
        accept_sqrt[:, i] = samp_sqrt.accept
        accept_min[:, i] = samp_min.accept
        accept_max[:, i] = samp_max.accept
        accept_lsb1[:, i] = samp_lsb1.accept.detach()
        accept_lsb2[:, i] = samp_lsb2.accept.detach()

        if i % skip == 0:
            print('Iteration {}'.format(i))

    summary = {
        'barker': summary_barker,
        'sqrt': summary_sqrt,
        'min': summary_min,
        'max': summary_max,
        'lsb1': summary_lsb1,
        'lsb2': summary_lsb2
    }

    torch.save(summary, os.path.join(path,'Summary.pt'))

    accept = {
        'barker': accept_barker,
        'sqrt': accept_sqrt,
        'min': accept_min,
        'max': accept_max,
        'lsb1': accept_lsb1,
        'lsb2': accept_lsb2
    }

    torch.save(accept, os.path.join(path,'Accept.pt'))

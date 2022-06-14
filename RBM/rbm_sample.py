import argparse
import rbm
import torch
import numpy as np
import samplers
import self_sampler
import mmd
import matplotlib.pyplot as plt
import os
device = torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')
import utils
import tensorflow_probability as tfp
import block_samplers
import time
import pickle


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)

    print('\n ---------- INITIALIZING PARAMETERS OF RBM ----------\n')

    if args.data == "mnist":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0) # [mnist_dataset_size, 784]
        init = 0.01 * torch.ones(args.n_visible)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        for x, _ in train_loader:
            x = x.to(device) # [batch_size, 784]

            xhat = model.gibbs_sample(v=x, n_steps=args.cd)

            d = model.logp_v_unnorm(x)
            m = model.logp_v_unnorm(xhat)

            obj = d - m
            loss = -obj.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr,d.mean(), m.mean(),
                                                                                               (d - m).mean()))
                if plot is not None:
                    plot("{}/training_{}.png".format(args.save_dir, itr), xhat)

            itr += 1

    else:
        model.W.data = torch.randn_like(model.W.data)# * (.05 ** .5)
        model.b_v.data = torch.randn_like(model.b_v.data)# * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data)# * 1.0
        viz = plot = None

    print('\n ---------- GETTING GROUND TRUTH SAMPLES FROM RBM ----------\n')

    # for mnist [n_samples + n_test_samples, 784]
    # for random [n_samples + n_test_samples, n_visible]
    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)

    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)

    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt mmd {} log-mmd {}".format(opt_stat.item(), opt_stat.log10().item()))

    print('\n ---------- RUNNING BLOCK GIBBS SAMPLING ALGORITHM OF RBM p(h|v),p(h|v) ----------\n')

    # Initialization of random samples
    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['rbm-gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []

    # Running simple Gibbs sampling and measuring the performance
    times['rbm-gibbs'] = []
    start_time = time.time()
    for i in range(args.n_steps):
        if i % args.print_every == 0:
            stat = kmmd.compute_mmd(new_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['rbm-gibbs'].append(log_stat)
            print("gt <--> gibbs iter {} mmd {} log-mmd {}".format(i, stat.item(), stat.log10().item()))
            times['rbm-gibbs'].append(time.time() - start_time)
        new_samples = model.gibbs_sample(new_samples, 1)
        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(new_samples.cpu().numpy()[0][None])
            else:
                xc = new_samples[0][None] # Take first sample by keeping dimension
                h = (xc != gt_samples).float().sum(-1) # Computing the Hamming distance wrt to the first sample and all gt_samples
                # for mnist h is a vector [n_samples]
                chain.append(h.detach().cpu().numpy()[None])
    if plot is not None:
        plot("{}/temp_{}_samples.png".format(args.save_dir, 'gt'), new_samples)
    chain = np.concatenate(chain, 0) # for mnist + dims [n_steps/subsample, 784], for mnist + hamming [n_steps/subsample, n_samples]

    # Computing ess statistic on dimension or on Hamming distances
    chains['rbm-gibbs'] = chain
    ess['rbm-gibbs'] = get_ess(chain, args.burn_in)
    print("ess = {} +/- {}".format(ess['rbm-gibbs'].mean(), ess['rbm-gibbs'].std()))

    print('\n ---------- RUNNING DIFFERENT SAMPLERS FOR p(v) ----------\n')

    # Running different samplers and measuring the performance
    temps = ['bg-2', 'hb-10-1', 'gwg', 'lsb1', 'lsb2']
    for temp in temps:

        # Setting the initial seed the same in order to ensure fair comparisons
        torch.manual_seed(args.seed)

        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(args.n_visible)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(args.n_visible, 1,
                                                approx=True, temp=2., n_samples=n_hops)
        elif temp == "lsb1":
            func = 'lsb1'
            lr = args.lr
            sampler = self_sampler.LSB(args.n_visible, func,  args.n_test_samples, 1, approx=True, lr=lr, burn_in=True, device=device, seed=args.seed)
        elif temp == "lsb2":
            func = 'lsb2'
            lr = args.lr
            sampler = self_sampler.LSB(args.n_visible, func,  args.n_test_samples, 1, approx=True, lr=lr, hidden=args.lsb_hidden, model=model, burn_in=True, device=device, seed=args.seed)
        else:
            raise ValueError("Invalid sampler...")

        sampler.burn_in = True

        x = model.init_dist.sample((args.n_test_samples,)).to(device)

        log_mmds[temp] = []
        ars[temp] = []
        hops[temp] = []
        times[temp] = []
        chain = []
        cur_time = 0.
        flag = True
        objective = torch.zeros(int(args.n_steps * args.burn_in))

        for i in range(args.n_steps):

            if i >= args.n_steps * args.burn_in:
                sampler.burn_in = False
                flag = False

            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            tmp = time.time() - st
            if i >= args.n_steps * args.burn_in:
                cur_time += tmp

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x[0][None]
                    h = (xc != gt_samples).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if (temp.find('lsb') != -1) and sampler.burn_in == True:
                objective[i] = sampler.obj.detach()

            if i % args.viz_every == 0 and sampler.burn_in == True:
                if (temp.find('lsb1') != -1):
                    # Analysis of the learnt function
                    data = torch.arange(start = 0.1, end = 10. + 0.1, step = 0.1, device=device)
                    output = sampler.barker(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'b-', label=r'$\frac{t}{1+t}$')
                    output = sampler.sqrt(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'c-', label=r'$\sqrt{t}$')
                    output = sampler.minim(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'g-', label=r'$\min\{1,t\}$')
                    output = sampler.maxim(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'k-', label=r'$\max\{1,t\}$')
                    output = sampler.func(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'r-', label=temp)

                    plt.title('Balancing function')
                    plt.savefig(os.path.join(args.save_dir, 'Function_{}_{}.png'.format(temp, i)))
                    plt.close()
                elif (temp.find('lsb2') != -1):
                    # Analysis of the learnt function
                    data = torch.arange(start = 0.1, end = 10. + 0.1, step = 0.1, device=device)
                    output = sampler.barker(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'b-', label=r'$\frac{t}{1+t}$')
                    output = sampler.sqrt(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'c-', label=r'$\sqrt{t}$')
                    output = sampler.minim(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'g-', label=r'$\min\{1,t\}$')
                    output = sampler.maxim(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'k-', label=r'$\max\{1,t\}$')
                    data = data.unsqueeze(1)
                    output = sampler.func(data).detach().cpu()
                    plt.plot(data.detach().cpu(), output, 'r-', label=temp)

                    plt.title('Balancing functions')
                    plt.savefig(os.path.join(args.save_dir, 'Function_{}_{}.png'.format(temp, i)))
                    plt.close()

            if i % args.print_every == 0:
                hard_samples = x
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log10().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                if (temp.find('lsb1') != -1):
                    print("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f} burn-in = {} theta = {}".format(temp, i, log_stat, cur_hops, flag, sampler.theta))
                else:
                    print("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f} burn-in = {}".format(temp, i, log_stat, cur_hops, flag))

            if i == 35000:
                if plot is not None:
                    plot("{}/temp_{}_samples_35000.png".format(args.save_dir, temp), x)

        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))

        plt.plot(objective)
        plt.xlabel('Iteration')
        plt.title('Objective function')
        plt.savefig(os.path.join(args.save_dir, 'Objective_{}.png'.format(temp)))
        plt.close()


    ess_temps = temps
    plt.clf()
    plt.boxplot([ess[temp] for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess.png".format(args.save_dir))

    plt.clf()
    plt.boxplot([ess[temp] / times[temp][-1] for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess_per_sec.png".format(args.save_dir))

    plt.clf()
    for temp in temps + ['rbm-gibbs']:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.xlabel('Likelihood Evaluations')
    plt.savefig("{}/results.png".format(args.save_dir))

    plt.clf()

    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    for temp in temps:
        plt.clf()
        plt.plot(chains[temp][:, 0])
        plt.savefig("{}/trace_{}.png".format(args.save_dir, temp))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'ess': ess,
            'hops': hops,
            'log_mmds': log_mmds,
            'chains': chains,
            'times': times
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./results")
    parser.add_argument('--data', choices=['mnist', 'random'], type=str, default='mnist')
    parser.add_argument('--n_steps', type=int, default=20000) # number of iterations to run the samplers
    parser.add_argument('--n_samples', type=int, default=500) # number of samples used to compare
    parser.add_argument('--n_test_samples', type=int, default=16) # real batch size
    parser.add_argument('--gt_steps', type=int, default=10000) # number of iterations required to get ground truth samples
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lsb_hidden', type=int, default=10)
    # rbm def
    parser.add_argument('--n_hidden', type=int, default=500)
    parser.add_argument('--n_visible', type=int, default=784) # 100 for random, 784 for mnist
    parser.add_argument('--print_every', type=int, default=100) # for all quantities except ESS
    parser.add_argument('--viz_every', type=int, default=100) # for plotting images
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001) # 0.001
    parser.add_argument('--cd', type=int, default=10)         # 10
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=20) # 100
    # for ess
    parser.add_argument('--subsample', type=int, default=1) # for ESS statistics
    parser.add_argument('--burn_in', type=float, default=.2)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--ess_statistic', type=str, default="hamming", choices=["hamming", "dims"])
    args = parser.parse_args()

    main(args)

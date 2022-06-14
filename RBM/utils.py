import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader

def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()


def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d


def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


def short_run_mcmc(logp_net, x_init, k, sigma, step_size=None):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    if step_size is None:
        step_size = (sigma ** 2.) / 2.
    for i in range(k):
        f_prime = torch.autograd.grad(logp_net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k


def get_data(args):
    if args.data == "mnist":
        transform = tr.Compose([tr.Resize(args.img_size), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
        train_data = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
        test_data = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, args.batch_size, shuffle=True, drop_last=True)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.img_size, args.img_size),
                                                         p, normalize=True, nrow=sqrt(x.size(0)))
        encoder = None
        viz = None
    else:
        raise ValueError

    return train_loader, test_loader, plot, viz

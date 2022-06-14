import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np
from scipy import stats

# Self-balancing for binary data
class LSB(nn.Module):
    def __init__(self, dim, func, batch, n_steps=10, approx=False, lr=1e-2, model=None,\
                hidden=10, burn_in=True, device=None, seed=None):
        super().__init__()
        torch.manual_seed(seed)
        self.batch = batch
        self.dim = dim
        self.n_steps = n_steps
        self.method = func
        self.lr = lr
        self.eps = 1e-8
        self.device = device
        self.burn_in = burn_in
        self.distr = model
        # Params for neural net
        self.max_value = 2
        self.hidden = hidden

        self.eta = torch.tensor(0., device=self.device, requires_grad=True)
        self.softplus = torch.nn.Softplus()

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m)

        if self.method == 'lsb1': # Learning to select

            self.softmax = torch.nn.Softmax(dim=0)
            self.softplus = torch.nn.Softplus()
            self.func = self.linear_normalized
            self.theta = torch.zeros((4,), requires_grad=True, device=self.device)
            self.optimizer = torch.optim.Adam([self.theta], lr=self.lr)
            self.optimizer.add_param_group({'params': self.eta})

        elif self.method == 'lsb2': # Learning the balancing function

            self.init_lr = self.lr
            self.init_iter = 5000
            self.model = MLP(self.hidden)
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.optimizer.add_param_group({'params': self.eta})
            self.func = self.nonlinear
            input_data = torch.rand((self.batch, 1), requires_grad=False, device=self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
            for i in range(self.init_iter):
                optimizer.zero_grad()
                loss = torch.mean((self.func(input_data) - self.sqrt(input_data))**2)
                if i % 100 == 0:
                    print(i, loss.item())
                loss.backward()
                optimizer.step()
            print('\nInitialization of lsb2 done!\n')

        else:
            raise Exception('Wrong selection of balancing function!')

    def step(self, x, model):

        x_cur = x

        for i in range(self.n_steps):

            if self.burn_in:
                # Resetting x_cur
                idx = torch.rand((self.batch,)) < self.eps
                rnd = torch.randint_like(x_cur, 2).float()
                x_cur[idx, :] = rnd[idx, :]

            # Computing Q(x'|x)
            forward_delta = self.diff_fn(x_cur, model) #[batch_size, dim] for mnist dim=784

            forward_delta = torch.exp(forward_delta)
            probs_forward = self.func(forward_delta)
            probs_forward = torch.log(probs_forward)
            cd_forward = dists.OneHotCategorical(logits=probs_forward, validate_args=False)
            changes = cd_forward.sample()

            self.proposal = cd_forward.log_prob(changes)

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

            # Computing Q(x|x')
            reverse_delta = self.diff_fn(x_delta, model)

            reverse_delta = torch.exp(reverse_delta)
            probs_reverse = self.func(reverse_delta)
            probs_reverse = torch.log(probs_reverse)
            cd_reverse = dists.OneHotCategorical(logits=probs_reverse, validate_args=False)

            proposal_reverse = cd_reverse.log_prob(changes)

            # Computing acceptance ratio
            temp = model(x_cur).squeeze()
            self.p_tilde = (temp - torch.max(temp)) # p(x)
            self.log_p_tilde = model(x_delta).squeeze() # log p(x')
            m_term = (self.log_p_tilde - temp)
            la = m_term + proposal_reverse - self.proposal # [batch_size]

            la = torch.minimum(la, torch.tensor(0.)) # Avoid NaN issue in backward phase
            self.accept = torch.minimum(la.exp(), torch.tensor(1.))

            if self.burn_in:
                # Computing Q(x^*|x)
                logits = torch.ones_like(probs_forward)
                cd_forward_star = dists.OneHotCategorical(logits=logits)
                changes_star = cd_forward_star.sample()

                self.proposal_star = cd_forward.log_prob(changes_star)

                x_delta_star = (1. - x_cur) * changes_star + x_cur * (1. - changes_star)

                # Computing Q(x|x^*)
                reverse_delta_star = self.diff_fn(x_delta_star, model)

                reverse_delta_star = torch.exp(reverse_delta_star)
                probs_reverse_star = self.func(reverse_delta_star)
                probs_reverse_star = torch.log(probs_reverse_star)
                cd_reverse_star = dists.OneHotCategorical(logits=probs_reverse_star, validate_args=False)

                proposal_reverse_star = cd_reverse_star.log_prob(changes_star)

                # Computing acceptance ratio
                log_p_tilde_star = model(x_delta_star).squeeze() # log p(x^*)
                m_term = (log_p_tilde_star - temp)

                la_star = m_term + proposal_reverse_star - self.proposal_star # [batch_size]

                la_star = torch.minimum(la_star, torch.tensor(0.)) # Avoid NaN issue in backward phase
                self.accept_star = torch.minimum(la_star.exp(), torch.tensor(1.))

            # print(la.exp())
            acc = (la.exp() > torch.rand_like(la)).clone()
            a = acc.float() # [batch_size]

            # Computing objective
            if self.burn_in:
                bias = torch.ones_like(self.proposal, requires_grad=False, device=self.device)
                bias[idx] = self.eps
                self.old_proposal = bias

                w = (self.p_tilde - self.proposal).exp().detach() / self.old_proposal
                accept = self.accept * self.proposal.exp()
                info = torch.log(accept) - self.log_p_tilde.detach()

                w_star = 1. / self.old_proposal
                accept_star = 1. - self.accept_star * self.proposal_star.exp()
                par = self.softplus(self.eta)
                info_star = par * accept_star - self.p_tilde.exp().detach() * (torch.log(par) + 1.)

                # Compute objective, accept or reject trajectory, update theta
                self.obj = (torch.sum(w * accept * info) + torch.sum(w_star * accept_star * info_star)) / self.batch

                self.optimizer.zero_grad()
                self.obj.backward() # Keep the whole graph for each call of sample_burn_in
                self.optimizer.step()

            # Updating solution
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur

    ##### Balancing functions
    def barker(self, inp):
        return inp / (1 + inp)

    def sqrt(self, inp):
        return torch.sqrt(inp)

    def minim(self, inp):
        return torch.minimum(torch.tensor(1., device=self.device), inp)

    def maxim(self, inp):
        return torch.maximum(torch.tensor(1., device=self.device), inp)
    
    def linear_normalized(self, inp):
        normalized = self.softmax(self.theta)
        return normalized[0] * self.barker(inp) + \
               normalized[1] * self.sqrt(inp) +   \
               normalized[2] * self.minim(inp) +  \
               normalized[3] * self.maxim(inp)

    def nonlinear(self, inp):

        length = 0
        if len(inp.shape) == 1:
            length = 1
            inp = inp.unsqueeze(1)
        elif len(inp.shape) == 3:
            length = 3
            b, w, h = inp.shape
            inp = torch.flatten(inp, start_dim=1)

        assert len(inp.shape) == 2, 'Input tensor must have 2 dims! Instead it has {} dims'.format(len(inp.shape))

        inp = inp.unsqueeze(2)

        assert torch.isnan(inp).any() == False, 'Found a Nan in input to LSB 2'
        assert torch.isinf(inp).any() == False, 'Found a Inf in input to LSB 2'

        value_1 = self.model(inp)

        assert torch.isnan(value_1).any() == False, 'Found a Nan in value_1 to LSB 2'
        assert torch.isinf(value_1).any() == False, 'Found a Inf in value_1 to LSB 2'

        value_2 = torch.maximum(inp, torch.tensor(0., requires_grad=False, device=self.device)) * self.model(1. / inp)

        assert torch.isnan(value_2).any() == False, 'Found a Nan in value_2 to LSB 2'
        assert torch.isinf(value_2).any() == False, 'Found a Inf in value_2 to LSB 2'

        result = value_1.squeeze(2)/2. + value_2.squeeze(2)/2.

        assert torch.isnan(result).any() == False, 'Found a Nan in result to LSB 2'
        assert torch.isinf(result).any() == False, 'Found a Inf in result to LSB 2'

        assert len(result.shape) == 2, 'Output tensor must have 2 dims! Instead it has {} dims'.format(len(result.shape))

        if length == 1:
            result = result.squeeze(1)
        elif length == 3:
            result = torch.reshape(result, (b, w, h))

        return result

class MLP(torch.nn.Module):
    
    def __init__(self, hidden):
        super(MLP, self).__init__()
        self.hidden = hidden
        self.linear1 = torch.nn.Linear(1, hidden)
        self.linear2 = torch.nn.Linear(hidden, 1)
        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        
    # inp is 3d tensor in NCW format
    def forward(self, inp):
        b, c, w = inp.shape
        inp = 1. / (1 + 1./inp) - 0.5 # Monotonic nonlinearity to map [0,infty) to [-0.5,0.5) 

        assert torch.isnan(inp).any() == False, 'Found a Nan in squash to LSB 2'
        assert torch.isinf(inp).any() == False, 'Found a Inf in squash to LSB 2'

        hid = self.relu(self.linear1(inp))

        assert torch.isnan(inp).any() == False, 'Found a Nan in linear1 to LSB 2'
        assert torch.isinf(inp).any() == False, 'Found a Inf in linear1 to LSB 2'

        inp = self.softplus(self.linear2(hid))
        return inp

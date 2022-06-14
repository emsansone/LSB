import torch
import copy
import random

##### Locally Self-Balancing Sampler
class LSB():

    ##### Function initializing elements used 
    ##### to compute and sample from Q(X'|X))
    # batch     - number of chains
    # func      - 'lsb1', 'lsb2'
    # lr        - learning rate
    # start_x   - dictionary with variables and their state value (currently accepting only binary values)
    # theta     - scalar vector for combining balancing functions for 'lsb1'
    #           state dictionary of parameters of the neural net for 'lsb2'
    # gamma     - scalar used to ensure monotonicity of balancng function
    # factor    - integer indicating the number of blocks in the monotonic network
    # block     - integer indicating the size of each block in the monotonic network
    # max_value - maximum value for random numbers in the monotonicity regularizer
    # model     - Markov model according to pgmpy library
    # evidence  - Dictionary containing evidence variables
    # burn_in   - Burn-in - True, Sampling - False
    def __init__(self, batch, func, lr=1e-3, start_x=None, theta=None, \
                gamma=.1, factor=20, block=20, max_value=2, model=None, evidence=None, burn_in=True):
        self.batch = batch
        self.method = func
        self.lr = lr
        self.eps = 1e-6
        self.max_eps = 1e36

        # Params for neural net
        self.gamma = gamma
        self.factor = factor
        self.max_value = max_value
        self.block = block
        self.hidden = self.factor * self.block

        self.burn_in = burn_in

        if self.method == 'lsb1': # Learning to select
            # Initializing theta to prefer sqrt balancing function
            self.velocity = torch.tensor(0., requires_grad=False)
            self.softmax = torch.nn.Softmax(dim=0)
            if theta is None:
                self.theta = torch.zeros((4,), requires_grad=True)
            else: 
                self.theta = theta.clone()
            self.func = self.linear_normalized
        elif self.method == 'lsb2': # Learning the balancing function
            self.init_lr = 1e-2
            self.init_iter = 5000
            self.mono = MonoNet(self.hidden, self.block)
            self.optimizer = torch.optim.SGD(self.mono.parameters(), lr=self.lr, momentum=0.9)
            self.func = self.nonlinear
            if theta is not None:
                self.mono.load_state_dict(theta)
            else:
                input_data = torch.rand((self.batch, 1, 1), requires_grad=False) * self.max_value
                optimizer = torch.optim.SGD(self.mono.parameters(), lr=self.init_lr, momentum=0.9)
                for i in range(self.init_iter):
                    optimizer.zero_grad()
                    loss = torch.mean((self.func(input_data) - self.maxim(input_data))**2)
                    if i % 100 == 0:
                        print(i, loss.item())
                    loss.backward(retain_graph=True)
                    optimizer.step()
                print('\nInitialization of lsb2 done!\n')
        else:
            raise Exception('Wrong selection of balancing function!')

        self.x = [copy.deepcopy(data) for data in start_x]
        self.vars_to_sample = list(self.x[0].keys() - evidence.keys())
        self.model = model

        self.ratio = torch.ones((self.batch, len(self.vars_to_sample)))
        for b in range(self.batch):
            var_id = 0
            for var in self.vars_to_sample:
                factors = self.model.get_factors(var)
                for factor in factors:
                    # Compute \phi(X)
                    phi = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi.variables]
                    phi.reduce(arg)
                    value = phi.values
                    # Compute \phi(X')
                    phi_1 = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi_1.variables if v != var]
                    val = (- (2*self.x[b][var] - 1) + 1) // 2
                    arg = arg + [(var, val)]
                    phi_1.reduce(arg)
                    value_1 = phi_1.values
                    value = torch.maximum(torch.tensor(value), torch.tensor(self.eps, requires_grad=False))       # Minimum factor value is eps
                    value_1 = torch.maximum(torch.tensor(value_1), torch.tensor(self.eps, requires_grad=False))   # Minimum factor value is eps
                    self.ratio[b, var_id] = self.ratio[b, var_id] * value_1 / value
                    self.ratio[b, var_id] = torch.minimum(self.ratio[b, var_id], torch.tensor(self.max_eps, requires_grad=False))
                var_id += 1
        self.g_ratio = self.func(self.ratio)
        self.norm_const = torch.sum(self.g_ratio, 1, keepdim=True)
        self.norm_const = torch.maximum(self.norm_const, torch.tensor(0., requires_grad=False))
        self.accept = torch.zeros((self.batch,))
        self.obj = None
        self.proposal = torch.zeros((self.batch,))

        self.log_p_tilde = torch.zeros((self.batch,))
        for b in range(self.batch):
            for factor in self.model.factors:
                phi = factor.copy()
                arg = [(v, self.x[b][v]) for v in phi.variables]
                phi.reduce(arg)
                if phi.values < self.eps:
                    tmp = torch.log(torch.tensor(self.eps))
                else:
                    tmp = torch.log(torch.tensor(phi.values))
                self.log_p_tilde[b] = self.log_p_tilde[b] + tmp


    ##### Function used to sample a new configuration
    def sample_burn_in(self):

        for i in range(self.batch // 2 + self.batch % 2, self.batch):
            for j in self.vars_to_sample:
                self.x[i][j] = 0.

        # Updating ratio, g_ratio and norm_const
        self.ratio = torch.ones((self.batch, len(self.vars_to_sample)))
        for b in range(self.batch):
            var_id = 0
            for var in self.vars_to_sample:
                factors = self.model.get_factors(var)
                for factor in factors:
                    # Compute \phi(X)
                    phi = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi.variables]
                    phi.reduce(arg)
                    value = phi.values
                    # Compute \phi(X')
                    phi_1 = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi_1.variables if v != var]
                    val = (- (2*self.x[b][var] - 1) + 1) // 2
                    arg = arg + [(var, val)]
                    phi_1.reduce(arg)
                    value_1 = phi_1.values
                    value = torch.maximum(torch.tensor(value), torch.tensor(self.eps, requires_grad=False))       # Minimum factor value is eps
                    value_1 = torch.maximum(torch.tensor(value_1), torch.tensor(self.eps, requires_grad=False))   # Minimum factor value is eps
                    self.ratio[b, var_id] = self.ratio[b, var_id] * value_1 / value
                    self.ratio[b, var_id] = torch.minimum(self.ratio[b, var_id], torch.tensor(self.max_eps, requires_grad=False))
                var_id += 1
        self.g_ratio = self.func(self.ratio)
        self.norm_const = torch.sum(self.g_ratio, 1, keepdim=True)
        self.norm_const = torch.maximum(self.norm_const, torch.tensor(0., requires_grad=False))

        # Backup log P_tilde(X)
        self.log_p_tilde_temp = self.log_p_tilde.clone()
        # Backup X
        self.x_temp = [copy.deepcopy(data) for data in self.x]
        # Initialize w, A, T
        self.w = torch.ones((self.batch,))
        self.A = torch.ones((self.batch,))
        self.T = torch.zeros((self.batch,))

        # Sampling an index
        probs = self.g_ratio / self.norm_const
        idx = torch.multinomial(probs, 1)[:, 0]
        self.var = [self.vars_to_sample[idx[i]] for i in range(self.batch)]

        # Compute Z(X') and update X
        temp_ratio = torch.ones(self.batch, len(self.vars_to_sample))
        for b in range(self.batch):
            self.x[b][self.var[b]] = (- (2*self.x[b][self.var[b]] - 1) + 1) // 2
            var_id = 0
            for var in self.vars_to_sample:
                factors = self.model.get_factors(var)
                for factor in factors:
                    # Compute \phi(X')
                    phi = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi.variables]
                    phi.reduce(arg)
                    value = phi.values
                    # Compute \phi(X'')
                    phi_1 = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi_1.variables if v != var]
                    val = (- (2*self.x[b][var] - 1) + 1) // 2
                    arg = arg + [(var, val)]
                    phi_1.reduce(arg)
                    value_1 = phi_1.values
                    # Compute local ratio and update the overall one (for all visited factors)
                    value = torch.maximum(torch.tensor(value), torch.tensor(self.eps, requires_grad=False))       # Minimum factor value is eps
                    value_1 = torch.maximum(torch.tensor(value_1), torch.tensor(self.eps, requires_grad=False))   # Minimum factor value is eps
                    temp_ratio[b, var_id] = temp_ratio[b, var_id] * value_1 / value
                    temp_ratio[b, var_id] = torch.minimum(temp_ratio[b, var_id], torch.tensor(self.max_eps, requires_grad=False))
                var_id += 1
        g_ratio = self.func(temp_ratio)
        norm_const_new = torch.sum(g_ratio, 1, keepdim=True)
        norm_const_new = torch.maximum(norm_const_new, torch.tensor(0., requires_grad=False))

        # Compute acceptance score
        self.accept = torch.minimum(torch.tensor(1.), self.norm_const / norm_const_new)

        # Compute proposal score
        batch_id = torch.arange(self.batch)
        self.proposal = self.g_ratio[batch_id, idx] / self.norm_const[:, 0]

        # Update log P_tilde(X')
        temp_log_p_tilde = self.log_p_tilde.clone()
        for b in range(self.batch):
            temp_val = (- (2*self.x[b][self.var[b]] - 1) + 1) // 2
            for factor in self.model.get_factors(self.var[b]):
                phi = factor.copy()
                arg = [(v, self.x[b][v]) for v in phi.variables if v != self.var[b]]
                arg = arg + [(self.var[b], temp_val)]
                phi.reduce(arg)
                if phi.values < self.eps:
                    tmp = torch.log(torch.tensor(self.eps))
                else:
                    tmp = torch.log(torch.tensor(phi.values))
                self.log_p_tilde[b] = self.log_p_tilde[b] - tmp
                phi = factor.copy()
                arg = [(v, self.x[b][v]) for v in phi.variables]
                phi.reduce(arg)
                if phi.values < self.eps:
                    tmp = torch.log(torch.tensor(self.eps))
                else:
                    tmp = torch.log(torch.tensor(phi.values))
                self.log_p_tilde[b] = self.log_p_tilde[b] + tmp

        # Update w, A, T
        self.w = self.w * (self.proposal / self.proposal.detach())
        self.A = self.A * self.accept[:, 0]
        self.T = self.T + torch.log(self.accept[:, 0]) + torch.log(self.proposal) - self.log_p_tilde.detach()

        # Update ratio, g_ratio and norm_const
        self.ratio = temp_ratio
        self.g_ratio = g_ratio
        self.norm_const = norm_const_new

        # Compute objective
        scalar = 2. * torch.ones((self.batch,), requires_grad=False) / self.batch
        scalar[(self.batch // 2 + self.batch % 2):] = scalar[(self.batch // 2 + self.batch % 2):]
        self.obj = torch.sum(self.w * self.A * self.T * scalar)
        if self.method == 'lsb2':
            ratio = torch.rand((self.batch, 1, 1), requires_grad=False) * self.max_value
            value_1_inv = self.mono(1. / ratio)
            value_2_inv = self.mono(1. / (ratio + self.eps))
            regularizer = torch.maximum((ratio * value_1_inv - (ratio + self.eps) * value_2_inv) / self.eps \
                                        , torch.tensor(0., requires_grad=False))
            self.obj = self.obj + self.gamma * torch.mean(regularizer)
            self.optimizer.zero_grad()
        else:
            self.theta.retain_grad() # This is required to retain the gradient for theta (otherwise it is not computed, as requires_grad=True was not for theta)
        self.obj.backward(retain_graph=True) # Keep the whole graph for each call of sample_burn_in

        # Accept or reject trajectory
        for b in range(self.batch):
            if torch.rand((1,)) > self.A[b]:
                self.log_p_tilde[b] = self.log_p_tilde_temp[b]
                self.x[b] = self.x_temp[b]

        # Update parameters using SGD with Momentum
        if self.method == 'lsb2':
            self.optimizer.step()
        else:
            with torch.no_grad():
                self.velocity = 0.9 * self.velocity + self.theta.grad
                self.theta -= self.lr * self.velocity
                self.theta.grad = None


    ##### Function used to sample a new configuration
    def sample(self):

        # Sampling an index
        probs = self.g_ratio / self.norm_const
        idx = torch.multinomial(probs, 1)[:, 0]
        self.var = [self.vars_to_sample[idx[i]] for i in range(self.batch)]

        # Compute Z(X')
        temp_ratio = torch.ones(self.batch, len(self.vars_to_sample))
        for b in range(self.batch):
            self.x[b][self.var[b]] = (- (2*self.x[b][self.var[b]] - 1) + 1) // 2 # Temporary modifying sampled var
            var_id = 0
            for var in self.vars_to_sample:
                factors = self.model.get_factors(var)
                for factor in factors:
                    # Compute \phi(X')
                    phi = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi.variables]
                    phi.reduce(arg)
                    value = phi.values
                    # Compute \phi(X'')
                    phi_1 = factor.copy()
                    arg = [(v, self.x[b][v]) for v in phi_1.variables if v != var]
                    val = (- (2*self.x[b][var] - 1) + 1) // 2
                    arg = arg + [(var, val)]
                    phi_1.reduce(arg)
                    value_1 = phi_1.values
                    # Compute local ratio and update the overall one (for all visited factors)
                    value = torch.maximum(torch.tensor(value), torch.tensor(self.eps, requires_grad=False))       # Minimum factor value is eps
                    value_1 = torch.maximum(torch.tensor(value_1), torch.tensor(self.eps, requires_grad=False))   # Minimum factor value is eps
                    temp_ratio[b, var_id] *= value_1 / value
                    temp_ratio[b, var_id] = torch.minimum(temp_ratio[b, var_id], torch.tensor(self.max_eps, requires_grad=False))
                var_id += 1
            self.x[b][self.var[b]] = (- (2*self.x[b][self.var[b]] - 1) + 1) // 2 # Restoring sampled var

            g_ratio = self.func(temp_ratio[b, :])
            norm_const_new = torch.sum(g_ratio)
            norm_const_new = torch.maximum(norm_const_new, torch.tensor(0.))

            self.accept[b] = torch.minimum(torch.tensor(1.), self.norm_const[b, 0] / norm_const_new)

            if torch.rand((1,),) < self.norm_const[b, 0] / norm_const_new:

                self.ratio[b, :] = temp_ratio[b, :]
                self.g_ratio[b, :] = g_ratio
                self.norm_const[b, 0] = norm_const_new

                self.x[b][self.var[b]] = (- (2*self.x[b][self.var[b]] - 1) + 1) // 2

    ##### Balancing functions
    def barker(self, inp):
        return inp / (1 + inp)

    def sqrt(self, inp):
        return torch.sqrt(inp)

    def minim(self, inp):
        return torch.minimum(torch.tensor(1.), inp)

    def maxim(self, inp):
        return torch.maximum(torch.tensor(1.), inp)
    
    def linear_normalized(self, inp):
        if self.burn_in == True:
            normalized = self.softmax(self.theta + torch.randn_like(self.theta))
        else:
            normalized = self.softmax(self.theta)
        return normalized[0] * self.barker(inp) + \
               normalized[1] * self.sqrt(inp) +   \
               normalized[2] * self.minim(inp) +  \
               normalized[3] * self.maxim(inp)

    def nonlinear(self, inp):
        length = 0
        if len(inp.shape) == 1:
            length = 1
            inp = inp.unsqueeze(0).unsqueeze(0)
        elif len(inp.shape) == 2:
            length = 2
            inp = inp.unsqueeze(1)

        value_1 = self.mono(inp)
        value_2 = torch.maximum(inp, torch.tensor(0., requires_grad=False)) * self.mono(1. / inp)

        result = torch.minimum(value_1, value_2)
        
        if length == 2:
            result = result.squeeze(1)
        elif length == 1:
            result = result.squeeze(0).squeeze(0)
        return result

class MonoNet(torch.nn.Module):
    
    def __init__(self, hidden, block):
        super(MonoNet, self).__init__()
        self.hidden = hidden
        self.block = block
        init_value = 1e-3
        self.weights = torch.nn.Parameter(init_value * torch.randn(self.hidden, 1, 1))
        self.bias = torch.nn.Parameter(init_value * torch.randn(self.hidden))
        self.softplus = torch.nn.Softplus()
        self.max = torch.nn.MaxPool1d(self.block, self.block, 0, 1)

    # inp is 3d tensor in NCW format
    def forward(self, inp):
        inp = 1. / (1 + 1./inp) - 0.5 # Monotonic nonlinearity to map [0,infty) to [-0.5,0.5) 
        inp = torch.nn.functional.conv1d(inp, self.weights.exp(), bias=self.bias)
        inp = self.softplus(inp)
        inp = inp.permute(0, 2, 1)
        inp = self.max(inp)
        inp = torch.min(inp, dim=2, keepdim=True)[0]
        inp = inp.permute(0, 2, 1)
        return inp

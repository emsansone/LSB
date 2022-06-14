import torch

##### Locally Self-Balancing Sampler
class LSBS():

    ##### Function initializing elements used 
    ##### to compute and sample from Q(X'|X))
    # batch     - number of chains
    # n         - size of square lattice
    # lamda     - scalar
    # alpha     - n x n matrix of bias coefficients
    # func      - 'lsb1', 'lsb2' 
    # lr        - learning rate
    # start_x   - batch x n x n initial tensor
    # theta     - scalar vector for combining balancing functions for 'lsb1'
    #           state dictionary of parameters of the neural net for 'lsb2'
    # hidden    - Number of hidden neurons
    # max_value - maximum value for random numbers in the monotonicity regularizer
    # device    - device to run the simulation
    # burn_in   - Burn-in - True, Sampling - False
    def __init__(self, batch, n, lamda, alpha, func, lr=1e-3, start_x=None, theta=None, \
                hidden=10, max_value=2, device=None, burn_in=True):

        self.batch = batch
        self.n = n
        self.lamda = lamda
        self.device = device
        self.alpha = alpha.to(self.device)
        self.method = func
        self.lr = lr
        self.eps = 1e-8
        # Params for neural net
        self.max_value = max_value
        self.hidden = hidden

        self.burn_in = burn_in

        self.eta = torch.tensor(0., device=self.device, requires_grad=True)
        self.softplus = torch.nn.Softplus()

        if self.method == 'lsb1': # Learning to select
            # Initializing theta to prefer sqrt balancing function
            self.softmax = torch.nn.Softmax(dim=0)
            self.func = self.linear_normalized
            if theta is None:
                self.theta = torch.zeros((4,), requires_grad=True, device=self.device)
                self.optimizer = torch.optim.Adam([self.theta], lr=self.lr)
                self.optimizer.add_param_group({'params': self.eta})
            else: 
                self.theta = theta.clone()
            
        elif self.method == 'lsb2': # Learning the balancing function
            self.init_lr = self.lr
            self.init_iter = 5000
            self.model = MLP(self.hidden)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.optimizer.add_param_group({'params': self.eta})
            self.func = self.nonlinear
            if theta is not None:
                self.model.load_state_dict(theta)
            else:
                input_data = torch.rand((self.batch, 1, 1), requires_grad=False, device=self.device)
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

        # Computing invariants
        if start_x == None:
            self.x = torch.randint(2, (batch, n, n), requires_grad=False, device=self.device)*2 - 1
        else:
            self.x = start_x.clone()

        temp_up = torch.cat((self.x[:, 1:, :], self.x[:, :1, :]), 1)       # Moving up 1 row
        temp_low = torch.cat((self.x[:, -1:,:], self.x[:, :-1, :]), 1)     # Moving down 1 row
        temp_left = torch.cat((self.x[:, :, 1:], self.x[:, :, :1]), 2)     # Moving left 1 col
        temp_right = torch.cat((self.x[:, :, -1:], self.x[:, :, :-1]), 2)  # Moving right 1 col
        self.ratio = torch.exp(-2. * self.alpha * self.x
                                -2. * self.lamda * self.x
                                * (temp_up + temp_low + temp_left + temp_right))
        # print(torch.median(self.ratio), torch.max(self.ratio))
        self.g_ratio = self.func(self.ratio)
        self.sum_rows = torch.sum(self.g_ratio, 2)
        self.sum_rows = torch.maximum(self.sum_rows, torch.tensor(0., requires_grad=False, device=self.device)) # Check to avoid numerical issues
        self.norm_const = torch.sum(self.sum_rows, 1)
        self.norm_const = torch.maximum(self.norm_const, torch.tensor(0., requires_grad=False, device=self.device)) # Check to avoid numerical issues
        self.accept = None
        self.obj = None

    def normalizing(self, batch_id, i, j):
        # Compute the neighbours
        i_up = i - 1
        i_up[i_up < 0] = self.n - 1
        i_low = i + 1
        i_low[i_low == self.n] = 0
        j_left = j - 1
        j_left[j_left < 0] = self.n - 1
        j_right = j + 1
        j_right[j_right == self.n] = 0

        # Compute new normalization constant Z(X')
        delta_i = self.func(1./self.ratio[batch_id, i, j])                                                \
                - self.g_ratio[batch_id, i, j]                                                            \
                - self.g_ratio[batch_id, i, j_left]                                                       \
                - self.g_ratio[batch_id, i, j_right]                                                      \
                + self.func(self.ratio[batch_id, i, j_left]                                               \
                            * torch.exp(4. * self.lamda * self.x[batch_id, i, j_left]                     \
                                        * self.x[batch_id, i, j]))                                        \
                + self.func(self.ratio[batch_id, i, j_right]                                              \
                            * torch.exp(4. * self.lamda * self.x[batch_id, i, j_right]                    \
                                        * self.x[batch_id, i, j]))                    
        delta_i_up = - self.g_ratio[batch_id, i_up, j]                                                    \
                    + self.func(self.ratio[batch_id, i_up, j]                                             \
                                * torch.exp(4. * self.lamda * self.x[batch_id, i_up, j]                   \
                                            * self.x[batch_id, i, j]))
        delta_i_low = - self.g_ratio[batch_id, i_low, j]                                                  \
                    + self.func(self.ratio[batch_id, i_low, j]                                            \
                                * torch.exp(4. * self.lamda * self.x[batch_id, i_low, j]                  \
                                            * self.x[batch_id, i, j]))

        return self.norm_const + delta_i + delta_i_up + delta_i_low



    ##### Function used to sample a new configuration (during burn-in)
    def sample_burn_in(self):
        
        #### Resetting at the beginning of each iteration (this will rarely occur)
        idx = torch.rand((self.batch,)) < self.eps
        rnd = torch.randint(2, (self.batch, self.n, self.n), requires_grad=False, device=self.device)*2 - 1
        self.x[idx, :, :] = rnd[idx, :, :]
        # Updating ratio, g_ratio, sum_rows and norm_const
        temp_up = torch.cat((self.x[:, 1:, :], self.x[:, :1, :]), 1)       # Moving up 1 row
        temp_low = torch.cat((self.x[:, -1:,:], self.x[:, :-1, :]), 1)     # Moving down 1 row
        temp_left = torch.cat((self.x[:, :, 1:], self.x[:, :, :1]), 2)     # Moving left 1 col
        temp_right = torch.cat((self.x[:, :, -1:], self.x[:, :, :-1]), 2)  # Moving right 1 col
        self.ratio = torch.exp(-2. * self.alpha * self.x                                                        \
                                -2. * self.lamda * self.x                                                       \
                                * (temp_up + temp_low + temp_left + temp_right))
        # print(torch.median(self.ratio), torch.max(self.ratio))
        self.g_ratio = self.func(self.ratio)
        self.sum_rows = torch.sum(self.g_ratio, 2)
        self.sum_rows = torch.maximum(self.sum_rows, torch.tensor(0., requires_grad=False, device=self.device)) # Check to avoid numerical issues
        self.norm_const = torch.sum(self.sum_rows, 1)
        self.norm_const = torch.maximum(self.norm_const, torch.tensor(0., requires_grad=False, device=self.device)) # Check to avoid numerical issues
        # Computing log P_tilde(X)
        self.log_p_tilde = torch.sum(self.alpha * self.x, dim=[1, 2])                                           \
                                            + 0.5 * self.lamda * torch.sum(self.x                               \
                                            * (temp_up + temp_low + temp_left + temp_right), dim=[1, 2])
        # Backup X
        self.x_temp = self.x.clone()

        batch_id = torch.arange(self.batch, device=self.device)

        # Sampling a pair of indices
        probs = self.sum_rows / torch.sum(self.sum_rows, 1, keepdim=True)  # [n, n]
        self.i = torch.multinomial(probs, 1)[:, 0]
        temp = torch.arange(0, self.batch, device=self.device)
        probs = self.g_ratio[temp, self.i, :] / torch.sum(self.g_ratio[temp, self.i, :], 1, keepdim=True) # [n, n] 
        self.j = torch.multinomial(probs, 1)[:, 0]

        # Compute the neighbours
        self.i_up = self.i - 1
        self.i_up[self.i_up < 0] = self.n - 1
        self.i_low = self.i + 1
        self.i_low[self.i_low == self.n] = 0
        self.j_left = self.j - 1
        self.j_left[self.j_left < 0] = self.n - 1
        self.j_right = self.j + 1
        self.j_right[self.j_right == self.n] = 0

        # Compute Z(X)
        norm_const_new = self.normalizing(batch_id, self.i, self.j)

        # Compute acceptance score
        self.accept = torch.minimum(torch.tensor(1., device=self.device), self.norm_const / norm_const_new)

        # Compute proposal score Q(X'|X)
        self.proposal = self.g_ratio[batch_id, self.i, self.j] / self.norm_const

        ####### 
        self.i_star = torch.randint(0, self.n, (self.batch,))
        self.j_star = torch.randint(0, self.n, (self.batch,))

        # Compute Z(X^*)
        norm_const_star = self.normalizing(batch_id, self.i_star, self.j_star)

        # Compute acceptance score
        self.accept_star = torch.minimum(torch.tensor(1., device=self.device), self.norm_const / norm_const_star)

        # Compute proposal score Q(X^*|X)
        self.proposal_star = self.g_ratio[batch_id, self.i_star, self.j_star] / self.norm_const
        ####### 


        # Compute unmnormalized true score
        self.p_tilde = self.log_p_tilde.clone()
        self.log_p_tilde = self.log_p_tilde                                                                        \
                           - 2. * self.alpha[batch_id, self.i, self.j] * self.x[batch_id, self.i, self.j]          \
                           - 2. * self.lamda * (self.x[batch_id, self.i, self.j_left]                              \
                                                + self.x[batch_id, self.i, self.j_right]                           \
                                                + self.x[batch_id, self.i_up, self.j]                              \
                                                + self.x[batch_id, self.i_low, self.j])

        bias = torch.ones_like(self.proposal, requires_grad=False, device=self.device)
        bias[idx] = self.eps
        self.old_proposal = bias

        # Update w, A, T
        self.w = (self.p_tilde - torch.max(self.p_tilde)).exp() / (self.old_proposal.detach() *  self.proposal.detach())
        self.A = self.accept * self.proposal
        self.T = torch.log(self.A) - self.log_p_tilde.detach()

        self.w_star = 1. / self.old_proposal.detach()
        self.A_star = (1. - self.accept_star * self.proposal_star)
        par = self.softplus(self.eta)
        self.T_star = par * self.A_star - (self.p_tilde - torch.max(self.p_tilde)).exp() * (torch.log(par) + 1.)

        self.x[batch_id, self.i, self.j] = - self.x[batch_id, self.i, self.j]

        # Compute objective
        self.obj = (torch.sum(self.w * self.A * self.T) + torch.sum(self.w_star * self.A_star * self.T_star)) / self.batch

        self.optimizer.zero_grad()
        self.obj.backward() # Keep the whole graph for each call of sample_burn_in
        self.optimizer.step()

        # Accept or reject trajectory
        id_reject = torch.rand((self.batch,), device=self.device) > self.accept
        if id_reject[id_reject == True].nelement() != 0:
            self.x[batch_id[id_reject], :, :] = self.x_temp[batch_id[id_reject], :, :]


    ##### Function used to sample a new configuration
    def sample(self):

        batch_id = torch.arange(self.batch, device=self.device)

        # Sampling a pair of indices
        probs = self.sum_rows / torch.sum(self.sum_rows, 1, keepdim=True)
        self.i = torch.multinomial(probs, 1)[:, 0]
        temp = torch.arange(0, self.batch, device=self.device)
        probs = self.g_ratio[temp, self.i, :] / torch.sum(self.g_ratio[temp, self.i, :], 1, keepdim=True)
        self.j = torch.multinomial(probs, 1)[:, 0]

        # Compute the neighbours
        self.i_up = self.i - 1
        self.i_up[self.i_up < 0] = self.n - 1
        self.i_low = self.i + 1
        self.i_low[self.i_low == self.n] = 0
        self.j_left = self.j - 1
        self.j_left[self.j_left < 0] = self.n - 1
        self.j_right = self.j + 1
        self.j_right[self.j_right == self.n] = 0

        # Compute new normalization constant Z(X')
        delta_i = self.func(1./self.ratio[batch_id, self.i, self.j])                                                \
                - self.g_ratio[batch_id, self.i, self.j]                                                            \
                - self.g_ratio[batch_id, self.i, self.j_left]                                                       \
                - self.g_ratio[batch_id, self.i, self.j_right]                                                      \
                + self.func(self.ratio[batch_id, self.i, self.j_left]                                               \
                            * torch.exp(4. * self.lamda * self.x[batch_id, self.i, self.j_left]                     \
                                        * self.x[batch_id, self.i, self.j]))                                        \
                + self.func(self.ratio[batch_id, self.i, self.j_right]                                              \
                            * torch.exp(4. * self.lamda * self.x[batch_id, self.i, self.j_right]                    \
                                        * self.x[batch_id, self.i, self.j]))                    
        delta_i_up = - self.g_ratio[batch_id, self.i_up, self.j]                                                    \
                    + self.func(self.ratio[batch_id, self.i_up, self.j]                                             \
                                * torch.exp(4. * self.lamda * self.x[batch_id, self.i_up, self.j]                   \
                                            * self.x[batch_id, self.i, self.j]))
        delta_i_low = - self.g_ratio[batch_id, self.i_low, self.j]                                                  \
                    + self.func(self.ratio[batch_id, self.i_low, self.j]                                            \
                                * torch.exp(4. * self.lamda * self.x[batch_id, self.i_low, self.j]                  \
                                            * self.x[batch_id, self.i, self.j]))
        norm_const_new = self.norm_const + delta_i + delta_i_up + delta_i_low

        self.accept = torch.minimum(torch.tensor(1., device=self.device), self.norm_const / norm_const_new)

        id_accept = torch.rand((self.batch,), device=self.device) < self.norm_const / norm_const_new

        # Acceptance
        if id_accept[id_accept == True].nelement() != 0:
            self.ratio[batch_id[id_accept], self.i[id_accept], self.j[id_accept]] =                                 \
                            1./self.ratio[batch_id[id_accept], self.i[id_accept], self.j[id_accept]]
            self.ratio[batch_id[id_accept], self.i_up[id_accept], self.j[id_accept]] =                              \
                            self.ratio[batch_id[id_accept], self.i_up[id_accept], self.j[id_accept]]                \
                            * torch.exp(4. * self.lamda                                                             \
                                        * self.x[batch_id[id_accept], self.i_up[id_accept], self.j[id_accept]]      \
                                        * self.x[batch_id[id_accept], self.i[id_accept], self.j[id_accept]])
            self.ratio[batch_id[id_accept], self.i_low[id_accept], self.j[id_accept]] =                             \
                            self.ratio[batch_id[id_accept], self.i_low[id_accept], self.j[id_accept]]               \
                            * torch.exp(4. * self.lamda                                                             \
                                        * self.x[batch_id[id_accept], self.i_low[id_accept], self.j[id_accept]]     \
                                        * self.x[batch_id[id_accept], self.i[id_accept], self.j[id_accept]])
            self.ratio[batch_id[id_accept], self.i[id_accept], self.j_left[id_accept]] =                            \
                            self.ratio[batch_id[id_accept], self.i[id_accept], self.j_left[id_accept]]              \
                            * torch.exp(4. * self.lamda                                                             \
                                        * self.x[batch_id[id_accept], self.i[id_accept], self.j_left[id_accept]]    \
                                        * self.x[batch_id[id_accept], self.i[id_accept], self.j[id_accept]])
            self.ratio[batch_id[id_accept], self.i[id_accept], self.j_right[id_accept]] =                           \
                            self.ratio[batch_id[id_accept], self.i[id_accept], self.j_right[id_accept]]             \
                            * torch.exp(4. * self.lamda                                                             \
                                        * self.x[batch_id[id_accept], self.i[id_accept], self.j_right[id_accept]]   \
                                        * self.x[batch_id[id_accept], self.i[id_accept], self.j[id_accept]])

            self.g_ratio[batch_id[id_accept], self.i[id_accept], self.j[id_accept]] =                               \
                            self.func(self.ratio[batch_id[id_accept], self.i[id_accept], self.j[id_accept]])
            self.g_ratio[batch_id[id_accept], self.i_up[id_accept], self.j[id_accept]] =                            \
                            self.func(self.ratio[batch_id[id_accept], self.i_up[id_accept], self.j[id_accept]])
            self.g_ratio[batch_id[id_accept], self.i_low[id_accept], self.j[id_accept]] =                           \
                            self.func(self.ratio[batch_id[id_accept], self.i_low[id_accept], self.j[id_accept]])
            self.g_ratio[batch_id[id_accept], self.i[id_accept], self.j_left[id_accept]] =                          \
                            self.func(self.ratio[batch_id[id_accept], self.i[id_accept], self.j_left[id_accept]])
            self.g_ratio[batch_id[id_accept], self.i[id_accept], self.j_right[id_accept]] =                         \
                            self.func(self.ratio[batch_id[id_accept], self.i[id_accept], self.j_right[id_accept]])

            self.sum_rows[batch_id[id_accept], self.i[id_accept]] =                                                 \
                            self.sum_rows[batch_id[id_accept], self.i[id_accept]] + delta_i[id_accept]
            self.sum_rows[batch_id[id_accept], self.i_up[id_accept]] =                                              \
                            self.sum_rows[batch_id[id_accept], self.i_up[id_accept]] + delta_i_up[id_accept]
            self.sum_rows[batch_id[id_accept], self.i_low[id_accept]] =                                             \
                            self.sum_rows[batch_id[id_accept], self.i_low[id_accept]] + delta_i_low[id_accept]
            self.sum_rows = torch.maximum(self.sum_rows, torch.tensor(0., device=self.device)) # Check to avoid numerical issues

            self.norm_const[id_accept] = norm_const_new[id_accept]
            self.norm_const = torch.maximum(self.norm_const, torch.tensor(0., device=self.device)) # Check to avoid numerical issues

            self.x[batch_id[id_accept], self.i[id_accept], self.j[id_accept]] =                                     \
                            - self.x[batch_id[id_accept], self.i[id_accept], self.j[id_accept]]


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
        # if self.burn_in == True:
        #     normalized = self.softmax(self.theta + torch.randn_like(self.theta))
        # else:
        #     normalized = self.softmax(self.theta)
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

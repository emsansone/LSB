import torch

##### Locally Balanced Sampler
# Useful functions
# sample()
class LBS():

    ##### Function initializing elements used 
    ##### to compute and sample from Q(X'|X))
    # batch   - number of chains
    # n       - size of square lattice
    # lamda   - scalar
    # alpha   - n x n matrix of bias coefficients
    # func    - 'barker', 'sqrt', 'min', 'max' 
    # start_x - batch x n x n initial tensor
    # device  - device to run the simulation
    def __init__(self, batch, n, lamda, alpha, func, start_x=None, device=None):
        self.batch = batch
        self.n = n
        self.lamda = lamda
        self.device = device
        self.alpha = alpha.to(self.device)
        if func == 'barker':
            self.func = self.barker
        elif func == 'sqrt':
            self.func = self.sqrt
        elif func == 'min':
            self.func = self.minim
        elif func == 'max':
            self.func = self.maxim
        else:
            raise Exception('Wrong selection of balancing function!')

        # Computing invariants
        if start_x == None:
            self.x = torch.randint(2, (batch, n, n), device=self.device)*2 - 1
        else:
            self.x = start_x.clone()

        temp_up = torch.cat((self.x[:, 1:, :], self.x[:, :1, :]), 1)        # Moving up 1 row
        temp_low = torch.cat((self.x[:, -1:,:], self.x[:, :-1, :]), 1)     # Moving down 1 row
        temp_left = torch.cat((self.x[:, :, 1:], self.x[:, :, :1]), 2)      # Moving left 1 col
        temp_right = torch.cat((self.x[:, :, -1:], self.x[:, :, :-1]), 2)  # Moving right 1 col
        self.ratio = torch.exp(-2. * self.alpha * self.x
                                -2. * self.lamda * self.x
                                * (temp_up + temp_low + temp_left + temp_right))
        self.g_ratio = self.func(self.ratio)
        self.sum_rows = torch.sum(self.g_ratio, 2)
        self.sum_rows[self.sum_rows < 0.] = 0. # Check to avoid numerical issues
        self.norm_const = torch.sum(self.sum_rows, 1)
        self.norm_const[self.norm_const < 0.] = 0. # Check to avoid numerical issues
        self.accept = None


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
            self.sum_rows[self.sum_rows < 0.] = 0. # Check to avoid numerical issues

            self.norm_const[id_accept] = norm_const_new[id_accept]
            self.norm_const[self.norm_const < 0.] = 0. # Check to avoid numerical issues

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


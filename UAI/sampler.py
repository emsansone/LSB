import torch
import copy

##### Locally Balanced Sampler
# Useful functions
# sample()
class LBS():

    ##### Function initializing elements used 
    ##### to compute and sample from Q(X'|X))
    # batch    - number of chains
    # func     - 'barker', 'sqrt', 'min', 'max' 
    # start_x  - dictionary with variables and their state value
    # model    - Markov model according to pgmpy library
    def __init__(self, batch, func, start_x=None, model=None):
        self.batch = batch
        self.eps = 1e-6
        self.max_eps = 1e6

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

        self.x = [copy.deepcopy(data) for data in start_x]
        self.vars_to_sample = list(set(self.x[0].keys()))
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
                    if value != value_1:
                        if value < self.eps:
                            value = self.eps
                        self.ratio[b, var_id] *= value_1 / value
                    self.ratio[b, var_id] = torch.minimum(self.ratio[b, var_id], torch.tensor(self.max_eps, requires_grad=False))
                var_id += 1
        self.g_ratio = self.func(self.ratio)
        self.norm_const = torch.sum(self.g_ratio, 1, keepdim=True)
        self.norm_const = torch.maximum(self.norm_const, torch.tensor(0.))
        self.accept = torch.zeros((self.batch,))

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
                    if value != value_1:
                        if value < self.eps:
                            value = self.eps
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


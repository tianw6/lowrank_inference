from low_rank_rnns.helpers import *
import torch.nn as nn
from math import sqrt, floor
import random
import time
import numpy as np

def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)

    ############################## Tian changed this
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # loss_tensor = ((target - output)).pow(2).mean(dim=-1)
    ##############################


    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


def accuracy_general(output, targets, mask):
    good_trials = (targets != 0).any(dim=1).squeeze()
    target_decisions = torch.sign((targets[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze())
    decisions = torch.sign((output[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze())
    return (target_decisions == decisions).type(torch.float32).mean()


def train(net, _input, _target, _mask, n_epochs, lr=1e-2, batch_size=32, plot_learning_curve=False, plot_gradient=False,
          mask_gradients=False, clip_gradient=None, early_stop=None, keep_best=False, cuda=False, resample=False,
          initial_states=None):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param mask_gradients: bool, set to True if training the SupportLowRankRNN_withMask for reduced models
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param early_stop: None or float, set to target loss value after which to immediately stop if attained
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors if desired
    :return: nothing
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            if cuda == True:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{cuda}')
    else:
        device = torch.device('cpu')

    device = torch.device('cpu')
    

    net.to(device=device)
    input = _input.to(device=device, dtype=torch.float32)   # TODO do we need _input
    target = _target.to(device=device, dtype=torch.float32)
    mask = _mask.to(device=device, dtype=torch.float32)
    if initial_states is not None:
        initial_states = initial_states.to(device=device, dtype=torch.float32)

    # Initialize setup to keep best network
    with torch.no_grad():
        output, h = net(input, initial_states=initial_states)
        initial_loss = loss_mse(output, target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    # Training loop
    for epoch in range(n_epochs):
        begin = time.time()
        losses = []  # losses over the whole epoch
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()



            ############################# Tian changed this
            # random_batch_idx = random.sample(range(num_examples), batch_size)

            random_batch_idx = np.arange(batch_size*i, batch_size*(i + 1))

            ##################################

            batch = input[random_batch_idx]
            if initial_states is not None:
                output, h = net(batch, initial_states=initial_states[random_batch_idx])
            else:
                output, h = net(batch)

            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            

            ################################## Tian changed this
            # add L2 regularization 
            loss += 0.001*torch.norm(net.w_rec_eff, p=2)/np.sqrt(torch.numel(net.w_rec_eff)) + 0.001*torch.norm(h, p=2)/np.sqrt(torch.numel(h))
            ##################################

            losses.append(loss.item())
            all_losses.append(loss.item())
            loss.backward()



            if clip_gradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad ** 2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()

            # print("batch %d:  loss=%.3f  (took %.2f s) *" % (i, loss, time.time() - begin))


        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print("epoch %d:  loss=%.3f  (took %.2f s) *" % (epoch, np.mean(losses), time.time() - begin))
        else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" % (epoch, np.mean(losses), time.time() - begin))
        if early_stop is not None and np.mean(losses) < early_stop:
            break

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())


class FullRankRNN(nn.Module):  # TODO rename biases train_biases, add to cloning function (important !!!!)

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True, wi_mask = None, wrec_mask = None, wo_mask = None,
                 wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None, b_init=None,
                 add_biases=False, non_linearity=torch.relu, output_non_linearity=torch.tanh):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so


        self.non_linearity = non_linearity
        self.output_non_linearity = output_non_linearity

        self.wi_mask = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=False)
        self.wo_mask = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=False)
        self.wrec_mask = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=False)

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        if not add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False


        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if b_init is None:
                self.b.zero_()
            else:
                self.b.copy_(b_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)

            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_()



            ############### define masks
            if wrec_mask is not None:
                self.wrec_mask.copy_(wrec_mask)
                temp = torch.normal(0,1,size = (hidden_size,hidden_size))
                # self.wrec.copy_(temp*wrec_mask)

            else:
                self.wrec_mask.set_(torch.ones_like(self.wrec_mask))

            if wi_mask is not None:
                self.wi_mask.copy_(wi_mask)  
                temp = torch.normal(0,1,size = (input_size,hidden_size))
                # self.wi.copy_(temp*wi_mask)                
            else:
                self.wi_mask.set_(torch.ones_like(self.wi_mask))

            if wo_mask is not None:
                self.wo_mask.copy_(wo_mask)
                temp = torch.normal(0,1,size = (hidden_size, output_size))
                # self.wo.copy_(temp*wo_mask)
            else: 
                self.wo_mask.set_(torch.ones_like(self.wo_mask))         
            ###############


        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()


    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so



    def effective_weight(self, w, mask, w_fix=0):
        """ compute the effective weight """
    
        w_eff = torch.abs(w) * mask + w_fix
        return w_eff

    def effective_IO_weight(self, w, mask):
        
        w_eff = w*mask

        return w_eff   




    def forward(self, input, return_dynamics=True, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors for each trial if desired
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(initial_states)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.hidden_size, device=self.wrec.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):


            # self.wrec = self.wrec
            # h = h + self.noise_std * noise[:, i, :] + self.alpha * \
            #     (-h + r.matmul(self.wrec*self.wrec_mask) + input[:, i, :].matmul(self.wi_full*self.wi_mask))




            # compute effective weight
            self.w_rec_eff = self.effective_weight(w=self.wrec, mask=self.wrec_mask)
            self.w_in_eff = self.effective_IO_weight(w=self.wi_full, mask=self.wi_mask)
            self.w_out_eff = self.effective_IO_weight(w=self.wo_full, mask=self.wo_mask)

            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.w_rec_eff) + self.b + input[:, i, :].matmul(self.w_in_eff))

            r = self.non_linearity(h)
            output[:, i, :] = r @ (self.w_out_eff)     



            # self.wrec = nn.Parameter(torch.abs(self.wrec) * self.wrec_mask)

            # h = h + self.noise_std * noise[:, i, :] + self.alpha * \
            #     (-h + r.matmul(self.wrec) + self.b + input[:, i, :].matmul(self.wi_full*self.wi_mask))

            # r = self.non_linearity(h)
            # # output[:, i, :] = self.output_non_linearity(h) @ (self.wo_full*self.wo_mask)
            # output[:, i, :] = r @ (self.wo_full*self.wo_mask)






            if return_dynamics:
                trajectories[:, i + 1, :] = h

        if not return_dynamics:
            return output
        else:
            return output, trajectories

    def clone(self):
        new_net = FullRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                              self.rho, self.train_wi, self.train_wo, self.train_wrec, self.train_h0, self.train_si,
                              self.train_so, self.wi_mask, self.wrec_mask, self.wo_mask, self.wi, self.wo, self.wrec, self.si, self.so, self.b, False,
                              self.non_linearity, self.output_non_linearity)
        return new_net














class LowRankRNN(nn.Module):
    """
    This class implements the low-rank RNN. Instead of being parametrized by an NxN connectivity matrix, it is
    parametrized by two Nxr matrices m and n such that the connectivity is m * n^T
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha, rank=1, train_m = True,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True,
                 wi_init=None, wo_init=None, m_init=None, n_init=None, w_rec_eff = None, wrec_mask = None, wi_mask = None, wo_mask = None, si_init=None, so_init=None, h0_init=None,
                 add_biases=False, non_linearity=torch.relu, output_non_linearity=torch.relu):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rank: int
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool
        :param train_so: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param m_init: torch tensor of shape (hidden_size, rank)
        :param n_init: torch tensor of shape (hidden_size, rank)
        :param si_init: torch tensor of shape (input_size)
        :param so_init: torch tensor of shape (output_size)
        :param h0_init: torch tensor of shape (hidden_size)
        """
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.train_m = train_m
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = non_linearity
        self.output_non_linearity = output_non_linearity


        self.w_rec_eff = w_rec_eff

        self.wrec_mask = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=False)
        self.wi_mask = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=False)
        self.wo_mask = nn.Parameter(torch.Tensor(hidden_size, output_size), requires_grad=False)



        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))

        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.m = nn.Parameter(torch.Tensor(int(hidden_size/3), rank))
        self.n = nn.Parameter(torch.Tensor(int(hidden_size/3), rank))

        if not train_m:
            self.m.requires_grad = False
            
        if not train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        if not add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if m_init is None:
                self.m.normal_()
            else:
                self.m.copy_(m_init)
            if n_init is None:
                self.n.normal_()
            else:
                self.n.copy_(n_init)
            self.b.zero_()     # TODO add biases initializer
            if wo_init is None:
                self.wo.normal_(std=4.)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)



            ############### define masks
            if wi_mask is not None:
                self.wi_mask.copy_(wi_mask)  
            else:
                self.wi_mask.set_(torch.ones_like(self.wi_mask))

            if wo_mask is not None:
                self.wo_mask.copy_(wo_mask)
            else: 
                self.wo_mask.set_(torch.ones_like(self.wo_mask))      

            if wrec_mask is not None:
                self.wrec_mask.copy_(wrec_mask)
            else: 
                self.wrec_mask.set_(torch.ones_like(self.wrec_mask))                     
            ###############        


        self.wrec, self.wi_full, self.wo_full = [None] * 3
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wrec = None   # For optimization purposes the full connectivity matrix is never computed explicitly
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so



    def effective_weight(self, n, m, mask, w_rec_eff, w_fix=0):
        """ compute the effective weight """
    
        w_eff = w_fix + w_rec_eff

        w_eff[:100,:100] = n.matmul(m.t()) 

        return w_eff

    def effective_IO_weight(self, w, mask):
        
        w_eff = w*mask

        return w_eff  


    def forward(self, input, return_dynamics=True, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: boolean
        :return: if return_dynamics=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(h)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):



            # compute effective weight
            self.w_rec_eff = self.effective_weight(n=self.n, m = self.m, w_rec_eff = self.w_rec_eff, mask=self.wrec_mask)
            self.w_in_eff = self.effective_IO_weight(w=self.wi_full, mask=self.wi_mask)
            self.w_out_eff = self.effective_IO_weight(w=self.wo_full, mask=self.wo_mask)



            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.w_rec_eff) / self.hidden_size +
                    input[:, i, :].matmul(self.w_in_eff) + self.b)

            # h = h + self.noise_std * noise[:, i, :] + self.alpha * \
            #     (-h + r.matmul(self.w_rec_eff) / self.hidden_size +
            #         input[:, i, :].matmul(self.w_in_eff) + self.b)



            r = self.non_linearity(h)

            ################# Tian removed the average
            # output[:, i, :] = h @ (self.w_out_eff) / self.hidden_size
            output[:, i, :] = h @ (self.w_out_eff)

            if return_dynamics:
                trajectories[:, i + 1, :] = h



        if not return_dynamics:
            return output
        else:
            return output, trajectories





    def clone(self):


        new_net = LowRankRNN(self.input_size, self.hidden_size, self.output_size, self.noise_std, self.alpha,
                             self.rank, self.train_m, self.train_wi, self.train_wo, self.train_wrec, self.train_h0, self.train_si,
                             self.train_so, self.wi, self.wo, self.m, self.n, self.w_rec_eff, self.wrec_mask, self.wi_mask, self.wo_mask, self.si, self.so, self.h0, False,
                             self.non_linearity, self.output_non_linearity)
        new_net._define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override
        """
        if 'rec_noise' in state_dict:
            del state_dict['rec_noise']
        super().load_state_dict(state_dict, strict)
        self._define_proxy_parameters()

    def svd_reparametrization(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self._define_proxy_parameters()

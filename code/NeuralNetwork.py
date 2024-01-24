
import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################################
'''
Classes for NormNet:
Nonresidual Feedforward Neural Networks that use Weight Normalization

Important Parameters: share, use_bias, use_l2_wn, use_1pathnorm
share - Ties length parameter across all nonfinal linear layers
use_bias - Incorporates bias parameters across all nonfinal linear layers
use_l2_wn - Whether to use L2 weight normalization or the default of L1 WN
use_1pathnorm - determines which function is called by compute_reg()
                if false, calls a function to compute sum of L2 norms of weights

PSiLON Net is implemented with share=True, use_bias = True, use_l2_wn=False, use_1pathnorm=True
Standard Net is implemented with share=False, use_bias = True, use_l2_wn=True, use_1pathnorm=False
'''

class NormLinear(nn.Module):
    def __init__(self, input_size, output_size, share=False, use_bias=True,
                 use_l2_wn=False):
        super().__init__()
        self.share = share
        self.use_bias = use_bias
        self.use_l2_wn = use_l2_wn
        if use_bias:
            input_size = input_size + 1

        g_len = 1 if share else output_size
        self.g = nn.Parameter(torch.ones(1,g_len))
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        nn.init.orthogonal_(self.weight)

        self.sparsify = False
        self.n_sparsification_iters = 1e8
        self.sparsification_counter = 0

    # Normalization functions. Done 'vertically' since forward pass is implemented as X*W
    def _l1_normalize(self, W):
        norm = torch.sum(torch.abs(W), dim = 0).unsqueeze(0)
        return self.g*W/norm
    def _l2_normalize(self, W):
        norm = torch.sqrt(torch.sum(W**2, dim = 0)).unsqueeze(0)
        return self.g*W/norm
    def _l1_project(self, W):
        S = torch.sign(W)
        Y = torch.abs(W)
        U, _ = torch.sort(Y, dim=0, descending=True)
        k_vals = torch.arange(1, U.shape[0]+1, device=U.device)[:,None]
        V = (torch.cumsum(U, dim=0)-1)/k_vals
        idx = torch.sum(V < U, dim=0).long()-1
        tau = V[idx,torch.arange(U.shape[1], device=U.device)].unsqueeze(0)
        return(self.g*S*F.relu(Y-tau))
    
    def _exact_sparsity(self, W):
        # returns sum of exact sparsities for each column vector and number of columns
        assert self.sparsify
        W = self._l1_project(W)
        s = torch.sum(W == 0, dim = 0)/W.shape[0]
        return torch.sum(s).item(), s.shape[0]

    def _near_sparsity(self, W):
        # returns sum of near sparsities for each column vector and number of columns
        d = W.shape[0]
        pv = torch.abs(W)
        pv = pv/torch.sum(pv, dim=0).unsqueeze(0)
        H = torch.sum(-torch.log(pv+1e-8)*pv, dim=0)
        ns = 1 - torch.exp(H)/d
        return torch.sum(ns).item(), ns.shape[0]
    
    def init_sparsify(self, n_iters):
        self.sparsify = True
        self.n_sparsification_iters = n_iters
        self.sparsification_counter = 0

    def get_sparsity(self, near=True):
        with torch.no_grad():
            if near:
                return self._near_sparsity(self.weight)
            else:
                return self._exact_sparsity(self.weight)
            
    def normalize(self, W):
        if self.use_l2_wn:
            return self._l2_normalize(W)
        else:
            return self._l1_normalize(W)

    def forward(self, X):
        if self.use_bias:
            X = torch.cat([torch.ones_like(X[:,[0]]), X], dim=1)
        W = self.normalize(self.weight)
        if self.sparsify:
            W_sparse = self._l1_project(self.weight)
            self.sparsification_counter += 1
            alpha = min(self.sparsification_counter/self.n_sparsification_iters,1)
            W = (1-alpha)*W + alpha*W_sparse
        Z = torch.mm(X,W)
        return Z
    
class NormBlock(nn.Module):
    def __init__(self, input_size, output_size, share=False, 
                 use_bias=True, use_l2_wn=False):
        super().__init__()
        self.linear = NormLinear(input_size, output_size, 
                                 share=share,
                                 use_bias=use_bias,
                                 use_l2_wn=use_l2_wn)
        self.nonlinearity = nn.ReLU()
    def forward(self, X):
        return self.linear(self.nonlinearity(X))
    def init_sparsify(self, n_iters):
        self.linear.init_sparsify(n_iters)
    def get_sparsity(self, near=True):
        return self.linear.get_sparsity(near)

class NormNet(nn.Module):
    '''
    General structure is
    1. Embedder: linear map: input_size -> hidden_size
    2. Representer: (n_hidden-1) layers of 
        (nonlinearity + linear map: hidden_size -> hidden_size) each
    3. Apply Nonlinearity
    4. Predictor 1 final linear map: hidden_size -> output_size
        - Add final bias
        - Any output nonlinearity should be applied after usage of this module
    '''
    def __init__(self, input_size, hidden_size, output_size, n_hidden=1, 
                 share=False, use_bias=True, use_l2_wn=False, use_1pathnorm=True):
        super().__init__()
        assert n_hidden >= 1
        self.share = share
        self.use_bias = use_bias
        self.use_l2_wn = use_l2_wn
        self.use_1pathnorm = use_1pathnorm

        self.embedder = NormLinear(input_size, hidden_size, share=share, 
                                   use_bias=use_bias, use_l2_wn=use_l2_wn)
        self.representer = [NormBlock(hidden_size, hidden_size, share=share, 
                                      use_bias=use_bias, use_l2_wn=use_l2_wn) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.predictor = NormLinear(hidden_size, output_size, 
                                    share=False, use_bias=False, 
                                    use_l2_wn=use_l2_wn)

        self.nonlinearity = nn.ReLU()
        self.final_bias = nn.Parameter(torch.zeros(1,output_size))

    def _get_formatted_weight(self, W):
        W = torch.transpose(W,0,1)
        if self.use_bias:
            v = torch.zeros_like(W[0,:])
            v[0] = 1
            W = torch.cat([v.unsqueeze(0), W], dim = 0)
        return W
    
    def _get_formatted_weight_final(self, W):
        W = torch.transpose(W,0,1)
        if self.use_bias:
            v = torch.zeros_like(W[:,0])
            W = torch.cat([v.unsqueeze(1), W], dim=1)
        return W

    def _compute_1pathnorm_explicit(self):
        # explicitly compute the 1-path-norm via the matrix product formula
        W1 = self.embedder.normalize(self.embedder.weight)
        W1 = self._get_formatted_weight(W1)
        reg = torch.mm(torch.abs(W1), torch.ones_like(W1[:,0]).unsqueeze(1))
        for module in self.representer:
            Wi = module.linear.normalize(module.linear.weight)
            Wi = self._get_formatted_weight(Wi)
            reg = torch.mm(torch.abs(Wi),reg)
        WK = self.predictor.normalize(self.predictor.weight)
        WK = self._get_formatted_weight_final(WK)
        reg = torch.mm(torch.abs(WK),reg)
        return torch.sum(reg)
    
    def _compute_1pathnorm_simple(self):
        # In the case of L1 normalization with parameter sharing, use simplified formula
        reg = self.embedder.g[0,0]
        for module in self.representer:
            reg = reg*module.linear.g[0,0]
        reg = torch.abs(reg)*torch.sum(torch.abs(self.predictor.g))
        return reg
    
    def _compute_l2norm(self):
        W1 = self.embedder.weight
        reg = torch.sum(torch.pow(W1,2))
        for module in self.representer:
            Wi =  module.linear.weight
            reg = reg + torch.sum(torch.pow(Wi,2))
        WK = self.predictor.weight
        reg = reg + torch.sum(torch.pow(WK,2))
        return torch.sum(reg)

    @torch.jit.export
    def compute_reg(self):
        if self.use_1pathnorm:
            if self.use_l2_wn or (not self.share):
                return self._compute_1pathnorm_explicit()
            else:
                return self._compute_1pathnorm_simple()
        else:
            return self._compute_l2norm()

    def init_sparsify(self, n_iter):
        self.embedder.init_sparsify(n_iter)
        self.predictor.init_sparsify(n_iter)
        for module in self.representer:
            module.init_sparsify(n_iter)

    @torch.jit.export
    def get_sparsity(self, near=True):
        s, l = 0, 0
        with torch.no_grad():
            s1, l1 = self.embedder.get_sparsity(near)
            s, l = s+s1, l+l1
            for module in self.representer:
                si, li = module.get_sparsity(near)
                s, l = s+si, l+li
            sK, lK = self.predictor.get_sparsity(near)
            s, l = s+sK, l+lK
        return s/l
    
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        Z = self.nonlinearity(Z)
        return Z
    
    def forward(self, X):
        H = self.get_representation(X)
        y_preactivation = self.predictor(H) + self.final_bias
        return y_preactivation

########################################################################################
'''
Classes for NormResNet:
CReLU Residual Neural Networks that use Weight Normalization
    - Has nearly identical structure as NormNet, adapted for the CReLU residual case

NEW Important Parameter: use_improved_1pathnorm
    - determines whether to use the improved bound or the original 1pathnorm bound
      when use_1pathnorm=True

PSiLON Net is implemented with share=True, use_bias = True, use_l2_wn=False,
                               use_1pathnorm=True, use_improved_1pathnorm=True
'''


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinearity = nn.ReLU()
    def forward(self, X):
        return self.nonlinearity(X), -self.nonlinearity(-X)

class CReLUNormLinear(nn.Module):
    def __init__(self, input_size, output_size, share = False, 
                 use_bias = True, use_l2_wn = False, init_zero = False):
        super().__init__()
        self.share = share
        self.use_bias = use_bias
        self.use_l2_wn = use_l2_wn
        if use_bias:
            input_size = input_size + 1

        g_len = 1 if share else output_size
        if init_zero:
            self.g = nn.Parameter(torch.zeros(1,g_len))
        else:
            self.g = nn.Parameter(torch.ones(1,g_len))

        # Initializes pos weight matrix to be an orthogonal matrix
        self.weight_pos = torch.empty(input_size, output_size)
        nn.init.orthogonal_(self.weight_pos)
        # initialize neg weight matrix to same values as pos
        self.weight_neg = torch.empty(input_size, output_size)
        self.weight_neg.copy_(self.weight_pos.data)
        self.weight_pos = nn.Parameter(self.weight_pos)
        self.weight_neg = nn.Parameter(self.weight_neg)

        self.sparsify = False
        self.n_sparsification_iters = 1e8
        self.sparsification_counter = 0

    # Normalization functions. Done "vertically" since forward pass is implemented as X*W
    def _l1_normalize(self, W_pos, W_neg):
        W_tilde = torch.maximum(torch.abs(W_pos), torch.abs(W_neg))
        norm = torch.sum(W_tilde, dim = 0)[None,:] 
        return self.g*W_pos/norm, self.g*W_neg/norm
    def _l2_normalize(self, W_pos, W_neg):
        W_tilde = torch.maximum(torch.abs(W_pos), torch.abs(W_neg))
        norm = torch.sqrt(torch.sum(W_tilde**2, dim = 0)).unsqueeze(0)
        return self.g*W_pos/norm, self.g*W_neg/norm
    def _l1_project(self, W_pos, W_neg):
        # get tau using only W_tilde
        W_tilde = torch.maximum(torch.abs(W_pos), torch.abs(W_neg))
        Y = torch.abs(W_tilde)
        U, _ = torch.sort(Y, dim=0, descending=True)
        k_vals = torch.arange(1, U.shape[0]+1, dtype=U.dtype, device=U.device)[:,None]
        V = (torch.cumsum(U, dim=0)-1)/k_vals
        idx = torch.sum(V < U, dim=0).long()-1
        tau = V[idx,torch.arange(U.shape[1], device=U.device)].unsqueeze(0)

        # use tau to project W_pos and W_neg
        W_pos_project = torch.sign(W_pos)*F.relu(torch.abs(W_pos)-tau)
        W_neg_project = torch.sign(W_neg)*F.relu(torch.abs(W_neg)-tau)
        return self.g*W_pos_project, self.g*W_neg_project
    
    def _exact_sparsity(self, W_pos, W_neg):
        # returns sum of exact sparsities for each column vector and number of columns
        assert self.sparsify
        W_pos, W_neg = self._l1_project(W_pos, W_neg)
        s_pos = torch.sum(W_pos == 0, dim = 0)/W_pos.shape[0]
        s_neg = torch.sum(W_neg == 0, dim = 0)/W_neg.shape[0]
        s = s_pos + s_neg
        return torch.sum(s).item(), s.shape[0]

    def _near_sparsity(self, W_pos, W_neg):
        # returns sum of near sparsities for each column vector and number of columns
        d = W_pos.shape[0]
        pv_pos = torch.abs(W_pos)
        pv_pos = pv_pos/torch.sum(pv_pos, dim=0).unsqueeze(0)
        pv_neg = torch.abs(W_neg)
        pv_neg = pv_neg/torch.sum(pv_neg, dim=0).unsqueeze(0)
        H_pos = torch.sum(-torch.log(pv_pos+1e-8)*pv_pos, dim=0)
        H_neg = torch.sum(-torch.log(pv_neg+1e-8)*pv_neg, dim=0)
        s_pos = 1 - torch.exp(H_pos)/d
        s_neg = 1 - torch.exp(H_neg)/d
        s = s_pos + s_neg
        return torch.sum(s).item(), s.shape[0]
    
    def init_sparsify(self, n_iters):
        self.sparsify = True
        self.n_sparsification_iters = n_iters
        self.sparsification_counter = 0

    def get_sparsity(self, near=True):
        with torch.no_grad():
            if near:
                return self._near_sparsity(self.weight_pos, self.weight_neg)
            else:
                return self._exact_sparsity(self.weight_pos, self.weight_neg)
            
    def normalize(self, W_pos, W_neg):
        if self.use_l2_wn:
            return self._l2_normalize(W_pos, W_neg)
        else:
            return self._l1_normalize(W_pos, W_neg)
        
    def forward(self, X_pos, X_neg):
        if self.use_bias:
            X_pos = torch.cat([torch.ones_like(X_pos[:,[0]]), X_pos], dim=1)
            X_neg = torch.cat([torch.zeros_like(X_neg[:,[0]]), X_neg], dim=1)

        W_pos, W_neg = self.normalize(self.weight_pos, self.weight_neg)
        if self.sparsify:
            W_pos_sparse, W_neg_sparse = self._l1_project(self.weight_pos, self.weight_neg)
            self.sparsification_counter += 1
            alpha = min(self.sparsification_counter/self.n_sparsification_iters,1)
            W_pos = (1-alpha)*W_pos + alpha*W_pos_sparse
            W_neg = (1-alpha)*W_neg + alpha*W_neg_sparse

        Z = torch.mm(X_pos, W_pos) + torch.mm(X_neg, W_neg)
        return Z

class NormResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, share=False, 
                 use_bias=True, use_l2_wn=False, init_zero=False):
        super().__init__()
        self.use_bias = use_bias
        self.linear = CReLUNormLinear(input_size, output_size, 
                                      share=share, 
                                      use_bias=use_bias, 
                                      init_zero=init_zero,
                                      use_l2_wn=use_l2_wn)
        self.activation = CReLU()
    def init_sparsify(self, n_iters):
        self.linear.init_sparsify(n_iters)
    def get_sparsity(self, near=True):
        return self.linear.get_sparsity(near)
    def forward(self, X):
        return X + self.linear(*self.activation(X))

class NormResNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1,
                 share=False, use_bias=True, use_l2_wn=False, use_1pathnorm=True,
                 use_improved_1pathnorm=True):
        super().__init__()
        '''
        General structure is
        1. Embedder: linear map: input_size -> hidden_size
        2. Representer: (n_hidden-1) layers of 
            (nonlinearity + linear map: hidden_size -> 2*hidden_size -> hidden_size) each
            Then, a last nonlinearity: hidden_size -> 2*hidden_size
        3. Predictor 1 final linear map: 2*hidden_size -> output_size)
            - final nonlinearity is to be applied after usage of this module
        '''
        assert n_hidden >= 1
        self.share = share
        self.use_bias = use_bias
        self.use_l2_wn = use_l2_wn
        self.use_1pathnorm = use_1pathnorm
        self.use_improved_1pathnorm = use_improved_1pathnorm

        self.embedder = NormLinear(input_size, hidden_size, 
                                   share=share, use_bias=use_bias,
                                    use_l2_wn=use_l2_wn)
        self.representer = [NormResidualBlock(hidden_size, hidden_size,
                                              share=share, use_bias=use_bias,
                                                use_l2_wn=use_l2_wn, init_zero=True) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.predictor = CReLUNormLinear(hidden_size, output_size, 
                                         share=False, use_bias=False, 
                                         use_l2_wn=use_l2_wn, init_zero=False)
        
        self.final_nonlinearity = CReLU()
        self.final_bias = nn.Parameter(torch.zeros(1,output_size))

    def _get_formatted_weight(self, W_pos, W_neg):
        W_pos, W_neg = torch.transpose(W_pos,0,1), torch.transpose(W_neg,0,1)
        if self.use_bias:
            v = torch.zeros_like(W_pos[0,:])
            v[0] = 1
            W_pos = torch.cat([v.unsqueeze(0), W_pos], dim = 0)
            W_neg = torch.cat([v.unsqueeze(0), W_neg], dim = 0)
        return W_pos, W_neg
    
    def _get_formatted_weight_final(self, W_pos, W_neg):
        W_pos, W_neg = torch.transpose(W_pos,0,1), torch.transpose(W_neg,0,1)
        if self.use_bias:
            v = torch.zeros_like(W_pos[:,0])
            W_pos = torch.cat([v.unsqueeze(1), W_pos], dim = 1)
            W_neg = torch.cat([v.unsqueeze(1), W_neg], dim = 1)
        return W_pos, W_neg

    def _compute_1pathnorm_explicit(self):
        # explicitly compute the 1-path-norm via the matrix product formula
        W1 = torch.transpose(self.embedder.normalize(self.embedder.weight), 0, 1)
        if self.use_bias:
            v = torch.zeros_like(W1[[0],:])
            W1 = torch.cat([v, W1], dim = 0)
        reg = torch.mm(torch.abs(W1), torch.ones_like(W1[:,[0]]))

        for module in self.representer:
            Wi_pos, Wi_neg = module.linear.normalize(module.linear.weight_pos, 
                                                     module.linear.weight_neg)
            Wi_pos, Wi_neg = self._get_formatted_weight(Wi_pos, Wi_neg)
            I = torch.eye(Wi_pos.shape[0], dtype=Wi_pos.dtype, 
                          layout=Wi_pos.layout, device=Wi_pos.device)
            I[0,0] = 0
            # needed matrix looks like | 1  0^T     | most stucture already taken
            #                          | b  I+W_pos | care of by design and other fxns
            reg = torch.mm(torch.abs(torch.cat([I+Wi_pos,I+Wi_neg], dim=1)), 
                           torch.cat([reg,reg], dim=0))

        WK_pos, WK_neg = self.predictor.normalize(self.predictor.weight_pos, 
                                                  self.predictor.weight_neg)
        WK_pos, WK_neg = self._get_formatted_weight_final(WK_pos, WK_neg)
        I = torch.eye(WK_pos.shape[0], dtype=WK_pos.dtype, 
                          layout=WK_pos.layout, device=WK_pos.device)
        I[0,0] = 0
        reg = reg = torch.mm(torch.abs(torch.cat([I+WK_pos,I+WK_neg], dim=1)),
                              torch.cat([reg,reg], dim=0))
        return torch.sum(reg)

    def _compute_improved_1pathnorm_explicit(self):
        # explicitly compute the 1-path-norm via the matrix product formula using improved formula
        W1 = torch.transpose(self.embedder.normalize(self.embedder.weight), 0, 1)
        if self.use_bias:
            v = torch.zeros_like(W1[[0],:])
            W1 = torch.cat([v, W1], dim = 0)
        reg = torch.mm(torch.abs(W1), torch.ones_like(W1[:,[0]]))
        for module in self.representer:
            Wi_pos, Wi_neg = module.linear.normalize(module.linear.weight_pos, 
                                                     module.linear.weight_neg)
            Wi_pos, Wi_neg = self._get_formatted_weight(Wi_pos, Wi_neg)
            Wi_tilde = torch.maximum(torch.abs(Wi_pos), torch.abs(Wi_neg))
            I = torch.eye(Wi_pos.shape[0], dtype=Wi_pos.dtype, 
                          layout=Wi_pos.layout, device=Wi_pos.device)
            I[0,0] = 0
            reg = torch.mm(I+Wi_tilde,reg)
        WK_pos, WK_neg = self.predictor.normalize(self.predictor.weight_pos, 
                                                  self.predictor.weight_neg)
        WK_pos, WK_neg = self._get_formatted_weight_final(WK_pos, WK_neg)
        WK_tilde = torch.maximum(torch.abs(WK_pos), torch.abs(WK_neg))
        I = torch.eye(WK_pos.shape[0], dtype=WK_pos.dtype, 
                          layout=WK_pos.layout, device=WK_pos.device)
        I[0,0] = 0
        reg = reg = torch.mm(I+WK_tilde,reg)
        return torch.sum(reg)
    
    def _compute_improved_1pathnorm_simple(self):
        reg = torch.abs(self.embedder.g[0,0])
        for module in self.representer:
            reg = reg*(1 + torch.abs(module.linear.g)[0,0])
        reg = reg*torch.sum(torch.abs(self.predictor.g))
        return reg
    
    def _compute_l2norm(self):
        W1 = self.embedder.weight
        reg = torch.sum(torch.pow(W1,2))
        for module in self.representer:
            Wi_pos, Wi_neg =  module.linear.weight_pos, module.linear.weight_neg
            reg = reg + torch.sum(torch.pow(Wi_pos,2)) + torch.sum(torch.pow(Wi_neg,2))
        WK_pos, WK_neg = self.predictor.weight_pos, self.predictor.weight_neg
        reg = reg + torch.sum(torch.pow(WK_pos,2)) + torch.sum(torch.pow(WK_neg,2))
        return torch.sum(reg)
    
    @torch.jit.export
    def compute_reg(self):
        if self.use_1pathnorm:
            if self.use_improved_1pathnorm:
                if self.use_l2_wn or (not self.share): # not PSiLON
                    return self._compute_improved_1pathnorm_explicit()
                else:
                    return self._compute_improved_1pathnorm_simple()
            else:
                return self._compute_1pathnorm_explicit()
        else:
            return self._compute_l2norm()
    
    def init_sparsify(self, n_iter):
        self.embedder.init_sparsify(n_iter)
        self.predictor.init_sparsify(n_iter)
        for module in self.representer:
            module.init_sparsify(n_iter)

    @torch.jit.export
    def get_sparsity(self, near=True):
        s, l = 0, 0
        with torch.no_grad():
            s1, l1 = self.embedder.get_sparsity(near)
            s, l = s+s1, l+l1
            for module in self.representer:
                si, li = module.get_sparsity(near)
                s, l = s+si, l+li
            sK, lK = self.predictor.get_sparsity(near)
            s, l = s+sK, l+lK
        return s/l
    
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        Z_pos, Z_neg = self.final_nonlinearity(Z)
        return Z_pos, Z_neg 
    def forward(self, X):
        Z_pos, Z_neg = self.get_representation(X)
        y_preactivation = self.predictor(Z_pos, Z_neg) + self.final_bias
        return y_preactivation
    
############################################################################################
'''
Basic MLP for a Baseline
'''

from collections import OrderedDict

class BasicNet(nn.Module):
    # Standard MLP with L2 weight regularization
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1, accelerator = 'cpu'):
        super().__init__()
        self.input_size = input_size
        assert n_hidden >= 1
        self.n_hidden = n_hidden
        self.embedder = nn.Linear(input_size, hidden_size)
        self.representer = [nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                       ('linear', nn.Linear(hidden_size,hidden_size))])) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.final_nonlinearity = nn.ReLU()
        self.predictor = nn.Linear(hidden_size, output_size)
  
    @torch.jit.export
    def compute_reg(self):
        W1 = self.embedder.weight
        reg = torch.sum(torch.pow(W1,2))
        for module in self.representer:
            Wi =  module[1].weight
            reg = reg + torch.sum(torch.pow(Wi,2))
        WK = self.predictor.weight
        reg = reg + torch.sum(torch.pow(WK,2))
        return torch.sum(reg)
    
    @torch.jit.export
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        Z = self.final_nonlinearity(Z)
        return Z
    def forward(self, X):
        H = self.get_representation(X)
        y_preactivation = self.predictor(H)
        return y_preactivation

import torch
import torch.nn as nn
from torch import optim
import lightning as L
from torch.optim import lr_scheduler


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self,y,yhat):
        return torch.sqrt(self.mse(yhat,y))

class LitNetwork(L.LightningModule):
    def __init__(self, model, lambda_, compute_loss, compute_eval, total_steps):
        super().__init__()
        self.model = model
        self.compute_loss = compute_loss
        self.compute_eval = compute_eval
        self.lambda_ = lambda_
        self.total_steps = total_steps

    def forward(self, X):
        return self.model(X)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                                lr = 2e-3, 
                                weight_decay = 0)
        sch = self.build_scheduler(optimizer)
        return [optimizer], [sch]
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y_hat= self.model(X)
        loss = self.compute_loss(y_hat[:,0], y)
        loss_reg = loss + self.lambda_*self.model.compute_reg()
        self.log('train_loss', loss)
        return loss_reg
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y_hat= self.model(X)
        loss = self.compute_eval(y_hat[:,0], y)
        self.log('val_loss', loss)
    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y_hat= self.model(X)
        loss = self.compute_eval(y_hat[:,0], y)
        self.log('test_loss', loss)
    def build_scheduler(self, optimizer):
        k = int(self.total_steps/2)
        scheduler1 = lr_scheduler.LinearLR(optimizer, 
                                           start_factor = 1/20,
                                           end_factor = 1, 
                                           total_iters = 500)
        scheduler2 = lr_scheduler.LinearLR(optimizer, 
                                           start_factor = 1,
                                           end_factor = 1, 
                                           total_iters = k-500)
        scheduler3 = lr_scheduler.LinearLR(optimizer, 
                                           start_factor = 1,
                                           end_factor = 1/20,
                                           total_iters = k)
        scheduler = lr_scheduler.SequentialLR(optimizer, 
                                              [scheduler1, scheduler2, scheduler3],
                                              milestones = [500,k])
        return scheduler
    def get_nsparsity(self):
        ns, l = 0, 0
        model = self.model
        with torch.no_grad():
            ns1, l1 = self.nsparsity(model.embedder.weight.data)
            ns = ns + ns1
            l = l + l1
            for module in model.representer:
                nsi, li = self.nsparsity(module.linear.weight.data)
                ns = ns + nsi
                l = l + li
            nsK, lK = self.nsparsity(model.predictor.weight.data)
            ns = ns + nsK
            l = l + lK
        return (ns/l).item()
    
    def nsparsity(self, W):
        # returns sum of n_sparsities for each column vector and number of columns
        d = W.shape[0]
        pv = torch.abs(W)
        pv = pv/(torch.sum(pv, dim=0)[None,:])
        H = torch.sum(-torch.log(pv+1e-8)*pv, dim=0)
        ns = 1 - torch.exp(H)/d
        return torch.sum(ns), ns.shape[0]
        


####################################################################################

class StandardNormLinear(nn.Module):
    def __init__(self, input_size, output_size, share = False):
        super().__init__()
        # Allows learnable rescaling g of weights that is shared across all weight vectors
        # Allows a learned shift term b to be applied after the linear layer (aka 'bias' term)
        self.share = share
        if share:
            self.g = nn.Parameter(torch.ones(1,1))
        else:
            self.g = nn.Parameter(torch.ones(1,output_size))
        self.bias = nn.Parameter(torch.zeros(1,output_size))

        # Initializes weight matrix to be an orthogonal matrix
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        nn.init.orthogonal_(self.weight)

    # Normalization function. Note normalization would normally be done horizontally
    # when writing W*X, but here is instead done vertically here based on the matrix
    # multiplication in the forward pass being implemented as X*W
    @torch.jit.export
    def normalize(self, W, b):
        norm = torch.sum(torch.abs(W), dim = 0)[None,:] 
        if self.share:
            norm += torch.abs(b)
            return self.g*W/norm, self.g*b/norm
        else:
            return self.g*W/norm, b
    def forward(self, X):
        W, b = self.normalize(self.weight, self.bias)
        Z = torch.mm(X,W) + b
        return Z
    
class StandardNormBlock(nn.Module):
    def __init__(self, input_size, output_size, share = False):
        super().__init__()
        self.linear = StandardNormLinear(input_size, output_size, 
                                         share = share)
        self.nonlinearity = nn.ReLU()
    def forward(self, X):
        return self.linear(self.nonlinearity(X))
    
##################################################################################

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinearity = nn.ReLU()
    def forward(self, X):
        return self.nonlinearity(X), -self.nonlinearity(-X)


class CReLUNormLinear(nn.Module):
    def __init__(self, input_size, output_size, share = False, init_zero = False):
        super().__init__()
        # Allows learnable rescaling g of weights that is shared across all weight vectors
        # Allows a learned shift term b to be applied after the linear layer (aka 'bias' term)
        self.share = share
        if share:
            if init_zero:
                self.g = nn.Parameter(torch.zeros(1,1))
            else:
                self.g = nn.Parameter(torch.ones(1,1))
        else:
            self.g = nn.Parameter(torch.ones(1,output_size))
        self.bias = nn.Parameter(torch.zeros(1,output_size))

        # Initializes weight matrix to be an orthogonal matrix
        self.weight_pos = torch.empty(input_size, output_size)
        nn.init.orthogonal_(self.weight_pos)
        self.weight_neg = torch.empty(input_size, output_size)
        self.weight_neg.copy_(self.weight_pos.data)
        self.weight_pos = nn.Parameter(self.weight_pos)
        self.weight_neg = nn.Parameter(self.weight_neg)

    # Normalization function. Note normalization would normally be done horizontally
    # when writing W*X, but here is instead done vertically here based on the matrix
    # multiplication in the forward pass being implemented as X*W
    @torch.jit.export
    def normalize(self, W_pos, W_neg, b):
        W_tilde = torch.maximum(torch.abs(W_pos), torch.abs(W_neg))
        norm = torch.sum(W_tilde, dim = 0)[None,:] 
        if self.share:
            norm += torch.abs(b)
            return self.g*W_pos/norm, self.g*W_neg/norm, self.g*b/norm
        else:
            return self.g*W_pos/norm, self.g*W_neg/norm, b
        
    def forward(self, X_pos, X_neg):
        W_pos, W_neg, b = self.normalize(self.weight_pos, self.weight_neg, self.bias)
        Z = torch.mm(X_pos, W_pos) + torch.mm(X_neg, W_neg) + b
        return Z
    
class ResidualNormBlock(nn.Module):
    def __init__(self, input_size, output_size, share = False, init_zero = False):
        super().__init__()
        self.linear = CReLUNormLinear(input_size, output_size, 
                                      share = share, init_zero = init_zero)
        self.activation = CReLU()
    def forward(self, X):
        return X + self.linear(*self.activation(X))

##################################################################################


class PSiLONNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1, accelerator = 'cpu'):
        super().__init__()
        '''
        General structure is
        1. Embedder: linear map: input_size -> hidden_size
        2. Representer: (n_hidden-1) layers of 
            (nonlinearity + linear map: hidden_size -> hidden_size) each
            Then, a last nonlinearity: hidden_size -> hidden_size
        3. Predictor 1 final linear map: hidden_size -> output_size)
            - final nonlinearity is to be applied after usage of this module
        '''
        assert n_hidden >= 1
        self.n_hidden = n_hidden
        self.embedder = StandardNormLinear(input_size, hidden_size, share = True)
        self.representer = [StandardNormBlock(hidden_size, hidden_size, share = True) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.final_nonlinearity = nn.ReLU()
        self.predictor = StandardNormLinear(hidden_size, output_size, share = False)
    
    @torch.jit.export
    def compute_reg(self):
        # The scale of the weights in the final layer is the only variable we must control
        # to control the 1-Path Norm (and thus a bound on the Lipschitz constant) under default
        # settings.  This is still approx true under other settings
        reg = self.embedder.g[0,0]
        for module in self.representer:
            reg = reg*module.linear.g[0,0]
        reg = torch.abs(reg)*torch.sum(torch.abs(self.predictor.g))
        return reg
    
    @torch.jit.export
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        return Z
    def forward(self, X):
        H = self.final_nonlinearity(self.get_representation(X))
        y_preactivation = self.predictor(H)
        return y_preactivation
    
class ResPSiLONNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1, accelerator = 'cpu'):
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
        self.n_hidden = n_hidden
        self.embedder = StandardNormLinear(input_size, hidden_size, share = True)
        self.representer = [ResidualNormBlock(hidden_size, hidden_size,
                                               share = True, init_zero = True) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.final_nonlinearity = CReLU()
        self.predictor = CReLUNormLinear(hidden_size, output_size, 
                                            share = False, init_zero = False)
    
    @torch.jit.export
    def compute_reg(self):
        # The scale of the weights in the final layer is the only variable we must control
        # to control the 1-Path Norm (and thus a bound on the Lipschitz constant) under default
        # settings.  This is still approx true under other settings
        reg = torch.abs(self.embedder.g[0,0])
        for module in self.representer:
            reg = reg*(1 + torch.abs(module.linear.g)[0,0])
        reg = reg*torch.sum(torch.abs(self.predictor.g))
        return reg
    
    @torch.jit.export
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        return Z
    def forward(self, X):
        Z_pos, Z_neg = self.final_nonlinearity(self.get_representation(X))
        y_preactivation = self.predictor(Z_pos, Z_neg)
        return y_preactivation
    
#####################################################################################
### Alternative Neural Networks for Comparison


class NoShareStandardNormLinear(nn.Module):
    def __init__(self, input_size, output_size, l2_norm = False):
        super().__init__()
        self.l2_norm = l2_norm
        # Allows learnable rescaling g of weights that is shared across all weight vectors
        # Allows a learned shift term b to be applied after the linear layer (aka 'bias' term)
        self.g = nn.Parameter(torch.ones(1,output_size))
        self.bias = nn.Parameter(torch.zeros(1,output_size))

        # Initializes weight matrix to be an orthogonal matrix
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        nn.init.orthogonal_(self.weight)

    # Normalization function. Note normalization would normally be done horizontally
    # when writing W*X, but here is instead done vertically here based on the matrix
    # multiplication in the forward pass being implemented as X*W
    @torch.jit.export
    def normalize(self, W):
        if self.l2_norm:
            norm = torch.sqrt(torch.sum(W**2, dim = 0))[None,:]
        else:
            norm = torch.sum(torch.abs(W), dim = 0)[None,:] 
        return self.g*W/norm
    
    def forward(self, X):
        W= self.normalize(self.weight)
        Z = torch.mm(X,W) + self.bias
        return Z
    
class NoShareStandardNormBlock(nn.Module):
    def __init__(self, input_size, output_size, l2_norm = False):
        super().__init__()
        self.linear = NoShareStandardNormLinear(input_size, output_size, l2_norm)
        self.nonlinearity = nn.ReLU()
    def forward(self, X):
        return self.linear(self.nonlinearity(X))
    

class LONNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1, accelerator = 'cpu'):
        super().__init__()
        self.accelerator = accelerator
        self.input_size = input_size
        assert n_hidden >= 1
        self.n_hidden = n_hidden
        self.embedder = NoShareStandardNormLinear(input_size, hidden_size)
        self.representer = [NoShareStandardNormBlock(hidden_size, hidden_size) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.final_nonlinearity = nn.ReLU()
        self.predictor = NoShareStandardNormLinear(hidden_size, output_size)

    @torch.jit.export
    def reshape_weight(self, W, bias):
        W = torch.transpose(W,0,1)
        W = torch.cat([W, torch.transpose(bias,0,1)], dim=1)
        v = torch.cat([torch.zeros(W.shape[1]-1), torch.ones(1)])[None,:]
        if self.accelerator == 'gpu':
            v = v.cuda()
        W = torch.cat([W, v], dim = 0)
        return W
    
    @torch.jit.export
    def reshape_weight_final(self, W):
        W = torch.transpose(W,0,1)
        v = torch.zeros(W.shape[0],1)
        if self.accelerator == 'gpu':
            v = v.cuda()
        W = torch.cat([W, torch.zeros(W.shape[0],1)], dim=1)
        return W
    
    @torch.jit.export
    def normalize(self, g, W):
        norm = torch.sum(torch.abs(W), dim = 0)[None,:] 
        return g*W/norm

    @torch.jit.export
    def compute_reg(self):
        # The scale of the weights in the final layer is the only variable we must control
        # to control the 1-Path Norm (and thus a bound on the Lipschitz constant) under default
        # settings.  This is still approx true under other settings
        W1 = self.reshape_weight(self.normalize(self.embedder.g,self.embedder.weight), 
                                 self.embedder.bias)
        reg = torch.mm(torch.abs(W1), torch.ones(self.input_size+1,1))
        for module in self.representer:
            Wi = self.reshape_weight(self.normalize(module.linear.g, module.linear.weight),
                                     module.linear.bias)
            reg = torch.mm(torch.abs(Wi),reg)
        WK = self.reshape_weight_final(self.normalize(self.predictor.g, self.predictor.weight))
        reg = torch.mm(torch.abs(WK),reg)
        return torch.sum(reg)
    
    @torch.jit.export
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        return Z
    def forward(self, X):
        H = self.final_nonlinearity(self.get_representation(X))
        y_preactivation = self.predictor(H)
        return y_preactivation
    



class L2NNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1, accelerator = 'cpu'):
        super().__init__()
        self.accelerator = accelerator
        self.input_size = input_size
        assert n_hidden >= 1
        self.n_hidden = n_hidden
        self.embedder = NoShareStandardNormLinear(input_size, hidden_size, l2_norm = True)
        self.representer = [NoShareStandardNormBlock(hidden_size, hidden_size, l2_norm = True) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.final_nonlinearity = nn.ReLU()
        self.predictor = NoShareStandardNormLinear(hidden_size, output_size, l2_norm = True)

    @torch.jit.export
    def reshape_weight(self, W, bias):
        W = torch.transpose(W,0,1)
        W = torch.cat([W, torch.transpose(bias,0,1)], dim=1)
        v = torch.cat([torch.zeros(W.shape[1]-1), torch.ones(1)])[None,:]
        if self.accelerator == 'gpu':
            v = v.cuda()
        W = torch.cat([W, v], dim = 0)
        return W
    
    @torch.jit.export
    def reshape_weight_final(self, W):
        W = torch.transpose(W,0,1)
        v = torch.zeros(W.shape[0],1)
        if self.accelerator == 'gpu':
            v = v.cuda()
        W = torch.cat([W, torch.zeros(W.shape[0],1)], dim=1)
        return W
    
    @torch.jit.export
    def normalize(self, g, W):
        norm = torch.sqrt(torch.sum(W**2, dim = 0))[None,:]
        return g*W/norm

    @torch.jit.export
    def compute_reg(self):
        # The scale of the weights in the final layer is the only variable we must control
        # to control the 1-Path Norm (and thus a bound on the Lipschitz constant) under default
        # settings.  This is still approx true under other settings
        W1 = self.reshape_weight(self.normalize(self.embedder.g,self.embedder.weight), 
                                 self.embedder.bias)
        reg = torch.mm(torch.abs(W1), torch.ones(self.input_size+1,1))
        for module in self.representer:
            Wi = self.reshape_weight(self.normalize(module.linear.g, module.linear.weight),
                                     module.linear.bias)
            reg = torch.mm(torch.abs(Wi),reg)
        WK = self.reshape_weight_final(self.normalize(self.predictor.g, self.predictor.weight))
        reg = torch.mm(torch.abs(WK),reg)
        return torch.sum(reg)
    
    @torch.jit.export
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        return Z
    def forward(self, X):
        H = self.final_nonlinearity(self.get_representation(X))
        y_preactivation = self.predictor(H)
        return y_preactivation
    




class StandardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  n_hidden = 1, accelerator = 'cpu'):
        super().__init__()
        self.input_size = input_size
        assert n_hidden >= 1
        self.n_hidden = n_hidden
        self.embedder = NoShareStandardNormLinear(input_size, hidden_size, l2_norm = True)
        self.representer = [NoShareStandardNormBlock(hidden_size, hidden_size, l2_norm = True) \
                            for _ in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.final_nonlinearity = nn.ReLU()
        self.predictor = NoShareStandardNormLinear(hidden_size, output_size, l2_norm = True)

    @torch.jit.export
    def reshape_weight(self, W, bias):
        W = torch.transpose(W,0,1)
        W = torch.cat([W, torch.transpose(bias,0,1)], dim=1)
        v = torch.cat([torch.zeros(W.shape[1]-1), torch.ones(1)])[None,:]
        W = torch.cat([W, v], dim = 0)
        return W
    
    @torch.jit.export
    def normalize(self, g, W):
        norm = torch.sqrt(torch.sum(W**2, dim = 0))[None,:]
        return g*W/norm

    @torch.jit.export
    def compute_reg(self):
        # The scale of the weights in the final layer is the only variable we must control
        # to control the 1-Path Norm (and thus a bound on the Lipschitz constant) under default
        # settings.  This is still approx true under other settings
        W1 = self.normalize(self.embedder.g, self.embedder.weight)
        reg = torch.sum(torch.pow(W1,2))
        for module in self.representer:
            Wi =  self.normalize(module.linear.g, module.linear.weight)
            reg = reg + torch.sum(torch.pow(Wi,2))
        WK = self.normalize(self.predictor.g, self.predictor.weight)
        reg = reg + torch.sum(torch.pow(WK,2))
        return torch.sum(reg)
    
    @torch.jit.export
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        return Z
    def forward(self, X):
        H = self.final_nonlinearity(self.get_representation(X))
        y_preactivation = self.predictor(H)
        return y_preactivation
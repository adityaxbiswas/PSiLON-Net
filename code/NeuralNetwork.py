
import torch
import torch.nn as nn
import torch.nn.functional as F


    
class NormLinear(nn.Module):
    def __init__(self, input_size, output_size, rescale = False, shift = False, type = 'l1'):
        super().__init__()

        # Constrains outgoing weights to a neuron to have l1 or l2 norm of 1
        if type == 'l1':
            self.norm_func = self.l1_normalize
        elif type == 'l2':
            self.norm_func = self.l2_normalize
        else:
            Exception("type can only take values 'l1' or 'l2'")

        # Allows learnable rescaling g of weights for full expressivity of a linear layer
        #   - g is learned on the log scale.  must be exponentiated when used
        # Allows a learned shift term b to be applied after the linear layer (aka 'bias' term)
        self.rescale = rescale
        self.shift = shift
        if rescale:
            self.g = nn.Parameter(torch.zeros(1,output_size))
        if shift:
            self.b = nn.Parameter(torch.zeros(1,output_size))

        # Initializes weight matrix to be an orthogonal matrix, followed by normalization
        self.weight = torch.empty(input_size, output_size)
        nn.init.orthogonal_(self.weight)
        self.weight = nn.Parameter(self.norm_func(self.weight))

    # Normalization function. Note normalization would normally be done along the columns
    # when writing W*X, but here is instead done along the rows here based on the matrix
    # multiplication in the forward pass being implemented as X*W
    def l1_normalize(self, W):
        norm = torch.sum(torch.abs(W), dim = 0)[None,:]
        return W/norm
    def l2_normalize(self, W):
        norm = (torch.sqrt(torch.sum(W**2, dim = 0))[None,:])
        return W/norm
    
    def forward(self, X):
        W = self.norm_func(self.weight)
        Z = torch.mm(X,W)
        if self.rescale:
            Z *= torch.exp(self.g)
        if self.shift:
            Z += self.b
        return Z


class CALLayer2(nn.Module):
    ''' Concatenated-Activation + Linear Layer'''
    def __init__(self, input_size, output_size, rescale = False, shift = False, 
                 type = 'l1', activation = F.relu):
        super().__init__()
        self.activation = activation
        # Doubles the number of incoming features using concatenated activation trick
        # Uses 'looks linear' initialization
        self.lineara = NormLinear(input_size, output_size, rescale, False, type)
        self.linearb = NormLinear(input_size, output_size, rescale, shift, type)
        with torch.no_grad():
            self.lineara.weight.copy_(-self.linearb.weight.data)
    def forward(self, H):
        return self.lineara(self.activation(H)) + self.linearb(self.activation(-H))



class CALLayer(nn.Module):
    ''' Concatenated-Activation + Linear Layer'''
    def __init__(self, input_size, output_size, rescale = False, shift = False, 
                 type = 'l1', activation = F.relu):
        super().__init__()
        self.activation = activation
        # Doubles the number of incoming features using concatenated activation trick
        # Uses 'looks linear' initialization
        self.linear = NormLinear(2*input_size, output_size, rescale, shift, type)
        with torch.no_grad():
            self.linear.weight[input_size:,:].copy_(-self.linear.weight[:input_size,:].data)
    def forward(self, H):
        H = torch.cat([self.activation(H), self.activation(-H)], axis = 1)
        # multiplies output by 2, since when using concatenated activation with relu,
        # approximately half the neurons are in an off state on average. For NormLinear layers
        # without scaling, this modification leads to the "effective" norm being about 1
        return 2*self.linear(H)
    
class OPLLayer(nn.Module):
    '''Orthogonal Permutaton Linear Unit + Linear Layer'''
    def __init__(self, input_size, output_size, rescale = False, shift = False, 
                 type = 'l1', activation = F.relu):
        super().__init__()
        self.linear = NormLinear(input_size, output_size, rescale, shift, type)
        self.d = int(input_size/2)
    def forward(self, H):
        H1, H2 = H[:,:self.d], H[:,self.d:]
        H_max = torch.maximum(H1, H2)
        H_min = torch.minimum(H1, H2)
        H = torch.cat([H_max, H_min], axis = 1)
        return self.linear(H)
    

class PathNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 n_hidden = 1, use_biases = False, final_type = 'l1', activation = F.relu):
        super().__init__()
        '''
        General structure is
        1. Embedder: linear map: input_size -> hidden_size
        2. Representer: (n_hidden-1) layers of 
            (nonlinearity + linear map: hidden_size -> 2*hidden_size -> hidden_size) each
        3. Predictor 1 final layer of 
            (nonlinearity + linear map: hidden_size -> 2*hidden_size -> output_size)
            - final nonlinearity is to be applied after usage of this module
        '''

        assert n_hidden >= 1
        self.embedder = NormLinear(input_size, hidden_size, 
                                   rescale = False, shift = use_biases, type = 'l1')
        self.representer = [CALLayer(hidden_size, hidden_size,
                                      rescale = False, shift = use_biases, 
                                      type = 'l1', activation = activation) \
                            for i in range(n_hidden-1)]
        self.representer = nn.ModuleList(self.representer)
        self.predictor = CALLayer(hidden_size, output_size,
                                  rescale = True, shift = True, 
                                  type = final_type, activation = activation)
        
    def compute_reg(self, lambda_):
        # The scale of the weights in the final layer is the only variable we must control
        # to control the 1-Path Norm (and thus a bound on the Lipschitz constant) under default
        # settings.  This is still approx true under other settings
        #   - g is given on the log scale, so must be exponentiated
        reg = torch.mean(torch.exp(self.predictor.linear.g)) 
        return 2*lambda_*reg
    
    def get_representation(self, X):
        Z = self.embedder(X)
        for module in self.representer:
            Z = module(Z)
        return Z
    def forward(self, X):
        H = self.get_representation(X)
        y_preactivation = self.predictor(H)
        return y_preactivation
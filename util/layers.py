import torch
import torch.nn as nn
import torch.nn.functional as F
import math
SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}



def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


def aggregate_mean(X, adj=None,Type=None, device='cuda'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    if adj is None:
        X_mean = torch.mean(X, dim=-2)
    else:
        if Type=='facility':
            
            adj = adj.permute(0, 2, 1)
           
        X_masked = torch.where(adj.unsqueeze(-1).bool(), X, torch.tensor(float('nan')).to(X.device))

        # 각 customer별로 연결된 텐서에 대해서만 max를 적용
        X_mean = torch.nanmean(X_masked, dim=2) # n_Facility 차원에 대해 max 적용
           

       

    return X_mean


def aggregate_max(X,adj=None,Type=None, device='cuda'):
    if adj is None:
        max = torch.max(X, dim=-2)[0]
    else:
       
        if Type=='facility':
           
            adj = adj.permute(0, 2, 1)
           
        X_masked = torch.where(adj.unsqueeze(-1).bool(), X, torch.tensor(float('-inf')).to(X.device))

        # 각 customer별로 연결된 텐서에 대해서만 max를 적용
        max, _ = torch.max(X_masked, dim=2) # n_Facility 차원에 대해 max 적용
    return max


def aggregate_min(X,  adj=None,Type=None, device='cuda'):
    if adj is None:
        min = torch.min(X, dim=-2)[0]
    else:
       
        if Type=='facility':
           
            adj = adj.permute(0, 2, 1)
           
        X_masked = torch.where(adj.unsqueeze(-1).bool(), X, torch.tensor(float('inf')).to(X.device))
        min, _ = torch.min(X_masked, dim=2) 
    return min



def aggregate_std(X,  adj=None,Type=None, device='cuda'):
    
    if adj is None:
        std = torch.std(X, dim=-2) # sqrt(mean_squares_X - mean_X^2)
    else:
        if Type=='facility':
               
                
        
            adj = adj.permute(0, 2, 1)
            
        X_masked = torch.where(adj.unsqueeze(-1).bool(), X, torch.tensor(float('nan')).to(X.device))
        not_nan_mask = ~torch.isnan(X_masked)

        # NaN이 아닌 값들만을 사용하여 평균을 계산합니다.
        mean_not_nan = torch.sum(X_masked * not_nan_mask, dim=2) / not_nan_mask.sum(dim=2)

        # 표준 편차 계산을 위한 준비: 평균을 빼고, 제곱합니다.
        diffs_squared = (X_masked - mean_not_nan.unsqueeze(2)) ** 2

        # NaN이 아닌 값들에 대해서만 diffs_squared을 고려합니다.
        diffs_squared_not_nan = torch.where(not_nan_mask, diffs_squared, torch.tensor(0.0, device=X_masked.device))

        # 표준 편차 계산
        std = torch.sqrt(torch.sum(diffs_squared_not_nan, dim=2) / not_nan_mask.sum(dim=2))
    return std



#X: B*n_customer *(4*U_out)
#adj: B*n_customer*n_facility
# avg_d: B*1
def scale_identity(X, adj, avg_d=None, Type=None):
    return X


def scale_amplification(X, adj, avg_d=None,Type=None):
    avg_d = avg_d.unsqueeze(1)
    if Type =='customer':
        #X: B*n_customer *(4*U_out)
        adj_cus=torch.sum(adj, dim=-1)   #B*n_customer
        scale=torch.log(adj_cus+1)/avg_d #B*n_customer
        scale=scale.unsqueeze(-1)
        X_scaled = torch.mul(scale, X)
        
    else: 
        #X: B*n_facility *(4*U_out)
        adj_fac=torch.sum(adj, dim=-2)   #B*n_facility
        scale=torch.log(adj_fac+1)/avg_d #B*n_facility
        scale=scale.unsqueeze(-1)
        X_scaled = torch.mul(scale, X)
   
    return X_scaled


def scale_attenuation(X, adj, avg_d=None,Type=None):
    avg_d = avg_d.unsqueeze(1)
    if Type =='customer':
         #X: B*n_customer *(4*U_out)
        adj_cus=torch.sum(adj, dim=-1)   #B*n_customer
        scale=torch.log(adj_cus+1)/avg_d #B*n_customer
        scale=scale.unsqueeze(-1)
        scale_inverse = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))# 0^(-1)으로 인한 nan을 해결하는 ㅋ드
        X_scaled = torch.mul(scale_inverse, X)
    else:
        #X: B*n_facility *(4*U_out)
        adj_fac=torch.sum(adj, dim=-2)   #B*n_facility
        scale=torch.log(adj_fac+1)/avg_d #B*n_facility
        scale=scale.unsqueeze(-1)
        scale_inverse = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))
        X_scaled = torch.mul(scale_inverse, X)
    return X_scaled

def get_avgd(adj):
    adj_cus=torch.sum(adj, dim=-1)   
    adj_fac=torch.sum(adj, dim=-2)   

    adj_cus=adj_cus+1
    adj_fac=adj_fac+1

    log_cus=torch.log(adj_cus)    #B*n_cus
    log_fac=torch.log(adj_fac)    #B*n_fac
    sum_d= torch.cat([log_cus,log_fac], dim=-1) #sum_d: B*(n_cus + n_fac)
    avg_d=torch.mean(sum_d, dim= -1)  #avg_g:B*1
    return avg_d


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        
        h = self.linear(x)
        
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                # h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
                original_shape = h.shape  # 원래 형태 저장
                h = h.view(-1, self.out_size)  # 배치 정규화를 위해 형태 변경
            
                # 배치 정규화 적용
                h = self.b_norm(h)
                
                # 형태를 원래대로 되돌림
                h = h.view(original_shape)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_size, hidden_size, out_size, layers, mid_activation='relu', last_activation='none',
                 dropout=0., mid_b_norm=False, last_b_norm= False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(in_size, hidden_size, activation=mid_activation, b_norm=mid_b_norm,
                                                device=device, dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_size, hidden_size, activation=mid_activation,
                                                    b_norm=mid_b_norm, device=device, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                                device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'

    
SCALERS = {'identify': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation    }



AGGREGATORS = {'mean': aggregate_mean, 'std': aggregate_std, 'max': aggregate_max, 'min': aggregate_min   }



class PNALayer(nn.Module):
    def __init__(self, Type,nhid_set, amplify, aggregators, scalers, U_layers,
                 M_layers):
        """
        :param in_features:     size of the input per node of the tower
        :param out_features:    size of the output per node of the tower
        :param aggregators:     set of aggregation functions each taking as input X (B x N x N x Din), adj (B x N x N), self_loop and device
        :param scalers:         set of scaling functions each taking as input X (B x N x Din), adj (B x N x N) and avg_d
        """
        super(PNALayer, self).__init__()

        self.nhid_set=nhid_set
        self.Type= Type

        U_init=self.nhid_set['facility_nhid']+self.nhid_set['customer_nhid']+1
        U_out=self.nhid_set['facility_nhid']+self.nhid_set['customer_nhid']
        if self.Type =='customer':
            
           
            M_init=3*len(aggregators)*U_out+self.nhid_set['customer_nhid']
            M_out=self.nhid_set['customer_nhid']
        if self.Type =='facility':
            
            M_init=3*len(aggregators)*U_out+self.nhid_set['facility_nhid']
            M_out=self.nhid_set['facility_nhid']
            

   
        self.aggregators = aggregators
        self.scalers=scalers
        self.amplify=amplify
        
        self.U = MLP(in_size=U_init, hidden_size=self.amplify*U_init, out_size=U_out,
                            layers=U_layers, mid_activation='relu', last_activation='none')
        self.M = MLP(in_size=M_init,
                             hidden_size=self.amplify*M_out, out_size=M_out, layers=M_layers,
                             mid_activation='relu', last_activation='none')
       

    def forward(self, c, f,  adj ):
    # def forward(self, root_x,child_x):
        # c : B*n_customer*customer_nhid 
        # f : B*n_faciilty*facility_nhid 
        
        # adj: B *n_customer*n_facility
        ( _, n_customer, n_facility) =adj.shape
        avg_d=get_avgd(adj)


        if self.Type =='customer':
            c_pre=c.unsqueeze(-2).repeat(1,1,n_facility,1)  #c : B*n_customer*n_facility*customer_nhid 
            f_pre=f.unsqueeze(-3).repeat(1,n_customer,1,1)
            adj_expanded = adj.unsqueeze(-1) # 마지막 차원 추가

            u_cat=torch.cat([c_pre,f_pre,adj_expanded], dim=-1) # u_cat:B*n_customer*n_facility*(customer_nhid+facility_nhid)
            U_result=self.U(u_cat)   #U_result : B*n_customer*n_facility*(customer_nhid+facility_nhid)

            # aggregation
            m = torch.cat([AGGREGATORS[aggregate]( U_result) for aggregate in self.aggregators], dim=-1) #m: B*n_cus*{4*(customer_nhid+facility_nhid)}
            m = torch.cat([SCALERS[scale](m,adj,avg_d, self.Type) for scale in self.scalers], dim=-1)#m: B*n_cus*{12*(customer_nhid+facility_nhid)}

            #post aggregation
            m_cat=torch.cat([c, m ], dim=-1)
        elif self.Type =='facility':
            f_pre=f.unsqueeze(-2).repeat(1,1,n_customer,1)  #c : B*n_customer*n_facility*customer_nhid 
            c_pre=c.unsqueeze(-3).repeat(1,n_facility,1,1)
            adj_permute = adj.permute(0, 2, 1)
            adj_expanded = adj_permute.unsqueeze(-1) # 마지막 차원 추가
            u_cat=torch.cat([f_pre,c_pre,adj_expanded], dim=-1) # u_cat:B*n_facility*n_customer*(customer_nhid+facility_nhid)
            U_result=self.U(u_cat)   #U_result : B*n_facility*n_customer*(customer_nhid+facility_nhid)

            # aggregation
            m = torch.cat([AGGREGATORS[aggregate]( U_result) for aggregate in self.aggregators], dim=-1) #m: B*n_facility*{4*(customer_nhid+facility_nhid)}
            m = torch.cat([SCALERS[scale](m,adj,avg_d, self.Type) for scale in self.scalers], dim=-1)#m: B*n_facility*{12*(customer_nhid+facility_nhid)}

            #post aggregation
            m_cat=torch.cat([f, m],  dim=-1)

       
        
        out = self.M(m_cat)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

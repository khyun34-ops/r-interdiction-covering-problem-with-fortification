
import torch
import torch.nn.functional as F
import numpy as np
import statistics
import random
import cplex
from torch.nn import Linear
from tqdm import tqdm
import time
from util.util import  GNNmodel, getloss
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

n=400
ndata=16
indexset=[1,3,4,5,6,8,9,10,14,15]
adj=[]
demand=[]
fac_init=[]
newadj, newdemand, newfac_init= [],[],[]
output_Adj=[]
output_demand=[]
output_fac=[]


with np.load("data/Test_adj{}_{}.npz".format(n,ndata)) as data:
    adj.append(data[data.files[0]])
with np.load("data/Test_demand{}_{}.npz".format(n,ndata)) as data:
    demand.append(data[data.files[0]])

with np.load("data/Test_fac{}_{}.npz".format(n,ndata)) as data:
    fac_init.append(data[data.files[0]])

# with np.load("data/Test_adj{}_10.npz".format(n,ndata)) as data:
#     newadj.append(data[data.files[0]])
# with np.load("data/Test_demand{}_10.npz".format(n,ndata)) as data:
#     newdemand.append(data[data.files[0]])

# with np.load("data/Test_fac{}_10.npz".format(n,ndata)) as data:
#     newfac_init.append(data[data.files[0]])





for i in range(ndata):
    if i in indexset:
        output_Adj.append(adj[0][i])
        output_demand.append(demand[0][i])
        output_fac.append(fac_init[0][i]
        )
# for i in range(10):
    
#     output_Adj.append(adj[0][i])
#     output_demand.append(demand[0][i])
#     output_fac.append(fac_init[0][i]
#     )
# for i in range(4):
    
#     output_Adj.append(newadj[0][i])
#     output_demand.append(newdemand[0][i])
#     output_fac.append(newfac_init[0][i]
#     )

np.savez_compressed('data/Final_adj{}_30.npz'.format( n), output_Adj)
np.savez_compressed('data/Final_demand{}_30.npz'.format(n), output_demand)
np.savez_compressed('data/Final_fac{}_30.npz'.format(n), output_fac)
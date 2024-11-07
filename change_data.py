import numpy as np
import random
from util.util import Adj_matrix_demand_generate
import time 
import cplex
from scipy.sparse import csr_matrix, save_npz

nnode=[100,150,200]
type='demand'

for n in nnode:
    adj=np.load("data/adj{}_4000.npz".format(n))
    adj0 = adj['arr_0']
    Neighbor_f=np.zeros((4000,int(0.2*n),1))

    if type=='adj':
       
        newadj=adj0[ : , : int(0.8*n), int(-0.2*n) :]
        asdf=0
        np.savez_compressed('data/newadj{}_4000.npz'.format(n), newadj)
    elif type == 'demand':
        demand=np.load("data/demand{}_4000.npz".format(n))
        demand0 = demand['arr_0']
        newdemand=np.expand_dims(demand0, axis=-1)
        zeros_array = np.zeros((4000, int(0.8*n), 1))
        sum_adj_cus=np.sum(adj0, axis= -1)
        sum_adj_fac=np.sum(adj0, axis= -2)

        for i in range(4000):
            for node_index in range(int(0.8*n)):
                if sum_adj_cus[i][node_index] !=0:
                    zeros_array[i][node_index][0]=1
            for node_index in range(int(0.2*n)):
                if sum_adj_cus[i][node_index] !=0:
                    Neighbor_f[i][node_index][0]=1

        result = np.concatenate((newdemand, zeros_array), axis=-1)
        
                    
   
        
       
        np.savez_compressed('data/newdemand{}_4000.npz'.format(n), result)
        np.savez_compressed('data/fac_init{}_4000.npz'.format(n), Neighbor_f)
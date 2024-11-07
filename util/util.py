import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import math
import matplotlib.pyplot as plt
from .layers import MLP, PNALayer




def Adj_matrix_demand_generate(n,p,criterion,mu,sigma_squared,show_plt):
    #adjaceny matrix 및 demand 생성 
    points = np.round(np.random.rand(n, 2), 4)

    adj_mat=np.zeros((n,n))

    index=0
    for i in range(n-p):
        for j in range(n-p,n):
            distance= math.sqrt((points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2)
            
            if distance<=criterion:
                index+=1
                adj_mat[i][j]=1
                adj_mat[j][i]=1

    # print(adj_mat)
    #adj_mat 형식    1,2...  n-p: custoemr   n-p+1,,,,,,n: facility


    demand=np.zeros(n-p)
    for i in range(n-p):
        # Generate a random number from a log-normal distribution
        random_number =  np.random.lognormal(mean=mu, sigma=np.sqrt(sigma_squared))

        # Round the number to the nearest integer
        rounded_number = int(round(random_number))
        if rounded_number<1:
            demand[i]=1
        else:
            demand[i]=rounded_number
    # print(demand)
    plot_points_and_edges(points,p, adj_mat,demand,show_plt)
    max_C=np.sum(demand)


    Neighbor = [[] for _ in range(n-p)]

    non_cover_customer=0
    for j in range(n-p):
        num_neighbor=0
        for i in range(p):
            if adj_mat[j][n-p+i] ==1:
                Neighbor[j].append(i)
                num_neighbor+=1
        if num_neighbor==0:
            non_cover_customer+=1
    return adj_mat, demand, Neighbor, max_C, non_cover_customer/n*1.25


def plot_points_and_edges(points,p, adjacency_matrix,demand,show_plt):
    if show_plt==True:
        
        plt.figure(figsize=(8, 6))
        n = len(points)
        # Plot points
        for i in range(n):
            if i<n-p:
                plt.scatter(points[i, 0], points[i, 1], c='green')
                plt.text(points[i][0], points[i][1], f' {demand[i]}', color='red') # demand 값 표시
            else:
                plt.scatter(points[i, 0], points[i, 1], c='blue')
                
            
        
        
        # Plot edges
        
        for i in range(n):
            for j in range(i+1, n):  # Avoid duplicating edges
                if adjacency_matrix[i, j] == 1:  # There's an edge between i and j
                    plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'r-')
        
        plt.title('Nodes and Edges')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.show()


def getloss(output, label):
    mse_loss_fn = nn.MSELoss(reduction='none')

    loss=mse_loss_fn(output,label).mean()
    
    return loss

#PNALayer 부분을 나한테 맞게 교체
class GNNmodel(nn.Module):

    def __init__(self,nhid_set,amplify, npna):
            super(GNNmodel,self).__init__()
            self.nhid_set=nhid_set
            self.npna=npna
            self.amplify=amplify
            #define embedding
            self.facility_emv=nn.Linear(1,self.nhid_set['facility_nhid'])
            self.customer_emv=nn.Linear(1,self.nhid_set['customer_nhid'])
            

            self.cus_pna=PNALayer(Type='customer', nhid_set=nhid_set,amplify=self.amplify, aggregators=['max','min','mean','std'],scalers=['identify','amplification','attenuation'],U_layers=2,M_layers=2)
            self.fac_pna=PNALayer(Type='facility', nhid_set=nhid_set,amplify=self.amplify, aggregators=['max','min','mean','std'],scalers=['identify','amplification','attenuation'],U_layers=2,M_layers=2)
            
            # self.f=MLP(in_size=root_nhid+child_nhid,hidden_size=2*(root_nhid+child_nhid),out_size=3,layers=3)
            
            self.f=MLP(in_size=nhid_set['facility_nhid'],hidden_size=12,out_size=1,layers=3,last_activation='Sigmoid')
    def forward(self, demand, fac_init,adj):
        #demand  :B*n_customer *2
        #fac_init :B*n_facility *2
        f_init=self.facility_emv(fac_init) # f : B*n_faciilty*facility_nhid 
        c_init=self.customer_emv(demand) # c : B*n_customer*customer_nhid 
       
        
        

        for _ in range(self.npna):
            c_after=self.cus_pna(c_init,f_init, adj)
            f_after=self.fac_pna(c_init,f_init, adj)
            
            c_init=c_after
            f_init=f_after
            
        
        
        output=self.f(f_init)
        return output


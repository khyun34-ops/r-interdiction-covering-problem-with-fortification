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


def Followers_problem(Z_current,adj, demand, parameter):
    p=parameter['p']
    r=parameter['r']
    
    Neighbor = [[] for _ in range(n-p)]

    
    for j in range(n-p):
        num_neighbor=0
        for i in range(p):
            if adj[j][i] ==1:
                Neighbor[j].append(i)
                
        
    problem=cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    u=["u{}".format(j) for j in range(n-p)]
    s=["s{}".format(i) for i in range(p)]
    for j in range(n-p):
        problem.variables.add(names=[u[j]],lb=[0],ub=[1],types=['B'], obj=[float(demand[j])])
    for i in range(p):
        problem.variables.add(names=[s[i]],lb=[0],ub=[1],types=['B'])
        
    #제약식 (6)
    var, coef= [],[]
    for i in range(p):
        var.append(s[i])
        coef.extend([1])
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                senses = "E", rhs=[r])
    #제약식 (7)
    for j in range(n-p):
        if len(Neighbor[j]) !=0:
            for i in Neighbor[j]:
                var=[u[j],s[i]]
                coef=[1,1]
                problem.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                        senses = "G", rhs=[1])
        
        
        else:
            var=[u[j]]
            coef=[1]
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                        senses = "E", rhs=[0])
    #제약식(8)
    for i in range(p):
        rhs=round(1-Z_current[i])
        var=[s[i]]
        coef=[1]
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                        senses = "L", rhs=[rhs])
    
    problem.solve()
    problem.write("testSA.lp")
    f_new=problem.solution.get_objective_value()
    
    return f_new

nset=[100,150,200,300,400,500]
ndata=10

Seed=[9]
nhid_set = {
    'facility_nhid': 8, 'customer_nhid': 8, 'root_nhid': 8}
amplify=1
npna=2

qr_set = {
    100: [(3, 3), (3, 5), (5, 3), (10, 7), (7, 10)],
    150: [(5, 5), (10, 12), (12, 10), (12, 15), (15, 12)],
    200: [(7,7),(12,12),(16,14),(14,16),(16,16)],
    300: [(10,10), (10,15),(20,16), (16,20),(26,26)],
    400: [(15,15),(15,20), (20,25),(25,20),(30,30)], 
    500: [(20,20), (30,30), (30,35), (35,30), (40,40)]
}


if torch.cuda.is_available():
    device=torch.device('cuda')





for seed in Seed:
    
    model=GNNmodel(nhid_set,amplify, npna)
    trained_parameter=torch.load("train_result/best_model{}.pth".format(seed))
    model.load_state_dict(trained_parameter['model_state_dict'])
    model.to(device)
    model.eval()
    result=open("result/D_Final_Learning_resuls{}.txt".format(seed),"w")
    for n , pairs in qr_set.items():
        n_candidate=int(0.05*n)
        adj=[]
        demand=[]
        fac_init=[]
        
                
        with np.load("data/Final_adj{}_{}.npz".format(n,ndata)) as data:
            adj.append(data[data.files[0]])
        with np.load("data/Final_demand{}_{}.npz".format(n,ndata)) as data:
            demand.append(data[data.files[0]])
    
        with np.load("data/Final_fac{}_{}.npz".format(n,ndata)) as data:
            fac_init.append(data[data.files[0]])

        adj_tensor = torch.tensor(adj[0]).float().to(device)
        demand_tensor = torch.tensor(demand[0]).unsqueeze(-1).float().to(device)
        fac_tensor= torch.tensor(fac_init[0]).float().to(device)
                       
        output =model( demand_tensor, fac_tensor, adj_tensor)

        
    
        
       
        for q,r in pairs:
            parameter={"p":int(0.2*n), "q":q, "r":r }  
            p=parameter['p']
            q=parameter['q']
            r=parameter['r']
            
            start_time=time.perf_counter()

            ratio=output[0].squeeze(-1).detach().cpu().numpy()
            current_adj=adj_tensor[0].cpu().numpy()
            demand=demand_tensor[0].squeeze(-1).detach().cpu().numpy()
            current_demand=demand_tensor[0].cpu().numpy()
            (n_customer, n_facility)=current_adj.shape
            Z_current= np.zeros(n_facility)

            greedy_start=time.perf_counter()
            for _ in range(q):
                np_index=0 
                #result중 가장 큰 n_candadate개의 index 출력
                indices = np.argsort(ratio)[-n_candidate:]
                candidate= np.zeros(2, dtype=int)
                for i in indices:
                    sumw=0
                    for j in range(n_customer):
                        sumw+=current_adj[j][i]*current_demand[j]
                    if sumw > candidate[0]:
                        candidate=(sumw,i)
                    
                Z_current[candidate[1]]=1
                for j in range(n_customer):
                    if current_adj[j][candidate[1]]==1:
                        current_demand[j]=0
                ratio[candidate[1]]= float('-inf')
            greedy_time=time.perf_counter()-greedy_start
            objective_value=Followers_problem(Z_current, current_adj, demand,parameter)
            end_time=time.perf_counter()
            result.write("{}      {}      {}      {}       {}\n".format(q,r , objective_value, greedy_time, end_time-start_time))
    result.close()
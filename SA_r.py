import numpy as np
import random
from util.util import Adj_matrix_demand_generate
import time 
import math
import cplex
from scipy.sparse import csr_matrix, save_npz

def Swap(Z):
    indices_of_1 = np.where(Z == 1)[0]
    indices_of_0 = np.where(Z == 0)[0]

    # 각각에서 랜덤하게 하나씩 인덱스 선택
    index_to_turn_0 = np.random.choice(indices_of_1, 1)[0]
    index_to_turn_1 = np.random.choice(indices_of_0, 1)[0]

    # 선택된 인덱스의 값을 변경
    Z[index_to_turn_0] = 0
    Z[index_to_turn_1] = 1
    
    return Z


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
        problem.variables.add(names=[u[j]],lb=[0],ub=[1],types=['B'], obj=[demand[j]])
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
    
n=818
ndata=1
T0=100
delta=0.99
n_iteration=5
qr_set=[ (10,10), (10,12), (12,10), (14,14), (14,16),(16,14), (20,20) ,(20,22),(22,20), (24,24),(24,26),(26,24), (30,30)]

adj=np.load("data/Real_adj.npz")['arr_0']
demand=np.load("data/Real_demand.npz")['arr_0']
result=open("result/Real_SAresults.txt","w")
    
    
for q,r in qr_set:
    start_time=time.perf_counter()
    f_total=0

    for _ in range(n_iteration):
        
        
        Tf=1
                
        parameter={"p":150, "q":q, "r":r }  
        p=parameter['p']
                
        (_,n_customer )=demand.shape
        n_facility=n-n_customer
        
        Z_init=np.zeros(n_facility)
        # 랜덤 위치 선택
        random_indices = np.random.choice(n_facility, parameter['r'], replace=False)
        # 선택된 위치의 값을 1로 변경
        Z_init[random_indices] = 1
        f_best=Followers_problem(Z_init, adj[0],demand[0],parameter)
        T=T0
        Z_best=Z_init
        
        Z_current=Z_init
        f_current=f_best
        while T> Tf:
            Z_prime=Swap(Z_current)
            f_prime=Followers_problem(Z_current, adj[0],demand[0],parameter)
            E=f_current-f_prime
            if E<0:
                Z_current=Z_prime
                f_current=f_prime
            else:
                low= np.random.rand()
                if low > math.exp(-abs(E)/T): 
                    Z_current=Z_prime
                    f_current=f_prime
            
            if f_current>f_best:
                f_best=f_current
                Z_best=Z_current
            T=T*delta
        if f_best>f_total:
            f_total=f_best
    end_time=time.perf_counter()
    result.write("{}    {}     {}      {}\n".format(q,r, f_total, end_time-start_time))
result.close()
        



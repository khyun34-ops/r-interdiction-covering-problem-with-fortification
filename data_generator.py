import numpy as np
import random
from util.util import Adj_matrix_demand_generate
import time 
import cplex
from scipy.sparse import csr_matrix, save_npz

global n,p,q,r
global Neighbor,demand
show_plt=False
# (15,0.05) (10,0.65)
# ratioset=[1,1.5,2]
# criterionset=[0.2,0.17, 0.135] 
# ratioset=[3,4,5]
# criterionset=[0.1,0.09, 0.08] 
ratioset=[30]
criterionset=[0.035] 
#수치 정의
timerecord=[]
ndata=10
test_index='Test'

for dataindex in range(1):
    
    ratio=ratioset[dataindex]
    criterion=criterionset[dataindex]
    n=int(100*ratio)
    p=int(20*ratio)
    
    r=10*ratio -random.randint(0,3)

    # r=int(2.5*ratio)

    # # criterion=0.081
    # criterion=0.135





    #lognormal 분포 파라미터
    mu=2.845565678551321
    sigma_squared=1.4759065198095778

    start_time=time.perf_counter()

    adj_mat_set=[]
    demand_set=[]
    label_set=[]
    fac_set=[]


    avg_neighbor=0
    for _ in range(ndata):
        #adj_mat   1,2,3,... n-p :customer n-p+1,....,n:facility, faciliyt들의 인덱스는 0,1,2,...,p-1 
        #demand : p-dimention numpy array, Neighbor:n-p dimension list, 각 list트는 인접한 facility들의 인덱스들의집합
        adj_mat, demand, Neighbor, max_C, non_cover_customer_ratio=Adj_matrix_demand_generate(n,p,criterion,mu,sigma_squared,show_plt)
        fac_init= np.ones(( p,1))
        adj_mat=adj_mat[ :n-p, n-p:]

        
        col_sums = adj_mat.sum(axis=0)
        for i in range(p):
            if col_sums[i]==0:
                fac_init[i][0]=0
        
            


        adj_mat_set.append(adj_mat)
        demand_set.append(demand)
        fac_set.append(fac_init)
        
        sum_neighbor=0
        neighbor_index=0
        for i in range(len(Neighbor)):
            if len(Neighbor[i]) !=0:
                neighbor_index+=1
                sum_neighbor+=len(Neighbor[i])
        avg_neighbor+=sum_neighbor/(neighbor_index*ndata)
        
        if test_index !="Test":
           

            
            lp_obj=np.zeros(p+1)
            lp_obj=np.zeros(p+1)
            
            
            
            for k in range(p+1):
                lp=cplex.Cplex()
                lp.objective.set_sense(lp.objective.sense.minimize)
                
                s,u =[],[]
                
                s=["s{}".format(i) for i in range(p)]
                u=["u{}".format(j) for j in range(n-p)]
                
                for i in range(p):
                    lp.variables.add(names=[s[i]],lb=[0],ub=[1],types=["C"])
                for j in range(n-p):
                    lp.variables.add(names=[u[j]],lb=[0],ub=[1],types=["C"], obj=[demand[j]])
                    
                x_bar=np.zeros(p)
                
                if k!=p:
                    x_bar[k]=1
                    
                #제약식(6)
                var=["s{}".format(i) for i in range(p)]
                coef=[1]*p
                lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                senses = "E", rhs=[r]) 
                
                #제약식(7)
                for j in range(n-p):
                    if len(Neighbor[j]) !=0:
                        for _ ,facility_index  in enumerate(Neighbor[j]):
                            var=[u[j],s[facility_index]]
                            coef=[1,1]
                            lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                            senses = "G", rhs=[1])
                    else:
                        var=["u{}".format(j)]
                        coef=[1]
                        lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                            senses = "E", rhs=[0])
                #제약식(8)
                for i in range(p):
                    var=["s{}".format(i)]
                    coef=[1]
                    lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                            senses = "L", rhs=[1-x_bar[i]])
                
                lp.solve()
                
                
                lp_obj[k]=lp.solution.get_objective_value()
                
            max_lp=np.max(lp_obj)
            nor_lp=np.zeros(p)

            for i in range(p):
                lp_result=(lp_obj[i]-lp_obj[p])/(max_lp-lp_obj[p])
                nor_lp[i]="{:.6f}".format(lp_result)
            label_set.append(nor_lp)
    
    if test_index !="Test":
        np.savez_compressed('data/{}_label{}_{}.npz'.format(test_index,n, ndata), label_set)

    # np.savez_compressed('data/{}_adj{}_{}.npz'.format(test_index, n, ndata), adj_mat_set)
    # np.savez_compressed('data/{}_demand{}_{}.npz'.format(test_index,n, ndata), demand_set)
    # np.savez_compressed('data/{}_fac{}_{}.npz'.format(test_index,n, ndata), fac_set)
    np.savez_compressed('data/Final_adj{}_{}.npz'.format( n, ndata), adj_mat_set)
    np.savez_compressed('data/Final_demand{}_{}.npz'.format(n, ndata), demand_set)
    np.savez_compressed('data/Final_fac{}_{}.npz'.format(n, ndata), fac_set)
    
                    
    print("avg_neighbor={}".format(avg_neighbor))
    timerecord.append(time.perf_counter()-start_time)

# with open("data/time_Record.txt", "w") as file :
#     for time in timerecord:
#         file.write( str(time)+"\n")
                
        
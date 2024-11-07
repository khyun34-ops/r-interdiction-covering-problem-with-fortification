import numpy as np
import random
from util.util import Adj_matrix_demand_generate
import time 
import cplex
from scipy.sparse import csr_matrix, save_npz




criterion=300
nnode=818
n_facility=150
n_customer=668
start_time=time.perf_counter()
file_path = "real_data_text.txt"
adj_mat_set=[]
demand_set=[]
label_set=[]
fac_set=[]


init_coordinate=np.zeros((818,2))
init_demand=np.zeros(818) 
init_fac= np.ones(( n_facility,1))


# 파일을 열고 각 줄을 읽어 데이터를 채웁니다.
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if i <818:
        # 줄바꿈 문자를 제거하고, 탭 또는 공백으로 분할합니다.
            data = line.strip().split('\t')  # 공백이나 탭으로 분리된 데이터를 가정합니다.
            
            # 좌표와 수요 데이터를 추출하여 NumPy 배열에 저장합니다.
            init_coordinate[i][0] = int(data[0])
            init_coordinate[i][1] = int(data[1])
            init_demand[i] = int(data[2])
random_indices = np.random.choice(nnode, n_facility, replace=False)

c_coordinate=np.zeros((n_customer,2))
f_coordinate=np.zeros((n_facility,2))
demand=np.zeros(n_customer)
c_index, f_index=0, 0

for i in range(818):
    if i in (random_indices):
        f_coordinate[f_index]=init_coordinate[i]
        f_index+=1
    else:
        c_coordinate[c_index]=init_coordinate[i]
        demand[c_index]=init_demand[i]
        c_index+=1
adj=np.zeros((n_customer,n_facility))
for i in range(n_customer):
    for j in range(n_facility):
        if (c_coordinate[i][0]-f_coordinate[j][0])**2+(c_coordinate[i][1]-f_coordinate[j][1])**2<criterion**2:
            adj[i][j]=1

avg_neighbor=np.sum(adj)/n_customer


col_sums = adj.sum(axis=0)
for i in range(n_facility):
    if col_sums[i]==0:
        init_fac[i][0]=0

adj_mat_set.append(adj)
demand_set.append(demand)
fac_set.append(init_fac)
np.savez_compressed('data/Real_adj.npz', adj_mat_set)
np.savez_compressed('data/Real_demand.npz', demand_set)
np.savez_compressed('data/Real_fac.npz', fac_set)

#adj_mat   1,2,3,... n-p :customer n-p+1,....,n:facility, faciliyt들의 인덱스는 0,1,2,...,p-1 
#demand : p-dimention numpy array, Neighbor:n-p dimension list, 각 list트는 인접한 facility들의 인덱스들의집합
# adj_mat, demand, Neighbor, max_C, non_cover_customer_ratio=Adj_matrix_demand_generate(n,p,criterion,mu,sigma_squared,show_plt)
# fac_init= np.ones(( p,1))
# adj_mat=adj_mat[ :n-p, n-p:]


# col_sums = adj_mat.sum(axis=0)
# for i in range(p):
#     if col_sums[i]==0:
#         fac_init[i][0]=0




# adj_mat_set.append(adj_mat)
# demand_set.append(demand)
# fac_set.append(fac_init)

# sum_neighbor=0
# neighbor_index=0
# for i in range(len(Neighbor)):
#     if len(Neighbor[i]) !=0:
#         neighbor_index+=1
#         sum_neighbor+=len(Neighbor[i])
# avg_neighbor+=sum_neighbor/(neighbor_index)

# if test_index !="Test":



#     lp_obj=np.zeros(p+1)
#     lp_obj=np.zeros(p+1)



#     for k in range(p+1):
#         lp=cplex.Cplex()
#         lp.objective.set_sense(lp.objective.sense.minimize)
    
#         s,u =[],[]
    
#         s=["s{}".format(i) for i in range(p)]
#         u=["u{}".format(j) for j in range(n-p)]
    
#         for i in range(p):
#             lp.variables.add(names=[s[i]],lb=[0],ub=[1],types=["C"])
#         for j in range(n-p):
#             lp.variables.add(names=[u[j]],lb=[0],ub=[1],types=["C"], obj=[demand[j]])
        
#         x_bar=np.zeros(p)
    
#         if k!=p:
#             x_bar[k]=1
        
#         #제약식(6)
#         var=["s{}".format(i) for i in range(p)]
#         coef=[1]*p
#         lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
#                         senses = "E", rhs=[r]) 
    
#         #제약식(7)
#         for j in range(n-p):
#             if len(Neighbor[j]) !=0:
#                 for _ ,facility_index  in enumerate(Neighbor[j]):
#                     var=[u[j],s[facility_index]]
#                     coef=[1,1]
#                     lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
#                                     senses = "G", rhs=[1])
#             else:
#                 var=["u{}".format(j)]
#                 coef=[1]
#                 lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
#                                     senses = "E", rhs=[0])
#         #제약식(8)
#         for i in range(p):
#             var=["s{}".format(i)]
#             coef=[1]
#             lp.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
#                                     senses = "L", rhs=[1-x_bar[i]])
    
#         lp.solve()
    
    
#         lp_obj[k]=lp.solution.get_objective_value()
    
#     max_lp=np.max(lp_obj)
#     nor_lp=np.zeros(p)

#     for i in range(p):
#         lp_result=(lp_obj[i]-lp_obj[p])/(max_lp-lp_obj[p])
#         nor_lp[i]="{:.6f}".format(lp_result)
#     label_set.append(nor_lp)

# # if test_index !="Test":
# # np.savez_compressed('data/{}_label{}_{}.npz'.format(test_index,n, ndata), label_set)

# # # np.savez_compressed('data/{}_adj{}_{}.npz'.format(test_index, n, ndata), adj_mat_set)
# # # np.savez_compressed('data/{}_demand{}_{}.npz'.format(test_index,n, ndata), demand_set)
# # # np.savez_compressed('data/{}_fac{}_{}.npz'.format(test_index,n, ndata), fac_set)
# # np.savez_compressed('data/Final_adj{}_{}.npz'.format( n, ndata), adj_mat_set)
# # np.savez_compressed('data/Final_demand{}_{}.npz'.format(n, ndata), demand_set)
# # np.savez_compressed('data/Final_fac{}_{}.npz'.format(n, ndata), fac_set)

        
# print("avg_neighbor={}".format(avg_neighbor))



    

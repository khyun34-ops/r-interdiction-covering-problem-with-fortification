import numpy as np
from util.util import Adj_matrix_demand_generate
import time 
import cplex

#수치 정의
ratio=1.5
global n,p,q,r
global Neighbor,demand
n=int(100*ratio)
p=int(20*ratio)
ndata=10

# q=int(2.5*ratio)
# r=int(2.5*ratio)
# r=3
# criterion=0.081
criterion=0.17

show_plt=False

R=[7]
#lognormal 분포 파라미터

mu=2.845565678551321
sigma_squared=1.4759065198095778

start_time=time.perf_counter()

avg_neighbor=0
for _ in range(ndata):
    #adj_mat   1,2,3,... n-p :customer n-p+1,....,n:facility, faciliyt들의 인덱스는 0,1,2,...,p-1 
    #demand : p-dimention numpy array, Neighbor:n-p dimension list, 각 list트는 인접한 facility들의 인덱스들의집합
    adj_mat, demand, Neighbor, max_C, non_cover_customer_ratio=Adj_matrix_demand_generate(n,p,criterion,mu,sigma_squared,show_plt)

    sum_neighbor=0
    neighbor_index=0
    for i in range(len(Neighbor)):
        if len(Neighbor[i]) !=0:
            neighbor_index+=1
            sum_neighbor+=len(Neighbor[i])
    avg_neighbor+=sum_neighbor/(neighbor_index*ndata)

    for r in R:
        ip_obj=np.zeros(p+1)
        lp_obj=np.zeros(p+1)
        type=["C","B"]
        for problem_type in type:
            start_time=time.perf_counter()
            for k in range(p+1):
                ip=cplex.Cplex()
                ip.objective.set_sense(ip.objective.sense.minimize)
                
                s,u =[],[]
                
                s=["s{}".format(i) for i in range(p)]
                u=["u{}".format(j) for j in range(n-p)]
                
                for i in range(p):
                    ip.variables.add(names=[s[i]],lb=[0],ub=[1],types=[problem_type])
                for j in range(n-p):
                    ip.variables.add(names=[u[j]],lb=[0],ub=[1],types=[problem_type], obj=[demand[j]])
                    
                x_bar=np.zeros(p)
                
                if k!=p:
                    x_bar[k]=1
                    
                #제약식(6)
                var=["s{}".format(i) for i in range(p)]
                coef=[1]*p
                ip.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                senses = "E", rhs=[r]) 
                
                #제약식(7)
                for j in range(n-p):
                    if len(Neighbor[j]) !=0:
                        for _ ,facility_index  in enumerate(Neighbor[j]):
                            var=[u[j],s[facility_index]]
                            coef=[1,1]
                            ip.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                            senses = "G", rhs=[1])
                    else:
                        var=["u{}".format(j)]
                        coef=[1]
                        ip.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                            senses = "E", rhs=[0])
                #제약식(8)
                for i in range(p):
                    var=["s{}".format(i)]
                    coef=[1]
                    ip.linear_constraints.add(lin_expr=[cplex.SparsePair(var,coef)], \
                                            senses = "L", rhs=[1-x_bar[i]])
                
                ip.solve()
                
                if problem_type=="C":
                    lp_obj[k]=ip.solution.get_objective_value()
                else:
                    ip_obj[k]=ip.solution.get_objective_value()
            if problem_type =="C":
                lp_time=time.perf_counter() -start_time
            else:
                ip_time=time.perf_counter() -start_time
            


        max_lp=np.max(lp_obj)
        max_ip=np.max(ip_obj)
        nor_lp=np.zeros(p)
        nor_ip=np.zeros(p)

        for i in range(p):
            lp_result=(lp_obj[i]-lp_obj[p])/(max_lp-lp_obj[p])
            ip_result=(ip_obj[i]-ip_obj[p])/(max_ip-ip_obj[p])
        
            nor_lp[i]="{:.6f}".format(lp_result)
            nor_ip[i]="{:.6f}".format(ip_result)
        non_zero_indexlp=[]
        zero_indexlp=[]
        non_zero_indexip=[]
        zero_indexip=[]
        for i in range(p):
            if nor_lp[i]>0.00000001:
                non_zero_indexlp.append(i) 
            else:
                zero_indexlp.append(i)

        for i in range(p):
            if nor_ip[i]>0.00000001:
                non_zero_indexip.append(i) 
            else:
                zero_indexip.append(i)

        len_ip=len(non_zero_indexip)
        len_lp=len(non_zero_indexlp)
        final_ip=np.zeros(len_ip)
        final_lp=np.zeros(len_lp)
        for i in range(len_ip):
            index=0
            if nor_ip[non_zero_indexip[i]]<0.1 and  nor_ip[non_zero_indexip[i]]>0:
                print("Ip에서 index{} 의 값은 매우 작다".format(non_zero_indexip[i]))
            for j in range(len_ip):
                if nor_ip[non_zero_indexip[i]]-nor_ip[non_zero_indexip[j]]==0 and i<j:
                    index+=1
                elif nor_ip[non_zero_indexip[i]]-nor_ip[non_zero_indexip[j]]>0:
                    index+=1
            final_ip[len_ip-1-index]=non_zero_indexip[i]
            
        for i in range(len_lp):
            index=0
            if nor_lp[non_zero_indexlp[i]]<0.1 and  nor_lp[non_zero_indexlp[i]]>0:
                print("Lp에서 index{} 의 값은 매우 작다".format(non_zero_indexlp[i]))
            for j in range(len_lp):
                if nor_lp[non_zero_indexlp[i]]-nor_lp[non_zero_indexlp[j]]==0 and i<j:
                    index+=1
                    
                elif nor_lp[non_zero_indexlp[i]]-nor_lp[non_zero_indexlp[j]]>0:
                    index+=1
            final_lp[len_lp-1-index]=non_zero_indexlp[i]
        sdfdf=0
            
                    



        # sort_lp = sorted(range(len(non_zero_indexlp)), key=lambda x: non_zero_indexlp[x], reverse=True)
        # sort_ip = sorted(range(len(non_zero_indexip)), key=lambda x: non_zero_indexip[x], reverse=True)
        
        # lp_sort=np.argsort(nor_lp)[::-1]
        # ip_sort=np.argsort(nor_ip)[::-1]

        # print("p={}".format(p))
        # print(sort_lp)
        # print(sort_ip)
        # print(final_lp)
        # print(final_ip)

        print("lp_time={} ".format(lp_time))
        print("ip_time={} ".format(ip_time))

print("avg_neighbor={}".format(avg_neighbor))
                
        


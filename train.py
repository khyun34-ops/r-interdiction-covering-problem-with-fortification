import torch
import torch.nn.functional as F
import numpy as np
import statistics
import random
from torch.nn import Linear
from tqdm import tqdm
import time

from util.util import  GNNmodel, getloss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(adj, demand, label, fac_init, n_train, batch_size,device='cuda'):
    train_datasets = []
    val_datasets = []
    for i in range(len(adj)):
        # 전체 데이터셋
        adj_tensor = torch.tensor(adj[i]).float()
        demand_tensor = torch.tensor(demand[i]).unsqueeze(-1).float()
        label_tensor = torch.tensor(label[i]).long()
        fac_tensor= torch.tensor(fac_init[i]).float()
        
        # 훈련 및 검증 데이터셋 분할
        train_dataset = TensorDataset(adj_tensor[:n_train], demand_tensor[:n_train], label_tensor[:n_train],fac_tensor[:n_train])
        val_dataset = TensorDataset(adj_tensor[n_train:], demand_tensor[n_train:], label_tensor[n_train:],fac_tensor[n_train:])
        
        # DataLoader 구성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=int(0.25*batch_size), shuffle=False)
        
        train_datasets.append(train_loader)
        val_datasets.append(val_loader)
    
    return train_datasets, val_datasets

nhid_set = {
    'facility_nhid': 8, 'customer_nhid': 8, 'root_nhid': 8,
}


file = open("train_result/train_parameter.txt", 'w')
for _ in range(5):


    seed = random.randint(1, 150) # 원하는 시드 값
    a = random.randint(1, 9)
    b = random.randint(5, 6)

    # a * 10^(-b)를 학습률로 계산
    lr = a * 10**(-b)
    # lr=5e-5
    a = random.randint(1, 9)
    b = random.randint(5, 6)
    wd=a * 10**(-b)
    patience=100
    epochs=3000
    batchsize=40
    nbatch=9600/batchsize

    amplify=1

    npna=2

    adj=[]
    demand=[]
    label=[]
    fac_init=[]
    nnode=[100,150,200]
    aggregators = ['mean', 'min', 'max', 'std']  # 사용할 집계 함수들
    scalers = ['identity', 'amplification', 'attenuation']  # 사용할 스케일러들
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시 모든 GPU에 시드 설정

    for i in range(3):
        with np.load("data/train_data/newadj{}_4000.npz".format(nnode[i])) as data:
            adj.append(data[data.files[0]])
        with np.load("data/train_data/demand{}_4000.npz".format(nnode[i])) as data:
            demand.append(data[data.files[0]])
        with np.load("data/train_data/label{}_4000.npz".format(nnode[i])) as data:
            label.append(data[data.files[0]])
        with np.load("data/train_data/fac_init{}_4000.npz".format(nnode[i])) as data:
            fac_init.append(data[data.files[0]])

    if torch.cuda.is_available():
        device=torch.device('cuda')

    print('Using device:', device)


    train_data, val_data = prepare_data(adj, demand, label, fac_init, n_train=3200, batch_size=batchsize, device=device)



    model=GNNmodel(nhid_set, amplify,npna)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    patience_counter = 0 
    best_val_loss = float('inf')
    for epoch in range( epochs):
        start_time = time.time()
        model.train()
        train_loss=0
        
        for i in range(3):
            for data in train_data[i]:
                optimizer.zero_grad()
                adj, demand, labels, fac_init =[d.to(device) for d in data]  # DataLoader에서 데이터 로드
                
                output =model( demand, fac_init, adj)
                loss= getloss(output.squeeze(-1).float(),labels.float())
                loss.backward()
                train_loss+=loss/nbatch
                optimizer.step()
                
        
        
        model.eval()
        val_loss=0
        for i in range(3):
            for data in val_data[i]:
                adj, demand, labels, fac_init =[d.to(device) for d in data]  # DataLoader에서 데이터 로드
                
                output =model( demand, fac_init, adj)
                loss= getloss(output.squeeze(-1).float(),labels.float())
                val_loss+=loss/nbatch
                
        end_time = time.time()    

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}',f"Duration: {end_time - start_time:.4f} seconds")
        
        if  0.00005< best_val_loss- val_loss:
            patience_counter = 0 
            best_val_loss = val_loss
            
        
            # file.write("seed={}, lr={}, wd={}, best_loss={}, epoch={}\n".format(seed,lr,wd,best_val_loss,epoch))
            torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        # 여기에 추가로 저장하고 싶은 정보를 넣을 수 있습니다.
    }, "train_result/best_model{}.pth".format(seed))
        else: 
            patience_counter += 1 
        
        if patience_counter >= patience:
            print("Early stopping triggered at epoch {epoch} due to no improvement in validation loss.")
            file.write("seed={}, lr={}, wd={}, best_loss={}, epoch={}\n".format(seed,lr,wd,best_val_loss,epoch))
            break  # 조기 종료 조건 충족
            

file.close()


    

# data=np.load("data/adj.npz")
# print(data.files)
# print(data['arr_0'])
# print(data['arr_0'][0])
# # 특정 배열에 접근하기
# 예를 들어, 'arr_0'이라는 이름의 배열에 접근하려면 다음과 같이 합니다.


# 사용 후 .npz 파일 객체 닫기
# data.close()
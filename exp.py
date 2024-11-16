from torchmetrics.classification import MulticlassF1Score
from client import Client
from typing import Any, Dict, List, Optional, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import torch
import random
import pickle
from train import *
from model import *
from utils.utils import *
from utils.metrics import *
from utils.clusters import *
import communication.gRPC as grpc
import threading
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

random.seed(1)
    
class Exp(object):
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.Client = Client(args)
        self.gRPCClient = grpc.gRPCClient(args.host, args.port)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.patience = self.args.patience
        self.clients_per_rounds = self.args.clients_per_rounds
        self.server = None
        self.epoch = self.args.n_client_epochs
        self.patience = 0
        self.n_clients = self.args.n_clients
        self.n_class = self.args.n_class
        self.mu = self.args.mu
        self.processes = []
        self.parent_conns = []
        self.device_dataloaders = []
        self.device_subset = []
        self.start_events = []
        self.states = {}
        self.Client._printClients()
        self.serve()
        print(args)
        
        torch.cuda.empty_cache()
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환 (0~255 값을 0~1로 스케일링)
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 평균과 표준편차로 정규화
        ])
        
        if self.args.data == "cifar10":
            self.train_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=True, download=True, transform=transform)
        elif self.args.data == "cifar100":
            self.train_dataset = datasets.CIFAR100(root='../datasets/cifar100/', train=True, download=True,  transform=transform)
        elif self.args.data == "FashionMNIST":
            self.train_dataset = datasets.FashionMNIST(root='../datasets/FashionMNIST/', train=True, download=True,  transform=transforms.ToTensor())
        elif self.args.data == "EMNIST":
            self.train_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=True, download=True,  transform=transforms.ToTensor())
            
    def serve(self):
        mp.set_start_method('spawn', force=True)
        p = mp.Process(target=grpc.serve, args=(2000 * 1024 * 1024, self.args.port))
        self.server = p
        p.start()

    def createClients(self, clients):        
        # 클라이언트별로 프로세스 생성
        for idx in range(self.n_clients):
            parent_conn, child_conn = mp.Pipe()
            start =mp.Event()
            if self.args.aggregator == "cluster":
                p = mp.Process(target=train_local_client_cluster, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, child_conn, self.n_class, start))
            elif self.args.aggregator == "fedavg":
                p = mp.Process(target=train_local_client_prox, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, child_conn, 0, start))
            elif self.args.aggregator == "fedprox":
                p = mp.Process(target=train_local_client_prox, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, child_conn, self.mu, start))
            elif self.args.aggregator == "fednova":
                p = mp.Process(target=train_local_client_nova, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, child_conn, start))
            
            self.processes.append(p)
            self.parent_conns.append(parent_conn)
            self.start_events.append(start)
            
        for i, p in enumerate(self.processes):
            print(f"Client {i} created! Data: {len(self.device_dataloaders[i])}")
            p.start()
            
        for i, p in enumerate(self.start_events):
            p.set()
            
        print(f"Training start!")
            
    def updateModel(self, model, conns, devices):
        if devices is None:
            for idx in range(self.n_clients):
                conns[idx].send(model)       
        else:
            for idx in range(self.n_clients):
                if idx in devices:
                    conns[idx].send([model, True])
                else:
                    conns[idx].send([model, False])
    
    def train(self, setting):
    # 멀티프로세싱 설정 (한 번만 호출)
        f = open(f"../metrics/results/{self.args.aggregator}/{self.args.aggregator}_{self.args.data}_{self.args.model_name}_{'non_iid' if self.args.non_iid else 'iid'}.csv", "a")
        f.write("round, avg_loss, avg_accuracy, avg_mae, avg_mse, avg_rae, avg_rmse, time\n")
        early_stopping = EarlyStopping(patience = self.args.patience, verbose=True)
        
        path = os.path.join("./checkpoints", setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        clients = self.Client._createClient()
        
        mp.set_start_method('spawn', force=True)
        
        if self.args.non_iid == 1:
            device_data_indices = split_dataset_non_iid(self.train_dataset, self.n_clients)
        else:
            ratio = generate_random_device_data_ratio(self.n_clients)
            device_data_indices = create_random_dataset(self.train_dataset, self.n_clients, self.n_class, ratio)
        
        for idx in range(self.n_clients):
            indices = device_data_indices[idx]
            batch_size = clients[idx][2]
            subset = Subset(self.train_dataset, indices)
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
            self.device_subset.append(subset)
            self.device_dataloaders.append(dataloader)
        
        self.createClients(clients)
        self.gRPCClient.setup(self.args.data, self.args.n_class, self.args.model_name)
        
        round = 1
        indice = None
        step_dict = {}
        s = time.time()
        
        # 닫아라
        if self.args.aggregator != "cluster":
            while round <= self.args.rounds:
                if indice is None:
                    indice = random.sample(range(self.n_clients), max(1, int(self.n_clients * self.args.frac)))
                    print(f"=================== selected devices : {indice} ===================")
                    self.states.clear()
                    step_dict.clear()
                
                self.drop_states(range(self.n_clients))
                while len(self.states.keys()) < len(indice):
                    for device in indice:
                        while self.parent_conns[device].poll(): 
                            res = self.parent_conns[device].recv()  # 데이터 받기
                            if self.args.aggregator == "fednova":
                                self.states[device], tau, _ = res
                                step_dict[device] = tau
                            else: 
                                self.states[device], _ = res
                    
                if len(self.states.keys()) >= len(indice):
                    res = self.gRPCClient.sendStates(self.states, self.args.aggregator, step_dict, False)
                    global_model, loss, accuracy, mae, mse, rse, rmse = pickle.loads(res.state), res.loss, res.accuracy, res.mae, res.mse, res.rse, res.rmse
                
                    print(f"============ Round {round} | Loss {loss} | Accuracy {accuracy} | Time {str(time.time() - s)} ============")
                    
                    f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(mae)+","+str(mse)+","+str(rse)+","+str(rmse)+","+str(time.time() - s)+"\n")
                    
                    self.updateModel(global_model, self.parent_conns, range(self.n_clients))
                    
                    indice = None
                    self.states.clear()
                    step_dict.clear()
                    round += 1
                    
                    if self.args.early_stopping == 1 :
                        early_stopping(loss, global_model, path)
                        
                if early_stopping.early_stop:
                    print("Early stopping")
                    round = 9999999
                    break
        
        else:
            # TODO : 한 바퀴 순환하면 클러스터 분배 다시 
            update_counter = UpdateCounter(patience= self.args.n_cluster / 2 + 1, verbose=True)
            round, n_cluster, update_round, selected_cluster_idx, self.patience = 1, self.args.n_cluster - 1, 1, 0, int(self.args.max_cluster + 1.5)
            cluster_dict = {} # cluster: list(devices)
            device_dict = {} # device: cluster
            labels_dict = {} # device: labels
            times_dict = {} # device: time
            pass_list = [] # 여기에 있는 클러스터는 업데이트 한 번 건너 뜀
            re_cluster = 0
            drop = False
            
            best_score = 100
            while round <= self.args.rounds:
                
                # device 학습 시간 / label 분포 측정
                if round == 1:
                    start_time = time.time()
                    visited = [False for _ in range(self.n_clients)]
                    while len(times_dict.keys()) < self.n_clients:
                        for device in range(self.n_clients):
                            while self.parent_conns[device].poll():
                                _, _, labels_dict[device] = self.parent_conns[device].recv()
                                times_dict[device] = time.time() - start_time
                                visited[device] = True
                    
                    similarity_scores = calc_scores(range(self.n_clients), labels_dict)
                    linked = linkage(squareform(similarity_scores), method='average')
                    
                # cluster : 2 ~ n / f1_score랑  loss값 평가해서 클러스터 내 일부 디바이스만 사용하기
                if round == update_round or update_counter.update:
                    if update_counter.update and n_cluster >= self.args.max_cluster:
                        print("재 클러스터")
                        
                        # 최적 모델 뿌려줌
                        self.updateModel(pickle.loads(self.gRPCClient.getGlobalModel().state), self.parent_conns, list(i for i in range(self.n_clients)))
                        self.drop_states(range(self.n_clients))
                        n_cluster = self.args.n_cluster - 1
                        
                        # 랜덤 샘플링
                        if re_cluster == 0:
                            for _ in range(3):
                                indice = random.sample(range(self.n_clients), max(1, int(self.n_clients * self.args.frac)))
                                print(f"=================== selected devices : {indice} ===================")
                                self.states.clear()
                                step_dict.clear()
                                buffer = {}
                                
                                while len(list(self.states.values())) < self.n_clients:
                                    for device in range(self.n_clients):
                                        while self.parent_conns[device].poll(): 
                                            self.states[device], _, _ = self.parent_conns[device].recv()
                                            if device in indice: buffer[device] = self.states[device]
                                        
                                res = self.gRPCClient.sendStates(buffer, self.args.aggregator, step_dict, drop)
                                global_model, loss, accuracy, mae, mse, rse, rmse = pickle.loads(res.state), res.loss, res.accuracy, res.mae, res.mse, res.rse, res.rmse
                                
                                res = self.gRPCClient.getGlobalModel()
                                if best_score > res.loss:
                                    self.updateModel(pickle.loads(res.state), self.parent_conns, list(i for i in range(self.n_clients)))
                                    best_score = res.loss
                                
                                update_counter.counter = 0
                                update_counter(loss, global_model, path)
                                
                                print(f"============ Round {round} | Loss {loss} | Accuracy {accuracy} | selected cluster {cur_cluster} | Time {str(time.time() - s)}  ============")   
                                f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(mae)+","+str(mse)+","+str(rse)+","+str(rmse)+","+str(time.time() - s)+"\n")
                                round += 1             
                        
                        similarity_scores = calc_scores(range(self.n_clients), labels_dict) if re_cluster else calc_scores(range(self.n_clients), self.states)
                        linked = linkage(squareform(similarity_scores), method='average') # 메서드 찾기
                        re_cluster = 0 if re_cluster else 1
                    
                    patience= [i + 1 for i in range(self.args.max_cluster + 1)]
                    selected_cluster_idx, idx, n_cluster = 0, 0, n_cluster + 1
                    update_counter.patience = patience[n_cluster]
                    update_counter.update = False
                    update_counter.counter = 0
                    protection = True # 업데이트 직후 한 번만 보호
                    schedule = []
                    max_time ={}
                    cluster_dict.clear(); device_dict.clear(); pass_list.clear()
                    
                    # 클러스터 분할
                    cluster_labels = fcluster(linked, n_cluster , criterion='maxclust') # device, cluster
                    
                    # 클러스터 스케줄링
                    for device, cluster in enumerate(cluster_labels):
                        if cluster_dict.get(cluster) is None:
                            cluster_dict[cluster] = [device]  
                        elif device not in cluster_dict[cluster]: 
                            cluster_devices = cluster_dict[cluster]
                            cluster_devices.append(device)
                            cluster_dict[cluster] = cluster_devices
                            
                        device_dict[device] = cluster
                        max_time[cluster] = times_dict[device] if max_time.get(cluster) is None else max(max_time[cluster], times_dict[device])
                    
                    for _ in range(n_cluster):
                        min_idx = np.argmin(list(max_time.values()))
                        cluster = list(max_time.keys())[min_idx]
                        schedule.append(cluster)
                        max_time.pop(cluster)
                        
                    print(f"cluster updated. {cluster_dict.items()}")
                    
                # 데이터 받아오기 / 나중에 클러스터 버퍼로 리팩터링하기
                cur_cluster = schedule[selected_cluster_idx]
                global_update, flag = True, True
                response = {}
                
                # 아웃라이어 제거 임시용 
                # cluster updated. dict_items([(2, [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]), (3, [3]), (1, [5, 14]), (4, [15])]) 아웃라이어는 사전에 쳐내야할 듯
                # if len(cluster_dict[cur_cluster]) < 2:
                #     pass_list.append(selected_cluster_idx)
                
                # loss 값이 낮으면 한 번 건너 뜀
                while selected_cluster_idx in pass_list: 
                    pass_list.remove(selected_cluster_idx)
                    selected_cluster_idx = (selected_cluster_idx + 1) % n_cluster
                    cur_cluster = schedule[selected_cluster_idx]
                
                # 클러스터에 속한 디바이스로부터 데이터를 받기
                while True:
                    for device in cluster_dict[cur_cluster]:
                        while self.parent_conns[device].poll(): 
                            self.states[device],  _, _ = self.parent_conns[device].recv() 
                            device_cluster = device_dict[device] # {device: cluster}
                            
                            if response.get(device_cluster) is None:
                                response[device_cluster] = [device]  
                            
                            elif device not in response[device_cluster]: 
                                x = response[device_cluster] 
                                x.append(device)
                                response[device_cluster] = x
                                
                    # 클러스터에 속한 디바이스로부터 데이터를 전부 받았는지 확인.
                    if response.get(cur_cluster) is not None:
                        flag = True
                        for cluster, devices in list(response.items()):
                            if len(cluster_dict[cluster]) != len(devices):
                                flag = False
                                break
                            
                        # 클러스터 데이터를 전부 받음.
                        if flag: 
                            # 이미 다 받았지만 전체적으로 한 번더 업데이트
                            while self.parent_conns[device].poll(): 
                                self.states[device], step_dict[device], _, _ = self.parent_conns[device].recv() 
                                device_cluster = device_dict[device]
                            break
                
                res = self.gRPCClient.sendStates(self.states, self.args.aggregator, step_dict, drop)
                global_model, loss, accuracy, mae, mse, rse, rmse = pickle.loads(res.state), res.loss, res.accuracy, res.mae, res.mse, res.rse, res.rmse
                best_score = min(loss, best_score)
                
                print(f"============ Round {round} | Cluster {cur_cluster} | Loss {loss} | Accuracy {accuracy} | Time {str(time.time() - s)}  ============")
                
                f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(mae)+","+str(mse)+","+str(rse)+","+str(rmse)+","+str(time.time() - s)+"\n")
                
                # 스케줄링된 디바이스가 모두 pass_list에 속해있는지 확인
                if selected_cluster_idx + 1 < n_cluster:
                    for x in (selected_cluster_idx + 1, n_cluster):
                        if x not in pass_list:
                            global_update = False
                            break
                
                 # 한 바퀴 돌면 전체 업데이트
                if global_update or selected_cluster_idx == n_cluster - 1:
                    self.updateModel(pickle.loads(self.gRPCClient.getGlobalModel().state), self.parent_conns, list(i for i in range(self.n_clients)))
                    protection = False
                else:
                    self.updateModel(global_model, self.parent_conns, list(self.states.keys()))
                    
                if self.args.early_stopping == 1:
                    early_stopping(loss, global_model, path)
                    
                    if early_stopping.early_stop:
                        print("Early stopping")
                        round = 9999999
                        break
                
                if loss > best_score:
                    pass_list.append(selected_cluster_idx)
                
                update_counter(loss, global_model, path)
                    
                if protection == True: update_counter.counter = 0
                
                selected_cluster_idx = (selected_cluster_idx + 1) % n_cluster
                step_dict.clear(); self.states.clear(); pass_list.clear()
                round += 1
                
        print(f"Train terminated. | time: {time.time() - s}")
        
        for idx in range(self.n_clients):
            self.processes[idx].terminate()
            self.processes[idx].join()
            
        self.server.terminate()
        self.server.join()
        f.close()
        
        torch.cuda.empty_cache()

    def drop_states(self, devices):
        for device in devices:
            while self.parent_conns[device].poll(): 
                self.parent_conns[device].recv()
        print(f"Drop complete.")
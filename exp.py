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
        self.min_clusters = self.args.min_clusters
        self.server = None
        self.epoch = self.args.n_client_epochs
        self.patience = 0
        self.n_clients = self.args.n_clients
        self.mu = self.args.mu
        self.eps = 1e-5
        self.processes = []
        self.threads = []
        self.parent_conns = []
        self.device_dataloaders = []
        self.device_subset = []
        self.start_events = []
        self.states = []
        self.n_class = self.args.n_class
        self.threshold = 15
        torch.cuda.empty_cache()
        
        print(args)
        self.Client._printClients()
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환 (0~255 값을 0~1로 스케일링)
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 평균과 표준편차로 정규화
        ])
        
        if self.args.data == "cifar10":
            self.train_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
        elif self.args.data == "cifar100":
            self.train_dataset = datasets.CIFAR100(root='../datasets/cifar100/', train=True, download=True,  transform=transform)
            test_dataset = datasets.CIFAR100(root='../datasets/cifar100/', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
        elif self.args.data == "FashionMNIST":
            self.train_dataset = datasets.FashionMNIST(root='../datasets/FashionMNIST/', train=True, download=True,  transform=transforms.ToTensor())
            test_dataset = datasets.FashionMNIST(root='../datasets/FashionMNIST/', train=False, download=True, transform=transforms.ToTensor())
            self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
        elif self.args.data == "EMNIST":
            self.train_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=True, download=True,  transform=transforms.ToTensor())
            test_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=False, download=True, transform=transforms.ToTensor())
            self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    
    def receiveParam(self, conn, states):
        receiver_thread = threading.Thread(target=states.append(conn.recv()))
        receiver_thread.start()  
    
    def serve(self):
        mp.set_start_method('spawn', force=True)
        p = mp.Process(target=grpc.serve, args=(2000 * 1024 * 1024, self.args.port))
        self.server = p
        p.start()
    
    def clearEnv(self):
        self.processes = []
        self.threads = []
        self.parent_conns = []
        self.device_dataloaders = []
        self.stop_events = []
        self.states = {}
    
    def createClients(self, clients):        
        # 클라이언트별로 프로세스 생성
        for idx in range(self.n_clients):
            parent_conn, child_conn = mp.Pipe()
            start =mp.Event()
            if self.args.aggregator == "cluster":
                p = mp.Process(target=train_local_client_cluster, args=(clients, self.device_dataloaders[idx], self.clients_per_rounds * self.epoch, idx, child_conn, self.n_class, self.mu, start))
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
            print(f"Client {i} training start!")
            p.set()
            
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
        self.clearEnv()
        f = open(f"../metrics/results/{self.args.aggregator}/{self.args.aggregator}_{self.args.data}_{self.args.model_name}_{'non_iid' if self.args.non_iid else 'iid'}.csv", "a")
        f.write("round, avg_loss, avg_accuracy, avg_mae, avg_mse, avg_rae, avg_rmse, time\n")
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
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
        
        round = 1
        indice = None
        step_dict = {}
        s = time.time()
        
        if self.args.aggregator != "cluster":
            while round <= self.args.rounds:
                if indice is None:
                    indice = random.sample(range(self.n_clients), max(1, int(self.n_clients * self.args.frac)))
                    print(f"=================== selected devices : {indice} ===================")
                    self.states.clear()
                    step_dict.clear()
                
                if round > 1:
                    self.drop_states()
                
                for device in indice:
                    if self.parent_conns[device].poll(timeout=0.1): 
                        res = self.parent_conns[device].recv()  # 데이터 받기
                        if self.args.aggregator == "fednova":
                            self.states[device], tau, _ = res
                            step_dict[device] = tau
                        else: 
                            self.states[device], _ = res
                    
                if len(self.states.keys()) >= len(indice):
                    states = []
                    steps = []
                    if self.args.aggregator == "fednova":
                        states, steps = list(self.states.values()), list(step_dict.values())
                    else:
                        states, steps = list(self.states.values()), [0 for _ in range(len(indice))]
                        
                    global_model, loss, accuracy, mae, mse, rse, rmse, f1_score = self.gRPCClient.sendStates(states, self.args.aggregator, steps, self.args.model_name, self.args.data, self.args.n_class)
                    global_model = pickle.loads(global_model)
                
                    print(f"============ Round {round} | Loss {loss} | Accuracy {accuracy} | Time {str(time.time() - s)} ============")
                    
                    f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(mae)+","+str(mse)+","+str(rse)+","+str(rmse)+","+str(time.time() - s)+"\n")
                    
                    self.updateModel(global_model, self.parent_conns)
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
        
        # TODO : 클러스터 디바이스 Fednova, 업데이트 횟수는 어케하지
        else:
            round, n_cluster, update_round, selected_cluster_idx = 1, 2, 1, 0
            cluster_dict = {} # cluster: list(devices)
            device_dict = {} # device: cluster
            labels_dict = {} # device: labels
            times_dict = {} # device: time
            update_dict = {} # cluster: update
            
            while round <= self.args.rounds:
                
                # device 학습 시간 / label 분포 측정
                if round == 1:
                    start_time = time.time()
                    visited = [False for _ in range(self.n_clients)]
                    while len(times_dict.keys()) < self.n_clients:
                        for device in range(self.n_clients):
                            if visited[device] == False and self.parent_conns[device].poll(timeout=0.1):
                                _, _, _, label = self.parent_conns[device].recv()
                                times_dict[device] = time.time() - start_time
                                labels_dict[device] = label
                                visited[device] = True
                    
                    similarity_scores = calc_scores(range(self.n_clients), labels_dict)
                    linked = linkage(squareform(similarity_scores), method='average')

                # cluster : 1 ~ n
                if round == update_round:
                    selected_cluster_idx, idx, n_cluster = 0, 0, n_cluster + 1
                    cluster_dict.clear(); device_dict.clear(); update_dict.clear()
                    
                    # 클러스터 분할
                    cluster_labels = fcluster(linked, n_cluster , criterion='maxclust') # device, cluster
                    
                    # 클러스터 스케줄링
                    max_time ={}
                    for device, cluster in enumerate(cluster_labels):
                        if cluster_dict.get(cluster) is None:
                            cluster_dict[cluster] = [device]  
                        elif device not in cluster_dict[cluster]: 
                            x = cluster_dict[cluster]
                            x.append(device)
                            cluster_dict[cluster] = x
                            
                        device_dict[device] = cluster
                        max_time[cluster] = times_dict[device] if max_time.get(cluster) is None else max(max_time[cluster], times_dict[device])
                    
                    schedule = [] # 클러스터 스케줄
                    for i in range(n_cluster):
                        min_idx = np.argmin(list(max_time.values()))
                        cluster = list(max_time.keys())[min_idx]
                        schedule.append(cluster)
                        max_time.pop(cluster)
                        update_dict[cluster] = 0
                        
                    update_round = update_round + n_cluster * 100
                
                # 데이터 받아오기 / 나중에 클러스터 버퍼로 리팩터링하기
                res_cluster = {}
                updates = {}
                flag = True
                cur_cluster = schedule[selected_cluster_idx]
                while True:
                    for device in cluster_dict[cur_cluster]:
                        if self.parent_conns[device].poll(timeout=0.1): 
                            self.states[device], step_dict[device], l, _ = self.parent_conns[device].recv()  # 데이터 받기
                            device_cluster = device_dict[device]
                            
                            if res_cluster.get(device_cluster) is None:
                                res_cluster[device_cluster] = [device]  
                            elif device not in res_cluster[device_cluster]: 
                                x = res_cluster[device_cluster] 
                                x.append(device)
                                res_cluster[device_cluster] = x
                            
                    if res_cluster.get(cur_cluster) is not None:
                        flag = True
                        for cluster, devices in list(res_cluster.items()):
                            if len(cluster_dict[cluster]) != len(devices):
                                flag = False
                                break
                            
                        # 업데이트 횟수 증가
                        if flag: 
                            for cluster in list(res_cluster.keys()):
                                update_dict[cluster] = update_dict[cluster] + 1
                                updates[cluster] = update_dict[cluster]
                            break
                #global_model = self.aggregation(update_dict=updates, client_states=self.states, device_dict=device_dict, weights=step_dict, cluster_dict=cluster_dict)
                states, steps = self.states, [0]
                res = self.gRPCClient.sendStates(states, self.args.aggregator, steps, self.args.model_name, self.args.data, self.args.n_class)
                global_model, loss, accuracy, mae, mse, rse, rmse, f1_score = pickle.loads(res.state), res.loss, res.accuracy, res.mae, res.mse, res.rse, res.rmse, res.f1_score
                 
                
                print(f"============ Round {round} | Loss {loss} | Accuracy {accuracy} | Time {str(time.time() - s)} ============")
                
                f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(mae)+","+str(mse)+","+str(rse)+","+str(rmse)+","+str(time.time() - s)+"\n")
                
                if selected_cluster_idx == n_cluster - 1:
                    self.updateModel(global_model, self.parent_conns, list(i for i in range(self.n_clients)))
                else:
                    self.updateModel(global_model, self.parent_conns, list(self.states.keys()))
                
                if self.args.early_stopping == 1:
                    early_stopping(loss, global_model, path)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    round = 9999999
                    break
                
                # drop states
                devices = []
                devices.extend(list(cluster_dict[cur_cluster]))
                self.drop_states(devices)
                
                selected_cluster_idx = (selected_cluster_idx + 1) % n_cluster
                step_dict.clear()
                self.states.clear()
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
            if self.parent_conns[device].poll(timeout=0.1): 
                self.parent_conns[device].recv()
                        
        print("=============== drop states ===================")
        print(devices)
        print("=============== drop states ===================")

    def aggregation(self, update_dict, client_states, device_dict, weights, cluster_dict, cur_cluster):
        cluster_steps = {} # cluster: total_steps
        # rebalanced_update_dict = {} # 업데이트 순서 반전
        # total_update = sum(list(update_dict.values()))
        
        # 모델 생성
        model_state = copy.deepcopy(list(client_states.values())[0])
        keys = model_state.keys()
        for key in keys:
            model_state[key] = torch.zeros_like(model_state[key].cpu().float()) 
            
        '''
        # cluster_update_cnt 초기화
        visited = [False for _ in range(max(list(update_dict.keys())) + 1)]
        while len(list(rebalanced_update_dict.keys())) < len(list(update_dict.keys())):
            max_update = -1
            max_cluster = -1
            min_update = 9999999
            min_cluster = 9999999
            
            for cluster, update in list(update_dict.items()): 
                if min_update > update and visited[cluster] == False:
                    min_update = update
                    min_cluster = cluster
                    
                if max_update < update and rebalanced_update_dict.get(cluster) is None:
                    max_update = update
                    max_cluster = cluster
                    
            rebalanced_update_dict[max_cluster] = min_update
            visited[min_cluster] = True
        '''
        
        '''
        # cluster_steps 초기화
        for device, cluster in list(device_dict.items()):
            if cluster_steps.get(cluster) is None:
                cluster_steps[cluster] = 0
            
            try:
                cluster_steps[cluster] = cluster_steps[cluster] + weights[device]
            except Exception as e:
                print(e)
                print(weights)
                print(device)
        '''
        
        # Aggregation 진행 (가중치 = device_step * update / (cluster_steps[cluster] * total_update))
        for devices in list(cluster_dict[cur_cluster]):
            for device in list(devices):
                device_step = weights[device]
                # update = rebalanced_update_dict[cluster]
                factor = device_step / (cluster_steps[cur_cluster])
            
                for key in keys:
                    model_state[key] += client_states[device][key].cpu().float() * factor
                    
        return model_state
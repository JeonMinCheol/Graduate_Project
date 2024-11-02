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
        self.threshold = 30
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
            print(f"Client {i} training start!")
            p.set()
            
    def updateModel(self, model, conns):
        for idx in range(self.n_clients):
            conns[idx].send(model)            
    
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
        
        s = time.time()
        if self.args.aggregator != "cluster":
            round = 1
            indice = None
            while round < self.args.rounds:
                if indice is None:
                    indice = random.sample(range(self.n_clients), max(1, int(self.n_clients * self.args.frac)))
                    print(f"=================== selected devices : {indice} ===================")
                    self.states.clear()
                
                for device in indice:
                    if self.parent_conns[device].poll(timeout=0.1): 
                        res = self.parent_conns[device].recv()  # 데이터 받기
                        if self.args.aggregator == "fednova":
                            self.states[device], tau, _ = res
                        else: 
                            self.states[device], _ = res
                        
                if len(self.states.keys()) >= max(1, int(self.n_clients * self.args.frac)):
                    states = []
                    steps = []
                    
                    for device in indice:
                        if self.args.aggregator == "fednova":
                            state, tau= self.states[device], tau
                        else:
                            state, tau= self.states[device], 0
                        
                        states.append(state); steps.append(tau)
                    
                    global_model = pickle.loads(self.gRPCClient.sendStates(states, self.args.aggregator, steps).state)
                
                    loss, accuracy, avg_mae, avg_mse, avg_rse, avg_rmse, f1_score = self.valid(global_model)
                    print(f"============ Round {round} | Loss {loss} | Accuracy {accuracy} | Time {str(time.time() - s)} ============")
                    
                    f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(avg_mae)+","+str(avg_mse)+","+str(avg_rse)+","+str(avg_rmse)+","+str(time.time() - s)+"\n")
                    
                    self.updateModel(global_model, self.parent_conns)
                    indice = None
                    self.states.clear()
                    round += 1
                    
                    if self.args.early_stopping == 1 :
                        early_stopping(loss, global_model, path)
                        
                if early_stopping.early_stop:
                    print("Early stopping")
                    round = 9999999
                    break
                        
        else:
            round = 1
            threshold = self.threshold # 걸리는 시간, sil_score, loss, 글로벌 모델 loss로 변동
            sil_scores = []
            labels = {}
            steps = {}
            times = []
            thresholds = []
            f1_scores = []
            losses = []
            best_score = 999
            
            while round < self.args.rounds:
                cur_score = 0
                st = time.time()
                
                while cur_score < threshold:
                    print(cur_score)
                    for device in range(self.n_clients):
                        if self.parent_conns[device].poll(timeout=0.1): 
                            res = self.parent_conns[device].recv()
                            state, step, l, label_counts = res
                            labels[device] = label_counts
                            steps[device] = step
                            self.states[device] = state
                            cur_score += 0.5
                            if cur_score >= threshold:
                                break
                
                for device in self.states.keys():
                    keys = self.states[device].keys()
                    for key in keys:
                        self.states[device][key] *= (steps[device] / sum(steps.values()))
                
                weights = []
                score_arr = [[0 for _ in range(len(self.states.keys()))] for _ in range(len(self.states.keys()))] 
                for k1, k2 in enumerate(self.states.keys()):
                    for j1, j2 in enumerate(self.states.keys()):
                        if j2 == k2: continue
                        param1 = self.states[k2]; param2 = self.states[j2]
                        C, L1 = calculate_similarity_and_distance(param1, param2)
                        score = (1 + C) / (1 + L1)
                        score_arr[k1][j1] = score
                
                states, cluster_labels, sil_score = make_clusters(self.states.keys(), self.states, steps, self.min_clusters)
                sil_scores.append(sil_score)
                self.states.clear()
                
                # TODO : 클러스터 별 평가 추가할 것 (가중 평균으로 가야할 듯)
                # f1 score, entropy, data의 양
                cluster_entropies, cluster_datas = calculateEntropy(labels, self.n_class, cluster_labels)
                total_entropies = sum(cluster_entropies)
                total_datas = 0
                for i, data in enumerate(cluster_datas):
                    total_datas += sum(data)    
                    #loss, accuracy, avg_mae, avg_mse, avg_rse, avg_rmse, f1_score = self.valid(states[i])
                    #f1_scores.append(f1_score)
                
                # total_f1_score = sum(f1_scores)
                for idx in range(len(cluster_entropies)):
                    en = cluster_entropies[idx] / total_entropies
                    cd = sum(cluster_datas[idx]) / total_datas
                    # f1 = float(f1_scores[idx].item() / total_f1_score.item())
                    weight =  en + cd 
                    weights.append(weight)
                
                for w in weights:
                    w /= sum(weights)
                
                print(weights)
                
                if len(states) == 0: 
                    print("len(states) == 0")
                    continue
                
                global_model = pickle.loads(self.gRPCClient.sendStates(states, self.args.aggregator, weights).state)
                loss, accuracy, avg_mae, avg_mse, avg_rse, avg_rmse, f1_score = self.valid(global_model)
                best_score = min(best_score, loss)
                print(f"============ Round {round} | Loss {loss} | Accuracy {accuracy} | Time {str(time.time() - s)} Sil {sil_score} ============")
                
                f.write(str(round)+","+str(loss)+","+str(accuracy)+","+str(avg_mae)+","+str(avg_mse)+","+str(avg_rse)+","+str(avg_rmse)+","+str(time.time() - s)+str(sil_score)+"\n")
                self.updateModel(global_model, self.parent_conns)
                
                spend_time = time.time() - st
                times.append(spend_time)
                
                if self.args.early_stopping == 1:
                    early_stopping(loss, global_model, path)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    round = 9999999
                    break
                
                # threshold 변경
                if round > 5:
                    if round % 2 == 0: self.threshold = max(10, self.threshold - 1)
                    t = threshold
                    threshold = max(self.threshold, threshold - loss / (sum(losses) / len(losses)) if best_score == loss else threshold + loss / (sum(losses) / len(losses)))
                    threshold = max(self.threshold, max(self.threshold, threshold - spend_time / (sum(times) / len(times)) if sum(times) / len(times) < spend_time else threshold + spend_time / (sum(times) / len(times))))
                    losses.append(loss)
                    
                    if round % 5 == 0:
                        self.threshold = sum(thresholds) / len(thresholds)
                        
                    if t != threshold:        
                        print(f"Threshold updated. ({t} --> {threshold})")
                
                round += 1
                thresholds.append(threshold)
                
        print(f"Train terminated. | time: {time.time() - s}")
        f.write(f"Train time: {time.time() - s}")
        
        for idx in range(self.n_clients):
            self.processes[idx].terminate()
            self.processes[idx].join()
            
        self.server.terminate()
        self.server.join()
        f.close()
        
        torch.cuda.empty_cache()
        
            
    def valid(self, val_model):
        if self.args.data == "EMNIST":
            model = CNN(1, self.n_class) if self.args.model_name == "cnn" else mobilenet(1, 1, self.n_class)
        elif self.args.data == "cifar10" or "cifar100":
            model = CNN(3, self.n_class) if self.args.model_name == "cnn" else mobilenet(1, 3, self.n_class)
        
        model.to(self.device)
        model.load_state_dict(val_model)
        f1_metric = MulticlassF1Score(num_classes=self.n_class, average='macro').to(self.device)
        
        with torch.no_grad():  # 그라디언트 계산 비활성화
            model.eval()
            criterion = nn.CrossEntropyLoss()
            correct = 0
            total = 0
            total_loss = 0.0
            mae, mse, rse, rmse = 0, 0, 0, 0
            
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 모델 출력 예측
                output = model(data)
                loss = criterion(output, target)
                
                a, b, c, d = metric(torch.argmax(output, dim=-1).cpu(), target.cpu())
                mae += a; mse += b; rse += c; rmse += d

                # 손실 집계
                total_loss += loss.item() * data.size(0)  # 배치 크기로 가중합
                
                # 예측값을 통한 정확도 계산
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # F1 Score 누적
                f1_metric.update(predicted, target) 
        
        # 평균 손실 및 정확도 계산
        avg_loss = total_loss / total
        accuracy = correct / total
        avg_mae = mae / total        
        avg_mse = mse / total        
        avg_rse = rse / total        
        avg_rmse = rmse / total     
        f1_score = f1_metric.compute()

        return avg_loss, accuracy, avg_mae, avg_mse, avg_rse, avg_rmse, f1_score
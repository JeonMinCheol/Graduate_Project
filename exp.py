from client import Client
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch
import random
import pickle
import time


import communication.gRPC as grpc
random.seed(54)

def train_local_client_score():
    pass

# 일반적인 연합학습 방식
def train_local_client_wait(clients, dataset, epochs, client_id, conn):
        train_delay, transfer_delay, batch_size, model, optimizer, loss_fn, host, idx, rounds, device_id = clients[client_id]
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.train()
        data_loader = DataLoader(dataset, batch_size)
        while True:
            for epoch in range(epochs):
                running_loss = 0.0
                for images, labels in data_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    pred = model(images)
                    loss = loss_fn(pred,labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                time.sleep(train_delay) # 딜레이
                
            clients[client_id][-2] += 1 # Iter 횟수 증가
            print(f"Model {client_id} - Rounds [{clients[client_id][-2]}], Loss: {running_loss/len(data_loader)}")
                
            conn.send(list(model.parameters())) # 부모에게 학습된 모델 전송
            
            command = conn.recv()
            
            if command == "terminate":
                conn.close()
                break
            elif command == "update":
                pass
            
class Exp(object):
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.Client = Client(args)
        self.gRPCClient = grpc.gRPCClient("localhost", 50051)
        self.Client._printClients()
        print(args)
        
    def train(self, datasets):
    # 멀티프로세싱 설정 (한 번만 호출)
        clients = self.Client._createClient()
        epoch = self.args.n_client_epochs
        rounds = self.args.rounds
        mp.set_start_method('spawn', force=True)
        processes = []
        parent_conns = []
        
        # 클라이언트별로 프로세스 생성
        for idx in range(self.args.n_clients):
            if self.args.selection == "wait":
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=train_local_client_wait, args=(clients, datasets[idx], epoch, idx, child_conn))
                processes.append(p)
                parent_conns.append(parent_conn)
                 
            elif self.args.selection == "score": # 바꾸기
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=train_local_client_wait, args=(clients, datasets[idx], epoch, idx, child_conn))
                processes.append(p)
                parent_conns.append(parent_conn)
                
            p.start()
            print(f"Client {idx} created! Data: {len(datasets[idx])}")
            
        if self.args.selection == "wait":
            for round in range(self.args.rounds):
                parameters = []
                indice = self.gRPCClient.randomSample(num_clients=self.args.n_clients, num_samples=int(self.args.n_clients * self.args.frac)).device_indices
                
                # 받아온 디바이스가 모델을 업데이트 했는지 확인
                for device in indice:
                    parameter = parent_conns[device].recv()
                    parameters.append(parameter)  # 업데이트 결과 수신
                    parent_conns[device].send("update")
                    
                #if len(models) == len(devices):
                #    pickle.dumps(models)
                    # TODO : grpc로 모델 전송 (gRPC client, server 1개씩 만들기)
                    # TODO : aggregator는 server 1, client 2
                    
                # TODO : 글로벌 모델 받을 때까지 대기
                self.updateModel()
        
        # 모든 프로세스가 끝날 때까지 기다림
        for p in processes:
            p.join()
    
    def _sendModel(self):
        # random selection
        self.random_indice = random.sample(range(0, self.n_clients), int(self.n_clients) * self.frac)
        for idx in self.random_indice:
            transfer_delay = self.clients[idx][1]
            data = pickle.dumps(self.clients[idx][3].state_dict())
            
            # grpc send
            time.sleep(transfer_delay)
            response = self.gRPC.request({"data": data, "topic": self.host, "id": str(idx)})
            if response.status == "200":
                print(f"Client {idx} sends successfully.")
    
    def updateModel(self):
        pass
        # kafka로 전달받음 : model
        # self.random_indice에 있는 디바이스 업데이트
        
        #self.kafkaBroker.deviceReceive()
        #indice = 1
        #model = None
        #for idx in indice:
        #    self.clients[idx][3] = copy.deepcopy(model).to(self.device)
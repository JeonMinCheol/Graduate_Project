from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import torch.nn.functional as F
import pickle
random.seed(54)

# 입력받은 설정의 클라이언트 생성
class Client:
    def __init__(self, args):
        self.args = args
        self.gpu = args.gpu
        self.n_clients = args.n_clients
        self.host = args.host
        self.epochs = args.n_client_epochs
        self.momentum = args.momentum
        self.lr = args.lr
        self.clients = self._createClient()
        self.frac = args.frac
        # self.gRPC = gRPCClient(args.core, 50000, 100 * 1024 * 1024)
        
    def _setup(self):
        setting = []
        # 지연시간, 통신시간, 배치크기
        for i in range(self.n_clients):
            if i < int(self.n_clients * 0.07):
                setting.append([0.5 + random.random(), random.random(), 16])
            elif i < int(self.n_clients * 0.18):
                setting.append([1.1 + random.random(), random.random(), 16])
            elif i < int(self.n_clients * 0.35):
                setting.append([1.8 + random.random(), random.random(), 8])
            elif i < int(self.n_clients * 0.70):
                setting.append([2.5 + random.random(), random.random(), 8])
            else:
                setting.append([3 + random.random(), random.random(), 4])
        return setting

    def _printClients(self):
        for client in self.clients:
            delay, transfer, batch, _, _, _, _, idx, round, _ = client
            print(f"client_id:{idx} | round:{round} | model:{self.args.model_name} | batch:{batch} | optimizer:{self.args.optim} | train delay:{delay} | transfer delay:{transfer}")
    
    def _createClient(self):
        # 지연시간, 통신시간, 배치크기, 모델, 옵티마이저, 로스함수 클라이언트 아이피, 인덱스, 라운드 횟수, gpu 
        settings = self._setup()
        clients = []
        
        for idx in range(self.n_clients):
            setting = settings[idx]
            if self.args.data == "MNIST" or self.args.data == "FashionMNIST":
                if self.args.model_name == "cnn":
                    model = CNN(1, 10)
                elif self.args.model_name == "simple":
                    model = SimpleModel()
                setting.extend([model, optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum), nn.CrossEntropyLoss(), self.host, idx, 0, idx % self.gpu if self.args.use_multiple_gpu else 0])            
            elif self.args.data == "cifar10":
                if self.args.model_name == "cnn":
                    model = CNN(3, 10)
                elif self.args.model_name == "simple":
                    model = SimpleModel()
                    
                setting.extend([model, optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum), nn.CrossEntropyLoss(), self.host, idx, 0, idx % self.gpu if self.args.use_multiple_gpu else 0])
                
            clients.append(setting)
        
        return clients
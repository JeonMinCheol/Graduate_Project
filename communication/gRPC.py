import grpc
from concurrent import futures
import communication.message_pb2_grpc as pb2_grpc
import communication.message_pb2 as pb2
import random
import pickle
import torch              
import torch.nn as nn
import copy
import traceback
from model import CNN, MobileNet    
from utils.metrics import *
from torchmetrics.classification import MulticlassF1Score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# grpc 오류 시 프로세스 종료 후 재실행

class gRPCClient(object):
    def __init__(self, host, server_port, MAX_MESSAGE_LENGTH  = 2000 * 1024 * 1024):
        self.host = host
        self.server_port = server_port
        self.MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH 
        
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(host, server_port),
            options=[
                ('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH)
            ])

        # 스텁 생성
        self.stub = pb2_grpc.grpcServiceStub(self.channel)

    def randomSample(self, num_clients, num_samples):
        request = pb2.RequestRandomIndices(num_clients=num_clients, num_samples=num_samples)
        return self.stub.randomSample(request)
    
    def sendStates(self, states, setting, weights, drop):
        try:
            serialized_states = pickle.dumps(states)
            serialized_weights = pickle.dumps(weights)
            request = pb2.SelectedStates(state=serialized_states, setting=setting, weights=serialized_weights, drop=drop)
            return self.stub.sendState(request)
        except Exception as e:
            print(traceback.format_exc())
    
    def setup(self, data, n_class, model_name):
        request = pb2.clientInformation(data=data, n_class=n_class, model_name=model_name)
        return self.stub.valSetup(request)
    
    def getGlobalModel(self):
        request = pb2.EmptyResponse(message="get global state")
        return self.stub.getGlobalModel(request)
    
class grpcServiceServicer(pb2_grpc.grpcServiceServicer):
    def __init__(self):
        super(grpcServiceServicer, self).__init__()
        self.global_state = None
        self.test_loader = None
        self.model = None
        self.val = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환 (0~255 값을 0~1로 스케일링)
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 평균과 표준편차로 정규화
        ])
    
    def valSetup(self, request, context):
        data, n_class, model_name = request.data, request.n_class, request.model_name
        try:
            if self.test_loader is None and self.model is None:
                if data == "cifar10":
                    test_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=False, download=True, transform=self.transform)
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
                    self.model = CNN(3, n_class) if model_name == "cnn" else MobileNet(1, 3, n_class)
                elif data == "cifar100":
                    test_dataset = datasets.CIFAR100(root='../datasets/cifar100/', train=False, download=True, transform=self.transform)
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
                    self.model = CNN(3, n_class) if model_name == "cnn" else MobileNet(1, 3, n_class)
                elif data == "EMNIST":
                    self.train_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=True, download=True,  transform=transforms.ToTensor())
                    test_dataset = datasets.EMNIST(root='../datasets/EMNIST/', split = 'byclass', train=False, download=True, transform=transforms.ToTensor())
                    self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
                    self.model = CNN(1, n_class) if model_name == "cnn" else MobileNet(1, 1, n_class)
            
                self.f1_metric = MulticlassF1Score(num_classes=n_class, average='macro').to(self.device)
                
        except Exception as e:
            print(e)
            context.set_details('Failed to deserialize states')
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.EmptyResponse(message="Error")
            
        return pb2.EmptyResponse(message="success")
        
    def randomSample(self, request, context):
        # 기존과 동일한 randomSample 구현
        num_clients = request.num_clients
        num_samples = request.num_samples

        if num_samples > num_clients:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('샘플 수는 디바이스 수보다 클 수 없습니다.')
            return pb2.ResponseRandomIndices()

        sampled_indices = random.sample(range(num_clients), num_samples)
        print(sampled_indices)
        return pb2.ResponseRandomIndices(device_indices=sampled_indices)

    def valid(self, val_model):    
        self.model.to(self.device)
        self.model.load_state_dict(val_model)
        
        with torch.no_grad():  # 그라디언트 계산 비활성화
            self.model.eval()
            criterion = nn.CrossEntropyLoss()
            correct = 0
            total = 0
            total_loss = 0.0
            mae, mse, rse, rmse = 0, 0, 0, 0
            
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 모델 출력 예측
                output = self.model(data)
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
                self.f1_metric.update(predicted, target) 
        
        # 평균 손실 및 정확도 계산
        avg_loss = total_loss / total
        accuracy = correct / total
        avg_mae = mae / total        
        avg_mse = mse / total        
        avg_rse = rse / total        
        avg_rmse = rmse / total     
        f1_score = self.f1_metric.compute()

        return {"loss" : avg_loss, "accuracy" : accuracy, "mae" : avg_mae, "mse" : avg_mse, "rse" : avg_rse, "rmse" : avg_rmse, "f1_score": f1_score}
    
    def sendState(self, request, context):
        try:
            # 변수 초기화
            client_states = pickle.loads(request.state)
            weights = pickle.loads(request.weights)
            setting = request.setting
            drop = request.drop
            
            # 모델 템플릿 생성
            model_state = copy.deepcopy(list(client_states.values())[0])
            keys = model_state.keys()
            
            weight_dict = {}
            if setting == "cluster" and drop:
                Beta = 0.1
                drop_size = max(1, int(Beta * len(list(client_states.values()))))
                for device in list(client_states.keys()):
                    val = self.valid(model_state)
                    weight_dict[device] = val["f1_score"]
                
                # 성능이 낮은 일부 디바이스 제거 (기준: Beta) 
                d = sorted(weight_dict.items(), key=lambda x: x[1])
                for _ in range(drop_size):
                    device = d.pop(0)[0]
                    client_states.pop(device)
                    print(f"Drop device {device} | f1_score: {weight_dict[device]}")
            
            # 집계
            for key in keys:
                model_state[key] = torch.zeros_like(model_state[key].to(self.device).float()) 
                for idx in range(len(client_states)):
                    if setting == "fednova": 
                        device = list(client_states.keys())[idx]
                        factor = weights[device] / sum(list(weights.values())) 
                    else: 
                        factor = 1 / len(list(client_states.keys()))
                    
                    model_state[key] += list(client_states.values())[idx][key].to(self.device).float() * factor
            
            # 성능 평가
            val = self.valid(model_state) 
            
            if setting == "cluster":
                if self.global_state is not None and val is not None:
                    best_state = model_state
                    best_loss = val["loss"]
                    if self.val["loss"] < val["loss"]:
                        for a in [0.6, 0.7, 0.8, 0.9]: # a = 0.7 ~ 0.9 이 중 가장 좋은 값으로 선택
                            state = copy.deepcopy(model_state)
                                
                            for key in keys:
                                state[key] = a * self.global_state[key] + (1 - a) * model_state[key]
                        
                            v = self.valid(state)
                            
                            if v["loss"] < best_loss:
                                best_loss = v["loss"]
                                best_state = state
                                val = v
                                
                        model_state = best_state 
                        if self.val["loss"] >= val["loss"]: 
                            self.global_state = model_state
                            self.val = val
                        
                    else:
                        self.global_state = model_state
                        self.val = val
                    
                else:
                        self.global_state = model_state
                        self.val = val
            else:
                self.global_state = model_state
                self.val = val
                
        except Exception as e:
            print(traceback.format_exc())
            context.set_details('Failed to deserialize states')
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.GlobalState(state=b"Error")
        
        return pb2.GlobalState(state=pickle.dumps(self.global_state),loss=val["loss"],accuracy=val["accuracy"],mae=val["mae"],mse=val["mse"],rse=val["rse"],rmse=val["rmse"],f1_score=val["f1_score"])
    
    def getGlobalModel(self, request, context):
        val = self.val
        return pb2.GlobalState(state=pickle.dumps(self.global_state),loss=val["loss"],accuracy=val["accuracy"],mae=val["mae"],mse=val["mse"],rse=val["rse"],rmse=val["rmse"],f1_score=val["f1_score"])
        
# gRPC server
def serve(MAX_MESSAGE_LENGTH, PORT):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ])
    pb2_grpc.add_grpcServiceServicer_to_server(grpcServiceServicer(), server)
    server.add_insecure_port(f'[::]:{PORT}')
    server.start()
    print("server start.")
    server.wait_for_termination()
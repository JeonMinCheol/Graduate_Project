import grpc
from concurrent import futures
import communication.message_pb2_grpc as pb2_grpc
import communication.message_pb2 as pb2
import random
import pickle
import torch              
import copy
import traceback
                
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

    # ================ Request ======================
    # selection=wait 설정에서 디바이스 선택을 위해 사용
    def randomSample(self, num_clients, num_samples):
        request = pb2.RequestRandomIndices(num_clients=num_clients, num_samples=num_samples)
        return self.stub.randomSample(request)
    
    # selection=wait 설정에서 파라미터 전송을 위해 사용
    def sendStates(self, states, setting, weights, cluster_dict, device_dict, update_dict):
        try:
            serialized_states = pickle.dumps(states)
            serialized_weights = pickle.dumps(weights)
            serialized_cluster_dict = pickle.dumps(cluster_dict)
            serialized_device_dict = pickle.dumps(device_dict)
            serialized_update_dict = pickle.dumps(update_dict)
            print(states.keys())
            print(cluster_dict.keys())
            print(device_dict.keys())
            print(update_dict.keys())
            print(weights.keys())
            
            request = pb2.SelectedStates(state=serialized_states, setting=setting, weights=serialized_weights, cluster_dict=serialized_cluster_dict, device_dict=serialized_device_dict, update_dict=serialized_update_dict)
            return self.stub.sendState(request)
        except Exception as e:
            print(traceback.format_exc())
    
    # selection=score 설정에서 파라미터 전송을 위해 사용
    def sendSingleState(self, state):
        if not isinstance(state, bytes):
            raise TypeError("Expected states to be bytes, got {}".format(type(state))
                            )
        request = pb2.SelectedState(state=state)
        return self.stub.sendSingleState(request)
    # ================================================
    
# Response service
class grpcServiceServicer(pb2_grpc.grpcServiceServicer):
    def __init__(self):
        super(grpcServiceServicer, self).__init__()
        self.global_state = None
        
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

    def sendState(self, request, context):
        try:
            client_states = pickle.loads(request.state)
            print(client_states.keys())
            cluster_dict = pickle.loads(request.cluster_dict)
            print(cluster_dict.keys())
            device_dict = pickle.loads(request.device_dict)
            print(device_dict.keys())
            update_dict = pickle.loads(request.update_dict)
            print(update_dict.keys())
            weights = pickle.loads(request.weights)
            print(weights.keys())
            setting = request.setting
            
            if setting == "cluster":
                cluster_steps = {} # cluster: total_steps
                rebalanced_update_dict = {} # 업데이트 순서 반전
                total_update = sum(list(update_dict.values()))
                print(update_dict.values())
                
                # 모델 생성
                model_state = copy.deepcopy(list(client_states.values())[0])
                keys = model_state.keys()
                for key in keys:
                    model_state[key] = torch.zeros_like(model_state[key].cpu().float()) 
                    
                print("# 모델 생성")
                
                # cluster_update_cnt 초기화
                while len(list(rebalanced_update_dict.keys())) < len(list(update_dict.keys())):
                    max_update = -1
                    max_cluster = -1
                    min_update = 9999999
                    min_cluster = 9999999
                    visited = [False for _ in range(len(list(update_dict.keys())) + 1)]
                    
                    for cluster, update in list(update_dict.items()): 
                        print(cluster)
                        if min_update > update and visited[cluster] == False:
                            min_update = update
                            min_cluster = cluster
                            
                        if max_update < update and rebalanced_update_dict.get(max_cluster) is None:
                            max_update = update
                            max_cluster = cluster
                            
                    rebalanced_update_dict[max_cluster] = min_update
                    print(rebalanced_update_dict.items())
                    visited[min_cluster] = True
                print("# cluster_update_cnt 초기화")
                
                # cluster_steps 초기화
                for device, cluster in list(device_dict.items()):
                    if cluster_steps.get(cluster) is None:
                        cluster_steps[cluster] = 0
                    
                    cluster_steps[cluster] = cluster_steps[cluster] + weights[device]
                print("# cluster_steps 초기화")
                
                print(rebalanced_update_dict.keys())
                # Aggregation 진행 (가중치 = device_step * update / (cluster_steps[cluster] * total_update))
                for cluster, devices in list(cluster_dict.items()):
                    for device in list(devices):
                        device_step = weights[device]
                        update = rebalanced_update_dict[cluster]
                        factor = device_step * update / (cluster_steps[cluster] * total_update)
                    
                        for key in keys:
                            model_state[key] += client_states[idx][key].cpu().float() * factor
            else:
                keys = client_states[0].keys()
                model_state = copy.deepcopy(client_states[0])
                
                for key in keys:
                    model_state[key] = torch.zeros_like(model_state[key].cpu().float()) 
                    for idx in range(len(client_states)):
                        if setting == "fednova":
                            factor = weights[idx] / sum(weights)
                        else:
                            factor = 1 / len(client_states)
                        model_state[key] += client_states[idx][key].cpu().float() * factor
                
                self.global_state = model_state
                    
        except Exception as e:
            print(traceback.format_exc())
            context.set_details('Failed to deserialize states')
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.GlobalState(state=b"Error")
        
        new_global_state = pickle.dumps(self.global_state)
        return pb2.GlobalState(state=new_global_state)
        
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
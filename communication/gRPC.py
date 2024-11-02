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
    def sendStates(self, states, setting, weights):
        try:
            serialized_states = pickle.dumps(states)
            request = pb2.SelectedStates(state=serialized_states, setting=setting, weights=weights)
            return self.stub.sendState(request)
        except Exception as e:
            print(setting)
            print(weights)
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
            setting = request.setting
            weights = request.weights
            keys = client_states[0].keys()
            model_state = copy.deepcopy(client_states[0])
            for key in keys:
                model_state[key] = torch.zeros_like(model_state[key].cpu().float()) 
                for idx in range(len(client_states)):
                    factor = 1 / len(client_states) if setting != "fednova" and setting != "cluster" else weights[idx] / sum(weights)
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
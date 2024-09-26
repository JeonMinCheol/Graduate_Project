import grpc
from concurrent import futures
import communication.message_pb2_grpc as pb2_grpc
import communication.message_pb2 as pb2
import random
                                             
class gRPCClient(object):
    def __init__(self, host, server_port, MAX_MESSAGE_LENGTH  = 100 * 1024 * 1024):
        self.host = host
        self.server_port = server_port
        self.MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH # 설정할 최대 메시지 크기 (예: 100MB)
        
        self.channel = grpc.insecure_channel(
            '{}:{}'.format("localhost", 50051),
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
    def sendParameters(self, parameters):
        request = pb2.SelectedParameters(parameters)
        return self.stub.sendParameter(request)
    
    # selection=score 설정에서 파라미터 전송을 위해 사용
    def sendSingleParameter(self, parameter):
        request = pb2.SelectedParameter(parameter)
        return self.stub.sendSingleParameter(request)
    # ================================================
    
# Response service
class grpcServiceServicer(pb2_grpc.grpcServiceServicer):
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

    def sendParameter(self, request, context):
        # 여러 개의 파라미터를 처리
        global_parameters = []
        
        for param in request:
            received_data = param.data
            print(f"서버에서 받은 파라미터: {received_data}")
            global_parameters.append(pb2.GlobalParameter(data=received_data))
        
        # 여러 개의 글로벌 파라미터를 반환
        return pb2.GlobalParameter(parameters=global_parameters)

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
from communication.gRPC import serve
if __name__ == "__main__":
    serve(200 * 1024 * 1024, 50051)
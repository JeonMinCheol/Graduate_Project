from client import Client
from utils import *
from exp import *
from communication.gRPC import serve

if __name__ == '__main__':
    args = arg_parser()
    suffix = "_train_dataset_7.pt"
    exp = Exp(args)
    dataset_list = sample(args.data_root + args.data + suffix, args.n_clients)
    exp.train(dataset_list)
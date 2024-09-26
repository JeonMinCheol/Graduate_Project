import torch
import random
random.seed(54)

from typing import Any, Dict, List
import argparse
import copy
import torch

def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] = weights_avg[key].detach().cpu()
            weights_avg[key] += weights[i][key].detach().cpu()
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=str, default="163.180.117.36")
    parser.add_argument("--host", type=str, default="163.180.117.64")

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--aggregator", type=str, default="fedavg")

    parser.add_argument("--n_clients", type=int, default=14)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--selection", type=str, default="wait")
    parser.add_argument("--n_client_epochs", type=int, default=10)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--use_multiple_gpu', action='store_true', help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpus')
    
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    return args

def loadDataset(path):
    dataset = torch.load(path)
    
    if isinstance(dataset, (tuple, list)):
        # 예시: (data, targets)로 저장된 경우
        data, targets = dataset
        dataset = torch.utils.data.TensorDataset(data, targets)
        print(f"Dataset Length: {len(dataset)}")
    return dataset

def sample(path, n_clients):
    data = loadDataset(path)
    
    Dataset = list(data)
    indice = list(range(len(data)))
    samples = list()
    num_data = len(Dataset) / (n_clients * 10)
    
    for _ in range(n_clients):
        x = []
        k = int(num_data * random.randrange(6, 11))
        random_indice = random.sample(indice, k)
        
        for index in random_indice:
            x.append(Dataset[index])
            indice.remove(index)
        
        samples.append(x)
    return samples
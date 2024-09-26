from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from data import *
from graduate_project.client.model import CNN, MLP
from utils import arg_parser, average_weights

# 서버는 집계 후 모델 추론만 (학습 X)

class FedAvg:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.args.data == "mnist":
            self.train_loaders, self.test_loaders, self.c_delays, self.t_delays, self.devices = get_data(
                root=self.args.data_root,
                n_clients=self.args.n_clients,
                datasets= 
            )
        elif self.args.data == "fashion_mnist":
            self.train_loaders, self.test_loaders, self.c_delays, self.t_delays, self.devices = get_data(
                root=self.args.data_root,
                n_clients=self.args.n_clients,
                datasets=FashionMNISTDataset
            )
        elif self.args.data == "cifar10":
            self.train_loaders, self.test_loaders, self.c_delays, self.t_delays, self.devices = get_data(
                root=self.args.data_root,
                n_clients=self.args.n_clients,
                datasets=CIFAR10Dataset
            )
        else:
            raise ValueError(f"Invalid data name, {self.args.data}")
        
        if self.args.model_name == "mlp":
            self.root_model = MLP(input_size=784, hidden_size=128, n_classes=10).to(self.device)
            self.target_acc = 0.97
        elif self.args.model_name == "cnn":
            if self.args.data == "cifar10":
                self.root_model = CNN(n_channels=3, n_classes=10).to(self.device)
            else:
                self.root_model = CNN(n_channels=1, n_classes=10).to(self.device)
            self.target_acc = 0.99
        else:
            raise ValueError(f"Invalid model name, {self.args.model_name}")

        self.reached_target_at = None  # type: int
        
    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """
        model = copy.deepcopy(root_model).to(self.devices[client_idx])
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

        for epoch in range(self.args.n_client_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.devices[client_idx]), target.to(self.devices[client_idx])
                optimizer.zero_grad()

                logits = model(data)
                
                loss = F.nll_loss(logits, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= idx
            epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.args.n_client_epochs} | Loss: {epoch_loss} | Acc: {epoch_acc}",
                end="\r",
            )
            
            # 시간 지연 반영            
            # time.sleep(self.c_delays[client_idx] + self.t_delays[client_idx])

        return model, epoch_loss / self.args.n_client_epochs

    def train(self) -> None:
        f = open(f"./metrics/{self.args.data}_fedavg.csv", "w")
        f.write("round,test/acc,train/loss,test/loss")
        
        """Train a server model."""
        train_losses = []

        for epoch in range(self.args.n_epochs):
            clients_models = []
            clients_losses = []

            # Randomly select clients
            m = max(int(self.args.frac * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            # Train clients
            self.root_model.train()

            for client_idx in idx_clients:
                # Set client in the sampler
                trainloader = self.train_loaders[client_idx]

                # Train client
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=trainloader,
                    client_idx=client_idx,
                )
                
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)

            # Update server model based on clients models
            updated_weights = average_weights(clients_models)
            self.root_model.load_state_dict(updated_weights)

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.test()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "test/loss": total_loss,
                    "test/acc": total_acc,
                    "round": epoch,
                }
                
                f.write(f"{epoch},{total_acc},{avg_train_loss},{total_loss}\n") 
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print(
                        f"\n -----> Target accuracy {self.target_acc} reached at round {epoch}! <----- \n"
                    )


                # Print results to CLI
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
                print(
                    f"---> Avg Test Loss: {total_loss} | Avg Test Accuracy: {total_acc}\n"
                )

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch}...")
                    break
                
        f.close()
                
    def test(self) -> Tuple[float, float]:
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for idx, (data, target) in enumerate(self.test_loaders[-1]):
            data, target = data.to(self.device), target.to(self.device)

            logits = self.root_model(data)
            loss = F.nll_loss(logits, target)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

        # calculate average accuracy and loss
        total_loss /= idx
        total_acc = total_correct / total_samples

        return total_loss, total_acc